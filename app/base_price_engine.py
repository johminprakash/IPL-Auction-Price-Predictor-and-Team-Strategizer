import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from top4_engine import STRAT_FIELDS, clean_text_cols, normalize_columns, train_top4_engine


ARTIFACT_NAME = "base_price_engine.pkl"


def load_raw_inputs(data_dir):
    sold = normalize_columns(pd.read_csv(Path(data_dir) / "auction_sold_final_csv.csv"))
    sold = clean_text_cols(sold, ["player", "team", "type", "nationality"])
    sold["season"] = pd.to_numeric(sold["season"], errors="coerce").astype("Int64")
    sold["price"] = pd.to_numeric(sold["price"], errors="coerce")
    return sold


def get_snapshot_before_year(master_df, year):
    snapshot = (
        master_df[master_df["season"] < year]
        .sort_values("season")
        .drop_duplicates("player", keep="last")
        .copy()
    )
    snapshot["player"] = snapshot["player"].astype(str).str.strip()
    return snapshot


def build_base_training_rows(master_df, sold_df):
    rows = []
    years = sorted(pd.to_numeric(sold_df["season"], errors="coerce").dropna().astype(int).unique())

    for year in years:
        sold_year = sold_df[sold_df["season"] == year].copy()
        snapshot = get_snapshot_before_year(master_df, year)

        merged = pd.merge(
            sold_year,
            snapshot.drop(columns=["team"], errors="ignore"),
            on="player",
            how="left",
        )

        for field in STRAT_FIELDS:
            if field not in merged.columns:
                merged[field] = 0.0

        merged["auction_year"] = year
        merged["price_cr"] = merged["price"].fillna(0.0) / 1e7
        rows.append(merged)

    if not rows:
        return pd.DataFrame()

    base_rows = pd.concat(rows, ignore_index=True)
    return base_rows


def fit_model(train_df):
    if len(train_df) < 20:
        return None

    x_train = train_df[STRAT_FIELDS].fillna(0.0)
    y_train = np.log1p(train_df["price_cr"].clip(lower=0.0))

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2,
    )
    model.fit(x_train, y_train)
    return model


def score_model(model, valid_df):
    if model is None or valid_df.empty:
        return None

    preds = np.expm1(model.predict(valid_df[STRAT_FIELDS].fillna(0.0)))
    actual = valid_df["price_cr"].fillna(0.0).to_numpy()
    mae = float(np.mean(np.abs(preds - actual)))
    rmse = float(np.sqrt(np.mean((preds - actual) ** 2)))
    return {"mae_cr": round(mae, 4), "rmse_cr": round(rmse, 4)}


def train_base_price_engine(data_dir, output_dir=None):
    output_dir = Path(output_dir or Path(data_dir).parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    top4_bundle = train_top4_engine(data_dir, output_dir)
    master_df = top4_bundle["processed_ml_master"].copy()
    sold_df = load_raw_inputs(data_dir)

    base_rows = build_base_training_rows(master_df, sold_df)
    study_models = {}
    study_summary = []

    if base_rows.empty:
        bundle = {
            "feature_columns": STRAT_FIELDS,
            "training_rows": base_rows,
            "models_by_target_year": study_models,
            "study_summary": study_summary,
            "final_model": None,
            "latest_target_year": None,
        }
        with open(output_dir / ARTIFACT_NAME, "wb") as f:
            pickle.dump(bundle, f)
        return bundle

    observed_years = sorted(base_rows["auction_year"].dropna().astype(int).unique())

    for target_year in observed_years[1:] + [observed_years[-1] + 1]:
        train_df = base_rows[base_rows["auction_year"] < target_year].copy()
        valid_df = base_rows[base_rows["auction_year"] == target_year].copy()

        model = fit_model(train_df)
        study_models[target_year] = model
        metrics = score_model(model, valid_df)

        study_summary.append({
            "target_year": int(target_year),
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
            "metrics": metrics,
        })

    latest_target_year = observed_years[-1] + 1
    final_model = study_models.get(latest_target_year)

    bundle = {
        "feature_columns": STRAT_FIELDS,
        "training_rows": base_rows,
        "models_by_target_year": study_models,
        "study_summary": study_summary,
        "final_model": final_model,
        "latest_target_year": latest_target_year,
    }

    with open(output_dir / ARTIFACT_NAME, "wb") as f:
        pickle.dump(bundle, f)

    return bundle


def load_base_price_bundle(output_dir):
    with open(Path(output_dir) / ARTIFACT_NAME, "rb") as f:
        return pickle.load(f)


def predict_base_prices(pool_df, base_bundle, target_year):
    pool_df = pool_df.copy()
    model = base_bundle["models_by_target_year"].get(target_year, base_bundle.get("final_model"))

    if model is None:
        pool_df["base_price_cr"] = pool_df[STRAT_FIELDS].mean(axis=1).fillna(0.0) / 10.0
        return pool_df

    pred_log = model.predict(pool_df[STRAT_FIELDS].fillna(0.0))
    pool_df["base_price_cr"] = np.expm1(pred_log).clip(min=0.0)
    return pool_df


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    bundle = train_base_price_engine(data_dir, base_dir)
    print(f"Saved {ARTIFACT_NAME} for target year {bundle['latest_target_year']}")