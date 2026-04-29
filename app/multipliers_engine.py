import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from top4_engine import (
    STRAT_FIELDS,
    clean_text_cols,
    normalize_columns,
    predict_target_profile,
    train_top4_engine,
)
from base_price_engine import predict_base_prices, train_base_price_engine


ARTIFACT_NAME = "multipliers_engine.pkl"

DEMAND_COLS = [
    "interested_team_count",
    "max_fit_score",
    "mean_fit_score",
    "sum_fit_score",
]

SCARCITY_COLS = [
    "role_pool_size",
    "better_role_options",
    "quality_pct_in_role",
    "demand_to_role_ratio",
]

PURSE_COLS = [
    "avg_interested_purse",
    "max_interested_purse",
    "high_purse_team_count",
    "total_slots_left",
]


def clean_purse_df(df):
    df = normalize_columns(df)
    df = clean_text_cols(df, ["team"])
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["purse"] = pd.to_numeric(df["purse"], errors="coerce").fillna(0.0)
    return df


def load_historical_inputs(data_dir):
    sold = normalize_columns(pd.read_csv(Path(data_dir) / "auction_sold_final_csv.csv"))
    unsold = normalize_columns(pd.read_csv(Path(data_dir) / "auction_unsold_final_csv.csv"))
    ret = normalize_columns(pd.read_csv(Path(data_dir) / "retentions_normalized.csv"))
    purse = clean_purse_df(pd.read_csv(Path(data_dir) / "purse_final.csv"))

    sold = clean_text_cols(sold, ["player", "team", "type", "nationality"])
    unsold = clean_text_cols(unsold, ["player", "type", "nationality"])
    ret = clean_text_cols(ret, ["player", "team", "type", "nationality"])

    sold["season"] = pd.to_numeric(sold["season"], errors="coerce").astype("Int64")
    sold["price"] = pd.to_numeric(sold["price"], errors="coerce")
    unsold["season"] = pd.to_numeric(unsold["season"], errors="coerce").astype("Int64")
    ret["season"] = pd.to_numeric(ret["season"], errors="coerce").astype("Int64")

    return sold, unsold, ret, purse


def get_snapshot_before_year(master_df, year):
    snapshot = (
        master_df[master_df["season"] < year]
        .sort_values("season")
        .drop_duplicates("player", keep="last")
        .copy()
    )
    snapshot["player"] = snapshot["player"].astype(str).str.strip()
    return snapshot


def build_team_states_for_year(year, retentions_df, purse_df, master_df, top4_bundle):
    ret_y = retentions_df[retentions_df["season"] == year].copy()
    purse_y = purse_df[purse_df["season"] == year].copy()
    snapshot = get_snapshot_before_year(master_df, year)

    ret_merged = pd.merge(
        ret_y,
        snapshot.drop(columns=["team"], errors="ignore"),
        on="player",
        how="left",
    )

    target_profile = predict_target_profile(top4_bundle, year, use_history_only=True)
    teams = sorted(set(ret_y["team"].dropna().unique()) | set(purse_y["team"].dropna().unique()))
    states = {}

    for team in teams:
        team_data = ret_merged[ret_merged["team"] == team].drop_duplicates("player")
        current_profile = team_data[STRAT_FIELDS].mean().fillna(0.0) if not team_data.empty else pd.Series({f: 0.0 for f in STRAT_FIELDS})
        gap = {f: max(0.0, float(target_profile[f] - current_profile[f])) for f in STRAT_FIELDS}
        purse = float(purse_y.loc[purse_y["team"] == team, "purse"].sum())
        retained_count = len(team_data)

        states[team] = {
            "team": team,
            "retained_count": retained_count,
            "slots_left": max(0, 25 - retained_count),
            "purse": purse,
            "gap": gap,
        }

    return states


def build_auction_pool_for_year(year, sold_df, unsold_df, master_df):
    sold_y = sold_df[sold_df["season"] == year].copy()
    unsold_y = unsold_df[unsold_df["season"] == year].copy()

    sold_y["actual_price_cr"] = sold_y["price"].fillna(0.0) / 1e7
    sold_y["was_sold"] = 1
    unsold_y["actual_price_cr"] = 0.0
    unsold_y["was_sold"] = 0

    sold_y = sold_y[["player", "type", "nationality", "actual_price_cr", "was_sold"]]
    unsold_y = unsold_y[["player", "type", "nationality", "actual_price_cr", "was_sold"]]

    pool = pd.concat([sold_y, unsold_y], ignore_index=True)
    pool = pool.drop_duplicates("player", keep="first")

    snapshot = get_snapshot_before_year(master_df, year)
    pool = pd.merge(
        pool,
        snapshot.drop(columns=["team"], errors="ignore"),
        on="player",
        how="left",
    )

    for field in STRAT_FIELDS:
        if field not in pool.columns:
            pool[field] = 0.0

    pool["type"] = pool["type"].fillna("Unknown")
    pool["quality_score"] = pool[STRAT_FIELDS].mean(axis=1).fillna(0.0)
    return pool


def compute_market_features(pool_df, team_states):
    pool_df = pool_df.copy()
    role_counts = pool_df["type"].fillna("Unknown").value_counts().to_dict()
    purses = [state["purse"] for state in team_states.values() if state["purse"] > 0]
    purse_cutoff = float(np.median(purses)) if purses else 0.0
    total_slots_left = int(sum(state["slots_left"] for state in team_states.values()))

    rows = []
    for _, row in pool_df.iterrows():
        fits = []
        interested_purses = []

        for state in team_states.values():
            if state["slots_left"] <= 0:
                continue

            fit_score = 0.0
            for field in STRAT_FIELDS:
                fit_score += max(0.0, state["gap"][field]) * (float(row.get(field, 0.0) or 0.0) / 100.0)

            if fit_score > 0:
                fits.append(fit_score)
                interested_purses.append(state["purse"])

        role = row.get("type", "Unknown")
        role_frame = pool_df[pool_df["type"] == role]
        role_pool_size = int(role_counts.get(role, len(pool_df)))
        better_role_options = int((role_frame["quality_score"] > row["quality_score"]).sum())
        quality_pct = 1.0 - (better_role_options / max(1, len(role_frame)))
        interested_team_count = len(fits)

        rows.append({
            "interested_team_count": interested_team_count,
            "max_fit_score": float(max(fits)) if fits else 0.0,
            "mean_fit_score": float(np.mean(fits)) if fits else 0.0,
            "sum_fit_score": float(np.sum(fits)) if fits else 0.0,
            "avg_interested_purse": float(np.mean(interested_purses)) if interested_purses else 0.0,
            "max_interested_purse": float(max(interested_purses)) if interested_purses else 0.0,
            "high_purse_team_count": int(sum(p > purse_cutoff for p in interested_purses)) if interested_purses else 0,
            "role_pool_size": role_pool_size,
            "better_role_options": better_role_options,
            "quality_pct_in_role": float(quality_pct),
            "demand_to_role_ratio": float(interested_team_count / max(1, role_pool_size)),
            "total_slots_left": total_slots_left,
        })

    return pd.concat([pool_df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def fit_rf(train_x, train_y):
    if len(train_x) < 20:
        return None
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2,
    )
    model.fit(train_x, train_y)
    return model


def build_multiplier_training_rows(data_dir, output_dir):
    top4_bundle = train_top4_engine(data_dir, output_dir)
    base_bundle = train_base_price_engine(data_dir, output_dir)
    master_df = top4_bundle["processed_ml_master"].copy()
    sold_df, unsold_df, ret_df, purse_df = load_historical_inputs(data_dir)

    rows = []
    seasons = sorted(sold_df["season"].dropna().astype(int).unique())
    for year in seasons:
        if year <= 2016:
            continue

        team_states = build_team_states_for_year(year, ret_df, purse_df, master_df, top4_bundle)
        if not team_states:
            continue

        pool = build_auction_pool_for_year(year, sold_df, unsold_df, master_df)
        pool = predict_base_prices(pool, base_bundle, year)
        pool = compute_market_features(pool, team_states)

        effective_actual = pool["actual_price_cr"].fillna(0.0)
        pool["log_multiplier_target"] = np.log((effective_actual + 0.1) / (pool["base_price_cr"] + 0.1))
        pool["auction_year"] = year
        rows.append(pool)

    if not rows:
        return top4_bundle, base_bundle, pd.DataFrame()

    return top4_bundle, base_bundle, pd.concat(rows, ignore_index=True)


def train_multiplier_engine(data_dir, output_dir=None):
    output_dir = Path(output_dir or Path(data_dir).parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    top4_bundle, base_bundle, training_rows = build_multiplier_training_rows(data_dir, output_dir)

    models_by_target_year = {}
    study_summary = []

    if not training_rows.empty:
        all_years = sorted(training_rows["auction_year"].dropna().astype(int).unique())
        for target_year in all_years[1:] + [all_years[-1] + 1]:
            train = training_rows[training_rows["auction_year"] < target_year].copy()
            valid = training_rows[training_rows["auction_year"] == target_year].copy()

            demand_model = fit_rf(train[DEMAND_COLS].fillna(0.0), train["log_multiplier_target"]) if not train.empty else None
            demand_pred = demand_model.predict(train[DEMAND_COLS].fillna(0.0)) if demand_model is not None else np.zeros(len(train))
            scarcity_target = train["log_multiplier_target"] - demand_pred

            scarcity_model = fit_rf(train[SCARCITY_COLS].fillna(0.0), scarcity_target) if not train.empty else None
            scarcity_pred = scarcity_model.predict(train[SCARCITY_COLS].fillna(0.0)) if scarcity_model is not None else np.zeros(len(train))
            purse_target = scarcity_target - scarcity_pred

            purse_model = fit_rf(train[PURSE_COLS].fillna(0.0), purse_target) if not train.empty else None

            metrics = None
            if not valid.empty and demand_model is not None:
                pred = demand_model.predict(valid[DEMAND_COLS].fillna(0.0))
                if scarcity_model is not None:
                    pred += scarcity_model.predict(valid[SCARCITY_COLS].fillna(0.0))
                if purse_model is not None:
                    pred += purse_model.predict(valid[PURSE_COLS].fillna(0.0))

                pred_price = (valid["base_price_cr"] + 0.1) * np.exp(pred) - 0.1
                actual = valid["actual_price_cr"].fillna(0.0).to_numpy()
                mae = float(np.mean(np.abs(pred_price - actual)))
                rmse = float(np.sqrt(np.mean((pred_price - actual) ** 2)))
                metrics = {"mae_cr": round(mae, 4), "rmse_cr": round(rmse, 4)}

            models_by_target_year[target_year] = {
                "demand_model": demand_model,
                "scarcity_model": scarcity_model,
                "purse_model": purse_model,
            }
            study_summary.append({
                "target_year": int(target_year),
                "train_rows": int(len(train)),
                "valid_rows": int(len(valid)),
                "metrics": metrics,
            })

        latest_target_year = all_years[-1] + 1
        final_models = models_by_target_year.get(latest_target_year, {})
    else:
        latest_target_year = None
        final_models = {}

    bundle = {
        "demand_cols": DEMAND_COLS,
        "scarcity_cols": SCARCITY_COLS,
        "purse_cols": PURSE_COLS,
        "training_rows": training_rows,
        "models_by_target_year": models_by_target_year,
        "study_summary": study_summary,
        "final_models": final_models,
        "latest_target_year": latest_target_year,
    }

    with open(output_dir / ARTIFACT_NAME, "wb") as f:
        pickle.dump(bundle, f)

    return bundle


def load_multiplier_bundle(output_dir):
    with open(Path(output_dir) / ARTIFACT_NAME, "rb") as f:
        return pickle.load(f)


def apply_multiplier_models(pool_df, multiplier_bundle, target_year):
    pool_df = pool_df.copy()
    models = multiplier_bundle["models_by_target_year"].get(target_year, multiplier_bundle.get("final_models", {}))

    demand_model = models.get("demand_model")
    scarcity_model = models.get("scarcity_model")
    purse_model = models.get("purse_model")

    if demand_model is None:
        pool_df["demand_multiplier"] = 1.0
        pool_df["scarcity_multiplier"] = 1.0
        pool_df["purse_multiplier"] = 1.0
        pool_df["final_price_cr"] = pool_df["base_price_cr"].fillna(0.0)
        return pool_df

    demand_log = demand_model.predict(pool_df[DEMAND_COLS].fillna(0.0))
    scarcity_log = scarcity_model.predict(pool_df[SCARCITY_COLS].fillna(0.0)) if scarcity_model is not None else np.zeros(len(pool_df))
    purse_log = purse_model.predict(pool_df[PURSE_COLS].fillna(0.0)) if purse_model is not None else np.zeros(len(pool_df))

    pool_df["demand_multiplier"] = np.exp(demand_log).clip(0.5, 3.0)
    pool_df["scarcity_multiplier"] = np.exp(scarcity_log).clip(0.5, 2.5)
    pool_df["purse_multiplier"] = np.exp(purse_log).clip(0.5, 2.5)
    pool_df["final_price_cr"] = (
        pool_df["base_price_cr"].fillna(0.0)
        * pool_df["demand_multiplier"]
        * pool_df["scarcity_multiplier"]
        * pool_df["purse_multiplier"]
    ).clip(lower=0.0)

    return pool_df


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    bundle = train_multiplier_engine(data_dir, base_dir)
    print(f"Saved {ARTIFACT_NAME} for target year {bundle['latest_target_year']}")
