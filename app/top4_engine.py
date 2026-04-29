import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression


STRAT_FIELDS = [
    "experience",
    "batting intent",
    "batting consistency",
    "pace wicket taker",
    "pace economy",
    "spin wicket taker",
    "spin economy",
]

ARTIFACT_NAME = "top4_engine.pkl"


def normalize_columns(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    rename_map = {}
    for col in df.columns:
        key = col.strip().lower()
        if "player name" == key:
            rename_map[col] = "player"
        elif key == "player":
            rename_map[col] = "player"
        elif key == "team":
            rename_map[col] = "team"
        elif key == "season" or key == "year":
            rename_map[col] = "season"
        elif key == "play type" or key == "type":
            rename_map[col] = "type"
        elif "winning bid" == key or key == "price" or key == "bid":
            rename_map[col] = "price"
        elif "nationality" in key:
            rename_map[col] = "nationality"

    return df.rename(columns=rename_map)


def clean_text_cols(df, cols):
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def load_csv(data_dir, filename):
    path = Path(data_dir) / filename
    df = pd.read_csv(path)
    df = normalize_columns(df)
    return df


def load_and_clean_stats(data_dir):
    stats_df = load_csv(data_dir, "ipl_stats_normalized.csv")
    stats_df = clean_text_cols(stats_df, ["player", "team"])

    numeric_cols = [
        "season",
        "sr_bat",
        "avg_bat",
        "wickets_bowl",
        "economy_bowl",
    ]
    for col in numeric_cols:
        if col in stats_df.columns:
            stats_df[col] = pd.to_numeric(stats_df[col], errors="coerce")

    stats_df["season"] = stats_df["season"].fillna(0).astype(int)
    return stats_df


def load_role_map(data_dir):
    sold_df = load_csv(data_dir, "auction_sold_final_csv.csv")
    unsold_df = load_csv(data_dir, "auction_unsold_final_csv.csv")

    sold_df = clean_text_cols(sold_df, ["player", "type", "nationality"])
    unsold_df = clean_text_cols(unsold_df, ["player", "type", "nationality"])

    role_frames = []
    for frame in [sold_df, unsold_df]:
        if "player" in frame.columns and "type" in frame.columns:
            role_frames.append(frame[["player", "type"]].dropna(subset=["player"]))

    role_map = {}
    if role_frames:
        role_map = (
            pd.concat(role_frames, ignore_index=True)
            .drop_duplicates("player", keep="last")
            .set_index("player")["type"]
            .to_dict()
        )

    return role_map


def build_master_dataframe(stats_df, role_map):
    final_rows = []
    seasons = sorted(stats_df["season"].dropna().unique())

    for year in seasons:
        context = stats_df[stats_df["season"] <= year].copy()
        current_year = stats_df[stats_df["season"] == year].copy()

        if context.empty or current_year.empty:
            continue

        benchmarks = {
            "exp": context.groupby("player")["season"].count().quantile(0.95) or 1,
            "intent": context["sr_bat"].quantile(0.95) or 140,
            "consist": context["avg_bat"].quantile(0.95) or 30,
            "wkt": context["wickets_bowl"].quantile(0.95) or 10,
            "eco_elite": context.loc[context["economy_bowl"] > 0, "economy_bowl"].quantile(0.05) or 6.5,
        }

        for _, row in current_year.iterrows():
            player = row["player"]
            player_type = str(role_map.get(player, "all-rounder")).lower()

            out = {field: 0.0 for field in STRAT_FIELDS}
            out.update({
                "player": player,
                "season": int(year),
                "team": row.get("team", ""),
            })

            player_history_len = len(context[context["player"] == player])
            out["experience"] = min(100, (player_history_len / max(benchmarks["exp"], 1)) * 100)

            sr_bat = float(row.get("sr_bat", 0) or 0)
            avg_bat = float(row.get("avg_bat", 0) or 0)
            wickets = float(row.get("wickets_bowl", 0) or 0)
            eco = float(row.get("economy_bowl", 0) or 0)

            out["batting intent"] = min(100, (sr_bat / max(benchmarks["intent"], 1)) * 100) if sr_bat > 0 else 0
            out["batting consistency"] = min(100, (avg_bat / max(benchmarks["consist"], 1)) * 100) if avg_bat > 0 else 0

            if "all-rounder" in player_type or "pace" in player_type:
                out["pace wicket taker"] = min(100, (wickets / max(benchmarks["wkt"], 1)) * 100) if wickets > 0 else 0
                if eco > 0:
                    out["pace economy"] = min(100, (benchmarks["eco_elite"] / eco) * 100)

            if "all-rounder" in player_type or "spin" in player_type:
                out["spin wicket taker"] = min(100, (wickets / max(benchmarks["wkt"], 1)) * 100) if wickets > 0 else 0
                if eco > 0:
                    out["spin economy"] = min(100, (benchmarks["eco_elite"] / eco) * 100)

            if "batter" in player_type or "wk" in player_type:
                out["pace economy"] = out["pace economy"] if out["pace economy"] > 0 else 0
                out["spin economy"] = out["spin economy"] if out["spin economy"] > 0 else 0

            final_rows.append(out)

    return pd.DataFrame(final_rows)


def build_top4_profiles(data_dir, master_df):
    standings_df = pd.read_csv(Path(data_dir) / "standings.csv")
    standings_df.columns = [str(col).strip() for col in standings_df.columns]

    standings_long = standings_df.melt(
        id_vars=standings_df.columns[0],
        var_name="season",
        value_name="team",
    )
    standings_long.columns = ["rank", "season", "team"]
    standings_long["season"] = pd.to_numeric(standings_long["season"], errors="coerce")
    standings_long["rank"] = pd.to_numeric(standings_long["rank"], errors="coerce")
    standings_long["team"] = standings_long["team"].astype(str).str.strip()

    team_profiles = (
        master_df.groupby(["season", "team"])[STRAT_FIELDS]
        .mean()
        .reset_index()
    )

    top4 = pd.merge(
        standings_long[standings_long["rank"] <= 4],
        team_profiles,
        on=["season", "team"],
        how="left",
    ).sort_values(["season", "rank"])

    return top4


def fit_target_models(top4_profiles):
    yearly = (
        top4_profiles.groupby("season")[STRAT_FIELDS]
        .mean()
        .reset_index()
        .sort_values("season")
    )

    models = {}
    for field in STRAT_FIELDS:
        model = LinearRegression()
        model.fit(yearly[["season"]], yearly[field])
        models[field] = model

    return yearly, models


def predict_target_profile(bundle, target_year, use_history_only=True):
    yearly = bundle["top4_yearly_mean"]
    train = yearly.copy()
    if use_history_only:
        train = train[train["season"] < target_year].copy()

    target = {}
    for field in STRAT_FIELDS:
        if len(train) >= 2:
            model = LinearRegression()
            model.fit(train[["season"]], train[field])
            target[field] = float(model.predict([[target_year]])[0])
        elif len(train) == 1:
            target[field] = float(train.iloc[0][field])
        else:
            target[field] = 0.0

    return target


def build_top4_bundle(data_dir):
    stats_df = load_and_clean_stats(data_dir)
    role_map = load_role_map(data_dir)
    master_df = build_master_dataframe(stats_df, role_map)
    top4_profiles = build_top4_profiles(data_dir, master_df)
    yearly_mean, _ = fit_target_models(top4_profiles)

    bundle = {
        "strat_fields": STRAT_FIELDS,
        "processed_ml_master": master_df,
        "top4_profiles": top4_profiles,
        "top4_yearly_mean": yearly_mean,
        "available_seasons": sorted(master_df["season"].dropna().unique().tolist()),
    }
    return bundle


def save_top4_bundle(bundle, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle["processed_ml_master"].to_csv(output_dir / "processed_ml_master.csv", index=False)
    bundle["top4_profiles"].to_csv(output_dir / "top4_profiles.csv", index=False)

    with open(output_dir / ARTIFACT_NAME, "wb") as f:
        pickle.dump(bundle, f)


def load_top4_bundle(output_dir):
    with open(Path(output_dir) / ARTIFACT_NAME, "rb") as f:
        return pickle.load(f)


def train_top4_engine(data_dir, output_dir=None):
    output_dir = output_dir or data_dir
    bundle = build_top4_bundle(data_dir)
    save_top4_bundle(bundle, output_dir)
    return bundle


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    bundle = train_top4_engine(data_dir, base_dir)
    print(f"Saved {ARTIFACT_NAME} with seasons: {bundle['available_seasons']}")
