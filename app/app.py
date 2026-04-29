from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import os
import subprocess

from top4_engine import (
    STRAT_FIELDS,
    clean_text_cols,
    load_top4_bundle,
    normalize_columns,
    predict_target_profile,
    train_top4_engine,
)
from base_price_engine import load_base_price_bundle, predict_base_prices, train_base_price_engine
from multipliers_engine import (
    apply_multiplier_models,
    clean_purse_df,
    compute_market_features,
    load_multiplier_bundle,
    train_multiplier_engine,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


st.set_page_config(layout="wide", page_title="IPL 2026 Strategist")
st.title("IPL 2026 Strategist")


def ensure_artifacts():
    top4_path = BASE_DIR / "top4_engine.pkl"
    base_path = BASE_DIR / "base_price_engine.pkl"
    mult_path = BASE_DIR / "multipliers_engine.pkl"

    status = st.empty()  # placeholder

    if not (top4_path.exists() and base_path.exists() and mult_path.exists()):
        status.write("⏳ Training models... please wait (first run only)")

        if not top4_path.exists():
            train_top4_engine(DATA_DIR, BASE_DIR)
        if not base_path.exists():
            train_base_price_engine(DATA_DIR, BASE_DIR)
        if not mult_path.exists():
            train_multiplier_engine(DATA_DIR, BASE_DIR)

        status.empty()  # remove message after done

    return (
        load_top4_bundle(BASE_DIR),
        load_base_price_bundle(BASE_DIR),
        load_multiplier_bundle(BASE_DIR),
    )

TOP4_BUNDLE, BASE_BUNDLE, MULT_BUNDLE = ensure_artifacts()
DF_MASTER = TOP4_BUNDLE["processed_ml_master"].copy()
DF_TOP4 = TOP4_BUNDLE["top4_profiles"].copy()


def infer_input_year(retention_df, purse_df):
    years = []
    if "season" in retention_df.columns:
        years.extend(pd.to_numeric(retention_df["season"], errors="coerce").dropna().astype(int).tolist())
    if "season" in purse_df.columns:
        years.extend(pd.to_numeric(purse_df["season"], errors="coerce").dropna().astype(int).tolist())

    if years:
        return int(max(years))

    hist_years = DF_MASTER["season"].dropna().astype(int)
    return int(hist_years.max() + 1)


def get_snapshot_before_year(year):
    snapshot = (
        DF_MASTER[DF_MASTER["season"] < year]
        .sort_values("season")
        .drop_duplicates("player", keep="last")
        .copy()
    )
    snapshot["player"] = snapshot["player"].astype(str).str.strip()
    return snapshot

def build_team_states(year, retention_df, purse_df):
    retention_df = retention_df.copy()
    purse_df = purse_df.copy()

    if "season" in retention_df.columns:
        retention_df = retention_df[pd.to_numeric(retention_df["season"], errors="coerce") == year].copy()
    if "season" in purse_df.columns:
        purse_df = purse_df[pd.to_numeric(purse_df["season"], errors="coerce") == year].copy()

    snapshot = get_snapshot_before_year(year)
    target_profile = predict_target_profile(TOP4_BUNDLE, year, use_history_only=True)

    retention_merged = pd.merge(
        retention_df,
        snapshot.drop(columns=["team"], errors="ignore"),
        on="player",
        how="left",
    )

    teams = sorted(set(retention_df["team"].dropna().unique()) | set(purse_df["team"].dropna().unique()))
    states = {}

    for team in teams:
        team_data = retention_merged[retention_merged["team"] == team].drop_duplicates("player")
        current_profile = team_data[STRAT_FIELDS].mean().fillna(0.0) if not team_data.empty else pd.Series({f: 0.0 for f in STRAT_FIELDS})
        gap = {f: max(0.0, float(target_profile[f] - current_profile[f])) for f in STRAT_FIELDS}
        purse = float(purse_df.loc[purse_df["team"] == team, "purse"].sum())
        retained_count = len(team_data)

        if "slots" in purse_df.columns and not purse_df.loc[purse_df["team"] == team, "slots"].empty:
            slots_left = int(pd.to_numeric(
                purse_df.loc[purse_df["team"] == team, "slots"],
                errors="coerce"
            ).fillna(0).max())
        else:
            slots_left = max(0, 25 - retained_count)

        states[team] = {
            "team": team,
            "retained_count": retained_count,
            "slots_left": slots_left,
            "slots_before_auction": slots_left,
            "purse": purse,
            "current_profile": current_profile,
            "gap": gap,
        }

    return states, target_profile



def build_input_pool(year, auction_df):
    auction_df = normalize_columns(auction_df)
    auction_df = clean_text_cols(auction_df, ["player", "type", "nationality"])
    auction_df = auction_df.drop_duplicates("player", keep="first").copy()

    snapshot = get_snapshot_before_year(year)
    pool = pd.merge(
        auction_df,
        snapshot.drop(columns=["team"], errors="ignore"),
        on="player",
        how="left",
    )

    for field in STRAT_FIELDS:
        if field not in pool.columns:
            pool[field] = 0.0

    if "type" not in pool.columns:
        pool["type"] = "Unknown"
    pool["type"] = pool["type"].fillna("Unknown")

    pool["quality_score"] = pool[STRAT_FIELDS].mean(axis=1).fillna(0.0)

    return pool

def run_teamwise_auction(pool_df, team_states):
    pool_df = pool_df.copy().reset_index(drop=True)

    pool_df["status"] = "Unsold"
    pool_df["predicted_price_cr"] = 0.0
    pool_df["winner_team"] = None
    pool_df["team_fit_score"] = 0.0

    runtime = {}
    for team, state in team_states.items():
        runtime[team] = {
            "initial_purse": float(state["purse"]),
            "purse_left": float(state["purse"]),
            "initial_slots": int(state["slots_before_auction"]),
            "slots_left": int(state["slots_before_auction"]),
            "bought_players": []
        }

    unsold_idx = set(pool_df.index.tolist())
    max_rounds = max((x["slots_left"] for x in runtime.values()), default=0)

    for _ in range(max_rounds):
        any_pick_this_round = False

        teams_in_order = sorted(
            runtime.keys(),
            key=lambda t: (runtime[t]["slots_left"], runtime[t]["purse_left"]),
            reverse=True
        )

        for team in teams_in_order:
            team_state = runtime[team]

            if team_state["slots_left"] <= 0 or team_state["purse_left"] <= 0:
                continue

            candidates = pool_df.loc[list(unsold_idx)].copy()
            if candidates.empty:
                continue

            gap = team_states[team]["gap"]

            def calc_fit(row):
                return sum(
                    max(0.0, gap[f]) * (float(row.get(f, 0.0) or 0.0) / 100.0)
                    for f in STRAT_FIELDS
                )

            candidates["fit_score"] = candidates.apply(calc_fit, axis=1)

            purse_ratio = 0.0
            if team_state["initial_purse"] > 0:
                purse_ratio = team_state["purse_left"] / team_state["initial_purse"]

            slot_ratio = 0.0
            if team_state["initial_slots"] > 0:
                slot_ratio = team_state["slots_left"] / team_state["initial_slots"]

            urgency_multiplier = 1 + 0.20 * purse_ratio + 0.20 * slot_ratio

            candidates["team_price"] = candidates["final_price_cr"] * urgency_multiplier
            candidates["team_price"] = candidates["team_price"].clip(lower=candidates["base_price_cr"])

            affordable = candidates[candidates["team_price"] <= team_state["purse_left"]].copy()
            if affordable.empty:
                continue

            # normal fit-based route
            if affordable["fit_score"].max() > 0:
                affordable["pick_score"] = (
                    affordable["fit_score"]
                    + 0.15 * affordable["quality_score"].fillna(0.0)
                    - 0.02 * affordable["team_price"]
                )

                affordable = affordable.sort_values(
                    ["pick_score", "fit_score", "quality_score", "team_price"],
                    ascending=[False, False, False, False]
                )
            else:
                # fallback route for teams like MI with low/filled gaps:
                # still buy best affordable quality/value players
                affordable["pick_score"] = (
                    affordable["quality_score"].fillna(0.0) / (affordable["team_price"] + 0.05)
                )

                affordable = affordable.sort_values(
                    ["pick_score", "quality_score", "team_price"],
                    ascending=[False, False, True]
                )

            pick = affordable.iloc[0]
            idx = int(pick.name)
            buy_price = round(float(pick["team_price"]), 2)

            pool_df.at[idx, "status"] = "Sold"
            pool_df.at[idx, "predicted_price_cr"] = buy_price
            pool_df.at[idx, "winner_team"] = team
            pool_df.at[idx, "team_fit_score"] = round(float(pick["fit_score"]), 2)

            runtime[team]["purse_left"] -= buy_price
            runtime[team]["slots_left"] -= 1
            runtime[team]["bought_players"].append(idx)

            unsold_idx.remove(idx)
            any_pick_this_round = True

        if not any_pick_this_round:
            break

    # use more of the purse on bought players
    for team, team_state in runtime.items():
        bought_idx = team_state["bought_players"]
        if not bought_idx:
            continue

        purse_left = float(team_state["purse_left"])
        if purse_left <= 0:
            continue

        extra_budget = purse_left * 0.85
        if extra_budget <= 0:
            continue

        weights = (
            pool_df.loc[bought_idx, "team_fit_score"].fillna(0.0)
            + pool_df.loc[bought_idx, "quality_score"].fillna(0.0)
            + 0.1
        )

        weights_sum = float(weights.sum())
        if weights_sum <= 0:
            continue

        increments = extra_budget * (weights / weights_sum)
        pool_df.loc[bought_idx, "predicted_price_cr"] = (
            pool_df.loc[bought_idx, "predicted_price_cr"] + increments
        ).round(2)

        runtime[team]["purse_left"] = round(runtime[team]["purse_left"] - float(extra_budget), 2)

    return pool_df, runtime

def build_team_summary_df(team_states, runtime):
    rows = []
    for team, state in team_states.items():
        run = runtime[team]
        bought_count = len(run["bought_players"])
        purse_before = float(state["purse"])
        purse_after = float(run["purse_left"])

        rows.append({
            "team": team,
            "retained_players": int(state["retained_count"]),
            "slots_before_auction": int(state["slots_before_auction"]),
            "players_bought": int(bought_count),
            "remaining_slots_after_auction": int(run["slots_left"]),
            "purse_before": round(purse_before, 2),
            "team_spent": round(purse_before - purse_after, 2),
            "purse_after": round(purse_after, 2)
        })
    return pd.DataFrame(rows)



def select_sold_players(pool_df, total_slots, total_purse):
    pool_df = pool_df.copy().sort_values(
        ["final_price_cr", "sum_fit_score", "player"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    pool_df["status"] = "Unsold"
    pool_df["predicted_price_cr"] = 0.0

    spent = 0.0
    sold_count = 0
    for idx, row in pool_df.iterrows():
        price = float(row["final_price_cr"])
        if sold_count >= total_slots:
            continue
        if price <= 0:
            continue
        if spent + price > total_purse:
            continue

        pool_df.at[idx, "status"] = "Sold"
        pool_df.at[idx, "predicted_price_cr"] = round(price, 2)
        spent += price
        sold_count += 1

    return pool_df


def suggest_players(team, team_states, sold_pool_df):
    state = team_states[team]
    gap = state["gap"]
    purse = state["purse"]

    available_players = sold_pool_df[sold_pool_df["predicted_price_cr"] > 0].copy()

    if available_players.empty:
        return pd.DataFrame(), 0.0

    suggestions = []
    for _, row in available_players.iterrows():
        fit_score = 0.0
        for field in STRAT_FIELDS:
            fit_score += max(0.0, gap[field]) * (float(row.get(field, 0.0) or 0.0) / 100.0)

        if fit_score <= 0:
            continue

        price = float(row["predicted_price_cr"])
        value_score = fit_score / (price + 0.01)

        suggestions.append({
            "player": row["player"],
            "fit_score": round(fit_score, 2),
            "price_cr": round(price, 2),
            "value_score": round(value_score, 4),
        })

    suggest_df = pd.DataFrame(suggestions)

    if suggest_df.empty:
        return suggest_df, 0.0

    suggest_df = suggest_df.sort_values(
        ["fit_score", "value_score", "price_cr"],
        ascending=[False, False, False]
    )

    selected = []
    spent = 0.0
    remaining_slots = int(state["slots_left"])

    for _, row in suggest_df.iterrows():
        if len(selected) >= remaining_slots:
            break
        if spent + float(row["price_cr"]) > purse:
            continue

        selected.append(row.to_dict())
        spent += float(row["price_cr"])

    return pd.DataFrame(selected), round(spent, 2)



tab1, tab2, tab3, tab4 = st.tabs([
    "Historical Analysis",
    "Target DNA",
    "Auction Price Engine",
    "Team Strategy",
])


with tab1:
    season = st.selectbox("Season", sorted(DF_TOP4["season"].dropna().unique(), reverse=True))
    season_data = DF_TOP4[DF_TOP4["season"] == season].sort_values("rank")

    fig = go.Figure()
    for _, row in season_data.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row[f] for f in STRAT_FIELDS],
            theta=STRAT_FIELDS,
            fill="toself",
            name=row["team"],
        ))
    st.plotly_chart(fig, use_container_width=True)


with tab2:
    next_year = int(DF_TOP4["season"].max() + 1)
    target_profile = predict_target_profile(TOP4_BUNDLE, next_year, use_history_only=True)
    st.subheader(f"{next_year} Top-4 Target DNA")
    st.table(pd.DataFrame([target_profile]).T.rename(columns={0: "Target"}))


with tab3:
    st.subheader("Auction Price Engine")

    auction_file = st.file_uploader("Auction CSV", type=["csv"], key="auction_file")
    retention_file = st.file_uploader("Retention CSV", type=["csv"], key="retention_file")
    purse_file = st.file_uploader("Purse CSV", type=["csv"], key="purse_file")

    if auction_file and retention_file and purse_file:
        auction_df = normalize_columns(pd.read_csv(auction_file))
        retention_df = normalize_columns(pd.read_csv(retention_file))
        purse_df = clean_purse_df(pd.read_csv(purse_file))

        auction_df = clean_text_cols(auction_df, ["player", "type", "nationality"])
        retention_df = clean_text_cols(retention_df, ["player", "team", "type", "nationality"])
        purse_df = clean_text_cols(purse_df, ["team"])

        current_year = infer_input_year(retention_df, purse_df)
        team_states, _ = build_team_states(current_year, retention_df, purse_df)
        total_slots = int(sum(state["slots_left"] for state in team_states.values()))
        total_purse = float(sum(state["purse"] for state in team_states.values()))

        pool_df = build_input_pool(current_year, auction_df)
        pool_df = predict_base_prices(pool_df, BASE_BUNDLE, current_year)
        pool_df["quality_score"] = pool_df[STRAT_FIELDS].mean(axis=1).fillna(0.0)
        pool_df = compute_market_features(pool_df, team_states)
        pool_df = apply_multiplier_models(pool_df, MULT_BUNDLE, current_year)
        pool_df, runtime = run_teamwise_auction(pool_df, team_states)

        team_summary_df = build_team_summary_df(team_states, runtime)
        total_spent = float(team_summary_df["team_spent"].sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Remaining Slots", total_slots)
        c2.metric("Total Purse", f"{round(total_purse, 2)} Cr")
        c3.metric("Total Spent", f"{round(total_spent, 2)} Cr")

        output_df = pool_df[[
            "player",
            "base_price_cr",
            "demand_multiplier",
            "scarcity_multiplier",
            "purse_multiplier",
            "predicted_price_cr",
            "winner_team",
            "status",
        ]].copy()



        output_df["base_price_cr"] = output_df["base_price_cr"].round(2)
        output_df["demand_multiplier"] = output_df["demand_multiplier"].round(2)
        output_df["scarcity_multiplier"] = output_df["scarcity_multiplier"].round(2)
        output_df["purse_multiplier"] = output_df["purse_multiplier"].round(2)

        st.dataframe(
            output_df.sort_values(
                ["predicted_price_cr", "base_price_cr", "player"],
                ascending=[False, False, True]
            ),
            use_container_width=True,
        )

        csv = output_df.sort_values(
            ["predicted_price_cr", "base_price_cr", "player"],
            ascending=[False, False, True]
        ).to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Predictions CSV",
            data=csv,
            file_name=f"auction_predictions_{current_year}.csv",
            mime="text/csv",
)


        st.session_state["current_year"] = current_year
        st.session_state["team_states"] = team_states
        st.session_state["target_pool"] = pool_df
        st.session_state["team_summary_df"] = team_summary_df


with tab4:
    st.subheader("Team Strategy")

    team_states = st.session_state.get("team_states")
    target_pool = st.session_state.get("target_pool")
    current_year = st.session_state.get("current_year")
    team_summary_df = st.session_state.get("team_summary_df")


    if team_states and target_pool is not None and team_summary_df is not None:
        teams = sorted(team_states.keys())
        team = st.selectbox("Select Team", teams)
        state = team_states[team]

        target_profile = predict_target_profile(TOP4_BUNDLE, current_year, use_history_only=True)
        suggest_df, team_spend = suggest_players(team, team_states, target_pool)

        c1, c2, c3, c4 = st.columns(4)
        team_summary = team_summary_df[team_summary_df["team"] == team].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Players Retained", int(team_summary["retained_players"]))
        c2.metric("Slots Before Auction", int(team_summary["slots_before_auction"]))
        c3.metric("Purse Before Auction", f"{round(float(team_summary['purse_before']), 2)} Cr")
        c4.metric("Purse After Auction", f"{round(float(team_summary['purse_before']) - team_spend, 2)} Cr")

        st.write(f"Suggested Players: {len(suggest_df)}")
        st.write(f"Remaining Slots After Suggestions: {max(0, int(team_summary['slots_before_auction']) - len(suggest_df))}")

        st.subheader("Current Profile")
        st.table(pd.DataFrame([state["current_profile"]])[STRAT_FIELDS].T.rename(columns={0: "Retained Squad"}))

        st.subheader("Target Profile")
        st.table(pd.DataFrame([target_profile])[STRAT_FIELDS].T.rename(columns={0: "Top-4 Target"}))

        gap_df = pd.DataFrame([state["gap"]]).T.rename(columns={0: "Gap"})
        st.subheader("Gap to Target")
        st.table(gap_df)

        st.subheader("Suggested Players")
        st.write(f"Suggested Players: {len(suggest_df)}")
        st.write(f"Team Spend: {round(team_spend, 2)} Cr")

        if suggest_df.empty:
            st.info("No sold players fit this team within its available purse.")
        else:
            st.dataframe(
                suggest_df[["player", "fit_score", "price_cr", "value_score"]],
                use_container_width=True,
            )

    else:
        st.info("Upload Auction CSV, Retention CSV, and Purse CSV in the Auction Price Engine tab first.")
