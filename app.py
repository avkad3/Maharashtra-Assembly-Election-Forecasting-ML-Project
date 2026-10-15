"""Streamlit dashboard: run with  `streamlit run app.py`."""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import (ALLIANCE_CHANGES, MAJORITY, PROCESSED_DIR,  # noqa: E402
                    REPORTS_DIR, alliance_changes)
from forecast import load_forecast_features, load_model  # noqa: E402

st.set_page_config(page_title="Maharashtra 2029 Forecast", page_icon="🗳️", layout="wide")


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_features(alliance_keys: tuple[str, ...]):
    return load_forecast_features(alliance_changes(alliance_keys))


@st.cache_data
def simulate(swing: float, swing_sd: float, n_sims: int, alliance_keys: tuple[str, ...]):
    df = get_features(alliance_keys)
    sim = get_model().simulate(df, swing=swing, swing_sd=swing_sd, n_sims=n_sims)
    seats = df[["ac_no", "ac_name", "district", "region", "ac_type", "winner_party",
                "winner_camp", "nda_margin"]].merge(sim["seat_probs"], on="ac_no")
    return seats, sim["seat_draws"], sim["summary"]


st.title("Maharashtra Assembly Election Forecast — 2029")
st.caption(
    "Seat-level ML model trained on 2014→2019 and 2019→2024 results for all 288 "
    "constituencies. NDA = BJP-led Mahayuti, UPA = Congress-led Maha Vikas Aghadi."
)

with st.sidebar:
    st.header("Scenario")
    swing = st.slider(
        "Statewide swing vs 2024 (pp)", -25.0, 10.0, 0.0, 0.5,
        help="Change in NDA-minus-UPA statewide vote margin relative to 2024 (+14.6 pp). "
             "Negative = swing towards UPA/MVA.")
    swing_sd = st.slider("Uncertainty in that swing (pp, SD)", 0.0, 10.0, 5.0, 0.5)
    n_sims = st.select_slider("Simulations", [1000, 2000, 5000, 10000], value=5000)
    st.header("Alliance changes since 2024")
    alliance_keys = tuple(
        key for key, a in ALLIANCE_CHANGES.items()
        if st.checkbox(a["label"], help=a["note"])
    )
    st.markdown(
        "**How to read this:** each simulation draws a statewide swing, then "
        "correlated region, district and seat errors sized from the model's "
        "real 2024 back-test error."
    )

seats, draws, summary = simulate(swing, swing_sd, n_sims, alliance_keys)

tab_fc, tab_seats, tab_bt, tab_hist = st.tabs(
    ["Forecast", "Constituency explorer", "2024 back-test", "History"])

with tab_fc:
    with st.expander("Current conditions (as of Oct 2026): what has changed since 2024", expanded=False):
        st.markdown(
            "- **2025-26 local body polls (Dec 2025 - Jan 2026):** Mahayuti swept the "
            "municipal corporations (BJP 1,416, Shiv Sena 397, NCP 160 corporators vs "
            "Congress 324, Shiv Sena (UBT) 153, NCP (SP) 36). BJP became the largest party "
            "in the BMC (89/227). No sign of a swing away from NDA, so **0 is a reasonable "
            "default swing**; seat counts alone don't give a vote-share number.\n"
            "- **Congress contested the local polls alone** and **Shiv Sena (UBT) allied "
            "with MNS** in Mumbai: toggle these in the sidebar.\n"
            "- **Ajit Pawar died in a plane crash on 28 Jan 2026**; Sunetra Pawar now leads "
            "NCP and holds Baramati. Its effect on the NCP vote can't be measured without "
            "polling, so it's not modelled.\n\n"
            "Local-body results are a different kind of election and are not fed into the model."
        )
    if alliance_keys:
        st.info("Applied: " + "; ".join(ALLIANCE_CHANGES[k]["label"] for k in alliance_keys)
                + ". " + " ".join(ALLIANCE_CHANGES[k]["note"] for k in alliance_keys))
    cols = st.columns(4)
    for col, camp in zip(cols, ["NDA", "UPA", "OTH"]):
        s = summary[camp]
        col.metric(f"{camp} seats (median)", s["median"], help=f"90% interval {s['p05']}–{s['p95']}")
        col.caption(f"90% interval: {s['p05']}–{s['p95']}")
    cols[3].metric("P(NDA majority)", f"{summary['NDA']['p_majority']:.0%}")
    cols[3].caption(f"P(UPA majority): {summary['UPA']['p_majority']:.0%} · majority = {MAJORITY}")

    st.subheader("Seat distribution across simulations")
    hist = pd.DataFrame({
        camp: draws[camp].value_counts().reindex(range(0, 289), fill_value=0)
        for camp in ["NDA", "UPA"]
    })
    st.bar_chart(hist, color=["#e07b39", "#3b6ea5"], height=280)

    st.subheader("Expected seats by region")
    region = seats.groupby("region")[["p_NDA", "p_UPA", "p_OTH"]].sum().round(1)
    region["seats"] = seats.groupby("region").size()
    region.columns = ["NDA", "UPA", "OTH", "Total seats"]
    st.dataframe(region, width="stretch")

    st.subheader("Battleground seats")
    bg = seats[(seats["p_NDA"].between(0.35, 0.65))].copy()
    bg = bg.assign(closeness=(bg["p_NDA"] - 0.5).abs()).sort_values("closeness")
    st.caption(f"{len(bg)} seats where neither camp has more than a 65% chance.")
    st.dataframe(
        bg[["ac_name", "district", "region", "winner_party", "p_NDA", "p_UPA", "p_OTH"]]
        .rename(columns={"winner_party": "2024 winner"}),
        width="stretch", hide_index=True)

with tab_seats:
    name = st.selectbox("Constituency", seats.sort_values("ac_no")["ac_name"])
    row = seats[seats["ac_name"] == name].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("P(NDA)", f"{row['p_NDA']:.0%}")
    c2.metric("P(UPA)", f"{row['p_UPA']:.0%}")
    c3.metric("P(Others)", f"{row['p_OTH']:.0%}")
    st.write(
        f"**{row['ac_name']}** ({row['ac_type']}) — {row['district'].title()} district, "
        f"{row['region'].title()} region. 2024 winner: **{row['winner_party']}** "
        f"({row['winner_camp']}), NDA-UPA margin {row['nda_margin']:+.1f} pp. "
        f"Model's expected 2029 margin: {row['expected_margin']:+.1f} pp."
    )
    hist_c = pd.read_csv(PROCESSED_DIR / "candidates.csv")
    hist_c = hist_c[(hist_c["ac_no"] == row["ac_no"]) & (hist_c["rank"] <= 3)]
    st.dataframe(hist_c[["year", "rank", "candidate", "party", "camp", "vote_share"]]
                 .round(2), width="stretch", hide_index=True)
    st.subheader("All constituencies")
    st.dataframe(seats.drop(columns=["ac_no"]).round(3), width="stretch",
                 hide_index=True)

with tab_bt:
    metrics = json.loads((REPORTS_DIR / "metrics.json").read_text())
    st.markdown(
        "Model fitted on **2014→2019 only**, then asked to predict every 2024 seat "
        "given the actual statewide swing. This is the honest out-of-time test."
    )
    rows = []
    for group in ("baselines", "classifiers", "regressors"):
        for name_, r in metrics[group].items():
            rows.append({"family": group, "model": name_, "accuracy": r["accuracy"],
                         "macro F1": r["macro_f1"], "margin MAE": r.get("margin_mae"),
                         "NDA": r["seats"]["NDA"], "UPA": r["seats"]["UPA"],
                         "OTH": r["seats"]["OTH"]})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(f"Actual 2024 seats: {metrics['actual_2024_seats']}")
    c1, c2 = st.columns(2)
    for col, fig in zip([c1, c2, c1], ["margin_pred_vs_actual_2024.png",
                                       "confusion_matrix_2024.png", "feature_importance.png"]):
        path = REPORTS_DIR / "figures" / fig
        if path.exists():
            col.image(str(path))

with tab_hist:
    c = pd.read_csv(PROCESSED_DIR / "candidates.csv")
    won = c[c["rank"] == 1]
    st.subheader("Seats won by camp")
    st.bar_chart(won.pivot_table(index="year", columns="camp", values="ac_no",
                                 aggfunc="count").fillna(0)[["NDA", "OTH", "UPA"]],
                 color=["#e07b39", "#9e9e9e", "#3b6ea5"])
    st.subheader("Seats won by party")
    st.dataframe(won.pivot_table(index="party", columns="year", values="ac_no",
                                 aggfunc="count").fillna(0).astype(int)
                 .sort_values(2024, ascending=False), width="stretch")
