"""Turn the tidy tables into seat-level features and next-election targets.

Each row describes one constituency after election ``t``; the targets describe
what happened at election ``t + 1`` (five years later):

    2014 features -> 2019 outcome   (training transition)
    2019 features -> 2024 outcome   (held-out test transition)
    2024 features -> 2029 outcome   (forecast, no target)

The key engineered feature is ``proj_margin``: last election's NDA-minus-UPA
margin shifted by the *statewide* swing. Statewide swing is the one thing a
seat-level model cannot learn from two transitions, so it is supplied from
outside (actual swing for back-testing, a scenario/opinion-poll value for the
forecast) and the model learns how each seat departs from it.
"""
import numpy as np
import pandas as pd

from config import CAMP_LABELS, FEATURE_PARENT, FEATURE_PARTIES, PROCESSED_DIR

NUMERIC_FEATURES = (
    ["nda_share", "upa_share", "oth_share", "nda_margin", "nda_lean", "win_margin",
     "turnout", "log_electors", "n_cand", "enop", "proj_margin"]
    + [f"{p.lower()}_share" for p in FEATURE_PARTIES]
)
CATEGORICAL_FEATURES = ["winner_camp", "ac_type", "region"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cands = pd.read_csv(PROCESSED_DIR / "candidates.csv")
    seats = pd.read_csv(PROCESSED_DIR / "seat_stats.csv")
    const = pd.read_csv(PROCESSED_DIR / "constituencies.csv")
    return cands, seats, const


def statewide_margin(cands: pd.DataFrame) -> pd.Series:
    """NDA minus UPA share of all votes cast in the state, per year (pct points)."""
    votes = cands.pivot_table(index="year", columns="camp", values="votes", aggfunc="sum")
    total = votes.sum(axis=1)
    return 100 * (votes["NDA"] - votes["UPA"]) / total


def seat_snapshot(cands: pd.DataFrame, seats: pd.DataFrame, const: pd.DataFrame,
                  year: int) -> pd.DataFrame:
    """Features describing every seat right after election ``year``."""
    c = cands[cands["year"] == year].copy()

    camp = c.pivot_table(index="ac_no", columns="camp", values="vote_share",
                         aggfunc="sum").reindex(columns=CAMP_LABELS, fill_value=0.0).fillna(0.0)
    df = pd.DataFrame({
        "ac_no": camp.index,
        "nda_share": camp["NDA"].values,
        "upa_share": camp["UPA"].values,
        "oth_share": camp["OTH"].values,
    })
    df["nda_margin"] = df["nda_share"] - df["upa_share"]
    df["nda_lean"] = df["nda_margin"] - statewide_margin(cands).loc[year]

    c["feat_party"] = c["party"].replace(FEATURE_PARENT)
    party = c.pivot_table(index="ac_no", columns="feat_party", values="vote_share", aggfunc="sum")
    party = party.reindex(columns=FEATURE_PARTIES).fillna(0.0)
    party.columns = [f"{p.lower()}_share" for p in party.columns]
    df = df.merge(party, left_on="ac_no", right_index=True)

    ranked = c.sort_values(["ac_no", "rank"])
    top = ranked[ranked["rank"] == 1].set_index("ac_no")
    second = ranked[ranked["rank"] == 2].set_index("ac_no")
    df["winner_camp"] = df["ac_no"].map(top["camp"])
    df["winner_party"] = df["ac_no"].map(top["party"])
    df["win_margin"] = df["ac_no"].map(top["vote_share"] - second["vote_share"])

    shares = c[c["party"] != "NOTA"].groupby("ac_no")["vote_share"]
    df["enop"] = df["ac_no"].map(shares.apply(lambda s: 1.0 / np.sum((s / s.sum()) ** 2)))

    s = seats[seats["year"] == year].set_index("ac_no")
    df["turnout"] = df["ac_no"].map(s["turnout"])
    df["log_electors"] = np.log(df["ac_no"].map(s["electors"]))
    df["n_cand"] = df["ac_no"].map(s["n_cand"])

    df = df.merge(const, on="ac_no")
    df["year"] = year
    return df


def build_transition(cands, seats, const, year: int, next_year: int | None,
                     swing: float | None = None) -> pd.DataFrame:
    """Features from ``year`` plus targets from ``next_year`` (if it exists).

    ``swing`` is the change in statewide NDA-UPA margin between the two
    elections. When omitted it is computed from the actual results.
    """
    df = seat_snapshot(cands, seats, const, year)
    sw_margin = statewide_margin(cands)
    if swing is None:
        swing = float(sw_margin.loc[next_year] - sw_margin.loc[year]) if next_year else 0.0
    df["statewide_swing"] = swing
    df["proj_margin"] = df["nda_margin"] + swing

    if next_year is not None and next_year in cands["year"].unique():
        nxt = seat_snapshot(cands, seats, const, next_year).set_index("ac_no")
        df["target_camp"] = df["ac_no"].map(nxt["winner_camp"])
        df["target_party"] = df["ac_no"].map(nxt["winner_party"])
        df["target_margin"] = df["ac_no"].map(nxt["nda_margin"])
        # How far the seat moved beyond the uniform statewide swing.
        df["target_deviation"] = df["target_margin"] - df["proj_margin"]
    df["next_year"] = next_year
    return df


def build_all() -> dict[str, pd.DataFrame]:
    cands, seats, const = load_tables()
    out = {
        "train": build_transition(cands, seats, const, 2014, 2019),
        "test": build_transition(cands, seats, const, 2019, 2024),
        "forecast": build_transition(cands, seats, const, 2024, 2029, swing=0.0),
    }
    for name, df in out.items():
        df.to_csv(PROCESSED_DIR / f"features_{name}.csv", index=False)
    return out


if __name__ == "__main__":
    sets = build_all()
    cands, _, _ = load_tables()
    print("Statewide NDA-UPA margin (pp):")
    print(statewide_margin(cands).round(2).to_string())
    for name, df in sets.items():
        print(name, df.shape)
    print(sets["test"][["ac_name", "nda_margin", "proj_margin", "target_margin",
                        "winner_camp", "target_camp"]].head())
