"""Clean the three raw result files into one tidy candidate-level table.

Outputs (data/processed):
    candidates.csv      year, ac_no, party, votes, vote_share, rank, camp
    constituencies.csv  static info per seat: name, reservation, district, region
    seat_stats.csv      year, ac_no, electors, total_votes, turnout, n_cand
"""
import pandas as pd

from config import CAMPS, PARTY_NAME_2024, PROCESSED_DIR, RAW_DIR


def _load_tcpd(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(RAW_DIR / f"mh_{year}.csv")
    df = df[df["Poll_No"] == 0]  # drop any re-polls/bye-polls
    cands = pd.DataFrame({
        "year": year,
        "ac_no": df["Constituency_No"].astype(int),
        "candidate": df["Candidate"],
        "party": df["Party"].astype(str).str.strip(),
        "votes": df["Votes"].astype(int),
    })
    seats = (
        df.groupby("Constituency_No")
        .agg(electors=("Electors", "first"), turnout=("Turnout_Percentage", "first"))
        .reset_index()
        .rename(columns={"Constituency_No": "ac_no"})
    )
    seats["year"] = year
    return cands, seats


def _load_eci_2024() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(RAW_DIR / "mh_2024.csv")
    party = df["Party"].astype(str).str.strip()
    unknown = sorted(set(party) - set(PARTY_NAME_2024))
    # Minor parties that never matter for camps keep a readable short label.
    mapped = party.map(PARTY_NAME_2024).fillna(party)
    cands = pd.DataFrame({
        "year": 2024,
        "ac_no": df["AC No"].astype(int),
        "candidate": df["Candidate"],
        "party": mapped,
        "votes": df["Total Votes"].astype(int),
    })

    el = pd.read_csv(RAW_DIR / "mh_2024_electors.csv")
    el = el[pd.to_numeric(el["Ac No"], errors="coerce").notna()].copy()
    el["ac_no"] = el["Ac No"].astype(int)
    el["electors"] = el["Total"].astype(int) + el["Total Service Electors"].astype(int)
    totals = cands.groupby("ac_no")["votes"].sum().rename("total_votes")
    seats = el[["ac_no", "electors"]].merge(totals, on="ac_no")
    seats["turnout"] = (100 * seats["total_votes"] / seats["electors"]).round(2)
    seats["year"] = 2024
    print(f"2024: {len(unknown)} minor party names kept verbatim")
    return cands, seats.drop(columns="total_votes")


def assign_camp(year: int, party: str) -> str:
    for camp, members in CAMPS[year].items():
        if party in members:
            return camp
    return "OTH"


def build() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frames, seat_frames = [], []
    for year in (2014, 2019):
        c, s = _load_tcpd(year)
        frames.append(c)
        seat_frames.append(s)
    c, s = _load_eci_2024()
    frames.append(c)
    seat_frames.append(s)

    cands = pd.concat(frames, ignore_index=True)
    totals = cands.groupby(["year", "ac_no"])["votes"].transform("sum")
    cands["vote_share"] = 100 * cands["votes"] / totals
    cands["rank"] = (
        cands.groupby(["year", "ac_no"])["votes"].rank(ascending=False, method="first").astype(int)
    )
    cands["camp"] = [assign_camp(y, p) for y, p in zip(cands["year"], cands["party"])]
    cands.to_csv(PROCESSED_DIR / "candidates.csv", index=False)

    seats = pd.concat(seat_frames, ignore_index=True)
    n_cand = cands[cands["party"] != "NOTA"].groupby(["year", "ac_no"]).size().rename("n_cand")
    total_votes = cands.groupby(["year", "ac_no"])["votes"].sum().rename("total_votes")
    seats = seats.merge(n_cand, on=["year", "ac_no"]).merge(total_votes, on=["year", "ac_no"])
    seats.to_csv(PROCESSED_DIR / "seat_stats.csv", index=False)

    # Static constituency attributes come from the richer TCPD 2019 file
    # (boundaries are identical for 2014-2024 under the 2008 delimitation).
    raw19 = pd.read_csv(RAW_DIR / "mh_2019.csv")
    const = (
        raw19.groupby("Constituency_No")
        .agg(
            ac_name=("Constituency_Name", "first"),
            ac_type=("Constituency_Type", "first"),
            district=("District_Name", "first"),
            region=("Sub_Region", "first"),
        )
        .reset_index()
        .rename(columns={"Constituency_No": "ac_no"})
    )
    const.to_csv(PROCESSED_DIR / "constituencies.csv", index=False)

    winners = cands[cands["rank"] == 1]
    print(winners.pivot_table(index="camp", columns="year", values="ac_no", aggfunc="count"))


if __name__ == "__main__":
    build()
