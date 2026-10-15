"""Voter-roll revision (SIR 2026) scenario.

The draft electoral roll published on 31 Aug 2026 left out 2.07 crore of
9.79 crore electors (21.1%), with rates from ~9% (Hingoli) to ~38% (Thane).
No past Maharashtra election had a comparable revision, so its effect cannot
be *learned* from history. Instead it is applied as an explicit, transparent
adjustment to each seat's expected margin, driven by two assumptions:

    genuine_share  fraction of excluded electors who were real, eligible
                   voters (the rest are dead / shifted / duplicate entries
                   who would not have voted in that seat anyway)
    lean           how those genuine voters leaned relative to the seat as a
                   whole, in NDA-minus-UPA points (negative = leaned UPA/MVA)

If a fraction f of a seat's voters is removed and they voted at margin
M + lean while the seat as a whole voted at margin M, the margin among the
remaining voters is

    (M - (M + lean) * f) / (1 - f)  =  M - lean * f / (1 - f)

so the seat shift is  -lean * f / (1 - f), with f = genuine_share * rate.
With lean = 0 the revision changes nothing (the neutral default).
"""
import numpy as np
import pandas as pd

from config import RAW_DIR

SOURCE = ("CEO Maharashtra, Press Note on Draft Publication of Electoral Roll in "
          "SIR-2026 (31 Aug 2026), Annexures 1-2")

# District names in the 2024 electors file -> names used in the press note.
DISTRICT_ALIASES = {"Dharashiv": "Osmanabad"}


def district_rates() -> pd.DataFrame:
    d = pd.read_csv(RAW_DIR / "sir_2026_draft_district.csv")
    d["exclusion_rate"] = d["electors_excluded"] / (d["electors_in_draft"] + d["electors_excluded"])
    return d


def seat_rates() -> pd.DataFrame:
    """Exclusion rate for every assembly seat (its district's rate)."""
    el = pd.read_csv(RAW_DIR / "mh_2024_electors.csv")
    el = el[pd.to_numeric(el["Ac No"], errors="coerce").notna()].copy()
    el["ac_no"] = el["Ac No"].astype(int)
    el["sir_district"] = el["District Name"].str.strip().replace(DISTRICT_ALIASES)
    rates = district_rates().set_index("district")["exclusion_rate"]
    el["exclusion_rate"] = el["sir_district"].map(rates)
    missing = el.loc[el["exclusion_rate"].isna(), "sir_district"].unique()
    if len(missing):
        raise ValueError(f"No SIR data for districts: {missing}")
    return el[["ac_no", "sir_district", "exclusion_rate"]]


def margin_shift(ac_nos, genuine_share: float, lean: float) -> np.ndarray:
    """NDA-minus-UPA margin shift (pp) for each seat in ``ac_nos``."""
    rates = seat_rates().set_index("ac_no")["exclusion_rate"]
    f = genuine_share * rates.reindex(np.asarray(ac_nos)).to_numpy()
    return -lean * f / (1.0 - f)
