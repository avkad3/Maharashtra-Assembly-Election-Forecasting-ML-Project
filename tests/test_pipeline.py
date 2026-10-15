import numpy as np
import pandas as pd
import pytest

from config import PROCESSED_DIR, TOTAL_SEATS
from features import FEATURES, build_all
from forecast import load_model
from preprocess import assign_camp, build


@pytest.fixture(scope="module")
def tables():
    build()
    sets = build_all()
    cands = pd.read_csv(PROCESSED_DIR / "candidates.csv")
    return cands, sets


def test_every_year_has_all_seats(tables):
    cands, _ = tables
    assert cands.groupby("year")["ac_no"].nunique().eq(TOTAL_SEATS).all()


def test_vote_shares_sum_to_100(tables):
    cands, _ = tables
    totals = cands.groupby(["year", "ac_no"])["vote_share"].sum()
    assert np.allclose(totals, 100.0)


@pytest.mark.parametrize("year,party,seats", [
    (2014, "BJP", 122), (2019, "BJP", 105), (2024, "BJP", 132),
    (2024, "SHS", 57), (2024, "NCP", 41), (2024, "SHS_UBT", 20), (2024, "INC", 16),
])
def test_seat_totals_match_official_results(tables, year, party, seats):
    cands, _ = tables
    won = cands[(cands["year"] == year) & (cands["rank"] == 1)]
    assert (won["party"] == party).sum() == seats


@pytest.mark.parametrize("year,party,camp", [
    (2019, "SHS", "NDA"), (2019, "NCP", "UPA"),
    (2024, "SHS", "NDA"), (2024, "NCP", "NDA"),
    (2024, "SHS_UBT", "UPA"), (2024, "NCP_SP", "UPA"), (2024, "VBA", "OTH"),
])
def test_camp_mapping_follows_realignment(year, party, camp):
    assert assign_camp(year, party) == camp


def test_features_complete_and_consistent(tables):
    _, sets = tables
    for name, df in sets.items():
        assert len(df) == TOTAL_SEATS
        assert not df[FEATURES].isna().any().any(), name
        assert np.allclose(df["proj_margin"], df["nda_margin"] + df["statewide_swing"])
    assert set(sets["test"]["target_camp"]) <= {"NDA", "UPA", "OTH"}
    assert "target_camp" not in sets["forecast"]


def test_simulation_is_coherent(tables):
    _, sets = tables
    model = load_model()
    sim = model.simulate(sets["forecast"], swing=0, n_sims=500)
    probs = sim["seat_probs"][["p_NDA", "p_UPA", "p_OTH"]].sum(axis=1)
    assert np.allclose(probs, 1.0, atol=1e-3)
    assert (sim["seat_draws"].sum(axis=1) == TOTAL_SEATS).all()


def test_swing_moves_seats_in_the_right_direction(tables):
    _, sets = tables
    model = load_model()
    low = model.simulate(sets["forecast"], swing=-15, n_sims=500)["summary"]["NDA"]["mean"]
    high = model.simulate(sets["forecast"], swing=5, n_sims=500)["summary"]["NDA"]["mean"]
    assert high > low


def test_alliance_changes_move_votes_between_camps(tables):
    from features import forecast_features

    base = forecast_features()
    with_mns = forecast_features({"MNS": "UPA"})
    assert np.allclose(with_mns["upa_share"], base["upa_share"] + base["mns_share"])
    assert (forecast_features({"INC": "OTH"})["nda_margin"] >= base["nda_margin"] - 1e-9).all()


def test_voter_roll_data_matches_press_note():
    from voter_roll import district_rates, seat_rates

    d = district_rates()
    assert d["electors_excluded"].sum() == 20_688_487
    assert d["electors_in_draft"].sum() == 77_165_562
    s = seat_rates()
    assert len(s) == TOTAL_SEATS and s["exclusion_rate"].between(0.05, 0.45).all()


def test_voter_roll_shift_direction():
    from voter_roll import margin_shift

    seats = range(1, TOTAL_SEATS + 1)
    assert np.allclose(margin_shift(seats, 0.2, 0.0), 0.0)        # neutral default
    assert (margin_shift(seats, 0.2, -20.0) > 0).all()            # MVA-leaning removed -> NDA up
    assert (margin_shift(seats, 0.2, 20.0) < 0).all()
