"""Model definitions and the Monte Carlo seat simulator.

The forecasting model has three parts:

1. ``regressor``  predicts each seat's deviation from uniform statewide swing,
   so expected NDA-UPA margin = last margin + statewide swing + deviation.
2. ``oth_classifier`` estimates the chance that a seat goes to neither camp
   (independents, MNS, AIMIM, VBA, ...).
3. ``simulate`` turns those point estimates into probabilities by drawing
   correlated uncertainty: one statewide swing per simulation, plus region,
   district and seat-level errors whose sizes come from out-of-sample
   residuals (see ``noise_components``).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import MAJORITY, RANDOM_STATE
from features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def make_regressor(model=None) -> Pipeline:
    if model is None:
        model = RandomForestRegressor(n_estimators=500, min_samples_leaf=5,
                                      random_state=RANDOM_STATE)
    return Pipeline([("prep", preprocessor()), ("model", clone(model))])


def make_oth_classifier() -> Pipeline:
    return Pipeline([("prep", preprocessor()),
                     ("model", LogisticRegression(C=0.3, max_iter=2000))])


def noise_components(df: pd.DataFrame, resid) -> dict:
    """Split out-of-sample margin errors into region, district and seat parts.

    Seat errors are strongly correlated within a region/district (alliance
    realignments, local leaders switching sides), and treating them as
    independent makes seat-total intervals far too narrow.
    """
    r = pd.Series(np.asarray(resid, dtype=float), index=df.index)
    r = r - r.mean()
    region_mean = r.groupby(df["region"]).transform("mean")
    district_mean = r.groupby(df["district"]).transform("mean") - region_mean
    seat = r - region_mean - district_mean
    return {
        "region_sd": float(r.groupby(df["region"]).mean().std()),
        "district_sd": float(district_mean.groupby(df["district"]).first().std()),
        "seat_sd": float(seat.std()),
    }


@dataclass
class ElectionModel:
    regressor: Pipeline
    oth_classifier: Pipeline
    seat_sd: float            # seat-specific SD of margin error (pp)
    region_sd: float = 0.0    # SD of region-wide error shared by its seats (pp)
    district_sd: float = 0.0  # SD of district-wide error shared by its seats (pp)

    def fit(self, df: pd.DataFrame) -> "ElectionModel":
        self.regressor.fit(df[FEATURES], df["target_deviation"])
        self.oth_classifier.fit(df[FEATURES], df["target_camp"] == "OTH")
        return self

    def expected_margin(self, df: pd.DataFrame, swing: float) -> np.ndarray:
        X = df[FEATURES].copy()
        X["proj_margin"] = df["nda_margin"] + swing
        return X["proj_margin"].to_numpy() + self.regressor.predict(X)

    def p_oth(self, df: pd.DataFrame) -> np.ndarray:
        return self.oth_classifier.predict_proba(df[FEATURES])[:, 1]

    def simulate(self, df: pd.DataFrame, swing: float = 0.0, swing_sd: float = 5.0,
                 n_sims: int = 5000, seed: int = RANDOM_STATE) -> dict:
        """Simulate ``n_sims`` elections; return seat-level and state-level results."""
        rng = np.random.default_rng(seed)
        n = len(df)
        mu = self.expected_margin(df, swing)
        p_oth = self.p_oth(df)

        regions, region_idx = np.unique(df["region"].to_numpy(), return_inverse=True)
        state = rng.normal(0.0, swing_sd, size=(n_sims, 1))
        region = rng.normal(0.0, self.region_sd, size=(n_sims, len(regions)))[:, region_idx]
        districts, district_idx = np.unique(df["district"].to_numpy(), return_inverse=True)
        district = rng.normal(0.0, self.district_sd,
                              size=(n_sims, len(districts)))[:, district_idx]
        noise = rng.normal(0.0, self.seat_sd, size=(n_sims, n))
        margin = mu[None, :] + state + region + district + noise

        oth = rng.random((n_sims, n)) < p_oth[None, :]
        nda = (~oth) & (margin > 0)
        upa = (~oth) & (margin <= 0)

        seats = pd.DataFrame({"NDA": nda.sum(1), "UPA": upa.sum(1), "OTH": oth.sum(1)})
        seat_probs = pd.DataFrame({
            "ac_no": df["ac_no"].to_numpy(),
            "expected_margin": mu.round(2),
            "p_NDA": nda.mean(0).round(3),
            "p_UPA": upa.mean(0).round(3),
            "p_OTH": oth.mean(0).round(3),
        })
        seat_probs["predicted_camp"] = seat_probs[["p_NDA", "p_UPA", "p_OTH"]].idxmax(axis=1).str[2:]
        summary = {
            camp: {
                "mean": round(float(seats[camp].mean()), 1),
                "median": int(seats[camp].median()),
                "p05": int(seats[camp].quantile(0.05)),
                "p95": int(seats[camp].quantile(0.95)),
                "p_majority": round(float((seats[camp] >= MAJORITY).mean()), 3),
            }
            for camp in seats.columns
        }
        return {"seat_probs": seat_probs, "seat_draws": seats, "summary": summary}
