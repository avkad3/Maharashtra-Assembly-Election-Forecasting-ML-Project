"""Forecast the next (2029) Maharashtra Assembly election from 2024 results.

The statewide swing is a *scenario input*: the change in NDA-minus-UPA vote
margin relative to 2024 (+14.6 pp for NDA/Mahayuti). 0 means "same statewide
mood as 2024"; negative values mean a swing toward UPA/MVA.

    python forecast.py                 # baseline (swing 0) + scenario sweep
    python forecast.py --swing -8      # custom central scenario
    python forecast.py --alliances thackeray_alliance congress_alone
                                       # replay 2024 under post-2024 alliances
"""
import argparse
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import (ALLIANCE_CHANGES, FIGURES_DIR, MAJORITY, MODELS_DIR, PROCESSED_DIR,
                    REPORTS_DIR, alliance_changes)
from features import build_all, forecast_features

SCENARIOS = [-20, -15, -10, -5, 0, 5]
SWING_SD = 5.0  # uncertainty around the chosen statewide swing (pp)


def load_model():
    return joblib.load(MODELS_DIR / "election_model.joblib")["model"]


def load_forecast_features(changes: dict[str, str] | None = None) -> pd.DataFrame:
    if changes:
        return forecast_features(changes)
    path = PROCESSED_DIR / "features_forecast.csv"
    if not path.exists():
        build_all()
    return pd.read_csv(path)


def run(swing: float, n_sims: int = 5000, changes: dict[str, str] | None = None) -> dict:
    model = load_model()
    df = load_forecast_features(changes)
    sim = model.simulate(df, swing=swing, swing_sd=SWING_SD, n_sims=n_sims)
    seats = df[["ac_no", "ac_name", "district", "region", "ac_type", "winner_party",
                "winner_camp", "nda_margin"]].rename(columns={
                    "winner_party": "party_2024", "winner_camp": "camp_2024",
                    "nda_margin": "nda_margin_2024"})
    sim["seat_probs"] = seats.merge(sim["seat_probs"], on="ac_no")
    return sim


def plot_distribution(draws: pd.DataFrame, swing: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"NDA": "#e07b39", "UPA": "#3b6ea5"}
    for camp, color in colors.items():
        ax.hist(draws[camp], bins=range(0, 289, 4), alpha=0.6, color=color, label=camp)
    ax.axvline(MAJORITY, color="black", ls="--", lw=1)
    ax.text(MAJORITY + 2, ax.get_ylim()[1] * 0.9, "majority (145)", fontsize=8)
    ax.set_xlabel("Seats won")
    ax.set_ylabel("Simulations")
    ax.set_title(f"2029 seat distribution (statewide swing {swing:+.0f} pp vs 2024)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "forecast_2029_distribution.png", dpi=150)
    plt.close(fig)


def plot_scenarios(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for camp, color in {"NDA": "#e07b39", "UPA": "#3b6ea5"}.items():
        ax.plot(table["swing"], table[f"{camp}_median"], marker="o", color=color, label=camp)
        ax.fill_between(table["swing"], table[f"{camp}_p05"], table[f"{camp}_p95"],
                        color=color, alpha=0.15)
    ax.axhline(MAJORITY, color="black", ls="--", lw=1)
    ax.set_xlabel("Statewide swing in NDA-UPA margin vs 2024 (pp)")
    ax.set_ylabel("Seats (median, 90% interval)")
    ax.set_title("2029 seats under different statewide swings")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "forecast_2029_scenarios.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--swing", type=float, default=0.0)
    parser.add_argument("--sims", type=int, default=5000)
    parser.add_argument("--alliances", nargs="*", default=[], choices=list(ALLIANCE_CHANGES),
                        help="post-2024 alliance changes to apply to the 2024 base")
    args = parser.parse_args()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    changes = alliance_changes(args.alliances)

    sim = run(args.swing, args.sims, changes)
    sim["seat_probs"].to_csv(REPORTS_DIR / "forecast_2029_seats.csv", index=False)
    plot_distribution(sim["seat_draws"], args.swing)

    rows = []
    for s in sorted(set(SCENARIOS + [args.swing])):
        summ = run(s, 2000, changes)["summary"]
        row = {"swing": s}
        for camp, st in summ.items():
            row.update({f"{camp}_{k}": v for k, v in st.items()})
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(REPORTS_DIR / "forecast_2029_scenarios.csv", index=False)
    plot_scenarios(table)

    # Same statewide mood, different alliance structures.
    combos = {"2024 alliances": []}
    combos.update({ALLIANCE_CHANGES[k]["label"]: [k] for k in ALLIANCE_CHANGES})
    combos["Both changes"] = list(ALLIANCE_CHANGES)
    alliance_rows = []
    for label, keys in combos.items():
        summ = run(args.swing, 2000, alliance_changes(keys))["summary"]
        alliance_rows.append({"alliances": label,
                              **{f"{c}_median": summ[c]["median"] for c in summ},
                              "NDA_p_majority": summ["NDA"]["p_majority"],
                              "UPA_p_majority": summ["UPA"]["p_majority"]})
    alliance_table = pd.DataFrame(alliance_rows)
    alliance_table.to_csv(REPORTS_DIR / "forecast_2029_alliance_scenarios.csv", index=False)

    with open(REPORTS_DIR / "forecast_2029_summary.json", "w") as f:
        json.dump({"swing": args.swing, "swing_sd": SWING_SD, "alliances": args.alliances,
                   "summary": sim["summary"]}, f, indent=2)
    print(f"Central scenario: swing {args.swing:+.1f} pp")
    print(json.dumps(sim["summary"], indent=2))
    print(table[["swing", "NDA_median", "NDA_p05", "NDA_p95", "UPA_median",
                 "NDA_p_majority", "UPA_p_majority"]].to_string(index=False))
    print(alliance_table.to_string(index=False))


if __name__ == "__main__":
    main()
