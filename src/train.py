"""Train, compare and back-test seat-level models; save the final model.

Protocol (temporal, no leakage):
    * model selection: 5-fold CV on the 2014->2019 transition only
    * back-test:       fit on 2014->2019, predict 2019->2024 given the actual
                       statewide swing (the standard "seats given votes" test)
    * final model:     refit on both transitions, saved for the 2029 forecast
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             mean_absolute_error)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline

from config import CAMP_LABELS, FIGURES_DIR, MODELS_DIR, RANDOM_STATE, REPORTS_DIR
from features import FEATURES, build_all
from model import (ElectionModel, make_oth_classifier, make_regressor, noise_components,
                   preprocessor)

CLASSIFIERS = {
    "logistic_regression": LogisticRegression(max_iter=2000, C=0.5),
    "random_forest": RandomForestClassifier(
        n_estimators=500, min_samples_leaf=3, random_state=RANDOM_STATE),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=2, learning_rate=0.05, random_state=RANDOM_STATE),
}

REGRESSORS = {
    "ridge": Ridge(alpha=10.0),
    "random_forest": RandomForestRegressor(
        n_estimators=500, min_samples_leaf=5, random_state=RANDOM_STATE),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=200, max_depth=2, learning_rate=0.05, random_state=RANDOM_STATE),
}


def seat_counts(labels) -> dict:
    s = pd.Series(labels).value_counts()
    return {c: int(s.get(c, 0)) for c in CAMP_LABELS}


def margin_to_camp(margin) -> np.ndarray:
    return np.where(np.asarray(margin) > 0, "NDA", "UPA")


def evaluate(y_true, y_pred) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", labels=CAMP_LABELS,
                                   zero_division=0), 4),
        "seats": seat_counts(y_pred),
    }


def brier(y_true: pd.Series, probs: pd.DataFrame) -> float:
    onehot = np.stack([(y_true == c).to_numpy() for c in CAMP_LABELS], axis=1)
    p = probs[[f"p_{c}" for c in CAMP_LABELS]].to_numpy()
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def compare_models(train, test) -> dict:
    X_tr, X_te = train[FEATURES], test[FEATURES]
    y_tr, y_te = train["target_camp"], test["target_camp"]
    out = {"baselines": {}, "classifiers": {}, "regressors": {}}

    out["baselines"]["persistence"] = evaluate(y_te, test["winner_camp"])
    us = evaluate(y_te, margin_to_camp(test["proj_margin"]))
    us["margin_mae"] = round(mean_absolute_error(test["target_margin"], test["proj_margin"]), 3)
    out["baselines"]["uniform_swing"] = us

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for name, model in CLASSIFIERS.items():
        pipe = Pipeline([("prep", preprocessor()), ("model", model)])
        cv_f1 = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1_macro").mean()
        res = evaluate(y_te, pipe.fit(X_tr, y_tr).predict(X_te))
        res["cv_macro_f1_train"] = round(cv_f1, 4)
        out["classifiers"][name] = res
        print(f"[clf] {name:20s} cv_f1={cv_f1:.3f} test_acc={res['accuracy']:.3f}")

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for name, model in REGRESSORS.items():
        pipe = make_regressor(model)
        cv_mae = -cross_val_score(pipe, X_tr, train["target_deviation"], cv=kf,
                                  scoring="neg_mean_absolute_error").mean()
        margin = test["proj_margin"] + pipe.fit(X_tr, train["target_deviation"]).predict(X_te)
        res = evaluate(y_te, margin_to_camp(margin))
        res["cv_deviation_mae_train"] = round(cv_mae, 3)
        res["margin_mae"] = round(mean_absolute_error(test["target_margin"], margin), 3)
        out["regressors"][name] = res
        print(f"[reg] {name:20s} cv_mae={cv_mae:.2f} test_mae={res['margin_mae']:.2f} "
              f"test_acc={res['accuracy']:.3f}")
    return out


def plot_backtest(test, seat_probs, name):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(test["target_camp"], seat_probs["predicted_camp"],
                                            labels=CAMP_LABELS, ax=ax, cmap="Blues",
                                            colorbar=False)
    ax.set_title(f"2024 back-test ({name} swing model)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix_2024.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(seat_probs["expected_margin"], test["target_margin"], s=12, alpha=0.6,
               color="#3b6ea5")
    lim = [-80, 80]
    ax.plot(lim, lim, color="grey", lw=1, ls="--")
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Predicted NDA-UPA margin, 2024 (pp)")
    ax.set_ylabel("Actual NDA-UPA margin, 2024 (pp)")
    ax.set_title("Seat margins: predicted vs actual")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "margin_pred_vs_actual_2024.png", dpi=150)
    plt.close(fig)


def plot_importance(reg, test) -> dict:
    imp = permutation_importance(reg, test[FEATURES], test["target_deviation"], n_repeats=20,
                                 scoring="neg_mean_absolute_error", random_state=RANDOM_STATE)
    s = pd.Series(imp.importances_mean, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(6, 6))
    s.tail(15).plot.barh(ax=ax, color="#3b6ea5")
    ax.set_xlabel("Increase in margin MAE when shuffled (pp)")
    ax.set_title("Permutation importance (2024 back-test)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)
    return s.sort_values(ascending=False).head(10).round(3).to_dict()


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    data = build_all()
    train, test = data["train"], data["test"]

    results = {"actual_2024_seats": seat_counts(test["target_camp"])}
    results.update(compare_models(train, test))
    best = min(results["regressors"],
               key=lambda k: results["regressors"][k]["cv_deviation_mae_train"])
    print(f"selected swing regressor: {best} (lowest training-CV MAE)")

    # --- Probabilistic back-test: everything sized from 2014->2019 only ------
    oof = cross_val_predict(make_regressor(REGRESSORS[best]), train[FEATURES],
                            train["target_deviation"],
                            cv=KFold(5, shuffle=True, random_state=RANDOM_STATE))
    cv_noise = noise_components(train, train["target_deviation"] - oof)
    bt_model = ElectionModel(make_regressor(REGRESSORS[best]), make_oth_classifier(),
                             **cv_noise).fit(train)
    swing_2024 = float(test["statewide_swing"].iloc[0])
    sim = bt_model.simulate(test, swing=swing_2024, swing_sd=0.0)
    sp = sim["seat_probs"]
    bt = evaluate(test["target_camp"], sp["predicted_camp"])
    bt["brier"] = round(brier(test["target_camp"].reset_index(drop=True), sp), 4)
    bt["margin_mae"] = round(mean_absolute_error(test["target_margin"], sp["expected_margin"]), 3)
    bt["seat_interval_90"] = sim["summary"]
    bt["noise_used"] = {k: round(v, 2) for k, v in cv_noise.items()}
    results["probabilistic_backtest"] = bt
    print(f"[bt ] acc={bt['accuracy']:.3f} brier={bt['brier']:.3f} "
          f"NDA 90%: {sim['summary']['NDA']['p05']}-{sim['summary']['NDA']['p95']} "
          f"(actual {results['actual_2024_seats']['NDA']})")

    out = test[["ac_no", "ac_name", "district", "region", "winner_camp", "target_camp",
                "target_party", "target_margin"]].merge(sp, on="ac_no")
    out.to_csv(REPORTS_DIR / "backtest_2024_predictions.csv", index=False)
    plot_backtest(test, sp, best)
    results["top_features"] = plot_importance(bt_model.regressor, test)

    # --- Final model on both labelled transitions ----------------------------
    full = pd.concat([train, test], ignore_index=True)
    # Error sizes for the forecast come from the honest out-of-time back-test.
    test_noise = noise_components(test, test["target_margin"] - sp["expected_margin"].to_numpy())
    final = ElectionModel(make_regressor(REGRESSORS[best]), make_oth_classifier(),
                          **test_noise).fit(full)
    joblib.dump({"model": final, "regressor_name": best}, MODELS_DIR / "election_model.joblib",
                compress=3)
    results["final_model"] = {"regressor": best,
                              "noise": {k: round(v, 2) for k, v in test_noise.items()},
                              "trained_on": ["2014->2019", "2019->2024"]}

    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved model (noise={results['final_model']['noise']}) and reports/metrics.json")


if __name__ == "__main__":
    main()
