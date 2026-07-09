"""Tests for the modelling layer.

The load-bearing one is `test_random_split_inflates_r2_but_group_split_does_not`.
It demonstrates, on data whose structure we control, the leak that a naive
cross-validation would introduce here, and shows that `grouped_cv` does not.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict

from lakeclarity import config, model, viz


FINGERPRINT_COLS = ["Bluemin", "Greenmin", "SWIR1stdDev"]


def _synthetic_training_frame(
    n_lakes: int = 40,
    n_obs: int = 24,
    between_sd: float = 0.30,
    within_sd: float = 0.10,
    feature_noise: float = 0.10,
    temporal_signal: float = 0.35,
    lake_fingerprint: bool = True,
    seed: int = 3,
) -> pd.DataFrame:
    """The real problem's optical structure, in miniature.

    Three properties are modelled deliberately:

    * A lake's mean clarity (``level``) varies a lot between lakes and its
      year-to-year movement is small. That is the measured ICC.
    * ``BluedivRedmedian`` is a *noisy* proxy for clarity, carrying the lake's
      level plus a weaker temporal component. Reflectance is not a clean readout.
    * Each lake carries an idiosyncratic optical fingerprint, constant across its
      observations and uninformative about clarity: bottom colour, CDOM
      composition, shoreline geometry. It identifies the lake without generalising
      to any other lake.

    The fingerprint is what a random train/test split leaks. Set
    ``lake_fingerprint=False`` for tests that need a clean signal.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for lake in range(n_lakes):
        level = rng.normal(0.55, between_sd)
        fingerprint = rng.normal(0, 1.0, size=len(FINGERPRINT_COLS))
        for t in range(n_obs):
            anomaly = rng.normal(0, within_sd)
            log_s = level + anomaly

            row = {c: rng.normal(0.05, 0.01) for c in config.FEATURES}
            row["BluedivRedmedian"] = (
                level + temporal_signal * anomaly + rng.normal(0, feature_noise)
            )
            if lake_fingerprint:
                for col, f in zip(FINGERPRINT_COLS, fingerprint):
                    row[col] = f + rng.normal(0, 0.01)

            row["Pixelcount"] = rng.integers(50, 3000)
            row["CLOUD_COVER"] = rng.uniform(0, 40)
            row["CLOUD_COVER_LAND"] = row["CLOUD_COVER"]
            row.update({
                "lagoslakeid": lake,
                "year": 1985 + t,
                "month": 7 if t % 3 == 0 else 6,
                "doy": 180 + (t % 30),
                "SATELLITE": ["LANDSAT_5", "LANDSAT_7", "LANDSAT_8"][t % 3],
                config.TARGET: 10**log_s,
                config.LOG_TARGET: log_s,
            })
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def train():
    return _synthetic_training_frame()


@pytest.fixture(scope="module")
def cv(train):
    return model.grouped_cv(train, n_splits=4, n_estimators=80)


def test_grouped_cv_returns_every_row_exactly_once(cv, train):
    assert len(cv.frame) == len(train)
    assert cv.fold_scores["n_test"].sum() == len(train)


def test_no_lake_appears_in_more_than_one_test_fold(cv):
    folds_per_lake = cv.frame.groupby("lagoslakeid")["fold"].nunique()
    assert (folds_per_lake == 1).all()


def test_random_split_inflates_r2_but_group_split_does_not(train):
    """The leak this module exists to prevent.

    Each lake carries an optical fingerprint that identifies it without saying
    anything about clarity. A random KFold puts the same lake in train and test,
    the forest reads the fingerprint, recalls that lake's mean, and reports a
    score it cannot reproduce on a lake it has never seen. Grouping by lake
    removes the shortcut and the honest score is materially lower.
    """
    from lakeclarity import features

    X = features.feature_matrix(train)
    y = train[config.LOG_TARGET]

    rf = model.RandomForestRegressor(**{**model.DEFAULT_RF, "n_estimators": 80})
    leaky_pred = cross_val_predict(rf, X, y, cv=KFold(4, shuffle=True, random_state=0))
    leaky_r2 = r2_score(y, leaky_pred)

    honest_r2 = model.grouped_cv(train, n_splits=4, n_estimators=80).pooled_metrics()["r2_log"]

    assert leaky_r2 > honest_r2 + 0.15, (
        f"expected the random split to be badly optimistic; "
        f"leaky={leaky_r2:.3f} honest={honest_r2:.3f}"
    )


def test_the_leak_disappears_when_lakes_have_no_fingerprint():
    """Control: with nothing to memorise, both splits agree."""
    from lakeclarity import features

    clean = _synthetic_training_frame(lake_fingerprint=False, seed=8)
    X = features.feature_matrix(clean)
    y = clean[config.LOG_TARGET]

    rf = model.RandomForestRegressor(**{**model.DEFAULT_RF, "n_estimators": 80})
    leaky_r2 = r2_score(y, cross_val_predict(rf, X, y, cv=KFold(4, shuffle=True, random_state=0)))
    honest_r2 = model.grouped_cv(clean, n_splits=4, n_estimators=80).pooled_metrics()["r2_log"]

    assert abs(leaky_r2 - honest_r2) < 0.12


def test_pooled_skill_and_per_lake_skill_are_different_numbers(cv):
    """The project's whole thesis, asserted on data where we know the truth.

    Reflectance encodes each lake's *level* strongly and its *movement* weakly, so
    the pooled correlation is good and the per-lake correlations are poor and
    broad. A single lake's r therefore says almost nothing about the model.
    """
    pooled = cv.pooled_metrics()
    per_lake = model.per_lake_skill(cv, min_obs=8)

    assert pooled["r2_log"] > 0.4
    assert pooled["pearson_r_pooled"] > 0.65
    assert len(per_lake) > 20

    # The gap. Pooled correlation is at least twice the typical per-lake one.
    assert pooled["pearson_r_pooled"] > 2 * per_lake["r"].median()
    assert per_lake["r"].std() > 0.1


def test_per_lake_r_distribution_contains_negative_values(cv):
    """A lake drawn from a skill-less model can easily produce r = -0.22."""
    per_lake = model.per_lake_skill(cv, min_obs=8)
    assert (per_lake["r"] < 0).any()


def test_pooled_metrics_reports_metres_not_only_log_space(cv):
    m = cv.pooled_metrics()
    assert m["rmse_m"] > 0
    assert "bias_m" in m
    assert m["n_lakes"] == 40


def test_importance_comparison_ranks_the_informative_predictor_first():
    """Without fingerprints, permutation importance finds the real predictor."""
    from lakeclarity import features

    clean = _synthetic_training_frame(lake_fingerprint=False, seed=8)
    X = features.feature_matrix(clean)
    y = clean[config.LOG_TARGET]
    rf = model.fit_random_forest(X, y, n_estimators=80)
    imp = model.importance_comparison(rf, X, y, n_repeats=3)
    assert imp.iloc[0]["feature"] == "BluedivRedmedian"
    assert {"gini", "permutation_mean", "permutation_std"} <= set(imp.columns)


def test_in_sample_importance_rewards_the_uninformative_fingerprint(train):
    """Why importance must never be read off a model fitted to all the data.

    The fingerprint columns carry no information about clarity whatsoever. Fitted
    in-sample, the forest still assigns them an order of magnitude more importance
    than the pure-noise predictors, because memorising a lake pays. Any feature
    ranking taken from a model that has seen every lake will reward this.
    """
    from lakeclarity import features

    X = features.feature_matrix(train)
    y = train[config.LOG_TARGET]
    rf = model.fit_random_forest(X, y, n_estimators=80)
    imp = model.importance_comparison(rf, X, y, n_repeats=3).set_index("feature")

    noise_cols = [
        c for c in config.FEATURES
        if c not in FINGERPRINT_COLS + ["BluedivRedmedian", "Pixelcount",
                                        "CLOUD_COVER", "CLOUD_COVER_LAND"]
    ]
    fingerprint_mean = imp.loc[FINGERPRINT_COLS, "gini"].mean()
    noise_mean = imp.loc[noise_cols, "gini"].mean()

    assert fingerprint_mean > 3 * noise_mean, (
        f"fingerprint gini {fingerprint_mean:.4f} vs noise {noise_mean:.4f}"
    )


def test_compare_to_national_labels_both_models(cv):
    national = cv.frame.copy()
    rng = np.random.default_rng(0)
    national["y_pred_m"] = national["y_pred_m"].mean() + rng.normal(0, 0.1, len(national))
    comp = model.compare_to_national(cv.frame, national)
    assert set(comp["model"]) == {"regional", "national"}
    # a constant-plus-noise national model must have near-zero within-lake skill
    assert abs(comp.loc[comp["model"] == "national", "r"].median()) < 0.2


def test_figures_render(cv, train):
    from lakeclarity import features

    viz.use_style()
    X = features.feature_matrix(train)
    y = train[config.LOG_TARGET]
    rf = model.fit_random_forest(X, y, n_estimators=60)
    imp = model.importance_comparison(rf, X, y, n_repeats=2)
    comp = model.compare_to_national(cv.frame, cv.frame.assign(y_pred_m=cv.frame["y_pred_m"].mean()))

    figs = [
        model.fig_fold_scores(cv),
        model.fig_observed_vs_predicted(cv),
        model.fig_residual_grid(cv),
        model.fig_importance(imp),
        model.fig_partial_dependence(rf, X, "BluedivRedmedian"),
        model.fig_per_lake_skill(comp),
    ]
    for fig in figs:
        assert fig.axes
        matplotlib.pyplot.close(fig)
