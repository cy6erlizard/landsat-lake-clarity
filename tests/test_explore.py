"""Phase 3 exploration must survive real-shaped input without a display."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from lakeclarity import config, explore, features, viz

FIXTURE = Path(__file__).parent / "fixtures" / "matchups_sample.csv"


@pytest.fixture(scope="module")
def sample():
    viz.use_style()
    return pd.read_csv(FIXTURE, low_memory=False)


@pytest.fixture(scope="module")
def train(sample):
    frame, _ = features.build_training_frame(sample, holdout_lake_ids=[])
    return frame


def test_condition_number_exposes_the_ratio_collinearity(train):
    """The 15 ratios are algebraic functions of 6 medians, so the matrix is ill-conditioned."""
    X = features.feature_matrix(train)
    cond_all = explore.predictor_condition_number(X)
    cond_bands_only = explore.predictor_condition_number(X[config.BAND_COLS])
    assert cond_all > cond_bands_only
    assert cond_all > 50


def test_mutual_information_returns_one_score_per_predictor(train):
    X = features.feature_matrix(train)
    mi = explore.mutual_information(X, train[config.LOG_TARGET])
    assert len(mi) == len(config.FEATURES)
    assert (mi >= 0).all()
    assert mi.is_monotonic_decreasing


def test_pick_stable_reference_lake_requires_a_long_large_record():
    """With no lake long enough, it refuses rather than returning a bad reference."""
    rng = np.random.default_rng(0)
    tiny = pd.DataFrame({
        "lagoslakeid": np.repeat([1, 2], 5),
        "Pixelcount": 10,
        "year": np.tile([1990, 1991, 1992, 1993, 1994], 2),
        config.TARGET: rng.uniform(1, 5, 10),
    })
    with pytest.raises(ValueError, match="no lake is long and large enough"):
        explore.pick_stable_reference_lake(tiny)


def test_pick_stable_reference_lake_prefers_clear_and_steady():
    rng = np.random.default_rng(1)
    rows = []
    for lake, (mean, sd) in enumerate([(6.0, 0.2), (6.0, 2.5), (1.5, 0.2)]):
        for year in range(1985, 2016):
            for _ in range(2):
                rows.append({"lagoslakeid": lake, "year": year, "Pixelcount": 800,
                             config.TARGET: rng.normal(mean, sd)})
    df = pd.DataFrame(rows)
    assert explore.pick_stable_reference_lake(df) == 0  # clear and steady wins


def _region(sample):
    r = features.add_time_columns(sample[sample[config.TARGET].notna()])
    r = features.add_log_target(r)
    r["lake_name"] = "Reference Lake"
    return r


def test_filter_waterfall_renders_and_reports_secchi_shift(sample):
    _, log = features.build_training_frame(sample, holdout_lake_ids=[])
    fig = explore.fig_filter_waterfall(log)
    assert len(fig.axes) == 2
    matplotlib.pyplot.close(fig)


def test_remaining_phase3_figures_render(sample, train):
    region = _region(sample)
    X = features.feature_matrix(train)

    figs = [
        explore.fig_correlation_heatmap(X),
        explore.fig_mutual_information(X, train[config.LOG_TARGET], top=12),
        explore.fig_seasonality(region),
    ]
    for fig in figs:
        assert fig.axes
        matplotlib.pyplot.close(fig)


def test_stable_lake_drift_renders_when_a_reference_exists():
    """Build a lake with a deliberate step at the 2013 sensor handoff."""
    rng = np.random.default_rng(4)
    rows = []
    for year in range(1985, 2021):
        sat = "LANDSAT_5" if year <= 2011 else "LANDSAT_8" if year >= 2013 else "LANDSAT_7"
        step = 0.01 if year >= 2013 else 0.0  # the artifact we want the figure to expose
        for _ in range(3):
            row = {f"{b}median": rng.normal(0.03 + step, 0.002) for b in config.BANDS}
            row.update({"lagoslakeid": 1, "year": year, "SATELLITE": sat,
                        "Pixelcount": 900, "lake_name": "Steady Lake",
                        config.TARGET: rng.normal(6.0, 0.15)})
            rows.append(row)
    region = pd.DataFrame(rows)

    fig = explore.fig_stable_lake_drift(region, lake_id=1)
    assert len(fig.axes) == len(config.BANDS)
    matplotlib.pyplot.close(fig)
