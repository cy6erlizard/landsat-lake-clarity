"""Phase 3: interrogate the features before fitting anything to them.

Three questions, each with a figure that can embarrass us.

1. Is the quality filter neutral? If negative median reflectance is a
   Collection 1 aerosol over-correction over dark water, then dropping those rows
   drops the clear end of the Secchi distribution and every model in this lineage
   trained on a biased sample. `fig_filter_waterfall` reports the mean Secchi on
   both sides of each filter step and lets the data answer.

2. Are the 15 band ratios independent predictors? They are algebraic functions of
   6 medians, so they carry roughly five degrees of freedom wearing fifteen hats.
   Random forests tolerate that, but it makes Gini importance meaningless, which
   is why Phase 4 reports permutation importance instead.

3. Is a forty-year clarity trend real, or is it the satellites? `fig_stable_lake_drift`
   plots each band's median reflectance for a large, clear, stable lake. In a lake
   that has not changed, this should be flat. If it steps at the sensor handoffs,
   any uncorrected trend is an instrument artifact.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from . import config, viz
from .features import FilterLog


def predictor_condition_number(X: pd.DataFrame) -> float:
    """How close to singular is the standardised predictor matrix?

    A large number is not a bug here. It is the quantitative statement that the
    ratio columns are algebraically dependent, and it is the reason Gini
    importance cannot be trusted on this feature set.
    """
    Z = (X - X.mean()) / X.std().replace(0, np.nan)
    Z = Z.dropna(axis=1, how="all").fillna(0.0)
    return float(np.linalg.cond(Z.to_numpy()))


def mutual_information(X: pd.DataFrame, y: pd.Series, seed: int = config.RANDOM_STATE) -> pd.Series:
    """Nonlinear univariate association of each predictor with log Secchi."""
    mi = mutual_info_regression(X.to_numpy(), y.to_numpy(), random_state=seed)
    return pd.Series(mi, index=X.columns).sort_values(ascending=False)


def pick_stable_reference_lake(region: pd.DataFrame, min_years: int = 25) -> int:
    """A large, clear lake with a long record and little Secchi movement.

    Whatever drift this lake's reflectance shows over forty years is not the lake.
    """
    g = region.groupby("lagoslakeid").agg(
        n=("Pixelcount", "size"),
        px=("Pixelcount", "median"),
        yrs=("year", "nunique"),
        secchi_mean=(config.TARGET, "mean"),
        secchi_sd=(config.TARGET, "std"),
    )
    g = g[(g["yrs"] >= min_years) & (g["px"] >= 200) & (g["n"] >= 40)]
    if g.empty:
        raise ValueError("no lake is long and large enough to serve as a drift reference")
    # clear, and as steady as the region offers
    g["score"] = g["secchi_mean"] / g["secchi_sd"].clip(lower=0.05)
    return int(g["score"].idxmax())


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_filter_waterfall(log: FilterLog):
    """F10. Rows removed at each step, and whether the survivors got clearer."""
    wf = log.to_frame()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [1.5, 1]})

    x = np.arange(len(wf))
    ax1.bar(x, wf["rows_before"], color=viz.GRID, width=0.7, label="removed")
    ax1.bar(x, wf["rows_after"], color=viz.CATEGORICAL[0], width=0.7, label="retained")
    ax1.set_ylabel("matchups")
    ax1.set_title("F10  The quality filter is not neutral")
    viz.headroom(ax1, 1.18)
    ax1.legend(loc="upper right")
    for xi, row in zip(x, wf.itertuples()):
        if row.pct_dropped >= 1:
            ax1.annotate(f"-{row.pct_dropped:.0f}%", (xi, row.rows_after),
                         ha="center", va="bottom", fontsize=8, color=viz.INK_SECONDARY)

    shift = wf["secchi_shift_m"]
    colors = [viz.STATUS["critical"] if s < -0.02 else
              viz.STATUS["good"] if s > 0.02 else viz.INK_MUTED for s in shift]
    ax2.bar(x, shift, color=colors, width=0.7)
    ax2.axhline(0, color=viz.AXIS, linewidth=1.0)
    ax2.set_ylabel("shift in mean Secchi (m)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(wf["step"], rotation=30, ha="right")
    ax2.set_title("What each step does to the clarity of the surviving sample")
    viz.annotate(
        ax2,
        "a negative bar means the step deleted clear water,\nbiasing training against exactly the lakes we predict",
        loc="lower left",
    )
    fig.tight_layout()
    return fig


def fig_correlation_heatmap(X: pd.DataFrame):
    """F11. Fifteen ratios, roughly five degrees of freedom."""
    order = config.BAND_COLS + config.RATIO_COLS + config.INDEX_COLS
    order = [c for c in order if c in X.columns]
    corr = X[order].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    im = ax.imshow(corr, cmap=viz.DIVERGING, vmin=-1, vmax=1, interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.02)
    cb.set_label("Spearman correlation")
    cb.outline.set_visible(False)

    ax.set_xticks(range(len(order)))
    ax.set_yticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=6.5)
    ax.set_yticklabels(order, fontsize=6.5)
    ax.grid(False)

    n_band = len([c for c in config.BAND_COLS if c in X.columns])
    for pos in (n_band - 0.5,):
        ax.axhline(pos, color=viz.INK, linewidth=1.0)
        ax.axvline(pos, color=viz.INK, linewidth=1.0)

    cond = predictor_condition_number(X[order])
    ax.set_title(f"F11  Predictor collinearity   (condition number {cond:,.0f})")
    return fig


def fig_mutual_information(X: pd.DataFrame, y: pd.Series, top: int = 20):
    """F12. Which predictors actually carry clarity, and does the physics hold?"""
    mi = mutual_information(X, y).head(top).iloc[::-1]
    colors = [viz.CATEGORICAL[1] if c in config.CDOM_RATIOS else viz.CATEGORICAL[0]
              for c in mi.index]

    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.32 * len(mi))))
    ax.barh(mi.index, mi.values, color=colors, height=0.74)
    ax.set_xlabel("mutual information with log10 Secchi (nats)")
    ax.set_title("F12  Univariate signal, ranked")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=viz.CATEGORICAL[1]),
               plt.Rectangle((0, 0), 1, 1, color=viz.CATEGORICAL[0])]
    ax.legend(handles, ["blue-family ratio (CDOM)", "other predictor"], loc="lower right")
    viz.annotate(
        ax,
        "CDOM absorbs in the blue and barely scatters.\nIf red and NIR terms dominate instead, this region\nis sediment-driven and the New Hampshire analogy weakens.",
        loc="lower left",
    )
    return fig


def fig_stable_lake_drift(region: pd.DataFrame, lake_id: int | None = None):
    """F13. The figure that decides whether any forty-year trend is believable."""
    lake_id = pick_stable_reference_lake(region) if lake_id is None else lake_id
    g = region[region["lagoslakeid"] == lake_id].sort_values("year")
    name = g["lake_name"].iloc[0] if "lake_name" in g.columns else f"lake {lake_id}"

    bands = config.BANDS
    fig, axes = plt.subplots(len(bands), 1, figsize=(9, 1.55 * len(bands)), sharex=True)

    for ax, band in zip(axes, bands):
        col = f"{band}median"
        annual = g.groupby("year")[col].median()
        for sat, sub in g.groupby("SATELLITE"):
            ax.scatter(sub["year"], sub[col], s=7, alpha=0.35, edgecolor="none",
                       color=viz.SATELLITE_COLORS.get(sat, viz.INK_MUTED),
                       label=sat if band == bands[0] else None)
        ax.plot(annual.index, annual.values, color=viz.INK, linewidth=1.4)
        ax.set_ylabel(band, fontsize=9)
        for yr in config.SENSOR_EVENTS:
            ax.axvline(yr, color=viz.INK_MUTED, linewidth=0.8, linestyle=(0, (4, 3)))

    axes[0].legend(loc="upper left", ncols=3, fontsize=7.5)
    axes[0].set_title(
        f"F13  Band drift on a stable reference lake: {name}\n"
        f"a lake that has not changed should be flat"
    )
    axes[-1].set_xlabel("year")
    fig.supylabel("median surface reflectance")
    fig.tight_layout()
    return fig


def fig_seasonality(region: pd.DataFrame):
    """F14. Why the client validates on July, and what that costs the model."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    by_doy = region.groupby("month")[config.TARGET].agg(["median", "count"])
    ax1.plot(by_doy.index, by_doy["median"], color=viz.CATEGORICAL[0], marker="o")
    ax1.axvspan(6.5, 7.5, color=viz.CATEGORICAL[2], alpha=0.18, zorder=0)
    ax1.annotate("July", xy=(7, ax1.get_ylim()[1]), ha="center", va="top",
                 fontsize=9, color=viz.INK_SECONDARY)
    ax1.set_xlabel("month")
    ax1.set_ylabel("median Secchi depth (m)")
    ax1.set_title("F14  Clarity has a season")

    ax2.bar(by_doy.index, by_doy["count"], color=viz.CATEGORICAL[0], width=0.72)
    ax2.set_xlabel("month")
    ax2.set_ylabel("matchups")
    ax2.set_title("and so does the sampling effort")
    viz.annotate(
        ax2,
        "a model trained across all months and\nvalidated on July must be checked for\nseasonal bias in its residuals",
        loc="upper left",
    )
    viz.headroom(ax2, 1.35)
    fig.tight_layout()
    return fig
