"""Phase 4: train the regional model, then attack it.

Two rules govern everything here.

**Group by lake, always.** A random split puts observations from the same lake in
both train and test. Because most of the variance in this dataset is between
lakes, that lets the model memorise lake identity from its reflectance signature
and report a cross-validated R-squared that means nothing. Every split in this
module is a `GroupKFold` on `lagoslakeid`.

**Report per-lake skill, not just pooled skill.** The pooled scatter is what the
published models report. The per-lake correlation is what the client measures. On
this data those two numbers are close to unrelated, and the gap between them is
the entire finding. `per_lake_skill` computes the second one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from . import config, eda, features, viz

log = logging.getLogger(__name__)

DEFAULT_RF = dict(
    n_estimators=500,
    min_samples_leaf=2,
    max_features="sqrt",
    n_jobs=-1,
    random_state=config.RANDOM_STATE,
)


@dataclass
class CVResult:
    """Out-of-fold predictions in log space, plus the fold each row belonged to."""
    frame: pd.DataFrame  # lagoslakeid, fold, y_true_log, y_pred_log, y_true_m, y_pred_m
    fold_scores: pd.DataFrame

    def pooled_metrics(self) -> dict[str, float]:
        f = self.frame
        return {
            "r2_log": r2_score(f["y_true_log"], f["y_pred_log"]),
            "rmse_log": float(np.sqrt(mean_squared_error(f["y_true_log"], f["y_pred_log"]))),
            "rmse_m": float(np.sqrt(mean_squared_error(f["y_true_m"], f["y_pred_m"]))),
            "bias_m": float((f["y_pred_m"] - f["y_true_m"]).mean()),
            "pearson_r_pooled": float(stats.pearsonr(f["y_true_log"], f["y_pred_log"])[0]),
            "n": len(f),
            "n_lakes": int(f["lagoslakeid"].nunique()),
        }


def fit_random_forest(X: pd.DataFrame, y: pd.Series, **kwargs) -> RandomForestRegressor:
    params = {**DEFAULT_RF, **kwargs}
    rf = RandomForestRegressor(**params)
    rf.fit(X, y)
    return rf


def grouped_cv(
    train: pd.DataFrame,
    n_splits: int = 5,
    **rf_kwargs,
) -> CVResult:
    """Out-of-fold predictions with lakes held out whole.

    A lake appears in exactly one test fold. Nothing about it is visible to the
    model that predicts it.
    """
    X = features.feature_matrix(train)
    y = train[config.LOG_TARGET]
    groups = train["lagoslakeid"]

    # Columns the residual and per-lake diagnostics need, carried through the
    # split alongside the predictions rather than rejoined by position afterwards.
    META = ["lagoslakeid", "year", "month", "doy", "SATELLITE", "Pixelcount", "CLOUD_COVER"]
    meta = train[[c for c in META if c in train.columns]].reset_index(drop=True)

    cv = GroupKFold(n_splits=n_splits)
    rows = []
    fold_rows = []

    for fold, (tr, te) in enumerate(cv.split(X, y, groups)):
        rf = fit_random_forest(X.iloc[tr], y.iloc[tr], **rf_kwargs)
        pred = rf.predict(X.iloc[te])

        block = meta.iloc[te].reset_index(drop=True)
        block["fold"] = fold
        block["y_true_log"] = y.iloc[te].to_numpy()
        block["y_pred_log"] = pred
        rows.append(block)

        fold_rows.append({
            "fold": fold,
            "n_train": len(tr),
            "n_test": len(te),
            "n_test_lakes": int(groups.iloc[te].nunique()),
            "r2_log": r2_score(y.iloc[te], pred),
        })

    frame = pd.concat(rows, ignore_index=True)
    frame["y_true_m"] = 10 ** frame["y_true_log"]
    frame["y_pred_m"] = 10 ** frame["y_pred_log"]

    assert len(frame) == len(train), "out-of-fold frame lost or duplicated rows"
    assert frame["lagoslakeid"].nunique() == train["lagoslakeid"].nunique()

    return CVResult(frame=frame, fold_scores=pd.DataFrame(fold_rows))


def per_lake_skill(cv: CVResult, min_obs: int = 8) -> pd.DataFrame:
    """Within-lake Pearson r between out-of-fold prediction and observation.

    This is the client's metric. A model can post `pooled_metrics()['r2_log'] =
    0.6` and have the median of this column sit at zero.
    """
    return eda.within_lake_correlation(
        cv.frame, observed_col="y_true_m", predicted_col="y_pred_m", min_obs=min_obs
    )


def importance_comparison(
    rf: RandomForestRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Gini next to permutation importance.

    Gini importance is biased toward high-cardinality and correlated predictors,
    and this feature set is fifteen algebraically dependent ratios. Showing both
    is the honest move; using Gini alone would be a mistake a reviewer would spot.
    """
    perm = permutation_importance(
        rf, X, y, n_repeats=n_repeats, random_state=config.RANDOM_STATE, n_jobs=-1
    )
    return pd.DataFrame({
        "feature": X.columns,
        "gini": rf.feature_importances_,
        "permutation_mean": perm.importances_mean,
        "permutation_std": perm.importances_std,
    }).sort_values("permutation_mean", ascending=False)


def compare_to_national(
    regional: pd.DataFrame,
    national: pd.DataFrame,
    observed_col: str = "y_true_m",
) -> pd.DataFrame:
    """Per-lake r for both models, on identical lakes and observations."""
    reg = eda.within_lake_correlation(regional, observed_col, "y_pred_m").assign(model="regional")
    nat = eda.within_lake_correlation(national, observed_col, "y_pred_m").assign(model="national")
    return pd.concat([reg, nat], ignore_index=True)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_fold_scores(cv: CVResult, national_fold_scores: pd.DataFrame | None = None):
    """F15. Fold-to-fold variance, because the mean R2 hides it."""
    fig, ax = plt.subplots(figsize=(7, 4.4))
    data = [cv.fold_scores["r2_log"]]
    labels = ["regional"]
    colors = [viz.MODEL_COLORS["regional"]]
    if national_fold_scores is not None:
        data.append(national_fold_scores["r2_log"])
        labels.append("national")
        colors.append(viz.MODEL_COLORS["national"])

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.45)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_edgecolor(viz.SURFACE)
    for median in bp["medians"]:
        median.set_color(viz.SURFACE)
        median.set_linewidth(1.8)

    for i, d in enumerate(data, start=1):
        ax.scatter(np.full(len(d), i) + np.random.default_rng(0).normal(0, 0.04, len(d)),
                   d, s=22, color=viz.INK, zorder=3, alpha=0.7)

    ax.set_ylabel("R-squared on held-out lakes (log space)")
    ax.set_title("F15  Pooled skill, per fold")
    ax.grid(axis="y")
    viz.annotate(ax, "lakes are held out whole;\nno lake appears in both train and test", loc="lower left")
    return fig


def fig_observed_vs_predicted(cv: CVResult):
    """F16. Random forests cannot extrapolate, so the clear end regresses to the mean."""
    f = cv.frame
    fig, ax = plt.subplots(figsize=(6.4, 6))
    hb = ax.hexbin(f["y_true_m"], f["y_pred_m"], gridsize=45, bins="log",
                   cmap=viz.SEQUENTIAL, mincnt=1, linewidths=0)
    cb = fig.colorbar(hb, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("matchups (log scale)")
    cb.outline.set_visible(False)

    lim = (0, np.nanpercentile(f["y_true_m"], 99.5))
    ax.plot(lim, lim, color=viz.INK, linewidth=1.4, linestyle=(0, (4, 3)), label="1:1")

    # Binned conditional mean, which is where the shrinkage becomes visible.
    bins = np.linspace(*lim, 18)
    mid = 0.5 * (bins[:-1] + bins[1:])
    binned = f.groupby(pd.cut(f["y_true_m"], bins), observed=True)["y_pred_m"].mean()
    ax.plot(mid[: len(binned)], binned.to_numpy(), color=viz.STATUS["critical"],
            linewidth=2.0, label="conditional mean of prediction")

    m = cv.pooled_metrics()
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("observed Secchi depth (m)")
    ax.set_ylabel("predicted Secchi depth (m)")
    ax.set_title("F16  Observed versus predicted, held-out lakes")
    ax.legend(loc="upper left")
    viz.annotate(ax, f"pooled R2 = {m['r2_log']:.3f}\nRMSE = {m['rmse_m']:.2f} m\nn = {m['n']:,}",
                 loc="lower right")
    ax.grid(axis="both")
    return fig


def fig_residual_grid(cv: CVResult):
    """F17. Residuals against everything that could be secretly driving them."""
    f = cv.frame.copy()
    f["residual_m"] = f["y_pred_m"] - f["y_true_m"]

    panels = [
        ("y_true_m", "observed Secchi (m)"),
        ("Pixelcount", "clear pixels"),
        ("CLOUD_COVER", "scene cloud cover (%)"),
        ("SATELLITE", "sensor"),
        ("doy", "day of year"),
        ("year", "year"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (col, label) in zip(axes.ravel(), panels):
        if col == "SATELLITE":
            order = [s for s in viz.SATELLITE_COLORS if s in f[col].unique()]
            data = [f.loc[f[col] == s, "residual_m"] for s in order]
            bp = ax.boxplot(data, tick_labels=order, patch_artist=True, widths=0.5, showfliers=False)
            for patch, s in zip(bp["boxes"], order):
                patch.set_facecolor(viz.SATELLITE_COLORS[s])
                patch.set_edgecolor(viz.SURFACE)
            for med in bp["medians"]:
                med.set_color(viz.SURFACE)
            ax.tick_params(axis="x", rotation=20)
        else:
            x = f[col]
            ax.scatter(x, f["residual_m"], s=4, alpha=0.15, color=viz.CATEGORICAL[0],
                       edgecolor="none")
            if col == "Pixelcount":
                ax.set_xscale("log")
            q = pd.qcut(x, 12, duplicates="drop")
            binned = f.groupby(q, observed=True)["residual_m"].median()
            centers = [iv.mid for iv in binned.index]
            ax.plot(centers, binned.to_numpy(), color=viz.STATUS["critical"], linewidth=1.8)
        ax.axhline(0, color=viz.AXIS, linewidth=1.0)
        ax.set_xlabel(label)
        ax.set_ylabel("prediction - observation (m)")

    fig.suptitle("F17  Residual diagnostics: a step by sensor is a manufactured trend",
                 x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    return fig


def fig_importance(imp: pd.DataFrame, top: int = 18):
    """F18. Permutation next to Gini, and the discrepancy is the point."""
    d = imp.head(top).iloc[::-1]
    y = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(8.6, max(5, 0.36 * len(d))))
    ax.barh(y + 0.19, d["permutation_mean"], height=0.36, color=viz.CATEGORICAL[0],
            xerr=d["permutation_std"], error_kw=dict(ecolor=viz.INK_MUTED, lw=0.8),
            label="permutation importance")
    ax.barh(y - 0.19, d["gini"], height=0.36, color=viz.CATEGORICAL[2], label="Gini importance")
    ax.set_yticks(y)
    ax.set_yticklabels(d["feature"], fontsize=8)
    ax.set_xlabel("importance")
    ax.set_title("F18  Permutation importance versus Gini")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    viz.annotate(
        ax,
        "Gini is biased toward correlated predictors, and the\n15 band ratios are algebraic functions of 6 medians.",
        loc="upper right",
    )
    return fig


def fig_partial_dependence(rf: RandomForestRegressor, X: pd.DataFrame, feature: str):
    """F19. What the model believes about the single most informative predictor."""
    pd_result = partial_dependence(rf, X, [feature], kind="average", grid_resolution=40)
    grid = pd_result["grid_values"][0]
    avg = pd_result["average"][0]

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(grid, 10**avg, color=viz.CATEGORICAL[0], linewidth=2.2)
    ax.set_xlabel(feature)
    ax.set_ylabel("partial dependence, Secchi depth (m)")
    ax.set_title(f"F19  Partial dependence on {feature}")
    rug = np.quantile(X[feature], np.linspace(0.01, 0.99, 40))
    ax.plot(rug, np.full_like(rug, ax.get_ylim()[0]), "|", color=viz.INK_MUTED,
            markersize=6, alpha=0.6)
    return fig


def fig_per_lake_skill(
    comparison: pd.DataFrame,
    client_r: float = -0.22,
):
    """F20. The headline.

    Distribution of within-lake correlation, one point per held-out lake, for the
    national model and the regional one. The client observed r = -0.22 on Squam
    and read it as an anomaly. If the national model's distribution is broad and
    centred near zero, -0.22 is not an anomaly at all: it is a typical draw from a
    model with no within-lake skill anywhere, and the fix is recalibration rather
    than a different lake.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    bins = np.linspace(-1, 1, 41)
    for name in ("national", "regional"):
        sub = comparison[comparison["model"] == name]
        if sub.empty:
            continue
        frac_neg = (sub["r"] < 0).mean()
        ax.hist(sub["r"], bins=bins, histtype="stepfilled", alpha=0.55,
                facecolor=viz.MODEL_COLORS[name], edgecolor=viz.MODEL_COLORS[name],
                linewidth=1.6,
                label=(f"{name}:  median r = {sub['r'].median():+.2f},  "
                       f"{frac_neg:.0%} of lakes negative  (n = {len(sub)})"))

    ax.axvline(0, color=viz.AXIS, linewidth=1.2)
    ax.axvline(client_r, color=viz.INK, linewidth=1.8, linestyle=(0, (4, 3)))

    viz.headroom(ax, 1.34)
    top = ax.get_ylim()[1]
    ax.annotate(
        f"Squam Lake\nnational model\nr = {client_r}",
        xy=(client_r, top * 0.50), xytext=(-0.96, top * 0.62),
        fontsize=9, color=viz.INK, ha="left", va="center", linespacing=1.4,
        arrowprops=dict(arrowstyle="->", color=viz.INK, linewidth=1.0,
                        connectionstyle="arc3,rad=-0.15"),
    )

    ax.set_xlabel("within-lake Pearson r between predicted and observed Secchi")
    ax.set_ylabel("held-out lakes")
    ax.set_title("F20  Can the model track a single lake through time?")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_xlim(-1, 1)
    return fig
