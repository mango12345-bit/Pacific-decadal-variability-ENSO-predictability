#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduce Figure 1, Appendix Figure 1, and Appendix Table 1.

This script contains the analysis only. It reads a single, portable NetCDF file
(`figure1_input.nc`) created by `build_figure1_input.py`.

Expected variables in figure1_input.nc
--------------------------------------
Dimensions:
    dataset : reanalysis product
    year    : calendar year

Variables (dataset, year):
    H               spring equatorial-Pacific subsurface heat-content predictor
    H_gradient      spring zonal-gradient predictor
    tau_x           spring western-Pacific zonal wind-stress predictor
    tau_y           spring eastern-Pacific meridional wind-stress predictor
    nino34_ond      October-December Niño-3.4 target
    TCD_G           annual tropical-Pacific zonal thermocline-depth gradient (m)

The file may contain NaNs outside the valid period of each dataset.

Outputs
-------
outputs/Figure1.png
outputs/Appendix_Figure1.png
outputs/Appendix_Table1.csv
outputs/Figure1_source_data.csv
outputs/Appendix_Figure1_source_data.csv

Notes
-----
* The four-predictor regression includes an intercept. The intercept is excluded
  from variance attribution because it does not contribute to temporal variance.
* Predictor variance contributions use symmetric covariance allocation:
      contribution_i = sum_j Cov(component_i, component_j) / Var(y_hat)
* Appendix Table 1 reports moving-window standardized beta and signed partial R,
  consistent with the current manuscript wording.
* HAC inference for overlapping 21-year windows uses maxlags=20.
* Linear-versus-quadratic alternatives are checked using AICc, purged blocked
  cross-validation, and HAC significance of the quadratic term. The table records
  which relationship is selected.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm


PREDICTORS = ("H", "H_gradient", "tau_x", "tau_y")
CORE_DATASETS = ("SODA-GODAS", "ORA-20C", "GECCO3")
WINDOW = 21
HAC_LAG = 20


def finite_rows(*arrays):
    """Return a joint finite mask for one-dimensional arrays."""
    mask = np.ones(len(np.asarray(arrays[0])), dtype=bool)
    for a in arrays:
        a = np.asarray(a)
        if a.ndim == 1:
            mask &= np.isfinite(a)
        else:
            mask &= np.all(np.isfinite(a), axis=1)
    return mask


def zscore(a: np.ndarray) -> np.ndarray:
    """Sample-standardize a 1-D or 2-D array column-wise."""
    a = np.asarray(a, dtype=float)
    mu = np.nanmean(a, axis=0)
    sd = np.nanstd(a, axis=0, ddof=1)
    sd = np.where((~np.isfinite(sd)) | (sd == 0), 1.0, sd)
    return (a - mu) / sd


def fit_four_predictor_model(X: np.ndarray, y: np.ndarray):
    """OLS regression with intercept."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = finite_rows(X, y)
    Xc = sm.add_constant(X[mask], has_constant="add")
    return sm.OLS(y[mask], Xc).fit(), X[mask], y[mask]


def variance_contributions(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Relative reconstructed-variance contribution of each predictor.

    If component_i = beta_i * X_i, then
        Var(y_hat) = sum_i sum_j Cov(component_i, component_j)
    and symmetric covariance allocation gives
        C_i = sum_j Cov(component_i, component_j) / Var(y_hat).
    """
    result, Xv, _ = fit_four_predictor_model(X, y)
    beta = np.asarray(result.params[1:], dtype=float)
    components = Xv * beta[None, :]
    cov = np.cov(components, rowvar=False, ddof=1)
    total = float(np.sum(cov))
    if not np.isfinite(total) or np.isclose(total, 0):
        return np.full(X.shape[1], np.nan)
    return np.sum(cov, axis=1) / total


def signed_partial_r(y: np.ndarray, X: np.ndarray, index: int) -> float:
    """
    Signed partial correlation between y and X[:, index], controlling for all
    other predictors.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    mask = finite_rows(X, y)
    y = y[mask]
    X = X[mask]

    other = [j for j in range(X.shape[1]) if j != index]
    Z = sm.add_constant(X[:, other], has_constant="add")
    ry = sm.OLS(y, Z).fit().resid
    rx = sm.OLS(X[:, index], Z).fit().resid
    if np.nanstd(ry) == 0 or np.nanstd(rx) == 0:
        return np.nan
    return float(stats.pearsonr(rx, ry).statistic)


def moving_window_diagnostics(
    X: np.ndarray,
    y: np.ndarray,
    tcd: np.ndarray,
    years: np.ndarray,
    window: int = WINDOW,
) -> pd.DataFrame:
    """Calculate all Figure-1/Table-1 diagnostics in the same moving windows."""
    rows = []
    n = min(len(X), len(y), len(tcd), len(years))
    X, y, tcd, years = X[:n], y[:n], tcd[:n], years[:n]

    for start in range(0, n - window + 1):
        sl = slice(start, start + window)
        Xw, yw, tw = X[sl], y[sl], tcd[sl]
        valid = finite_rows(Xw, yw, tw)
        if valid.sum() < max(8, X.shape[1] + 3):
            continue

        Xv, yv, tv = Xw[valid], yw[valid], tw[valid]
        contributions = variance_contributions(Xv, yv)

        # Standardized beta is estimated within each 21-year window.
        Xz = zscore(Xv)
        yz = zscore(yv)
        standardized = sm.OLS(
            yz, sm.add_constant(Xz, has_constant="add")
        ).fit()

        rows.append(
            {
                "start_year": int(years[start]),
                "end_year": int(years[start + window - 1]),
                "TCD_G_mean_m": float(np.nanmean(tv)),
                "contribution_H": float(contributions[0]),
                "contribution_tau_x": float(contributions[2]),
                "beta_H": float(standardized.params[1]),
                "beta_tau_x": float(standardized.params[3]),
                "partial_R_H": signed_partial_r(yv, Xz, 0),
                "partial_R_tau_x": signed_partial_r(yv, Xz, 2),
            }
        )

    return pd.DataFrame(rows)


def aicc(result) -> float:
    n = int(result.nobs)
    k = int(len(result.params))
    if n <= k + 1:
        return np.inf
    return float(result.aic + 2 * k * (k + 1) / (n - k - 1))


def purged_block_cv_rmse(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    block_size: int = 10,
    purge: int = 20,
) -> float:
    """
    Contiguous block CV with a purge gap around each held-out block.

    Purging reduces leakage caused by strong overlap among adjacent 21-year
    moving-window samples.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = finite_rows(x, y)
    x, y = x[mask], y[mask]
    n = len(y)
    pred = np.full(n, np.nan)

    def design(v):
        if degree == 1:
            return sm.add_constant(v, has_constant="add")
        if degree == 2:
            return sm.add_constant(np.column_stack([v, v**2]), has_constant="add")
        raise ValueError("degree must be 1 or 2")

    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        test = np.arange(start, stop)
        lo = max(0, start - purge)
        hi = min(n, stop + purge)
        train = np.ones(n, dtype=bool)
        train[lo:hi] = False

        # Fall back to ordinary blocked CV if purging leaves too few points.
        min_train = 8 if degree == 2 else 6
        if train.sum() < min_train:
            train[:] = True
            train[test] = False

        if train.sum() < min_train:
            continue

        res = sm.OLS(y[train], design(x[train])).fit()
        pred[test] = res.predict(design(x[test]))

    valid = np.isfinite(pred)
    if valid.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((pred[valid] - y[valid]) ** 2)))


def relationship_fit(x: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    """
    Compare linear and quadratic relationships.

    Quadratic is selected only when all three conditions are met:
      (1) AICc improves by >= 2;
      (2) purged blocked-CV RMSE improves;
      (3) quadratic term is significant at p < 0.05 using HAC covariance.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = finite_rows(x, y)
    x, y = x[mask], y[mask]

    X1 = sm.add_constant(x, has_constant="add")
    X2 = sm.add_constant(np.column_stack([x, x**2]), has_constant="add")
    linear = sm.OLS(y, X1).fit()
    quadratic = sm.OLS(y, X2).fit()
    q_hac = quadratic.get_robustcov_results(
        cov_type="HAC", maxlags=min(HAC_LAG, max(1, len(y) - 2))
    )

    cv1 = purged_block_cv_rmse(x, y, 1)
    cv2 = purged_block_cv_rmse(x, y, 2)
    p_quad = float(q_hac.pvalues[2])

    choose_quad = (
        aicc(quadratic) <= aicc(linear) - 2
        and np.isfinite(cv2)
        and np.isfinite(cv1)
        and cv2 < cv1
        and p_quad < 0.05
    )

    xx = np.linspace(np.nanmin(x), np.nanmax(x), 300)
    selected = quadratic if choose_quad else linear
    XX = (
        sm.add_constant(np.column_stack([xx, xx**2]), has_constant="add")
        if choose_quad
        else sm.add_constant(xx, has_constant="add")
    )

    return {
        "name": "Quadratic" if choose_quad else "Linear",
        "linear": linear,
        "quadratic": quadratic,
        "x_curve": xx,
        "y_curve": selected.predict(XX),
        "AICc_linear": aicc(linear),
        "AICc_quadratic": aicc(quadratic),
        "CV_RMSE_linear": cv1,
        "CV_RMSE_quadratic": cv2,
        "p_quadratic_HAC": p_quad,
    }


def linear_hac_stats(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Linear slope and HAC inference for overlapping moving-window series."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = finite_rows(x, y)
    x, y = x[mask], y[mask]

    pearson = stats.pearsonr(x, y)
    model = sm.OLS(y, sm.add_constant(x, has_constant="add")).fit()
    hac = model.get_robustcov_results(
        cov_type="HAC", maxlags=min(HAC_LAG, max(1, len(y) - 2))
    )
    return {
        "n": len(y),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "slope": float(model.params[1]),
        "slope_se_HAC": float(hac.bse[1]),
        "p_HAC": float(hac.pvalues[1]),
    }


def load_dataset(ds: xr.Dataset, dataset_name: str):
    """Extract one reanalysis product from the portable input file."""
    names = [str(v) for v in ds["dataset"].values]
    if dataset_name not in names:
        raise KeyError(
            f"{dataset_name!r} not found. Available datasets: {', '.join(names)}"
        )
    sub = ds.sel(dataset=dataset_name)
    years = sub["year"].values.astype(int)
    X = np.column_stack([sub[v].values for v in PREDICTORS])
    y = sub["nino34_ond"].values
    tcd = sub["TCD_G"].values

    valid = finite_rows(X, y, tcd)
    return X[valid], y[valid], tcd[valid], years[valid]


def make_figure1(diag: pd.DataFrame, output: Path):
    """Main Figure 1: SODA-GODAS background-state relationships."""
    x = diag["TCD_G_mean_m"].to_numpy() / 100.0
    panels = [
        ("contribution_H", "Explained variance contribution of H"),
        ("contribution_tau_x", r"Explained variance contribution of $\tau_x$"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
    for label, (key, ylabel) in zip(("a", "b"), panels):
        y = diag[key].to_numpy()
        fit = relationship_fit(x, y)
        st = linear_hac_stats(x, y)

        axes[0 if label == "a" else 1].scatter(x, y, s=28, alpha=0.8)
        ax = axes[0 if label == "a" else 1]
        ax.plot(fit["x_curve"], fit["y_curve"], linewidth=2)
        ax.set_xlabel(r"Mean TCD_G ($\times 10^2$ m)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"({label})", loc="left", fontweight="bold")
        ax.grid(linestyle="--", linewidth=0.4, alpha=0.5)
        ax.text(
            0.04,
            0.96,
            f"r = {st['pearson_r']:.2f}\n"
            f"p(HAC) = {st['p_HAC']:.3g}\n"
            f"{fit['name']} relationship",
            transform=ax.transAxes,
            va="top",
        )

    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def make_appendix_figure1(
    diagnostics: Dict[str, pd.DataFrame],
    output: Path,
) -> pd.DataFrame:
    """
    Cross-reanalysis slopes of explained-variance contribution versus TCD_G.

    Returns the source-data table used by the plot.
    """
    rows = []
    for name, diag in diagnostics.items():
        x = diag["TCD_G_mean_m"].to_numpy() / 100.0
        for key, predictor in [
            ("contribution_H", "H"),
            ("contribution_tau_x", "tau_x"),
        ]:
            st = linear_hac_stats(x, diag[key].to_numpy())
            rows.append(
                {
                    "dataset": name,
                    "predictor": predictor,
                    **st,
                }
            )

    df = pd.DataFrame(rows)

    # Ensemble mean is the arithmetic mean across the three independent products.
    em_rows = []
    for predictor in ("H", "tau_x"):
        d = df[df["predictor"] == predictor]
        em_rows.append(
            {
                "dataset": "EM",
                "predictor": predictor,
                "slope": d["slope"].mean(),
                # This reproduces the original manuscript plotting convention:
                # mean uncertainty across products, not uncertainty of the ensemble mean.
                "slope_se_HAC": d["slope_se_HAC"].mean(),
            }
        )
    plot_df = pd.concat([df, pd.DataFrame(em_rows)], ignore_index=True)

    order = list(CORE_DATASETS) + ["EM"]
    xloc = np.arange(len(order))
    width = 0.34

    fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=300)
    for offset, predictor, label in [
        (-width / 2, "H", "H"),
        (width / 2, "tau_x", r"$\tau_x$"),
    ]:
        d = plot_df[plot_df["predictor"] == predictor].set_index("dataset").reindex(order)
        ax.bar(
            xloc + offset,
            d["slope"].to_numpy(),
            width,
            yerr=d["slope_se_HAC"].to_numpy(),
            capsize=4,
            label=label,
        )

    ax.axhline(0, linewidth=1)
    ax.set_xticks(xloc)
    ax.set_xticklabels(order)
    ax.set_ylabel(r"Regression coefficient ($10^2$ m$^{-1}$)")
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return plot_df


def make_appendix_table1(diagnostics: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Appendix Table 1: standardized beta and signed partial R vs mean TCD_G.
    """
    rows = []
    for dataset_name, diag in diagnostics.items():
        x = diag["TCD_G_mean_m"].to_numpy() / 100.0
        for metric, predictor in [
            ("beta_H", "H"),
            ("beta_tau_x", "tau_x"),
            ("partial_R_H", "H"),
            ("partial_R_tau_x", "tau_x"),
        ]:
            y = diag[metric].to_numpy()
            st = linear_hac_stats(x, y)
            fit = relationship_fit(x, y)
            rows.append(
                {
                    "Dataset": dataset_name,
                    "Diagnostic": (
                        "Standardized beta" if metric.startswith("beta_") else "Partial R"
                    ),
                    "Predictor": predictor,
                    "Estimate_with_TCD_G": st["pearson_r"],
                    "HAC_p": st["p_HAC"],
                    "Relationship": fit["name"],
                    "Linear_slope": st["slope"],
                    "Linear_slope_SE_HAC": st["slope_se_HAC"],
                    "Pearson_p": st["pearson_p"],
                    "AICc_linear": fit["AICc_linear"],
                    "AICc_quadratic": fit["AICc_quadratic"],
                    "CV_RMSE_linear": fit["CV_RMSE_linear"],
                    "CV_RMSE_quadratic": fit["CV_RMSE_quadratic"],
                    "p_quadratic_HAC": fit["p_quadratic_HAC"],
                }
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="figure1_input.nc",
        help="Portable input NetCDF created by build_figure1_input.py",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for figures and tables",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(input_path)

    diagnostics = {}
    for name in CORE_DATASETS:
        X, y, tcd, years = load_dataset(ds, name)
        diagnostics[name] = moving_window_diagnostics(X, y, tcd, years)

    # Figure 1 uses the primary SODA-GODAS record.
    make_figure1(diagnostics["SODA-GODAS"], out / "Figure1.png")

    # Source data for main Figure 1.
    diagnostics["SODA-GODAS"].to_csv(
        out / "Figure1_source_data.csv", index=False
    )

    # Appendix Figure 1.
    appendix_fig_source = make_appendix_figure1(
        diagnostics, out / "Appendix_Figure1.png"
    )
    appendix_fig_source.to_csv(
        out / "Appendix_Figure1_source_data.csv", index=False
    )

    # Appendix Table 1.
    table1 = make_appendix_table1(diagnostics)
    table1.to_csv(out / "Appendix_Table1.csv", index=False)

    print(f"Finished. Outputs written to: {out.resolve()}")
    print("\nAppendix Table 1:")
    cols = [
        "Dataset", "Diagnostic", "Predictor",
        "Estimate_with_TCD_G", "HAC_p", "Relationship"
    ]
    print(table1[cols].to_string(index=False))


if __name__ == "__main__":
    main()
