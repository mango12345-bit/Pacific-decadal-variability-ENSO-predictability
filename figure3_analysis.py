#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduce Figure 3, Appendix Figure 2, Appendix Figure 4, and Appendix Table 2.

Public/repository script. It reads only `figure3_input.nc`.

Expected variables
------------------
Dimension:
    year

Variables (year):
    TCD_G_detrended   detrended 21-year-running-mean TCD_G (m)
    AMO               standardized 21-year-running-mean detrended AMO
    PDO               standardized 21-year-running-mean detrended PDO

Scalar variables:
    warming_beta      standardized warming-forced regression coefficient
    warming_beta_se   standard error of warming_beta

Definitions used here
---------------------
APJPT(t) = standardize[AMO(t) * PDO(t)]

For reconstruction of Y(t):
    AMO(t-8)
    PDO(t)
    APJPT(t-4)

BRM = AMO(t-8) + PDO(t)
TRM = AMO(t-8) + PDO(t) + APJPT(t-4)

Outputs
-------
outputs/Figure3.png
outputs/Appendix_Figure2.png
outputs/Appendix_Figure4.png
outputs/Appendix_Table2.csv
plus source-data CSV files.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm


MAX_LAG = 15


def zscore(x):
    x = np.asarray(x, dtype=float)
    return (x - np.nanmean(x)) / np.nanstd(x, ddof=1)


def finite_mask(*arrs):
    m = np.ones(len(np.asarray(arrs[0])), dtype=bool)
    for a in arrs:
        a = np.asarray(a)
        if a.ndim == 1:
            m &= np.isfinite(a)
        else:
            m &= np.all(np.isfinite(a), axis=1)
    return m


def lead_lag_correlation(target, index, max_lag=MAX_LAG):
    """
    Correlation convention:
    positive lag => index leads target by that many years.
    """
    target = np.asarray(target, dtype=float)
    index = np.asarray(index, dtype=float)

    rows = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            y = target[:lag]
            x = index[-lag:]
        elif lag > 0:
            y = target[lag:]
            x = index[:-lag]
        else:
            y = target
            x = index

        valid = finite_mask(y, x)
        r, p = stats.pearsonr(x[valid], y[valid])
        rows.append({"lag": lag, "r": r, "p": p, "n": int(valid.sum())})
    return pd.DataFrame(rows)


def residualize(y, controls):
    y = np.asarray(y, dtype=float)
    controls = np.asarray(controls, dtype=float)
    if controls.ndim == 1:
        controls = controls[:, None]
    valid = finite_mask(y, controls)
    out = np.full(len(y), np.nan)
    res = sm.OLS(
        y[valid], sm.add_constant(controls[valid], has_constant="add")
    ).fit()
    out[valid] = res.resid
    return out


def partial_correlation(y, x, controls):
    ry = residualize(y, controls)
    rx = residualize(x, controls)
    valid = finite_mask(ry, rx)
    r, p = stats.pearsonr(rx[valid], ry[valid])
    return float(r), float(p), int(valid.sum()), rx, ry


def standardized_fit(y, X):
    """
    OLS using predictors standardized over the supplied sample.
    Target remains in physical units (m), matching the reconstruction framework.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    valid = finite_mask(y, X)
    yv, Xv = y[valid], X[valid]
    mu = Xv.mean(axis=0)
    sd = Xv.std(axis=0, ddof=1)
    sd[sd == 0] = 1
    Xz = (Xv - mu) / sd
    res = sm.OLS(yv, sm.add_constant(Xz, has_constant="add")).fit()
    pred = res.fittedvalues
    return res, pred, yv, mu, sd


def exhaustive_leave_two_out(y, X):
    """
    Classical exhaustive leave-two-out CV over every pair.

    Scaling is fit on the training fold only to avoid leakage.
    Each observation receives the mean of all predictions made when it is held out.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    valid = finite_mask(y, X)
    y, X = y[valid], X[valid]

    n = len(y)
    pred_sum = np.zeros(n)
    pred_n = np.zeros(n, dtype=int)

    for i, j in combinations(range(n), 2):
        test = np.array([i, j])
        train = np.ones(n, dtype=bool)
        train[test] = False

        mu = X[train].mean(axis=0)
        sd = X[train].std(axis=0, ddof=1)
        sd[sd == 0] = 1

        Xtr = (X[train] - mu) / sd
        Xte = (X[test] - mu) / sd

        res = sm.OLS(
            y[train], sm.add_constant(Xtr, has_constant="add")
        ).fit()
        pred = res.predict(sm.add_constant(Xte, has_constant="add"))

        pred_sum[test] += pred
        pred_n[test] += 1

    pred = pred_sum / pred_n
    r = stats.pearsonr(pred, y).statistic
    rmse = np.sqrt(np.mean((pred - y) ** 2))
    mae = np.mean(np.abs(pred - y))
    return pred, float(r), float(rmse), float(mae)


def align_reconstruction(year, y, amo, pdo):
    apjpt = zscore(amo * pdo)

    # Y(t), starting 8 years after the first available smoothed state.
    Y = y[8:]
    years = year[8:]

    AMO8 = amo[:-8]
    PDO0 = pdo[8:]
    APJPT4 = apjpt[4:-4]

    # Same-lag controls used only for stricter partial-regression test.
    AMO4 = amo[4:-4]
    PDO4 = pdo[4:-4]

    assert len(Y) == len(AMO8) == len(PDO0) == len(APJPT4)

    return {
        "year": years,
        "Y": Y,
        "AMO8": AMO8,
        "PDO0": PDO0,
        "APJPT4": APJPT4,
        "AMO4": AMO4,
        "PDO4": PDO4,
        "APJPT_all": apjpt,
    }


def make_appendix_figure2(y, amo, pdo, apjpt, out):
    frames = {
        "AMO": lead_lag_correlation(y, amo),
        "PDO": lead_lag_correlation(y, pdo),
        "APJPT": lead_lag_correlation(y, apjpt),
    }

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), dpi=300)
    for ax, (name, df), label in zip(axes, frames.items(), ["a", "b", "c"]):
        ax.bar(df["lag"], df["r"])
        ax.axhline(0, linewidth=1)
        ax.set_xlim(-MAX_LAG - 0.5, MAX_LAG + 0.5)
        ax.set_ylabel("Correlation")
        ax.set_xlabel(f"{name} lead TCD_G (years)")
        ax.set_title(f"({label}) {name}–TCD_G lead–lag relationship", loc="left")
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)

        # Mark p < 0.01 with asterisks, avoiding color-dependent significance encoding.
        sig = df["p"] < 0.01
        for _, row in df[sig].iterrows():
            offset = 0.025 if row["r"] >= 0 else -0.025
            va = "bottom" if row["r"] >= 0 else "top"
            ax.text(row["lag"], row["r"] + offset, "*", ha="center", va=va)

    fig.tight_layout()
    fig.savefig(out / "Appendix_Figure2.png", bbox_inches="tight")
    plt.close(fig)

    source = []
    for name, df in frames.items():
        x = df.copy()
        x.insert(0, "index", name)
        source.append(x)
    pd.concat(source, ignore_index=True).to_csv(
        out / "Appendix_Figure2_source_data.csv", index=False
    )


def make_appendix_figure4(aligned, out):
    Y = aligned["Y"]
    X = aligned["APJPT4"]

    controls_core = np.column_stack([aligned["AMO8"], aligned["PDO0"]])
    controls_strict = np.column_stack(
        [aligned["AMO8"], aligned["PDO0"], aligned["AMO4"], aligned["PDO4"]]
    )

    r1, p1, n1, x1, y1 = partial_correlation(Y, X, controls_core)
    r2, p2, n2, x2, y2 = partial_correlation(Y, X, controls_strict)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), dpi=300)
    source_rows = []

    specs = [
        (axes[0], x1, y1, r1, p1, "a", "AMO(t−8), PDO(t) removed"),
        (
            axes[1], x2, y2, r2, p2, "b",
            "AMO(t−8), PDO(t), AMO(t−4), PDO(t−4) removed",
        ),
    ]

    for ax, xr_, yr_, r, p, lab, title in specs:
        valid = finite_mask(xr_, yr_)
        xv, yv = xr_[valid], yr_[valid]
        coef = np.polyfit(xv, yv, 1)
        xx = np.linspace(xv.min(), xv.max(), 100)
        ax.scatter(xv, yv, s=25, alpha=0.8)
        ax.plot(xx, coef[0] * xx + coef[1], linestyle="--", linewidth=1.5)
        ax.axhline(0, linewidth=0.8, linestyle=":")
        ax.axvline(0, linewidth=0.8, linestyle=":")
        ax.set_xlabel("Residual APJPT")
        ax.set_ylabel("Residual TCD_G")
        ax.set_title(f"({lab}) {title}", loc="left", fontsize=10.5)
        ax.text(
            0.04, 0.95,
            f"partial r = {r:.2f}\np = {p:.3g}",
            transform=ax.transAxes, va="top"
        )

        for xxv, yyv in zip(xv, yv):
            source_rows.append(
                {
                    "panel": lab,
                    "residual_APJPT": xxv,
                    "residual_TCD_G": yyv,
                    "partial_r": r,
                    "p": p,
                }
            )

    fig.tight_layout()
    fig.savefig(out / "Appendix_Figure4.png", bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(source_rows).to_csv(
        out / "Appendix_Figure4_source_data.csv", index=False
    )


def make_table2(aligned, out):
    y = aligned["Y"]
    X_brm = np.column_stack([aligned["AMO8"], aligned["PDO0"]])
    X_trm = np.column_stack(
        [aligned["AMO8"], aligned["PDO0"], aligned["APJPT4"]]
    )

    rows = []
    prediction_source = {"year": aligned["year"], "observed": y}

    for name, X in [("BRM", X_brm), ("TRM", X_trm)]:
        res, fitted, yv, _, _ = standardized_fit(y, X)
        corr_full = stats.pearsonr(fitted, yv).statistic

        pred, corr_cv, rmse, mae = exhaustive_leave_two_out(y, X)
        prediction_source[f"{name}_LTOCV"] = pred

        rows.append(
            {
                "Model": name,
                "Predictors": "AMO+PDO" if name == "BRM" else "AMO+PDO+APJPT",
                "Corr_full": corr_full,
                "Corr_LTOCV": corr_cv,
                "RMSE_LTOCV": rmse,
                "MAE_LTOCV": mae,
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(out / "Appendix_Table2.csv", index=False)
    pd.DataFrame(prediction_source).to_csv(
        out / "Appendix_Table2_predictions.csv", index=False
    )
    return table


def make_figure3(aligned, full_year, full_y, amo, pdo, warming_beta, warming_se, out):
    y = aligned["Y"]
    years = aligned["year"]
    X_brm = np.column_stack([aligned["AMO8"], aligned["PDO0"]])
    X_trm = np.column_stack(
        [aligned["AMO8"], aligned["PDO0"], aligned["APJPT4"]]
    )

    res_brm, fit_brm, _, _, _ = standardized_fit(y, X_brm)
    res_trm, fit_trm, _, _, _ = standardized_fit(y, X_trm)

    # Full timeline for panel a; regression timeline starts 8 years later.
    apjpt_all = aligned["APJPT_all"]

    fig = plt.figure(figsize=(12, 7), dpi=300)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], hspace=0.35, wspace=0.32)

    # (a)
    ax = fig.add_subplot(grid[0, 0])
    ax.plot(full_year, full_y, linestyle="--", linewidth=1.5, label="Detrended TCD_G")
    ax2 = ax.twinx()
    ax2.plot(full_year, amo, linewidth=1.2, label="AMO")
    ax2.plot(full_year, pdo, linewidth=1.2, linestyle="--", label="PDO")
    ax2.plot(full_year, apjpt_all, linewidth=1.2, linestyle=":", label="APJPT")
    ax.set_xlabel("Year")
    ax.set_ylabel("Detrended TCD_G (m)")
    ax2.set_ylabel("Standardized decadal index")
    ax.set_title("(a) Detrended TCD_G and climate variability", loc="left")
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.5)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, ncol=2, fontsize=8)

    # (b)
    ax = fig.add_subplot(grid[0, 1])
    ax.scatter(fit_brm, y, s=22, label="BRM")
    ax.scatter(fit_trm, y, s=22, marker="s", label="TRM")
    lim = [
        min(np.min(y), np.min(fit_brm), np.min(fit_trm)),
        max(np.max(y), np.max(fit_brm), np.max(fit_trm)),
    ]
    ax.plot(lim, lim, linewidth=1, linestyle=":")
    ax.set_xlabel("Reconstructed detrended TCD_G (m)")
    ax.set_ylabel("Observed detrended TCD_G (m)")
    ax.set_title("(b) Reconstruction performance", loc="left")
    ax.legend(frameon=False)

    # (c)
    ax = fig.add_subplot(grid[1, :])
    brm_coef = res_brm.params[1:]
    trm_coef = res_trm.params[1:]
    brm_se = res_brm.bse[1:]
    trm_se = res_trm.bse[1:]

    positions = np.arange(4)
    width = 0.28

    ax.bar(
        positions[:2] - width / 2,
        brm_coef,
        width,
        yerr=1.96 * brm_se,
        capsize=3,
        label="BRM",
    )
    ax.bar(
        positions[:3] + width / 2,
        trm_coef,
        width,
        yerr=1.96 * trm_se,
        capsize=3,
        label="TRM",
    )
    ax.bar(
        positions[3],
        warming_beta,
        width,
        yerr=1.96 * warming_se,
        capsize=3,
        label="Warming forced",
    )
    ax.axhline(0, linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(["AMO", "PDO", "APJPT", "Warming forced"])
    ax.set_ylabel("Standardized regression coefficient")
    ax.set_title("(c) Regression coefficients with 95% confidence intervals", loc="left")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.legend(frameon=False, ncol=3)

    fig.tight_layout()
    fig.savefig(out / "Figure3.png", bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(
        {
            "year": years,
            "TCD_G_detrended": y,
            "BRM_reconstructed": fit_brm,
            "TRM_reconstructed": fit_trm,
            "AMO_t_minus_8": aligned["AMO8"],
            "PDO_t": aligned["PDO0"],
            "APJPT_t_minus_4": aligned["APJPT4"],
        }
    ).to_csv(out / "Figure3_source_data.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="figure3_input.nc")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(args.input)
    year = ds["year"].values.astype(int)
    y = ds["TCD_G_detrended"].values.astype(float)
    amo = ds["AMO"].values.astype(float)
    pdo = ds["PDO"].values.astype(float)

    valid = finite_mask(y, amo, pdo)
    year, y, amo, pdo = year[valid], y[valid], amo[valid], pdo[valid]

    warming_beta = float(ds["warming_beta"].values)
    warming_se = float(ds["warming_beta_se"].values)

    apjpt = zscore(amo * pdo)
    aligned = align_reconstruction(year, y, amo, pdo)

    make_figure3(aligned, year, y, amo, pdo, warming_beta, warming_se, out)
    make_appendix_figure2(y, amo, pdo, apjpt, out)
    make_appendix_figure4(aligned, out)
    table2 = make_table2(aligned, out)

    print(f"Finished. Outputs written to {out.resolve()}")
    print("\nAppendix Table 2:")
    print(table2.to_string(index=False))


if __name__ == "__main__":
    main()
