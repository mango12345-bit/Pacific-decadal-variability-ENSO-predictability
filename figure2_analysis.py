#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduce Figure 2 from a compact public input file.

Public input
------------
figure2_input.nc

Expected variables on dimension `window_end_year`:
    TCD_G_mean       21-year running-mean TCD_G (m)
    TCD_G_trend      fitted long-term linear component of TCD_G_mean (m)
    TCD_G_internal   detrended residual of TCD_G_mean (m)
    GWI              21-year running-mean standardized global warming index

Outputs
-------
outputs/Figure2.png
outputs/Figure2_source_data.csv
outputs/Figure2_statistics.txt

This public script does not read or redistribute the original SODA2.2.4,
GODAS, or Berkeley Earth files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm


def linear_fit(x, y):
    """OLS with intercept."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    X = sm.add_constant(x[valid], has_constant="add")
    return sm.OLS(y[valid], X).fit(), x[valid], y[valid]


def variance_decomposition(gwi, tcd_trend, tcd_internal):
    """
    Variance terms used in Figure 2c.

    The warming-associated component is the fitted TCD_G trend reconstructed
    from GWI. Internal variability is the detrended residual.
    """
    fit, x, y = linear_fit(gwi, tcd_trend)
    forced = fit.predict(sm.add_constant(x, has_constant="add"))

    var_forced = float(np.var(forced, ddof=1))
    var_internal = float(np.var(tcd_internal, ddof=1))
    var_total = var_forced + var_internal

    # Approximate 95% uncertainty propagated from the slope standard error,
    # retaining the convention of the original analysis.
    slope = float(fit.params[1])
    slope_se = float(fit.bse[1])
    var_x = float(np.var(x, ddof=1))
    var_forced_err95 = 1.96 * abs(2.0 * slope * var_x) * slope_se

    return {
        "fit": fit,
        "forced_series": forced,
        "var_forced": var_forced,
        "var_forced_err95": var_forced_err95,
        "var_internal": var_internal,
        "var_total": var_total,
    }


def make_figure(ds, output):
    years = ds["window_end_year"].values.astype(int)
    tcd_mean = ds["TCD_G_mean"].values.astype(float)
    tcd_trend = ds["TCD_G_trend"].values.astype(float)
    tcd_internal = ds["TCD_G_internal"].values.astype(float)
    gwi = ds["GWI"].values.astype(float)

    fit, xv, yv = linear_fit(gwi, tcd_trend)
    pearson = stats.pearsonr(xv, yv)
    vd = variance_decomposition(gwi, tcd_trend, tcd_internal)

    fig = plt.figure(figsize=(12.5, 6.2), dpi=300)
    grid = plt.GridSpec(7, 22, wspace=0.8, hspace=0.7)

    # -------------------------------------------------------------------------
    # (a) Low-frequency TCD_G and GWI
    # -------------------------------------------------------------------------
    ax = fig.add_subplot(grid[1:6, 0:13])
    h1, = ax.plot(
        years, tcd_mean, linestyle="--", linewidth=1.5, label="TCD_G mean state"
    )
    h2, = ax.plot(
        years, tcd_trend, linestyle="-", linewidth=2.0, label="Long-term trend"
    )
    ax.set_ylabel("Mean state of TCD_G (m)")
    ax.set_xlabel("Ending year of 21-year window")
    ax.set_title("(a) 21-year running mean of TCD_G and GWI", loc="left")
    ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.5)

    ax2 = ax.twinx()
    h3, = ax2.plot(years, gwi, linewidth=1.5, label="GWI")
    ax2.set_ylabel("Global warming index (standardized)")
    ax.legend(
        [h1, h2, h3],
        ["TCD_G mean state", "Long-term trend", "GWI"],
        frameon=False,
        fontsize=9,
        loc="best",
    )

    # -------------------------------------------------------------------------
    # (b) TCD_G trend versus GWI
    # -------------------------------------------------------------------------
    axb = fig.add_subplot(grid[0:3, 15:22])
    axb.scatter(gwi, tcd_trend, marker="o", s=18, alpha=0.7)

    order = np.argsort(gwi)
    x_sorted = gwi[order]
    X_sorted = sm.add_constant(x_sorted, has_constant="add")
    yfit_sorted = fit.predict(X_sorted)
    axb.plot(x_sorted, yfit_sorted, linewidth=2.2)

    axb.set_xlabel("Global warming index (standardized)")
    axb.set_ylabel("TCD_G long-term trend (m)")
    axb.set_title("(b) TCD_G trend and GWI", loc="left")
    axb.text(
        0.04, 0.95,
        f"slope = {fit.params[1]:.2f} ± {fit.bse[1]:.2f}\n"
        f"r = {pearson.statistic:.2f}, p = {pearson.pvalue:.2g}",
        transform=axb.transAxes,
        va="top",
    )
    axb.grid(linestyle=":", linewidth=0.5, alpha=0.5)

    # -------------------------------------------------------------------------
    # (c) Variance decomposition
    # -------------------------------------------------------------------------
    axc = fig.add_subplot(grid[4:7, 15:22])
    values = [vd["var_forced"], vd["var_internal"], vd["var_total"]]
    errors = [vd["var_forced_err95"], 0.0, 0.0]
    xpos = np.arange(3)

    axc.bar(xpos, values, yerr=errors, capsize=3)
    axc.set_xticks(xpos)
    axc.set_xticklabels(["Warming-associated", "Internal", "Total"])
    axc.set_ylabel("Variance (m$^2$)")
    axc.set_title("(c) Variance decomposition", loc="left")
    axc.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)

    return {
        "slope": float(fit.params[1]),
        "slope_se": float(fit.bse[1]),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "var_forced": vd["var_forced"],
        "var_forced_err95": vd["var_forced_err95"],
        "var_internal": vd["var_internal"],
        "var_total": vd["var_total"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="figure2_input.nc")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(args.input)

    required = {"TCD_G_mean", "TCD_G_trend", "TCD_G_internal", "GWI"}
    missing = required.difference(ds.data_vars)
    if missing:
        raise KeyError(f"Missing variables in input file: {sorted(missing)}")

    stats_out = make_figure(ds, out / "Figure2.png")

    pd.DataFrame(
        {
            "window_end_year": ds["window_end_year"].values,
            "TCD_G_mean": ds["TCD_G_mean"].values,
            "TCD_G_trend": ds["TCD_G_trend"].values,
            "TCD_G_internal": ds["TCD_G_internal"].values,
            "GWI": ds["GWI"].values,
        }
    ).to_csv(out / "Figure2_source_data.csv", index=False)

    with open(out / "Figure2_statistics.txt", "w", encoding="utf-8") as f:
        for key, value in stats_out.items():
            f.write(f"{key}: {value}\n")

    print(f"Finished. Outputs written to {out.resolve()}")
    for key, value in stats_out.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
