#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduce Appendix Figure 3: piControl lead-lag relationships between
AMO/PDO and the tropical-Pacific subsurface mean-state index.

Public input
------------
A directory containing one NetCDF file per CMIP6 piControl model:

    <MODEL>_piControl_indices.nc

Each file must contain:
    AMO_rm21
    PDO_rm21
    Subsurface_rm21

These variables should already represent:
    annual index -> linear detrending -> 21-year running mean

This script does not detrend or smooth the indices again.

Lead convention
---------------
Positive lag means that AMO/PDO leads the subsurface mean state:

    corr(index(t), subsurface(t + lag))

Outputs
-------
Appendix_Figure3.png
Appendix_Figure3_source_data.csv
Appendix_Figure3_multimodel_summary.csv
Appendix_Figure3_best_lag_summary.csv

The figure shows:
    gray lines  : individual models
    black line  : multi-model mean
    gray shading: ±1 inter-model standard deviation
    red star    : lag with strongest absolute multi-model-mean correlation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats as stats
import matplotlib.pyplot as plt


DEFAULT_MAX_LAG = 15
MISSING_ABS_THRESHOLD = 1.0e20
MISSING_LOW_THRESHOLD = -900.0


def clean_missing(x):
    """Convert common NetCDF/NCL missing values to NaN."""
    x = np.asarray(x, dtype=float).squeeze()
    x = np.where(np.abs(x) > MISSING_ABS_THRESHOLD, np.nan, x)
    x = np.where(x <= MISSING_LOW_THRESHOLD, np.nan, x)
    return x


def read_1d_variable(ds, name):
    """Read one required 1-D index variable."""
    if name not in ds:
        raise KeyError(f"Required variable {name!r} not found.")
    x = clean_missing(ds[name].values)
    if x.ndim != 1:
        raise ValueError(
            f"{name!r} must be 1-D after squeeze; got shape {x.shape}."
        )
    return x


def corr_pairwise_nan(x, y, min_pairs=6):
    """Pearson correlation after pairwise removal of missing values."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < min_pairs:
        return np.nan
    return float(stats.pearsonr(x[valid], y[valid]).statistic)


def lead_lag_corr(index, target, max_lag=DEFAULT_MAX_LAG):
    """
    Lead-lag correlation.

    lag > 0:
        index leads target:
        corr(index(t), target(t + lag))

    lag < 0:
        target leads index.
    """
    index = np.asarray(index, dtype=float)
    target = np.asarray(target, dtype=float)

    n = min(len(index), len(target))
    index = index[:n]
    target = target[:n]

    lags = np.arange(-max_lag, max_lag + 1, dtype=int)
    corrs = np.full(lags.size, np.nan, dtype=float)

    for i, lag in enumerate(lags):
        if lag > 0:
            x = index[:-lag]
            y = target[lag:]
        elif lag < 0:
            x = index[-lag:]
            y = target[:lag]
        else:
            x = index
            y = target

        corrs[i] = corr_pairwise_nan(x, y)

    return lags, corrs


def model_name(path):
    suffix = "_piControl_indices.nc"
    return path.name[:-len(suffix)] if path.name.endswith(suffix) else path.stem


def strongest_abs_lag(lags, mean_corr):
    """Return lag and r at strongest absolute multi-model-mean correlation."""
    valid = np.isfinite(mean_corr)
    if not valid.any():
        return np.nan, np.nan, None
    valid_idx = np.where(valid)[0]
    idx = valid_idx[np.nanargmax(np.abs(mean_corr[valid]))]
    return int(lags[idx]), float(mean_corr[idx]), int(idx)


def collect_model_correlations(input_dir, max_lag):
    files = sorted(input_dir.glob("*_piControl_indices.nc"))
    if not files:
        raise FileNotFoundError(
            f"No '*_piControl_indices.nc' files found in {input_dir}"
        )

    rows = []
    amo_all = []
    pdo_all = []
    used_models = []

    for path in files:
        model = model_name(path)
        print(f"Processing {model}")

        ds = xr.open_dataset(path)
        amo = read_1d_variable(ds, "AMO_rm21")
        pdo = read_1d_variable(ds, "PDO_rm21")
        subsurface = read_1d_variable(ds, "Subsurface_rm21")

        n = min(len(amo), len(pdo), len(subsurface))
        amo = amo[:n]
        pdo = pdo[:n]
        subsurface = subsurface[:n]

        lags, amo_corr = lead_lag_corr(amo, subsurface, max_lag=max_lag)
        _, pdo_corr = lead_lag_corr(pdo, subsurface, max_lag=max_lag)

        if np.all(~np.isfinite(amo_corr)) and np.all(~np.isfinite(pdo_corr)):
            print(f"  WARNING: {model}: no valid correlations; skipped.")
            continue

        amo_all.append(amo_corr)
        pdo_all.append(pdo_corr)
        used_models.append(model)

        for lag, r_amo, r_pdo in zip(lags, amo_corr, pdo_corr):
            rows.append(
                {
                    "model": model,
                    "lag_year": int(lag),
                    "corr_AMO_subsurface": r_amo,
                    "corr_PDO_subsurface": r_pdo,
                }
            )

    if not used_models:
        raise RuntimeError("No valid model available for analysis.")

    return (
        lags,
        np.asarray(amo_all, dtype=float),
        np.asarray(pdo_all, dtype=float),
        used_models,
        pd.DataFrame(rows),
    )


def multimodel_summary(lags, amo_mat, pdo_mat):
    amo_mean = np.nanmean(amo_mat, axis=0)
    pdo_mean = np.nanmean(pdo_mat, axis=0)

    # ddof=1 is appropriate when at least two models are available.
    amo_std = np.nanstd(amo_mat, axis=0, ddof=1)
    pdo_std = np.nanstd(pdo_mat, axis=0, ddof=1)

    return pd.DataFrame(
        {
            "lag_year": lags,
            "AMO_multi_model_mean": amo_mean,
            "AMO_multi_model_std": amo_std,
            "PDO_multi_model_mean": pdo_mean,
            "PDO_multi_model_std": pdo_std,
        }
    )


def best_lag_summary(lags, summary):
    rows = []
    for label, col in [
        ("AMO", "AMO_multi_model_mean"),
        ("PDO", "PDO_multi_model_mean"),
    ]:
        lag, r, _ = strongest_abs_lag(lags, summary[col].to_numpy())
        rows.append(
            {
                "index": label,
                "best_abs_lag_year": lag,
                "corr_at_best_abs_lag": r,
            }
        )
    return pd.DataFrame(rows)


def plot_panel(ax, lags, corr_mat, mean_corr, std_corr, label, panel_label):
    for row in corr_mat:
        ax.plot(lags, row, linewidth=0.8, alpha=0.55)

    ax.fill_between(
        lags,
        mean_corr - std_corr,
        mean_corr + std_corr,
        alpha=0.18,
        linewidth=0,
        label=r"$\pm$1 inter-model s.d.",
    )
    ax.plot(
        lags,
        mean_corr,
        linewidth=2.0,
        label="Multi-model mean",
    )

    best_lag, best_r, best_idx = strongest_abs_lag(lags, mean_corr)
    if best_idx is not None:
        ax.scatter(
            [best_lag],
            [best_r],
            marker="*",
            s=120,
            zorder=5,
        )
        ax.annotate(
            f"lag = {best_lag} yr\nr = {best_r:.2f}",
            xy=(best_lag, best_r),
            xytext=(8, 8 if best_r >= 0 else -28),
            textcoords="offset points",
            fontsize=10,
        )

    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1, linestyle=":")
    ax.set_xlim(lags.min(), lags.max())
    ax.set_ylim(-1, 1)
    ax.set_xlabel(f"{label} lead (years)")
    ax.set_ylabel("Correlation")
    ax.set_title(
        f"({panel_label}) {label} vs subsurface mean state",
        loc="left",
        fontweight="bold",
    )
    ax.grid(linestyle=":", linewidth=0.6, alpha=0.5)
    ax.legend(frameon=False, fontsize=9)


def make_figure(lags, amo_mat, pdo_mat, summary, output):
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), dpi=300)

    plot_panel(
        axes[0],
        lags,
        amo_mat,
        summary["AMO_multi_model_mean"].to_numpy(),
        summary["AMO_multi_model_std"].to_numpy(),
        "AMO",
        "a",
    )
    plot_panel(
        axes[1],
        lags,
        pdo_mat,
        summary["PDO_multi_model_mean"].to_numpy(),
        summary["PDO_multi_model_std"].to_numpy(),
        "PDO",
        "b",
    )

    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Appendix Figure 3 from processed piControl indices."
    )
    parser.add_argument(
        "--input-dir",
        default="picontrol_indices",
        help="Directory containing *_piControl_indices.nc files.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for figure and source-data files.",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=DEFAULT_MAX_LAG,
        help="Maximum lead/lag in years (default: 15).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lags, amo_mat, pdo_mat, models, source = collect_model_correlations(
        input_dir, args.max_lag
    )
    summary = multimodel_summary(lags, amo_mat, pdo_mat)
    summary["n_models"] = len(models)

    best = best_lag_summary(lags, summary)

    source.to_csv(
        output_dir / "Appendix_Figure3_source_data.csv",
        index=False,
    )
    summary.to_csv(
        output_dir / "Appendix_Figure3_multimodel_summary.csv",
        index=False,
    )
    best.to_csv(
        output_dir / "Appendix_Figure3_best_lag_summary.csv",
        index=False,
    )

    make_figure(
        lags,
        amo_mat,
        pdo_mat,
        summary,
        output_dir / "Appendix_Figure3.png",
    )

    print(f"\nUsed models: {len(models)}")
    print(", ".join(models))
    print("\nStrongest multi-model-mean relationships:")
    print(best.to_string(index=False))
    print(f"\nOutputs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
