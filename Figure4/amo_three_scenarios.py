#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate three near-term AMO scenarios used in the future TCD_G projection.

The historical AMO index is processed exactly as in the analysis workflow:

    monthly AMO
        -> linear detrending
        -> standardization
        -> annual mean
        -> 21-year running mean

Three 10-year scenarios are then generated from the latest available
21-year-running-mean AMO state:

1. Persistence
       AMO(t+k) = AMO(t)

2. Damped persistence
       AMO(t+k) = phi_k * AMO(t)

   where phi_k is estimated from historical lag-k pairs using a no-intercept
   regression and clipped to [0, 1].

3. Recent linear trend
       A linear trend is fitted to the most recent 30 values of the
       21-year-running-mean AMO series and extrapolated for 10 future leads.

Year convention
---------------
The 21-year running mean is labeled by its center year in the output table.
For convenience, window_start and window_end are also written explicitly.

Example:
    target_center_year = 2016
    window_start       = 2006
    window_end         = 2026

The manuscript may display the window-ending year (2026) to emphasize the
forecast period.

Expected input
--------------
data/AMO_index_1854-2025.nc

Expected variable
-----------------
AMO_index

Usage
-----
python amo_three_scenarios.py \
    --input data/AMO_index_1854-2025.nc \
    --output outputs/AMO_future_10yr_three_methods_wide.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import detrend


FILE_START_YEAR = 1854
ANALYSIS_START_YEAR = 1920
ANALYSIS_END_YEAR = 2025
RUNNING_WINDOW = 21
FORECAST_LEADS = 10
TREND_WINDOW = 30


def standardize(x: np.ndarray) -> np.ndarray:
    """Standardize a 1-D series using the full selected historical period."""
    x = np.asarray(x, dtype=float)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd == 0:
        raise ValueError("Cannot standardize a constant or invalid AMO series.")
    return (x - np.nanmean(x)) / sd


def running_mean_forward(x: np.ndarray, window: int) -> np.ndarray:
    """
    Forward-indexed running mean used in the original analysis script.

    The first returned value is mean(x[0:window]). A center-year label is
    assigned later as start_year + window//2.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    if len(x) < window:
        raise ValueError(
            f"Series length {len(x)} is shorter than running window {window}."
        )

    return np.array(
        [np.nanmean(x[i:i + window]) for i in range(len(x) - window + 1)],
        dtype=float,
    )


def read_monthly_amo(
    path: Path,
    variable: str,
    file_start_year: int,
    analysis_start_year: int,
    analysis_end_year: int,
) -> np.ndarray:
    """
    Read a monthly AMO time series by positional index.

    This avoids depending on the NetCDF dimension name or datetime decoding.
    """
    ds = xr.open_dataset(path, decode_times=False)

    if variable not in ds:
        raise KeyError(
            f"Variable {variable!r} not found in {path}. "
            f"Available variables: {list(ds.data_vars)}"
        )

    da = ds[variable]
    arr = np.asarray(da.values, dtype=float).squeeze()

    if arr.ndim != 1:
        raise ValueError(
            f"{variable!r} must be one-dimensional after squeeze; "
            f"found shape {arr.shape}."
        )

    i0 = (analysis_start_year - file_start_year) * 12
    i1 = (analysis_end_year - file_start_year + 1) * 12

    if i0 < 0 or i1 > len(arr):
        raise ValueError(
            f"Requested {analysis_start_year}-{analysis_end_year}, but the "
            f"input series has {len(arr)} monthly samples from "
            f"file_start_year={file_start_year}."
        )

    out = arr[i0:i1]

    expected = (analysis_end_year - analysis_start_year + 1) * 12
    if len(out) != expected:
        raise ValueError(
            f"Extracted {len(out)} months; expected {expected}."
        )

    if not np.all(np.isfinite(out)):
        raise ValueError(
            "Selected AMO input contains NaN or infinite values. "
            "Please inspect the uploaded time series."
        )

    return out


def monthly_to_annual_mean(monthly: np.ndarray) -> np.ndarray:
    """Convert consecutive Jan-Dec monthly values to annual means."""
    monthly = np.asarray(monthly, dtype=float)
    if len(monthly) % 12 != 0:
        raise ValueError(
            f"Monthly series length {len(monthly)} is not divisible by 12."
        )
    return monthly.reshape(-1, 12).mean(axis=1)


def estimate_damped_persistence(
    amo_rm21: np.ndarray,
    n_leads: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate lead-dependent damped persistence.

    For each lead k:
        y(t+k) = phi_k * y(t)

    phi_k is estimated without an intercept and clipped to [0, 1].
    """
    last_value = float(amo_rm21[-1])
    forecast = []
    phi_values = []

    for k in range(1, n_leads + 1):
        x_hist = amo_rm21[:-k]
        y_hist = amo_rm21[k:]

        valid = np.isfinite(x_hist) & np.isfinite(y_hist)
        denominator = np.sum(x_hist[valid] ** 2)

        if valid.sum() < 3 or denominator <= 0:
            raise ValueError(
                f"Cannot estimate damped persistence coefficient for lead {k}."
            )

        phi_k = np.sum(x_hist[valid] * y_hist[valid]) / denominator
        phi_k = float(np.clip(phi_k, 0.0, 1.0))

        phi_values.append(phi_k)
        forecast.append(phi_k * last_value)

    return np.asarray(forecast), np.asarray(phi_values)


def generate_scenarios(
    amo_rm21: np.ndarray,
    center_years: np.ndarray,
    n_leads: int = FORECAST_LEADS,
    trend_window: int = TREND_WINDOW,
) -> pd.DataFrame:
    """Generate persistence, damped-persistence, and recent-trend scenarios."""
    last_value = float(amo_rm21[-1])
    last_center_year = int(center_years[-1])

    future_center_years = np.arange(
        last_center_year + 1,
        last_center_year + n_leads + 1,
        dtype=int,
    )

    persistence = np.repeat(last_value, n_leads)

    damped, phi = estimate_damped_persistence(
        amo_rm21,
        n_leads=n_leads,
    )

    if len(amo_rm21) < trend_window:
        raise ValueError(
            f"Need at least {trend_window} running-mean values for the recent "
            f"linear-trend scenario; found {len(amo_rm21)}."
        )

    recent_year = center_years[-trend_window:]
    recent_value = amo_rm21[-trend_window:]

    slope, intercept = np.polyfit(recent_year, recent_value, 1)
    recent_trend = slope * future_center_years + intercept

    out = pd.DataFrame(
        {
            "lead": np.arange(1, n_leads + 1, dtype=int),
            "target_center_year": future_center_years,
            "window_start": future_center_years - RUNNING_WINDOW // 2,
            "window_end": future_center_years + RUNNING_WINDOW // 2,
            "display_year": future_center_years + RUNNING_WINDOW // 2,
            "AMO_persistence": persistence,
            "AMO_damped_persistence": damped,
            "AMO_recent_linear_trend": recent_trend,
            "damped_phi": phi,
            "last_available_center_year": last_center_year,
            "last_available_window_start": last_center_year - RUNNING_WINDOW // 2,
            "last_available_window_end": last_center_year + RUNNING_WINDOW // 2,
            "last_available_AMO_z_rm21": last_value,
            "recent_trend_slope_per_year": slope,
            "recent_trend_intercept": intercept,
            "recent_trend_n_points": trend_window,
        }
    )

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Generate the three AMO scenarios used in the TCD_G projection."
    )
    parser.add_argument(
        "--input",
        default="data/AMO_index_1854-2025.nc",
        help="Monthly historical AMO NetCDF file.",
    )
    parser.add_argument(
        "--variable",
        default="AMO_index",
        help="AMO variable name in the NetCDF file.",
    )
    parser.add_argument(
        "--file-start-year",
        type=int,
        default=FILE_START_YEAR,
        help="Calendar year corresponding to the first 12 monthly samples.",
    )
    parser.add_argument(
        "--analysis-start-year",
        type=int,
        default=ANALYSIS_START_YEAR,
    )
    parser.add_argument(
        "--analysis-end-year",
        type=int,
        default=ANALYSIS_END_YEAR,
    )
    parser.add_argument(
        "--trend-window",
        type=int,
        default=TREND_WINDOW,
        help="Number of recent 21-year-running-mean values used for linear extrapolation.",
    )
    parser.add_argument(
        "--output",
        default="outputs/AMO_future_10yr_three_methods_wide.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input AMO file not found: {input_path}")

    monthly = read_monthly_amo(
        input_path,
        variable=args.variable,
        file_start_year=args.file_start_year,
        analysis_start_year=args.analysis_start_year,
        analysis_end_year=args.analysis_end_year,
    )

    # Preserve the original workflow exactly:
    # monthly detrend -> monthly standardization -> annual mean -> RM21.
    monthly_detrended = detrend(monthly, type="linear")
    monthly_standardized = standardize(monthly_detrended)
    annual = monthly_to_annual_mean(monthly_standardized)
    amo_rm21 = running_mean_forward(annual, RUNNING_WINDOW)

    rm21_start_year = np.arange(
        args.analysis_start_year,
        args.analysis_start_year + len(amo_rm21),
        dtype=int,
    )
    rm21_center_year = rm21_start_year + RUNNING_WINDOW // 2

    result = generate_scenarios(
        amo_rm21,
        rm21_center_year,
        n_leads=FORECAST_LEADS,
        trend_window=args.trend_window,
    )

    result.to_csv(output_path, index=False)

    print("AMO scenario generation completed.")
    print("Input:", input_path)
    print("Output:", output_path)
    print(
        f"Last available AMO window: "
        f"{int(result['last_available_window_start'].iloc[0])}-"
        f"{int(result['last_available_window_end'].iloc[0])}"
    )
    print("\nFuture scenarios:")
    print(
        result[
            [
                "lead",
                "target_center_year",
                "window_start",
                "window_end",
                "display_year",
                "AMO_persistence",
                "AMO_damped_persistence",
                "AMO_recent_linear_trend",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
