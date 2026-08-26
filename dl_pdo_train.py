# -*- coding: utf-8 -*-
"""
Train the 20-member DL-PDO ensemble used in the manuscript.

This public training script preserves the workflow used in the study:

1. Read monthly CMIP6 and observational time-series indices:
   AMO, PDO, AO, OHC, and central-Pacific SSHA.
2. Fit detrending/standardization parameters using training/calibration data only.
3. Convert monthly indices to annual/seasonal indices.
4. Apply a centered 21-year running mean.
5. Construct 10-year input / 10-year target sequence samples.
6. Pretrain the residual Seq2Seq-attention model on CMIP6.
7. Apply two-step transfer learning to observations.
8. Train a 20-member ensemble using CMIP6 model bootstrap resampling.
9. Estimate member-specific lead-wise Delta-PDO calibration coefficients.
10. Save the final H5 models, ensemble metadata, hindcast diagnostics, and
    near-term PDO prediction.

By default, the script reads the previously selected hyperparameters from:
    metadata/best_optuna_params_cmip6.csv

To repeat the Optuna search instead, run:
    python dl_pdo_train.py --run-optuna

Expected repository layout
--------------------------
DL_PDO/
├── dl_pdo_train.py
├── data/
│   ├── AMO_index_model.nc
│   ├── pdo_model_series.nc
│   ├── ao_model_series.nc
│   ├── ohc_index_model.nc
│   ├── ssha_model_index_cp.nc
│   ├── AMO_index_1854-2025.nc
│   ├── pdo_index.1854-2025.nc
│   ├── ao_series.nc
│   ├── ohc_index_cp.nc
│   └── ssha_index_cp.nc
├── metadata/
│   └── best_optuna_params_cmip6.csv
├── models/
└── outputs/

Notes
-----
* The neural network predicts Delta PDO relative to a persistence baseline.
* Final PDO = persistence + member-specific calibrated Delta PDO.
* The predictand is the detrended, standardized centered 21-year-running-mean PDO.
* For a centered 21-year mean, center year Y represents the window Y-10 ... Y+10.
* The observation test period is excluded from scaling, Optuna tuning, and
  transfer learning.
"""

import os
import random
import warnings
import argparse
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats as sts
import tensorflow as tf
import optuna
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K

# =============================================================================
# Command-line configuration
# =============================================================================
parser = argparse.ArgumentParser(
    description="Train the 20-member DL-PDO ensemble used in the manuscript."
)
parser.add_argument(
    "--data-dir",
    default="data",
    help="Directory containing the 10 uploaded NetCDF time-series files."
)
parser.add_argument(
    "--metadata-dir",
    default="metadata",
    help="Directory containing best_optuna_params_cmip6.csv and receiving ensemble metadata."
)
parser.add_argument(
    "--model-dir",
    default="models",
    help="Directory in which trained H5 ensemble members are saved."
)
parser.add_argument(
    "--output-dir",
    default="outputs",
    help="Directory for hindcast, forecast, and diagnostic outputs."
)
parser.add_argument(
    "--run-optuna",
    action="store_true",
    help="Repeat the Optuna hyperparameter search instead of reading the saved best-parameter CSV."
)
parser.add_argument(
    "--n-trials",
    type=int,
    default=50,
    help="Number of Optuna trials when --run-optuna is supplied."
)
parser.add_argument(
    "--n-ensemble",
    type=int,
    default=20,
    help="Number of ensemble members to train (manuscript setting: 20)."
)
parser.add_argument(
    "--cpu",
    action="store_true",
    help="Disable GPU use."
)
args = parser.parse_args()

DATA_DIR = Path(args.data_dir)
METADATA_DIR = Path(args.metadata_dir)
MODEL_SAVE_DIR = Path(args.model_dir)
OUTPUT_DIR = Path(args.output_dir)

for directory in (METADATA_DIR, MODEL_SAVE_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Runtime / GPU configuration
# =============================================================================
USE_GPU = not args.cpu             # GPU by default; use --cpu to disable
GPU_MEMORY_GROWTH = True        # GPU内存增长
RANDOM_SEED = 42                # 随机种子

if USE_GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            if GPU_MEMORY_GROWTH:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU devices available: {}".format(len(gpus)))
            for i, gpu in enumerate(gpus):
                print("  GPU {}: {}".format(i, gpu.name))
        except RuntimeError as e:
            print("GPU configuration warning: {}".format(e))
    else:
        print("No GPU found; using CPU.")
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print("CPU mode.")

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
tf.config.optimizer.set_jit(True)
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

HIST_STEPS = 10
FUTURE_STEPS = 10
RUN_WRAP = 21
RUNMEAN_MODE = "center"     # 21-year centered running mean; year Y = mean(Y-10 ... Y+10).
SPLIT_MODE = "center_year"  # "center_year" keeps more obs samples; "strict_window" uses real 21-year window boundaries.
MAX_MISSING_FRAC_MODEL = 0.05
MAX_MISSING_FRAC_OBS = 0.05

N_TRIALS = args.n_trials
N_ENSEMBLE = args.n_ensemble

# Ensemble diversity strategy.
# "model_bootstrap"    : resample CMIP6 training models with replacement.
# "model_subsample"    : sample a large fraction of CMIP6 training models without replacement;
#                        this is more stable than bootstrap and is recommended when ensemble
#                        members differ too strongly in smoke tests.
# "sample_bootstrap"   : resample individual samples; less physically clean because samples overlap.
# "none"               : all members use the same source training set; spread then comes only from NN stochasticity.
ENSEMBLE_SOURCE_RESAMPLING = "model_bootstrap"
FIRST_MEMBER_USE_FULL_SOURCE = True
SOURCE_SUBSAMPLE_FRACTION = 0.85
MIN_UNIQUE_MODEL_FRACTION = 0.70
BOOTSTRAP_MAX_ATTEMPTS = 100

# Persistence-correction target.
# The network predicts only the future correction relative to the latest
# observed/input PDO low-frequency state:
#     PDO_hat(t+k) = PDO_last_input + Delta_hat(t+k)
# This is more appropriate for 21-year centered-running-mean PDO, for which
# persistence is a very strong baseline.
CORRECTION_L2_WEIGHT = 0.10
DELTA_TREND_WEIGHT = 0.15
DELTA_MSE_WEIGHT = 0.85

# v5-leadwise setting:
# Keep the v5 Seq2Seq-attention architecture, but train it directly on Delta PDO
# with a simple MSE loss. Then apply a lead-dependent amplitude calibration:
#     Delta_final(k) = alpha_k * Delta_raw(k)
# estimated from the observation fine-tune validation subset.
USE_SIMPLE_DELTA_MSE_LOSS = True
USE_LEADWISE_DELTA_CALIBRATION = True
DELTA_ALPHA_MIN = 0.0
DELTA_ALPHA_MAX = 1.0

# Conservative lead-wise calibration. These are applied only on the observation
# fine-tune validation subset, never on the test subset.
# Positive scaling preserves the predicted Delta-PDO phase/sign information, but
# reduces amplitude over-shooting, which is the main reason why the previous v5
# had high correlation but larger RMSE than persistence.
DELTA_ALPHA_SHRINK = 0.85
MIN_CALIB_DELTA_CORR = 0.10
MIN_CALIB_SIGN_ACCURACY = 0.50

# Residual architecture settings.
# The model predicts Delta PDO. A linear AR-style residual head based on recent
# PDO history is added to the nonlinear Seq2Seq-attention Delta branch:
#     Delta_pred = Delta_nonlinear + Delta_AR_residual
# This is useful for smooth low-frequency sequences because recent PDO tendency
# often provides a strong first-order correction to persistence.
USE_OUTPUT_AR_RESIDUAL = True

MODEL_START_YEAR = 1850
MODEL_END_YEAR = 2014
OBS_START_YEAR = 1871
OBS_END_YEAR = 2025
OBS_TRANSFER_END = 1977
OBS_TEST_START = 1978


MODEL_NAMES = [
    "ACCESS-CM2","AWI-ESM-1-1-LR","BCC-CSM2-MR","CAMS-CSM1-0","CanESM5","CAS-ESM2-0",
    "CESM2-FV2","CESM2-WACCM-FV2","CESM2-WACCM","CESM2","CIESM",
    "CMCC-ESM2","E3SM-1-0","E3SM-1-1","EC-Earth3","FGOALS-f3-L",
    "FGOALS-g3","GFDL-ESM4","GISS-E2-1-G","GISS-E2-1-H","INM-CM5-0",
    "IPSL-CM6A-LR","KIOST-ESM","MIROC6","MPI-ESM1-2-HR","MRI-ESM2-0",
    "NESM3","NorCPM1","TaiESM1"
]

FEATURES = ["AMO", "PDO", "AO", "OHC", "SSHA"]

MODEL_FILES = {
    "AMO": DATA_DIR / "AMO_index_model.nc",
    "PDO": DATA_DIR / "pdo_model_series.nc",
    "AO": DATA_DIR / "ao_model_series.nc",
    "OHC": DATA_DIR / "ohc_index_model.nc",
    "SSHA": DATA_DIR / "ssha_model_index_cp.nc",
}
MODEL_VARS = {
    "AMO": "AMO_index",
    "PDO": "eof_ts",
    "AO": "eof_ts",
    "OHC": "ohc_index",
    "SSHA": "ssha_index",
}

OBS_FILES = {
    "AMO": DATA_DIR / "AMO_index_1854-2025.nc",
    "PDO": DATA_DIR / "pdo_index.1854-2025.nc",
    "AO": DATA_DIR / "ao_series.nc",
    "OHC": DATA_DIR / "ohc_index_cp.nc",
    "SSHA": DATA_DIR / "ssha_index_cp.nc",
}
OBS_VARS = {
    "AMO": "AMO_index",
    "PDO": "PDO_index",
    "AO": "eof_ts",
    "OHC": "ohc_index",
    "SSHA": "ssha_index",
}
OBS_FILE_START_YEAR = {
    "AMO": 1854,
    "PDO": 1854,
    "AO": 1854,
    "OHC": 1871,
    "SSHA": 1871,
}
INDEX_SEASON = {
    "AMO": "ANN",
    "PDO": "ANN",
    "AO": "DJF",
    "OHC": "ANN",
    "SSHA": "MAM",
}


missing_records = []


def sanitize_missing_values(x):
    """Convert NetCDF fill values, infinities, and very large sentinels to NaN."""
    arr = np.asarray(x, dtype=np.float32).copy()
    arr[~np.isfinite(arr)] = np.nan
    arr[np.abs(arr) > 1.0e19] = np.nan
    arr[arr <= -9.0e33] = np.nan
    return arr


def checked_fill_1d(x, source, feature, index=-1, max_missing_frac=0.05):
    """
    Check a raw monthly series before any detrending/scaling.
    If the series is all missing or has too many missing values, stop.
    Small missing gaps are linearly interpolated.
    """
    arr = sanitize_missing_values(x)
    n = arr.size
    nmiss = int(np.isnan(arr).sum())
    frac = nmiss / n if n > 0 else 1.0

    missing_records.append({
        "source": source,
        "feature": feature,
        "index": int(index),
        "n": int(n),
        "n_missing": int(nmiss),
        "missing_fraction": float(frac),
    })

    if n == 0:
        raise ValueError(f"{source} {feature} index={index}: empty series.")

    if nmiss == n:
        pd.DataFrame(missing_records).to_csv(
            os.path.join(OUTPUT_DIR, "missing_diagnostics_until_error.csv"),
            index=False,
        )
        raise ValueError(f"{source} {feature} index={index}: all values are missing.")

    if frac > max_missing_frac:
        pd.DataFrame(missing_records).to_csv(
            os.path.join(OUTPUT_DIR, "missing_diagnostics_until_error.csv"),
            index=False,
        )
        raise ValueError(
            f"{source} {feature} index={index}: missing fraction {frac:.3f} "
            f"exceeds threshold {max_missing_frac:.3f}."
        )

    if nmiss > 0:
        print(
            f"WARNING: {source} {feature} index={index} has {nmiss}/{n} "
            f"missing values ({frac:.3%}); filling by linear interpolation."
        )

    out = pd.Series(arr).interpolate(method="linear", limit_direction="both").values.astype(np.float32)

    if not np.all(np.isfinite(out)):
        raise ValueError(f"{source} {feature} index={index}: interpolation left NaNs.")

    return out


def checked_fill_panel(a, source, feature, max_missing_frac=0.05):
    a = sanitize_missing_values(a)
    if a.ndim == 1:
        return checked_fill_1d(a, source, feature, -1, max_missing_frac)
    if a.ndim != 2:
        raise ValueError(f"{source} {feature}: expected 1D/2D array, got shape {a.shape}.")
    out = np.empty_like(a, dtype=np.float32)
    for i in range(a.shape[0]):
        out[i, :] = checked_fill_1d(a[i, :], source, feature, i, max_missing_frac)
    return out


def save_missing_diagnostics():
    out_csv = os.path.join(OUTPUT_DIR, "missing_diagnostics_raw_monthly.csv")
    pd.DataFrame(missing_records).to_csv(out_csv, index=False)
    print("Saved missing diagnostics:", out_csv)


# Backward-compatible aliases used in the scaler.
def fill_1d(x):
    return pd.Series(sanitize_missing_values(x)).interpolate(method="linear", limit_direction="both").values.astype(np.float32)


def fill_panel(a):
    a = sanitize_missing_values(a)
    if a.ndim == 1:
        return fill_1d(a)
    out = np.empty_like(a, dtype=np.float32)
    for i in range(a.shape[0]):
        out[i] = fill_1d(a[i])
    return out


@dataclass
class ScaleParams:
    slope: float
    intercept: float
    mean: float
    std: float
    ref_year0: float


class MonthlyDetrendStandardizer:
    def __init__(self, name=""):
        self.name = name
        self.params = None

    def fit(self, x_monthly, year_axis):
        x = np.asarray(x_monthly, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        x = fill_panel(x)
        year_axis = np.asarray(year_axis, dtype=np.float64)

        t = year_axis - year_axis[0]
        tt = np.tile(t[None, :], (x.shape[0], 1)).ravel()
        yy = x.ravel()
        valid = np.isfinite(tt) & np.isfinite(yy)
        slope, intercept = np.polyfit(tt[valid], yy[valid], 1)

        detr = yy[valid] - (slope * tt[valid] + intercept)
        mu = float(np.mean(detr))
        sd = float(np.std(detr))
        if not np.isfinite(sd) or sd == 0:
            sd = 1.0

        self.params = ScaleParams(float(slope), float(intercept), mu, sd, float(year_axis[0]))
        return self

    def transform(self, x_monthly, year_axis):
        if self.params is None:
            raise RuntimeError("Scaler has not been fitted.")
        x = np.asarray(x_monthly, dtype=np.float64)
        ndim = x.ndim
        if x.ndim == 1:
            x = x[None, :]
        x = fill_panel(x)
        year_axis = np.asarray(year_axis, dtype=np.float64)
        t = year_axis - self.params.ref_year0
        trend = self.params.slope * t + self.params.intercept
        z = (x - trend[None, :] - self.params.mean) / self.params.std
        z = z.astype(np.float32)
        return z[0] if ndim == 1 else z


def month_year_axis(start_year, end_year):
    n = (end_year - start_year + 1) * 12
    return start_year + np.arange(n) / 12.0


def subset_months(x, file_start_year, start_year, end_year):
    i0 = (start_year - file_start_year) * 12
    i1 = (end_year - file_start_year + 1) * 12
    arr = np.asarray(x)
    if i0 < 0:
        raise ValueError(f"Requested start_year={start_year} is earlier than file_start_year={file_start_year}.")
    if arr.shape[-1] < i1:
        raise ValueError(
            f"Requested {start_year}-{end_year}, but the series has only {arr.shape[-1]} months "
            f"from file_start_year={file_start_year}."
        )
    return arr[i0:i1] if arr.ndim == 1 else arr[:, i0:i1]


def monthly_to_yearly(x, start_year, end_year, season):
    x = np.asarray(x, dtype=np.float32)
    ndim = x.ndim
    if x.ndim == 1:
        x = x[None, :]
    nyr = end_year - start_year + 1
    years = np.arange(start_year, end_year + 1)
    out = []
    for i in range(nyr):
        if season == "ANN":
            block = x[:, i*12:i*12+12]
        elif season == "MAM":
            block = x[:, i*12+2:i*12+5]
        elif season == "DJF":
            if i == 0:
                block = x[:, 0:2]
            else:
                block = np.concatenate([x[:, i*12-1:i*12], x[:, i*12:i*12+2]], axis=1)
        else:
            raise ValueError("Unknown season.")
        out.append(np.nanmean(block, axis=1))
    out = np.asarray(out, dtype=np.float32).T
    return (out[0] if ndim == 1 else out), years


def runmean_yearly(x, years, window=21, mode="center"):
    x = np.asarray(x, dtype=np.float32)
    ndim = x.ndim
    if x.ndim == 1:
        x = x[None, :]
    years = np.asarray(years, dtype=int)
    vals, yrs = [], []

    if mode == "center":
        if window % 2 != 1:
            raise ValueError("Centered running mean requires an odd window.")
        half = window // 2
        for i in range(half, x.shape[1] - half):
            vals.append(np.nanmean(x[:, i-half:i+half+1], axis=1))
            yrs.append(years[i])
    elif mode == "trailing":
        for i in range(window - 1, x.shape[1]):
            vals.append(np.nanmean(x[:, i-window+1:i+1], axis=1))
            yrs.append(years[i])
    elif mode == "forward":
        for i in range(0, x.shape[1] - window + 1):
            vals.append(np.nanmean(x[:, i:i+window], axis=1))
            yrs.append(years[i])
    else:
        raise ValueError("RUNMEAN_MODE must be center, trailing, or forward.")

    vals = np.asarray(vals, dtype=np.float32).T
    yrs = np.asarray(yrs, dtype=int)
    return (vals[0] if ndim == 1 else vals), yrs


def preprocess(x_raw, file_start_year, start_year, end_year, season, scaler,
               fit=False, fit_raw=None, fit_year_axis=None):
    x = subset_months(x_raw, file_start_year, start_year, end_year)
    years_m = month_year_axis(start_year, end_year)
    if fit:
        scaler.fit(x if fit_raw is None else fit_raw, years_m if fit_year_axis is None else fit_year_axis)
    z_month = scaler.transform(x, years_m)
    y_annual, years = monthly_to_yearly(z_month, start_year, end_year, season)
    y_run, run_years = runmean_yearly(y_annual, years, RUN_WRAP, RUNMEAN_MODE)
    return y_run, run_years


def read_model_index(name):
    arr = xr.open_dataset(MODEL_FILES[name])[MODEL_VARS[name]][:, :].data.astype(np.float32)

    if arr.ndim != 2:
        raise ValueError(f"CMIP6 {name}: expected 2D model x time array, got {arr.shape}.")
    if arr.shape[0] != len(MODEL_NAMES):
        raise ValueError(
            f"CMIP6 {name}: first dimension is {arr.shape[0]}, "
            f"but expected {len(MODEL_NAMES)} models."
        )

    expected_months = (MODEL_END_YEAR - MODEL_START_YEAR + 1) * 12
    if arr.shape[1] < expected_months:
        raise ValueError(f"CMIP6 {name}: only {arr.shape[1]} months, expected at least {expected_months}.")
    arr = arr[:, :expected_months]

    return checked_fill_panel(arr, "CMIP6", name, MAX_MISSING_FRAC_MODEL)


def read_obs_index(name):
    da = xr.open_dataset(OBS_FILES[name])[OBS_VARS[name]]
    arr = da.values.astype(np.float32)

    if arr.ndim == 2:
        # EOF files are usually mode x time; if not, this still selects the shorter dimension as mode.
        arr = arr[0, :] if arr.shape[0] <= arr.shape[1] else arr[:, 0]
    elif arr.ndim != 1:
        raise ValueError(f"OBS {name}: expected 1D or 2D array, got {arr.shape}.")

    expected_months = (OBS_END_YEAR - OBS_FILE_START_YEAR[name] + 1) * 12
    if arr.shape[0] < expected_months:
        raise ValueError(
            f"OBS {name}: only {arr.shape[0]} months from {OBS_FILE_START_YEAR[name]}, "
            f"but OBS_END_YEAR={OBS_END_YEAR} requires at least {expected_months} months."
        )
    arr = arr[:expected_months]

    return checked_fill_1d(arr, "OBS", name, -1, MAX_MISSING_FRAC_OBS)


def split_models(n_models, val_fraction=0.2, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(n_models)
    rng.shuffle(idx)
    n_val = max(1, int(round(n_models * val_fraction)))
    return np.sort(idx[n_val:]), np.sort(idx[:n_val])


def make_samples_one(features, pdo, years):
    x_list, y_list, issue, ystart, yend = [], [], [], [], []
    n = len(pdo)
    for i in range(n - HIST_STEPS - FUTURE_STEPS + 1):
        xb = np.stack([features[k][i:i+HIST_STEPS] for k in FEATURES], axis=-1)
        yb = pdo[i+HIST_STEPS:i+HIST_STEPS+FUTURE_STEPS, None]
        if np.all(np.isfinite(xb)) and np.all(np.isfinite(yb)):
            x_list.append(xb)
            y_list.append(yb)
            issue.append(years[i+HIST_STEPS-1])
            ystart.append(years[i+HIST_STEPS])
            yend.append(years[i+HIST_STEPS+FUTURE_STEPS-1])
    return (np.asarray(x_list, dtype=np.float32),
            np.asarray(y_list, dtype=np.float32),
            np.asarray(issue, dtype=int),
            np.asarray(ystart, dtype=int),
            np.asarray(yend, dtype=int))


def make_samples_panel(features, pdo, years, model_idx):
    xs, ys, issues, starts, ends, mids = [], [], [], [], [], []
    for m in model_idx:
        fm = {k: features[k][m, :] for k in FEATURES}
        x, y, issue, start, end = make_samples_one(fm, pdo[m, :], years)
        xs.append(x); ys.append(y); issues.append(issue); starts.append(start); ends.append(end)
        mids.append(np.full(len(issue), m, dtype=int))
    return (np.concatenate(xs), np.concatenate(ys),
            np.concatenate(issues), np.concatenate(starts), np.concatenate(ends), np.concatenate(mids))


def make_delta_targets(x_hist, y_future):
    """
    Convert absolute PDO targets to correction targets.

    x_hist      : samples x HIST_STEPS x features
    y_future   : samples x FUTURE_STEPS x 1, absolute PDO_rm21
    return     : Delta PDO relative to the latest input PDO state
    """
    baseline = persistence_baseline(x_hist)
    return (y_future - baseline).astype(np.float32)


def add_delta_to_persistence(x_hist, delta_pred):
    """
    Convert predicted Delta PDO back to absolute PDO prediction.
    """
    return (persistence_baseline(x_hist) + delta_pred).astype(np.float32)


def delta_seq_loss(y_true_delta, y_pred_delta):
    """
    Simple Delta-PDO loss.

    The target is:
        Delta PDO = future PDO - persistence PDO

    A simple MSE loss is used here because the main goal is to learn the
    persistence-to-observation correction without over-constraining the phase
    evolution.
    """
    return tf.reduce_mean(tf.square(y_true_delta - y_pred_delta))


# Backward-compatible name used below when compiling.
seq_loss = delta_seq_loss

def build_model(lstm_units=64, dense_units=64, dropout_rate=0.2,
                l2_reg=1e-4, num_heads=2, key_dim=16):
    """
    Residual-enhanced v5 persistence-correction Seq2Seq model.

    Output target:
        Delta PDO = future PDO - persistence PDO

    Final prediction outside the model:
        PDO_hat = PDO_persistence + Delta_hat

    Residual design:
    1. Encoder self-attention residual:
        enc = LayerNorm(enc + SelfAttention(enc))
    2. Decoder residual block:
        decoder state is updated by a residual projection of decoder-attention features.
    3. Output AR residual head:
        a simple linear head from recent PDO history predicts a low-frequency
        Delta correction. The nonlinear Seq2Seq-attention branch then adds to it.

    The last item is the most important change for smooth decadal sequences:
    it gives the model an easy linear pathway for persistence-correction, while
    the nonlinear branch only needs to learn the remaining modulation from
    AMO/AO/OHC/SSHA and cross-feature information.
    """
    inp = Input(shape=(HIST_STEPS, len(FEATURES)), name="hist_inputs")

    # ------------------------------------------------------------------
    # Linear AR-style residual branch from recent PDO history.
    # PDO is feature index 1 in [AMO, PDO, AO, OHC, SSHA].
    # This branch predicts Delta PDO directly and acts as a stable baseline
    # correction to persistence.
    # ------------------------------------------------------------------
    pdo_hist = layers.Lambda(lambda x: x[:, :, 1:2], name="extract_pdo_for_delta_residual")(inp)
    pdo_hist_flat = layers.Flatten(name="pdo_hist_flatten")(pdo_hist)

    # Explicit recent-tendency features. The Dense layer could in principle
    # learn these from pdo_hist_flat, but giving the model a direct tendency path
    # often stabilizes prediction of smooth low-frequency sequences.
    pdo_diff = layers.Lambda(
        lambda x: x[:, 1:, :] - x[:, :-1, :],
        name="pdo_recent_differences"
    )(pdo_hist)
    pdo_diff_flat = layers.Flatten(name="pdo_diff_flatten")(pdo_diff)
    pdo_start_end = layers.Lambda(
        lambda x: tf.concat([x[:, -1, :], x[:, -1, :] - x[:, 0, :]], axis=-1),
        name="pdo_last_and_10yr_change"
    )(pdo_hist)

    pdo_ar_features = layers.Concatenate(name="pdo_ar_feature_concat")(
        [pdo_hist_flat, pdo_diff_flat, pdo_start_end]
    )

    ar_delta = layers.Dense(
        FUTURE_STEPS,
        activation="linear",
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        name="delta_ar_dense"
    )(pdo_ar_features)
    ar_delta = layers.Reshape((FUTURE_STEPS, 1), name="delta_ar_reshape")(ar_delta)

    # ------------------------------------------------------------------
    # Nonlinear Seq2Seq-attention branch.
    # ------------------------------------------------------------------
    enc, h, c = layers.LSTM(
        lstm_units, return_sequences=True, return_state=True,
        dropout=dropout_rate, recurrent_dropout=0.0,
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        name="encoder_lstm"
    )(inp)

    self_attn = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        output_shape=lstm_units,
        dropout=dropout_rate,
        name="encoder_self_attention"
    )(enc, enc)

    enc = layers.Add(name="encoder_self_attention_add")([enc, self_attn])
    enc = layers.LayerNormalization(name="encoder_self_attention_norm")(enc)

    dec_seed = layers.RepeatVector(FUTURE_STEPS, name="decoder_repeat_context")(h)
    dec = layers.LSTM(
        lstm_units, return_sequences=True,
        dropout=dropout_rate, recurrent_dropout=0.0,
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        name="decoder_lstm"
    )(dec_seed, initial_state=[h, c])

    cross = layers.Attention(name="decoder_encoder_attention")([dec, enc])
    z = layers.Concatenate(name="decoder_attention_concat")([dec, cross])

    # Residual decoder block. Project attention features back to lstm_units so
    # they can be added to the decoder trajectory.
    z_res = layers.TimeDistributed(
        layers.Dense(lstm_units, activation="tanh"),
        name="decoder_residual_projection"
    )(z)
    z_res = layers.Dropout(dropout_rate, name="decoder_residual_dropout")(z_res)
    z = layers.Add(name="decoder_residual_add")([dec, z_res])
    z = layers.LayerNormalization(name="decoder_residual_norm")(z)

    z = layers.TimeDistributed(
        layers.Dense(dense_units, activation="tanh"),
        name="td_dense_1"
    )(z)
    z = layers.Dropout(dropout_rate, name="td_dropout")(z)

    nonlinear_delta = layers.TimeDistributed(
        layers.Dense(1, dtype="float32"),
        name="pdo_delta_nonlinear_out"
    )(z)

    if USE_OUTPUT_AR_RESIDUAL:
        delta = layers.Add(name="pdo_delta_final_add")([nonlinear_delta, ar_delta])
    else:
        delta = nonlinear_delta

    return Model(inp, delta, name="PDO_v5_Residual_Persistence_Correction")

def make_optimizer(name, lr):
    if name == "Adam":
        return tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    if name == "Nadam":
        return tf.keras.optimizers.Nadam(learning_rate=lr, clipnorm=1.0)
    if name == "RMSprop":
        return tf.keras.optimizers.RMSprop(learning_rate=lr, clipnorm=1.0)
    raise ValueError(name)


def lead_skill(y_pred, y_true):
    yp = y_pred[:, :, 0] if y_pred.ndim == 3 else y_pred
    yt = y_true[:, :, 0] if y_true.ndim == 3 else y_true
    rows = []
    for j in range(yp.shape[1]):
        p, t = yp[:, j], yt[:, j]
        valid = np.isfinite(p) & np.isfinite(t)
        if valid.sum() > 2 and np.std(p[valid]) > 0 and np.std(t[valid]) > 0:
            r, pv = sts.pearsonr(p[valid], t[valid])
        else:
            r, pv = np.nan, np.nan
        rmse = np.sqrt(mean_squared_error(t[valid], p[valid])) if valid.sum() else np.nan
        rows.append({"lead": j+1, "corr": r, "p_value": pv, "rmse": rmse, "n": int(valid.sum())})
    return pd.DataFrame(rows)


def add_rmse_skill_score(skill_model, skill_reference, reference_name="persistence"):
    out = skill_model.copy()
    ref_rmse = np.asarray(skill_reference["rmse"], dtype=float)
    mod_rmse = np.asarray(out["rmse"], dtype=float)
    out[f"rmse_skill_vs_{reference_name}"] = 1.0 - mod_rmse / ref_rmse
    return out


def persistence_baseline(x_hist):
    last = x_hist[:, -1, 1:2]
    return np.repeat(last[:, None, :], FUTURE_STEPS, axis=1).astype(np.float32)


def set_output_only_trainable(model):
    # First transfer step adapts only the Delta-output heads and nearby decoder projection.
    keep = {
        "delta_ar_dense",
        "decoder_residual_projection",
        "td_dense_1",
        "pdo_delta_nonlinear_out",
    }
    for layer in model.layers:
        layer.trainable = layer.name in keep


def set_all_trainable(model):
    for layer in model.layers:
        layer.trainable = True


# =============================================================================
# Data preprocessing
# =============================================================================
required_files = list(MODEL_FILES.values()) + list(OBS_FILES.values())
missing_files = [str(path) for path in required_files if not Path(path).exists()]
if missing_files:
    raise FileNotFoundError(
        "The following required uploaded time-series files are missing:\n  - "
        + "\n  - ".join(missing_files)
    )

print("Reading CMIP6 indices...")
raw_model = {k: read_model_index(k) for k in FEATURES}
save_missing_diagnostics()
n_models = len(MODEL_NAMES)
train_models, val_models = split_models(n_models, 0.2, SEED)
print("RUNMEAN_MODE:", RUNMEAN_MODE)
print("CMIP6 train models:", train_models)
print("CMIP6 validation models:", val_models)

model_month_years = month_year_axis(MODEL_START_YEAR, MODEL_END_YEAR)
model_run = {}
model_run_years = None

for k in FEATURES:
    scaler = MonthlyDetrendStandardizer("CMIP6_" + k)
    scaler.fit(raw_model[k][train_models, :], model_month_years)
    run, years = preprocess(
        raw_model[k], MODEL_START_YEAR, MODEL_START_YEAR, MODEL_END_YEAR,
        INDEX_SEASON[k], scaler, fit=False
    )
    model_run[k] = run
    model_run_years = years if model_run_years is None else model_run_years
    if not np.array_equal(model_run_years, years):
        raise ValueError("CMIP6 run-mean year mismatch for " + k)

x_source_train, y_source_train, *_ = make_samples_panel(model_run, model_run["PDO"], model_run_years, train_models)
x_source_val, y_source_val, *_ = make_samples_panel(model_run, model_run["PDO"], model_run_years, val_models)
print("Source train:", x_source_train.shape, y_source_train.shape)
print("Source val:", x_source_val.shape, y_source_val.shape)
y_delta_source_train = make_delta_targets(x_source_train, y_source_train)
y_delta_source_val = make_delta_targets(x_source_val, y_source_val)
print("Source delta train:", y_delta_source_train.shape)

print("Reading observations...")
raw_obs = {k: read_obs_index(k) for k in FEATURES}
save_missing_diagnostics()
obs_run = {}
obs_run_years = None

for k in FEATURES:
    scaler = MonthlyDetrendStandardizer("OBS_" + k)
    fit_raw = subset_months(raw_obs[k], OBS_FILE_START_YEAR[k], OBS_START_YEAR, OBS_TRANSFER_END)
    fit_year_axis = month_year_axis(OBS_START_YEAR, OBS_TRANSFER_END)
    run, years = preprocess(
        raw_obs[k], OBS_FILE_START_YEAR[k], OBS_START_YEAR, OBS_END_YEAR,
        INDEX_SEASON[k], scaler, fit=True, fit_raw=fit_raw, fit_year_axis=fit_year_axis
    )
    obs_run[k] = run
    obs_run_years = years if obs_run_years is None else obs_run_years
    if not np.array_equal(obs_run_years, years):
        raise ValueError("OBS run-mean year mismatch for " + k)

x_obs_all, y_obs_all, issue_year, target_start, target_end = make_samples_one(obs_run, obs_run["PDO"], obs_run_years)

half_window = RUN_WRAP // 2 if RUNMEAN_MODE == "center" else 0

if SPLIT_MODE == "center_year":
    # Split by the center-year labels of the smoothed PDO sequence.
    # This is the practical main setting because the observed sample size is limited.
    transfer_mask = target_end <= OBS_TRANSFER_END
    test_mask = (target_start >= OBS_TEST_START) & (target_end <= obs_run_years[-1])
elif SPLIT_MODE == "strict_window":
    # Split by actual 21-year window boundaries.
    # Conservative, but the observed test sample size becomes much smaller.
    transfer_mask = (target_end + half_window) <= OBS_TRANSFER_END
    test_mask = ((target_start - half_window) >= OBS_TEST_START) & ((target_end + half_window) <= OBS_END_YEAR)
else:
    raise ValueError("SPLIT_MODE must be 'center_year' or 'strict_window'.")

print("Split mode:", SPLIT_MODE)

x_obs_transfer = x_obs_all[transfer_mask]
y_obs_transfer = y_obs_all[transfer_mask]
x_obs_test = x_obs_all[test_mask]
y_obs_test = y_obs_all[test_mask]
issue_test = issue_year[test_mask]
target_start_test = target_start[test_mask]
target_end_test = target_end[test_mask]

n_val_ft = max(3, int(round(len(x_obs_transfer) * 0.2)))
x_obs_ft_train = x_obs_transfer[:-n_val_ft]
y_obs_ft_train = y_obs_transfer[:-n_val_ft]
x_obs_ft_val = x_obs_transfer[-n_val_ft:]
y_obs_ft_val = y_obs_transfer[-n_val_ft:]

print("Obs transfer:", x_obs_transfer.shape, y_obs_transfer.shape)
print("Obs test:", x_obs_test.shape, y_obs_test.shape)
print("Obs fine-tune train/val:", x_obs_ft_train.shape, x_obs_ft_val.shape)
y_delta_obs_ft_train = make_delta_targets(x_obs_ft_train, y_obs_ft_train)
y_delta_obs_ft_val = make_delta_targets(x_obs_ft_val, y_obs_ft_val)
y_delta_obs_test = make_delta_targets(x_obs_test, y_obs_test)
if len(x_obs_transfer) < 10:
    raise ValueError("Too few observation transfer samples. Use SPLIT_MODE=\'center_year\' or adjust OBS_TRANSFER_END.")
if len(x_obs_test) < 5:
    print(f"WARNING: only {len(x_obs_test)} observed test samples; lead-wise correlations may be unstable.")


# =============================================================================
# Optuna: hyperparameters selected only on CMIP6 source data
# =============================================================================
def objective(trial):
    K.clear_session()
    params = {
        "lstm_units": trial.suggest_int("lstm_units", 32, 128, step=32),
        "dense_units": trial.suggest_int("dense_units", 16, 96, step=16),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.35, step=0.05),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "epochs": trial.suggest_int("epochs", 40, 160, step=20),
        "optimizer": trial.suggest_categorical("optimizer", ["Adam", "Nadam", "RMSprop"]),
        "l2_reg": trial.suggest_categorical("l2_reg", [1e-5, 1e-4, 5e-4]),
        "num_heads": trial.suggest_categorical("num_heads", [1, 2, 4]),
        "key_dim": trial.suggest_categorical("key_dim", [8, 16, 32]),
    }
    model = build_model(
        params["lstm_units"], params["dense_units"], params["dropout_rate"],
        params["l2_reg"], params["num_heads"], params["key_dim"]
    )
    model.compile(loss=seq_loss, optimizer=make_optimizer(params["optimizer"], params["learning_rate"]))
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, mode="min"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, mode="min"),
    ]
    model.fit(
        x_source_train, y_delta_source_train,
        validation_data=(x_source_val, y_delta_source_val),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        verbose=0,
        callbacks=callbacks,
    )
    pred_delta = model.predict(x_source_val, verbose=0)
    return float(np.mean((pred_delta - y_delta_source_val) ** 2))


BEST_PARAMS_FILE = METADATA_DIR / "best_optuna_params_cmip6.csv"

if args.run_optuna:
    print("Starting Optuna search on CMIP6...")
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS)
    best_params = study.best_params
    pd.DataFrame([best_params]).to_csv(BEST_PARAMS_FILE, index=False)
    print("Saved selected hyperparameters:", BEST_PARAMS_FILE)
else:
    if not BEST_PARAMS_FILE.exists():
        raise FileNotFoundError(
            f"Saved hyperparameters not found: {BEST_PARAMS_FILE}\n"
            "Provide the uploaded CSV or run with --run-optuna."
        )
    best_df = pd.read_csv(BEST_PARAMS_FILE)
    if len(best_df) != 1:
        raise ValueError(
            f"{BEST_PARAMS_FILE} must contain exactly one row of selected hyperparameters."
        )
    best_params = best_df.iloc[0].to_dict()

    # Restore integer/categorical types after CSV reading.
    for key in ["lstm_units", "dense_units", "batch_size", "epochs", "num_heads", "key_dim"]:
        best_params[key] = int(best_params[key])
    for key in ["dropout_rate", "learning_rate", "l2_reg"]:
        best_params[key] = float(best_params[key])
    best_params["optimizer"] = str(best_params["optimizer"])

    print("Using saved selected hyperparameters:", BEST_PARAMS_FILE)

print("Best params:", best_params)


# =============================================================================
# Source pretraining + two-step transfer learning
# =============================================================================
ensemble_records = []


def get_source_training_for_member(member_id: int):
    """
    Build source training data for one ensemble member.

    The first member can use the full CMIP6 source training set as the central
    member. Other members use source-model resampling. This avoids interpreting
    a purely repeated fit as a meaningful uncertainty estimate.
    """
    if FIRST_MEMBER_USE_FULL_SOURCE and member_id == 0:
        ensemble_records.append({
            "member": member_id + 1,
            "strategy": "full_source_central_member",
            "selected_model_indices": ",".join([str(int(x)) for x in train_models]),
            "selected_model_names": ",".join([MODEL_NAMES[int(x)] for x in train_models]),
            "n_unique_models": int(len(np.unique(train_models))),
            "n_source_samples": int(x_source_train.shape[0]),
        })
        return x_source_train, y_source_train

    if ENSEMBLE_SOURCE_RESAMPLING == "none":
        ensemble_records.append({
            "member": member_id + 1,
            "strategy": "same_source_training",
            "selected_model_indices": ",".join([str(int(x)) for x in train_models]),
            "selected_model_names": ",".join([MODEL_NAMES[int(x)] for x in train_models]),
            "n_unique_models": int(len(np.unique(train_models))),
            "n_source_samples": int(x_source_train.shape[0]),
        })
        return x_source_train, y_source_train

    rng = np.random.default_rng(SEED + 10000 + member_id)

    if ENSEMBLE_SOURCE_RESAMPLING == "model_bootstrap":
        min_unique = max(1, int(np.ceil(len(train_models) * MIN_UNIQUE_MODEL_FRACTION)))

        boot_models = None
        for _ in range(BOOTSTRAP_MAX_ATTEMPTS):
            candidate = rng.choice(train_models, size=len(train_models), replace=True)
            if len(np.unique(candidate)) >= min_unique:
                boot_models = candidate
                break
        if boot_models is None:
            boot_models = candidate
            print(
                "WARNING: bootstrap member",
                member_id + 1,
                "did not reach the requested unique-model fraction; using last draw.",
            )

        x_boot, y_boot, *_ = make_samples_panel(model_run, model_run["PDO"], model_run_years, boot_models)

        ensemble_records.append({
            "member": member_id + 1,
            "strategy": "cmip6_model_bootstrap",
            "selected_model_indices": ",".join([str(int(x)) for x in boot_models]),
            "selected_model_names": ",".join([MODEL_NAMES[int(x)] for x in boot_models]),
            "n_unique_models": int(len(np.unique(boot_models))),
            "n_source_samples": int(x_boot.shape[0]),
        })

        print("Member", member_id + 1, "CMIP6 bootstrap model indices:", boot_models)
        print("Member", member_id + 1, "unique CMIP6 models:", len(np.unique(boot_models)))
        return x_boot, y_boot

    if ENSEMBLE_SOURCE_RESAMPLING == "model_subsample":
        n_pick = max(2, int(round(len(train_models) * SOURCE_SUBSAMPLE_FRACTION)))
        n_pick = min(n_pick, len(train_models))
        picked_models = rng.choice(train_models, size=n_pick, replace=False)

        x_sub, y_sub, *_ = make_samples_panel(model_run, model_run["PDO"], model_run_years, picked_models)

        ensemble_records.append({
            "member": member_id + 1,
            "strategy": "cmip6_model_subsample_without_replacement",
            "selected_model_indices": ",".join([str(int(x)) for x in picked_models]),
            "selected_model_names": ",".join([MODEL_NAMES[int(x)] for x in picked_models]),
            "n_unique_models": int(len(np.unique(picked_models))),
            "n_source_samples": int(x_sub.shape[0]),
        })

        print("Member", member_id + 1, "CMIP6 subsample model indices:", picked_models)
        return x_sub, y_sub

    if ENSEMBLE_SOURCE_RESAMPLING == "sample_bootstrap":
        boot_idx = rng.choice(np.arange(x_source_train.shape[0]), size=x_source_train.shape[0], replace=True)
        x_boot = x_source_train[boot_idx]
        y_boot = y_source_train[boot_idx]

        ensemble_records.append({
            "member": member_id + 1,
            "strategy": "cmip6_sample_bootstrap",
            "selected_model_indices": "sample_bootstrap",
            "selected_model_names": "sample_bootstrap",
            "n_unique_models": int(len(np.unique(source_train_member_id[boot_idx]))) if "source_train_member_id" in globals() else -1,
            "n_source_samples": int(x_boot.shape[0]),
        })

        print("Member", member_id + 1, "CMIP6 sample bootstrap size:", len(boot_idx))
        return x_boot, y_boot

    raise ValueError(
        "ENSEMBLE_SOURCE_RESAMPLING must be 'model_bootstrap', "
        "'model_subsample', 'sample_bootstrap', or 'none'."
    )

def latest_obs_input():
    xb = np.stack([obs_run[k][-HIST_STEPS:] for k in FEATURES], axis=-1)
    return xb[None, :, :].astype(np.float32), int(obs_run_years[-1])


def _safe_corr_1d(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() > 2 and np.std(a[valid]) > 0 and np.std(b[valid]) > 0:
        return float(sts.pearsonr(a[valid], b[valid])[0])
    return np.nan


def calibrate_delta_alpha_by_lead(delta_pred_val, delta_true_val):
    """
    Estimate one amplitude factor alpha_k for each lead.

    Raw alpha:
        alpha_k = argmin || alpha * Delta_pred(k) - Delta_true(k) ||^2

    Then alpha is:
      1. clipped to [DELTA_ALPHA_MIN, DELTA_ALPHA_MAX],
      2. set to zero if validation Delta phase skill is poor,
      3. mildly shrunk to reduce test-time amplitude overshoot.

    This preserves phase evolution when the validation subset supports it, while
    avoiding unnecessary corrections at short leads where persistence is already
    very hard to beat.
    """
    pred = np.asarray(delta_pred_val, dtype=np.float32)
    true = np.asarray(delta_true_val, dtype=np.float32)

    if not USE_LEADWISE_DELTA_CALIBRATION:
        alpha = np.ones(FUTURE_STEPS, dtype=np.float32)
        info = {"delta_alpha_mode": "none"}
        for j in range(FUTURE_STEPS):
            info[f"alpha_raw_lead{j+1}"] = 1.0
            info[f"alpha_lead{j+1}"] = 1.0
            info[f"calib_delta_corr_lead{j+1}"] = _safe_corr_1d(pred[:, j, 0], true[:, j, 0])
            info[f"calib_delta_signacc_lead{j+1}"] = np.nan
        return alpha, info

    alpha = np.zeros(FUTURE_STEPS, dtype=np.float32)
    info = {"delta_alpha_mode": "leadwise_ls_gated_shrunk"}

    for j in range(FUTURE_STEPS):
        p = pred[:, j, 0]
        t = true[:, j, 0]
        valid = np.isfinite(p) & np.isfinite(t)

        if valid.sum() < 3 or np.sum(p[valid] ** 2) <= 0:
            a_raw = 0.0
        else:
            a_raw = float(np.sum(p[valid] * t[valid]) / np.sum(p[valid] ** 2))

        corr = _safe_corr_1d(p, t)

        valid_sign = valid & (np.abs(t) > 1.0e-8)
        if valid_sign.sum() > 0:
            sign_acc = float(np.mean(np.sign(p[valid_sign]) == np.sign(t[valid_sign])))
        else:
            sign_acc = np.nan

        a = float(np.clip(a_raw, DELTA_ALPHA_MIN, DELTA_ALPHA_MAX))

        # Gate unreliable lead corrections using validation phase information.
        if (not np.isfinite(corr)) or (corr < MIN_CALIB_DELTA_CORR):
            a = 0.0
        if np.isfinite(sign_acc) and sign_acc < MIN_CALIB_SIGN_ACCURACY:
            a = 0.0

        # Mild shrinkage guards against amplitude over-shooting. Positive
        # shrinkage does not change Delta sign/correlation; it only reduces RMSE
        # risk when the validation subset is small.
        a = a * DELTA_ALPHA_SHRINK

        alpha[j] = a

        info[f"alpha_raw_lead{j+1}"] = float(a_raw)
        info[f"alpha_lead{j+1}"] = float(a)
        info[f"calib_delta_corr_lead{j+1}"] = corr
        info[f"calib_delta_signacc_lead{j+1}"] = sign_acc

    return alpha, info

def delta_sign_accuracy(delta_pred, delta_true):
    yp = delta_pred[:, :, 0] if delta_pred.ndim == 3 else delta_pred
    yt = delta_true[:, :, 0] if delta_true.ndim == 3 else delta_true

    rows = []
    for j in range(yp.shape[1]):
        p = yp[:, j]
        t = yt[:, j]
        valid = np.isfinite(p) & np.isfinite(t) & (np.abs(t) > 1.0e-8)
        if valid.sum() > 0:
            acc = float(np.mean(np.sign(p[valid]) == np.sign(t[valid])))
        else:
            acc = np.nan
        rows.append({"lead": j + 1, "delta_sign_accuracy": acc, "n": int(valid.sum())})
    return pd.DataFrame(rows)


def train_member(member_id, params):
    K.clear_session()
    tf.random.set_seed(SEED + member_id)
    np.random.seed(SEED + member_id)
    random.seed(SEED + member_id)

    model = build_model(
        params["lstm_units"], params["dense_units"], params["dropout_rate"],
        params["l2_reg"], params["num_heads"], params["key_dim"]
    )

    x_source_member, y_source_member = get_source_training_for_member(member_id)
    y_delta_source_member = make_delta_targets(x_source_member, y_source_member)

    callbacks_source = [
        EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, mode="min"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, mode="min"),
    ]

    print(f"\nMember {member_id+1}: CMIP6 pretraining")
    model.compile(loss=seq_loss, optimizer=make_optimizer(params["optimizer"], params["learning_rate"]))
    model.fit(
        x_source_member, y_delta_source_member,
        validation_data=(x_source_val, y_delta_source_val),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        verbose=2,
        callbacks=callbacks_source,
    )

    callbacks_transfer = [
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, mode="min"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-7, mode="min"),
    ]
    batch_ft = min(params["batch_size"], max(4, len(x_obs_ft_train)))

    print(f"Member {member_id+1}: transfer step 1, output layers only")
    set_output_only_trainable(model)
    model.compile(loss=seq_loss, optimizer=tf.keras.optimizers.Adam(params["learning_rate"] * 0.5, clipnorm=1.0))
    model.fit(
        x_obs_ft_train, y_delta_obs_ft_train,
        validation_data=(x_obs_ft_val, y_delta_obs_ft_val),
        epochs=120,
        batch_size=batch_ft,
        verbose=2,
        callbacks=callbacks_transfer,
    )

    print(f"Member {member_id+1}: transfer step 2, all layers")
    set_all_trainable(model)
    model.compile(loss=seq_loss, optimizer=tf.keras.optimizers.Adam(params["learning_rate"] * 0.05, clipnorm=1.0))
    model.fit(
        x_obs_ft_train, y_delta_obs_ft_train,
        validation_data=(x_obs_ft_val, y_delta_obs_ft_val),
        epochs=160,
        batch_size=batch_ft,
        verbose=2,
        callbacks=callbacks_transfer,
    )

    # Calibrate Delta amplitude by lead using the observation fine-tune
    # validation subset. This keeps v5's phase-evolution information while
    # reducing lead-dependent amplitude bias.
    pred_delta_val = model.predict(x_obs_ft_val, verbose=0)
    delta_alpha, alpha_info = calibrate_delta_alpha_by_lead(
        pred_delta_val,
        y_delta_obs_ft_val,
    )
    print(f"Member {member_id+1}: lead-wise Delta alpha = {np.round(delta_alpha, 3)}")

    if len(ensemble_records) > 0:
        ensemble_records[-1].update(alpha_info)

    pred_delta_test = model.predict(x_obs_test, verbose=0) * delta_alpha[None, :, None]
    pred_test = add_delta_to_persistence(x_obs_test, pred_delta_test)

    x_future, issue = latest_obs_input()
    pred_delta_future = model.predict(x_future, verbose=0) * delta_alpha[None, :, None]
    pred_future = add_delta_to_persistence(x_future, pred_delta_future)

    model_path = MODEL_SAVE_DIR / f"pdo_transfer_obs_finetuned_member_{member_id+1}.h5"
    model.save(model_path, include_optimizer=False)
    print("Saved final observation-finetuned H5 model:", model_path)
    return model, pred_test.astype(np.float32), pred_future.astype(np.float32), issue


models, pred_tests, pred_futures = [], [], []
future_issue_year = None
for m in range(N_ENSEMBLE):
    model, pt, pf, issue = train_member(m, best_params)
    models.append(model)
    pred_tests.append(pt)
    pred_futures.append(pf)
    future_issue_year = issue

pred_tests = np.asarray(pred_tests, dtype=np.float32)
pred_futures = np.asarray(pred_futures, dtype=np.float32)

ensemble_meta_df = pd.DataFrame(ensemble_records)
ensemble_meta_df.to_csv(
    METADATA_DIR / "ensemble_member_source_bootstrap_metadata.csv",
    index=False,
)

# Compact lead-wise alpha summary across ensemble members.
alpha_rows = []
for j in range(FUTURE_STEPS):
    col = f"alpha_lead{j+1}"
    raw_col = f"alpha_raw_lead{j+1}"
    corr_col = f"calib_delta_corr_lead{j+1}"
    sign_col = f"calib_delta_signacc_lead{j+1}"

    row = {"lead": j + 1}
    if col in ensemble_meta_df:
        row["alpha_mean"] = float(np.nanmean(ensemble_meta_df[col]))
        row["alpha_median"] = float(np.nanmedian(ensemble_meta_df[col]))
        row["alpha_min"] = float(np.nanmin(ensemble_meta_df[col]))
        row["alpha_max"] = float(np.nanmax(ensemble_meta_df[col]))
    if raw_col in ensemble_meta_df:
        row["alpha_raw_mean"] = float(np.nanmean(ensemble_meta_df[raw_col]))
    if corr_col in ensemble_meta_df:
        row["calib_delta_corr_mean"] = float(np.nanmean(ensemble_meta_df[corr_col]))
    if sign_col in ensemble_meta_df:
        row["calib_delta_signacc_mean"] = float(np.nanmean(ensemble_meta_df[sign_col]))
    alpha_rows.append(row)

pd.DataFrame(alpha_rows).to_csv(
    METADATA_DIR / "leadwise_delta_alpha_summary.csv",
    index=False,
)

pred_test_mean = np.mean(pred_tests, axis=0)
std_ddof = 1 if pred_tests.shape[0] > 1 else 0
pred_test_std = np.std(pred_tests, axis=0, ddof=std_ddof)
future_persist_value = float(latest_obs_input()[0][0, -1, 1])
pred_future_delta_members = pred_futures[:, 0, :, 0] - future_persist_value
pred_future_mean = np.mean(pred_futures, axis=0)[0, :, 0]
pred_future_std = np.std(pred_futures, axis=0, ddof=std_ddof)[0, :, 0]
pred_future_p05 = np.percentile(pred_futures[:, 0, :, 0], 5, axis=0)
pred_future_p50 = np.percentile(pred_futures[:, 0, :, 0], 50, axis=0)
pred_future_p95 = np.percentile(pred_futures[:, 0, :, 0], 95, axis=0)
pred_future_delta_mean = np.mean(pred_future_delta_members, axis=0)
pred_future_delta_p05 = np.percentile(pred_future_delta_members, 5, axis=0)
pred_future_delta_p50 = np.percentile(pred_future_delta_members, 50, axis=0)
pred_future_delta_p95 = np.percentile(pred_future_delta_members, 95, axis=0)

persist = persistence_baseline(x_obs_test)
skill_model = lead_skill(pred_test_mean, y_obs_test)
skill_persist = lead_skill(persist, y_obs_test)
skill_model = add_rmse_skill_score(skill_model, skill_persist, "persistence")

# Skill for predicted PDO change relative to latest input PDO state.
pred_delta_test_mean = pred_test_mean - persist
skill_delta_model = lead_skill(pred_delta_test_mean, y_delta_obs_test)
zero_delta = np.zeros_like(y_delta_obs_test, dtype=np.float32)
skill_delta_zero = lead_skill(zero_delta, y_delta_obs_test)
skill_delta_model = add_rmse_skill_score(skill_delta_model, skill_delta_zero, "zero_delta")
skill_model.to_csv(os.path.join(OUTPUT_DIR, "obs_test_skill_transfer_model.csv"), index=False)
skill_persist.to_csv(os.path.join(OUTPUT_DIR, "obs_test_skill_persistence_baseline.csv"), index=False)
skill_delta_model.to_csv(os.path.join(OUTPUT_DIR, "obs_test_skill_delta_model.csv"), index=False)
skill_delta_zero.to_csv(os.path.join(OUTPUT_DIR, "obs_test_skill_delta_zero_baseline.csv"), index=False)
delta_sign_accuracy(pred_delta_test_mean, y_delta_obs_test).to_csv(
    os.path.join(OUTPUT_DIR, "obs_test_delta_sign_accuracy.csv"),
    index=False,
)

# Save skill for each ensemble member. This is important for diagnosing whether
# a few members dominate the ensemble spread.
member_skill_rows = []
for mm in range(pred_tests.shape[0]):
    dfm = lead_skill(pred_tests[mm], y_obs_test)
    dfm = add_rmse_skill_score(dfm, skill_persist, "persistence")
    dfm.insert(0, "member", mm + 1)
    member_skill_rows.append(dfm)
pd.concat(member_skill_rows, ignore_index=True).to_csv(
    os.path.join(OUTPUT_DIR, "obs_test_skill_by_member.csv"),
    index=False,
)

# Member-wise Delta skill, useful for identifying members with good phase-evolution skill.
member_delta_skill_rows = []
for mm in range(pred_tests.shape[0]):
    dfm = lead_skill(pred_tests[mm] - persist, y_delta_obs_test)
    dfm = add_rmse_skill_score(dfm, skill_delta_zero, "zero_delta")
    dfm.insert(0, "member", mm + 1)
    member_delta_skill_rows.append(dfm)
pd.concat(member_delta_skill_rows, ignore_index=True).to_csv(
    os.path.join(OUTPUT_DIR, "obs_test_delta_skill_by_member.csv"),
    index=False,
)


# Save observed test predictions and actual observations in CSV format.
# Long format is recommended for analysis/plotting.
pred_rows = []
for s in range(pred_test_mean.shape[0]):
    for j in range(FUTURE_STEPS):
        center_year = int(target_start_test[s] + j)
        pred_rows.append({
            "sample": s,
            "issue_center_year": int(issue_test[s]),
            "lead": j + 1,
            "target_center_year": center_year,
            "target_window_start": center_year - half_window if RUNMEAN_MODE == "center" else center_year,
            "target_window_end": center_year + half_window if RUNMEAN_MODE == "center" else center_year,
            "sample_target_start_center": int(target_start_test[s]),
            "sample_target_end_center": int(target_end_test[s]),
            "sample_target_window_start": int(target_start_test[s] - half_window),
            "sample_target_window_end": int(target_end_test[s] + half_window),
            "y_pred_mean": float(pred_test_mean[s, j, 0]),
            "y_pred_std": float(pred_test_std[s, j, 0]),
            "y_true": float(y_obs_test[s, j, 0]),
            "pdo_persistence": float(persist[s, j, 0]),
            "y_pred_delta": float(pred_test_mean[s, j, 0] - persist[s, j, 0]),
            "y_true_delta": float(y_delta_obs_test[s, j, 0]),
        })
pd.DataFrame(pred_rows).to_csv(
    os.path.join(OUTPUT_DIR, "obs_test_predictions_vs_observations_long.csv"),
    index=False,
)

# Save member-wise test predictions as well.
member_pred_rows = []
for mm in range(pred_tests.shape[0]):
    for s in range(pred_tests.shape[1]):
        for j in range(FUTURE_STEPS):
            center_year = int(target_start_test[s] + j)
            member_pred_rows.append({
                "member": mm + 1,
                "sample": s,
                "issue_center_year": int(issue_test[s]),
                "lead": j + 1,
                "target_center_year": center_year,
                "target_window_start": center_year - half_window if RUNMEAN_MODE == "center" else center_year,
                "target_window_end": center_year + half_window if RUNMEAN_MODE == "center" else center_year,
                "y_pred_member": float(pred_tests[mm, s, j, 0]),
                "y_true": float(y_obs_test[s, j, 0]),
                "y_pred_delta_member": float(pred_tests[mm, s, j, 0] - persist[s, j, 0]),
                "y_true_delta": float(y_delta_obs_test[s, j, 0]),
            })
pd.DataFrame(member_pred_rows).to_csv(
    os.path.join(OUTPUT_DIR, "obs_test_predictions_by_member_long.csv"),
    index=False,
)

# Wide format is convenient for a quick manual check.
wide_rows = []
for s in range(pred_test_mean.shape[0]):
    row = {
        "sample": s,
        "issue_center_year": int(issue_test[s]),
        "target_start_center": int(target_start_test[s]),
        "target_end_center": int(target_end_test[s]),
        "target_window_start": int(target_start_test[s] - half_window),
        "target_window_end": int(target_end_test[s] + half_window),
    }
    for j in range(FUTURE_STEPS):
        row[f"pred_lead{j+1}"] = float(pred_test_mean[s, j, 0])
        row[f"true_lead{j+1}"] = float(y_obs_test[s, j, 0])
    wide_rows.append(row)
pd.DataFrame(wide_rows).to_csv(
    os.path.join(OUTPUT_DIR, "obs_test_predictions_vs_observations_wide.csv"),
    index=False,
)

print("\nTransfer model skill:")
print(skill_model.to_string(index=False))
print("\nPersistence baseline skill:")
print(skill_persist.to_string(index=False))

# Save hindcast
lead = np.arange(1, FUTURE_STEPS + 1)
ds_test = xr.Dataset(
    {
        "y_pred_members": (("member", "sample", "lead"), pred_tests[:, :, :, 0]),
        "y_pred_mean": (("sample", "lead"), pred_test_mean[:, :, 0]),
        "y_pred_std": (("sample", "lead"), pred_test_std[:, :, 0]),
        "y_true": (("sample", "lead"), y_obs_test[:, :, 0]),
        "issue_year": (("sample",), issue_test),
        "target_start_year": (("sample",), target_start_test),
        "target_end_year": (("sample",), target_end_test),
    },
    coords={"member": np.arange(1, pred_tests.shape[0] + 1), "sample": np.arange(pred_test_mean.shape[0]), "lead": lead},
    attrs={
        "description": "Observed PDO hindcast from a persistence-correction transfer-learning model",
        "predictand": "detrended standardized 21-year-running-mean PDO",
        "runmean_mode": RUNMEAN_MODE,
        "features": ", ".join(FEATURES),
        "delta_calibration": "leadwise_least_squares" if USE_LEADWISE_DELTA_CALIBRATION else "none",
        "output_ar_residual": str(USE_OUTPUT_AR_RESIDUAL),
        "delta_alpha_max": str(DELTA_ALPHA_MAX),
        "delta_alpha_shrink": str(DELTA_ALPHA_SHRINK),
        "min_calib_delta_corr": str(MIN_CALIB_DELTA_CORR),
        "min_calib_sign_accuracy": str(MIN_CALIB_SIGN_ACCURACY),
        "leakage_control": "observation test period was not used for scaling, Optuna, or fine-tuning",
    },
)
ds_test.to_netcdf(os.path.join(OUTPUT_DIR, "obs_test_hindcast_predictions.nc"))

# Save future prediction
future_years = np.arange(future_issue_year + 1, future_issue_year + FUTURE_STEPS + 1)
future_window_start = future_years - half_window if RUNMEAN_MODE == "center" else future_years
future_window_end = future_years + half_window if RUNMEAN_MODE == "center" else future_years
ds_future = xr.Dataset(
    {
        "pdo_pred_members": (("member", "lead"), pred_futures[:, 0, :, 0]),
        "pdo_pred_mean": (("lead",), pred_future_mean),
        "pdo_pred_std": (("lead",), pred_future_std),
        "pdo_pred_p05": (("lead",), pred_future_p05),
        "pdo_pred_p50": (("lead",), pred_future_p50),
        "pdo_pred_p95": (("lead",), pred_future_p95),
        "pdo_persistence_baseline": (("lead",), np.repeat(future_persist_value, FUTURE_STEPS)),
        "pdo_pred_delta_mean": (("lead",), pred_future_delta_mean),
        "pdo_pred_delta_p05": (("lead",), pred_future_delta_p05),
        "pdo_pred_delta_p50": (("lead",), pred_future_delta_p50),
        "pdo_pred_delta_p95": (("lead",), pred_future_delta_p95),
        "target_center_year": (("lead",), future_years),
        "target_window_start": (("lead",), future_window_start),
        "target_window_end": (("lead",), future_window_end),
    },
    coords={"member": np.arange(1, pred_futures.shape[0] + 1), "lead": lead},
    attrs={
        "description": "Future 10-year PDO prediction from latest observed predictors using persistence plus predicted correction",
        "issue_year": int(future_issue_year),
        "predictand": "detrended standardized 21-year-running-mean PDO",
        "runmean_mode": RUNMEAN_MODE,
        "features": ", ".join(FEATURES),
        "delta_calibration": "leadwise_least_squares" if USE_LEADWISE_DELTA_CALIBRATION else "none",
        "output_ar_residual": str(USE_OUTPUT_AR_RESIDUAL),
        "delta_alpha_max": str(DELTA_ALPHA_MAX),
        "delta_alpha_shrink": str(DELTA_ALPHA_SHRINK),
        "min_calib_delta_corr": str(MIN_CALIB_DELTA_CORR),
        "min_calib_sign_accuracy": str(MIN_CALIB_SIGN_ACCURACY),
    },
)
ds_future.to_netcdf(os.path.join(OUTPUT_DIR, "future_10yr_pdo_prediction.nc"))
future_df = pd.DataFrame({
    "lead": lead,
    "target_center_year": future_years,
    "target_window_start": future_window_start,
    "target_window_end": future_window_end,
    "pdo_pred_mean": pred_future_mean,
    "pdo_pred_std": pred_future_std,
    "pdo_pred_p05": pred_future_p05,
    "pdo_pred_p50": pred_future_p50,
    "pdo_pred_p95": pred_future_p95,
    "pdo_persistence_baseline": np.repeat(future_persist_value, FUTURE_STEPS),
    "pdo_pred_delta_mean": pred_future_delta_mean,
    "pdo_pred_delta_p05": pred_future_delta_p05,
    "pdo_pred_delta_p50": pred_future_delta_p50,
    "pdo_pred_delta_p95": pred_future_delta_p95,
})
for mm in range(pred_futures.shape[0]):
    future_df[f"pdo_pred_member_{mm+1}"] = pred_futures[mm, 0, :, 0]
future_df.to_csv(os.path.join(OUTPUT_DIR, "future_10yr_pdo_prediction.csv"), index=False)

# Plot skill
fig, axs = plt.subplots(2, 1, figsize=(9.5, 4.8), dpi=450)
axs[0].plot(skill_model["lead"], skill_model["corr"], "o-", label="ML model")
axs[0].plot(skill_persist["lead"], skill_persist["corr"], "o--", label="PDO persistence")
axs[0].axhline(0, color="k", lw=0.8, ls=":")
axs[0].axhline(0.5, color="gray", lw=0.8, ls=":")
axs[0].set_xlim(0.5, FUTURE_STEPS + 0.5)
axs[0].set_xticks(lead)
axs[0].set_ylabel("Correlation skill")
axs[0].set_title("Observed PDO hindcast skill", loc="left")
axs[0].grid(True, ls=":", lw=0.6)
axs[0].legend(frameon=False, fontsize=9)

axs[1].plot(skill_model["lead"], skill_model["rmse"], "o-", label="ML model")
axs[1].plot(skill_persist["lead"], skill_persist["rmse"], "o--", label="PDO persistence")
axs[1].set_xlim(0.5, FUTURE_STEPS + 0.5)
axs[1].set_xticks(lead)
axs[1].set_xlabel("Lead time (year)")
axs[1].set_ylabel("RMSE")
axs[1].grid(True, ls=":", lw=0.6)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "obs_test_prediction_skill.png"), bbox_inches="tight")
plt.close()

print("Saved outputs to:", OUTPUT_DIR)
print("Done.")
