# DL-PDO Model

This directory contains the processed input time series, training code,
selected hyperparameters, ensemble metadata, and trained models used for the
deep-learning PDO (DL-PDO) prediction presented in the manuscript.

The DL-PDO framework uses five climate predictors:

- Atlantic Multidecadal Oscillation (AMO)
- Pacific Decadal Oscillation (PDO)
- Arctic Oscillation (AO)
- Ocean heat content (OHC)
- Sea surface height anomaly (SSHA)

The model is first pretrained using CMIP6 simulations and subsequently
fine-tuned using observational data. A 20-member ensemble is used for the
final PDO prediction.

## Directory structure

```text
DL_PDO/
├── README.md
├── dl_pdo_train.py
│
├── data/
│   ├── AMO_index_model.nc
│   ├── pdo_model_series.nc
│   ├── ao_model_series.nc
│   ├── ohc_index_model.nc
│   ├── ssha_model_index_cp.nc
│   │
│   ├── AMO_index_1854-2025.nc
│   ├── pdo_index.1854-2025.nc
│   ├── ao_series.nc
│   ├── ohc_index_cp.nc
│   └── ssha_index_cp.nc
│
├── models/
│   ├── pdo_transfer_obs_finetuned_member_1.h5
│   ├── ...
│   └── pdo_transfer_obs_finetuned_member_20.h5
│
└── metadata/
    ├── best_optuna_params_cmip6.csv
    ├── ensemble_member_source_bootstrap_metadata.csv
    └── leadwise_delta_alpha_summary

## Training data
The data/ directory contains the processed climate-index time series used
as inputs to the DL-PDO framework.

## CMIP6 data
The following files contain indices derived from the CMIP6 simulations:

AMO_index_model.nc — AMO index
pdo_model_series.nc — PDO index
ao_model_series.nc — AO index
ohc_index_model.nc — OHC index
ssha_model_index_cp.nc — central-Pacific SSHA index

These processed CMIP6 time series are used for model pretraining and for
constructing the model-bootstrap ensemble.

## Observational data
The following files contain the corresponding observational indices:

AMO_index_1854-2025.nc — observational AMO index
pdo_index.1854-2025.nc — observational PDO index
ao_series.nc — observational AO index
ohc_index_cp.nc — observational OHC index
ssha_index_cp.nc — observational central-Pacific SSHA index

These observational time series are used for transfer learning, model
calibration, evaluation, and the final near-term PDO prediction.

The processed time series used in the study are provided directly in this
repository so that the DL-PDO training procedure can be reproduced without
requiring users to reconstruct all climate indices from the original gridded
datasets.

## Model architecture and training

The DL-PDO model is a residual sequence-to-sequence neural network with
attention.
The model uses a 10-year history of the five predictors (AMO, PDO, AO, OHC,
and SSHA) to predict PDO evolution over the subsequent 10 years.

The training procedure consists of the following main steps:
1.Preprocess the CMIP6 and observational climate indices.
2.Apply detrending and standardization using parameters estimated from the
appropriate training/calibration periods.
3.Construct the low-frequency climate indices using a centered 21-year
running mean.
4.Construct 10-year input and 10-year target sequences.
5.Pretrain the neural network using CMIP6 simulations.
6.Apply transfer learning using observational data.
7.Generate a 20-member ensemble using bootstrap resampling of the CMIP6
training models.
8.Apply member-specific, lead-dependent calibration to the predicted PDO
changes.
9.Evaluate the ensemble using the observational test period and generate
the near-term PDO prediction.

Full methodological details are provided in the Methods section of the
manuscript.

## Training script

dl_pdo_train.py implements the training, transfer-learning, ensemble,
calibration, evaluation, and prediction workflow used in the study.

By default, the script reads the selected hyperparameters from:
metadata/best_optuna_params_cmip6.csv and trains the 20-member ensemble.

Run:
python dl_pdo_train.py

To repeat the Optuna hyperparameter search instead of using the provided
selected hyperparameters, run:
python dl_pdo_train.py --run-optuna

For example, to perform 50 Optuna trials:
python dl_pdo_train.py --run-optuna --n-trials 50

Re-running neural-network training may not reproduce model weights
bit-for-bit across different hardware, TensorFlow versions, CUDA versions,
or GPU configurations. The trained models used for the manuscript are
therefore provided separately in the models/ directory.

## Selected hyperparameters
metadata/best_optuna_params_cmip6.csv contains the hyperparameters selected
from the CMIP6-based Optuna optimization and used in the final training
workflow.

Providing these selected hyperparameters allows the final ensemble training
to be repeated without requiring users to rerun the complete hyperparameter
search.

## Ensemble metadata and calibration
metadata/ensemble_member_source_bootstrap_metadata.csv records information
for each ensemble member, including the CMIP6 models selected during bootstrap
resampling and the member-specific lead-wise calibration coefficients.

The lead-wise calibration coefficients are used to calibrate the predicted
change in PDO relative to the persistence baseline.

metadata/leadwise_delta_alpha_summary.csv provides a summary of these
lead-dependent calibration coefficients across the ensemble.

## Trained models

The models/ directory contains the 20 final observation-fine-tuned ensemble
members used in the manuscript:
pdo_transfer_obs_finetuned_member_1.h5
...
pdo_transfer_obs_finetuned_member_20.h5

These files contain the trained neural-network models corresponding to the
final DL-PDO ensemble used for the reported prediction.

They are provided to preserve the exact trained model states used in the
study and to avoid requiring users to repeat the computationally expensive
training procedure solely to inspect the final models.

## Software requirements

The DL-PDO framework was implemented in Python. The principal required
packages are:

NumPy
pandas
xarray
SciPy
scikit-learn
TensorFlow
Optuna
netCDF4

A CUDA-compatible GPU is recommended for reproducing the complete training
procedure, although CPU execution is also supported.

Reproducibility notes

The supplied processed time series represent the model inputs used in the
study. The original gridded observational and CMIP6 datasets are therefore
not required to reproduce the neural-network training workflow from these
processed inputs.

The original datasets remain subject to the terms and conditions of their
respective data providers. Dataset descriptions, references, and availability
information are provided in the manuscript and its Data Availability
statement.

Because neural-network optimization can exhibit small platform-dependent
differences, exact numerical reproduction of independently retrained model
weights is not guaranteed. The 20 trained models used for the manuscript are
provided to document the exact ensemble used for the reported results.
