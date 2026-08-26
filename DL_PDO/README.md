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

# DL-PDO Model

This directory contains the processed input time series, training code, selected hyperparameters, ensemble metadata, and trained models used for the deep-learning PDO (DL-PDO) prediction presented in the manuscript.

The DL-PDO framework uses five climate predictors:

- Atlantic Multidecadal Oscillation (AMO)
- Pacific Decadal Oscillation (PDO)
- Arctic Oscillation (AO)
- Ocean heat content (OHC)
- Sea surface height anomaly (SSHA)

The model is first pretrained using CMIP6 simulations and subsequently fine-tuned using observational data. A 20-member ensemble is used for the final PDO prediction.

---

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
│   ├── AMO_index_1854-2025.nc
│   ├── pdo_index.1854-2025.nc
│   ├── ao_series.nc
│   ├── ohc_index_cp.nc
│   └── ssha_index_cp.nc
│
├── models/
│   ├── pdo_transfer_obs_finetuned_member_1.h5
│   ├── pdo_transfer_obs_finetuned_member_2.h5
│   ├── ...
│   └── pdo_transfer_obs_finetuned_member_20.h5
│
└── metadata/
    ├── best_optuna_params_cmip6.csv
    ├── ensemble_member_source_bootstrap_metadata.csv
    └── leadwise_delta_alpha_summary.csv
```

---

## Input data

The `data/` directory contains the processed climate-index time series used as inputs to the DL-PDO framework.

The five predictors are AMO, PDO, AO, OHC, and SSHA. Both CMIP6-derived and observational time series are provided.

### CMIP6-derived indices

The following files contain the climate indices derived from CMIP6 simulations and used for model pretraining:

- `AMO_index_model.nc` — AMO index
- `pdo_model_series.nc` — PDO index
- `ao_model_series.nc` — AO index
- `ohc_index_model.nc` — OHC index
- `ssha_model_index_cp.nc` — central-Pacific SSHA index

These processed CMIP6 time series are used for neural-network pretraining and for constructing the model-bootstrap ensemble.

### Observational indices

The following files contain the corresponding observational indices:

- `AMO_index_1854-2025.nc` — observational AMO index
- `pdo_index.1854-2025.nc` — observational PDO index
- `ao_series.nc` — observational AO index
- `ohc_index_cp.nc` — observational OHC index
- `ssha_index_cp.nc` — observational central-Pacific SSHA index

These observational time series are used for transfer learning, calibration, evaluation, and the final near-term PDO prediction.

The processed time series used in the study are provided directly in this repository so that the DL-PDO training and prediction workflow can be reproduced without requiring users to reconstruct all climate indices from the original gridded datasets.

---

## DL-PDO framework

The DL-PDO model is a residual sequence-to-sequence neural network with attention.

The model uses a 10-year history of five climate predictors:

1. AMO
2. PDO
3. AO
4. OHC
5. SSHA

to predict PDO evolution over the subsequent 10 years.

The predictand is formulated as the change in PDO relative to a persistence baseline. The final PDO prediction is obtained by combining the persistence component with the neural-network-predicted PDO change.

The climate indices are processed using the procedures described in the Methods of the manuscript, including detrending, standardization, temporal averaging, and centered 21-year running means.

---

## Training procedure

The DL-PDO training workflow consists of the following main steps:

1. Read the processed CMIP6 and observational climate-index time series.
2. Apply the preprocessing procedures used in the manuscript.
3. Construct the low-frequency climate indices using centered 21-year running means.
4. Construct 10-year input sequences and 10-year PDO target sequences.
5. Pretrain the neural network using CMIP6 simulations.
6. Apply transfer learning using observational data.
7. Generate a 20-member ensemble using bootstrap resampling of the CMIP6 training models.
8. Apply member-specific, lead-dependent calibration to the predicted PDO changes.
9. Evaluate the ensemble using the observational test period.
10. Generate the near-term PDO prediction used in the manuscript.

Full methodological details are provided in the Methods section of the manuscript.

---

## Training script

`dl_pdo_train.py` implements the complete DL-PDO workflow, including:

- data preprocessing;
- CMIP6 pretraining;
- observational transfer learning;
- ensemble construction;
- lead-dependent calibration;
- hindcast evaluation;
- persistence-baseline evaluation;
- trained-model output; and
- near-term PDO prediction.

By default, the script reads the selected hyperparameters from:

```text
metadata/best_optuna_params_cmip6.csv
```

and uses these parameters to train the ensemble.

To reproduce the training workflow, run from the `DL_PDO` directory:

```bash
python dl_pdo_train.py
```

The manuscript uses a 20-member ensemble, which is the default configuration of the script.

For testing purposes, a smaller ensemble can be trained using, for example:

```bash
python dl_pdo_train.py --n-ensemble 2
```

---

## Hyperparameter optimization

The hyperparameters used for the final DL-PDO model were selected using Optuna based on the CMIP6 training and validation data.

The selected hyperparameters are provided in:

```text
metadata/best_optuna_params_cmip6.csv
```

Providing the selected hyperparameters allows the final ensemble-training workflow to be repeated without requiring users to rerun the complete hyperparameter search.

To repeat the Optuna hyperparameter optimization, run:

```bash
python dl_pdo_train.py --run-optuna
```

The number of Optuna trials can also be specified. For example:

```bash
python dl_pdo_train.py --run-optuna --n-trials 50
```

The resulting selected hyperparameters are saved to the metadata directory.

---

## Ensemble construction

The final DL-PDO prediction is based on a 20-member ensemble.

Ensemble diversity is generated through bootstrap resampling of the CMIP6 models used during pretraining. Each ensemble member is subsequently fine-tuned using the observational training data.

Information describing the source-model bootstrap sampling for each ensemble member is provided in:

```text
metadata/ensemble_member_source_bootstrap_metadata.csv
```

This file records the CMIP6 model selection associated with each ensemble member together with the member-specific lead-dependent calibration coefficients used in the final prediction.

---

## Lead-dependent calibration

The neural network predicts the future change in PDO relative to the persistence baseline.

A member-specific and lead-dependent calibration is subsequently applied to the predicted PDO changes.

The calibration coefficients for individual ensemble members are included in:

```text
metadata/ensemble_member_source_bootstrap_metadata.csv
```

A summary of the lead-dependent calibration coefficients across the ensemble is provided in:

```text
metadata/leadwise_delta_alpha_summary.csv
```

The final PDO prediction is obtained by adding the calibrated predicted PDO change to the corresponding persistence baseline.

---

## Trained models

The `models/` directory contains the 20 final observation-fine-tuned neural-network models used for the results reported in the manuscript:

```text
pdo_transfer_obs_finetuned_member_1.h5
pdo_transfer_obs_finetuned_member_2.h5
...
pdo_transfer_obs_finetuned_member_20.h5
```

These files preserve the trained states of the final DL-PDO ensemble used in the study.

The trained models are provided in addition to the training code because independently retraining a neural network may produce small numerical differences depending on hardware, TensorFlow version, CUDA version, and other computational settings.

Providing the trained models therefore documents the exact ensemble used to obtain the reported DL-PDO prediction, while `dl_pdo_train.py` provides the complete workflow required to reproduce the model-training procedure.

---

## Model evaluation and prediction

The training script evaluates the final ensemble using the observational test period.

The evaluation includes comparison with a persistence baseline and calculation of the prediction skill of the ensemble.

After training and calibration, the script also generates the near-term PDO ensemble prediction used in the manuscript.

Prediction outputs and associated diagnostic files are written to the `outputs/` directory when the training script is executed.

---

## Year convention

The PDO predictand is based on a centered 21-year running mean.

A conventional centered 21-year mean assigned to year *Y* represents the interval from *Y − 10* to *Y + 10*.

For presentation of the near-term prediction in the manuscript, the prediction year is labeled by the **ending year of the corresponding 21-year averaging window** to emphasize the future period represented by the prediction.

For example, a value displayed at **2026** represents the mean over **2006–2026**, whose conventional centered-year label is 2016.

This labeling convention affects only the presentation of the prediction years and does not alter the underlying model calculation.

---

## Software requirements

The DL-PDO framework was implemented in Python.

The principal required packages are:

- NumPy
- pandas
- xarray
- SciPy
- scikit-learn
- TensorFlow
- Optuna
- netCDF4

A CUDA-compatible GPU is recommended for reproducing the complete training procedure because training the 20-member neural-network ensemble is computationally intensive.

CPU execution is also supported but may require substantially more computation time.

---

## Reproducibility

Two complementary components are provided to facilitate reproducibility.

### Training reproducibility

The combination of:

```text
dl_pdo_train.py
data/
metadata/best_optuna_params_cmip6.csv
```

provides the inputs and workflow required to repeat the DL-PDO training procedure.

### Exact trained ensemble

The combination of:

```text
models/
metadata/ensemble_member_source_bootstrap_metadata.csv
```

documents the trained 20-member ensemble and the member-specific calibration information used for the results reported in the manuscript.

Because neural-network optimization can exhibit small platform-dependent differences, independently retrained model weights are not expected to be identical bit-for-bit across all computational environments. The trained `.h5` models are therefore provided to preserve the model states used in the study.

---

## Data provenance

The processed CMIP6-derived and observational climate-index time series supplied in the `data/` directory correspond to the inputs used in the DL-PDO analysis reported in the manuscript.

These processed indices are provided to facilitate direct reproduction of the neural-network training and prediction workflow.

The repository does not redistribute the complete original gridded CMIP6 and observational datasets from which these indices were derived.

The original datasets remain subject to the terms and conditions of their respective data providers. Full dataset descriptions, references, processing definitions, and availability information are provided in the manuscript and its Data Availability statement.

---

## Notes

The definitions of the AMO, PDO, AO, OHC, and SSHA indices; preprocessing procedures; training and validation periods; neural-network architecture; transfer-learning strategy; ensemble construction; calibration procedure; and prediction framework follow those described in the Methods of the manuscript.

This directory is intended to provide both the processed inputs required to reproduce the DL-PDO training workflow and the trained ensemble used to obtain the PDO prediction reported in the manuscript.
