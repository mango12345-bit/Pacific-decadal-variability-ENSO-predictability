# Code and Data for Pacific Decadal Variability and ENSO Predictability

This repository contains the analysis code, processed data, and trained deep-learning models used to reproduce the main and supplementary results of the associated manuscript.

The repository is organized by manuscript figure. Processed input data used in the published analyses are provided together with the corresponding Python scripts to facilitate direct reproduction of the results.

In addition, the repository contains the training resources and trained ensemble members for the deep-learning Pacific Decadal Oscillation prediction framework (DL-PDO).

---

## Repository structure

```text
.
├── README.md
│
├── Figure1/
│   ├── README.md
│   ├── figure1_analysis.py
│   └── figure1_input.nc
│
├── Figure2/
│   ├── README.md
│   ├── figure2_analysis.py
│   └── figure2_input.nc
│
├── Figure3/
│   ├── README.md
│   ├── figure3_analysis.py
│   └── figure3_input.nc
│
├── Figure4/
│   ├── README.md
│   ├── figure4_analysis.py
│   ├── amo_three_scenarios.py
│   └── data/
│       ├── pdo_index.1854-2025.nc
│       ├── AMO_index_1854-2025.nc
│       ├── future_10yr_pdo_prediction.csv
│       ├── AMO_future_10yr_three_methods_wide.csv
│       ├── global_warming_index_annual_raw.csv
│       └── GWI_future_10yr_recent30_quadratic_with_uncertainty.csv
│
├── Appendix_Figure3/
│   ├── README.md
│   ├── appendix_figure3_picontrol_leadlag.py
│   └── data/
│       └── [processed CMIP6 piControl input data]
│
└── DL_PDO/
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
    │   ├── ...
    │   └── pdo_transfer_obs_finetuned_member_20.h5
    │
    └── metadata/
        ├── best_optuna_params_cmip6.csv
        ├── ensemble_member_source_bootstrap_metadata.csv
        └── leadwise_delta_alpha_summary.csv
```

Individual directories contain their own `README.md` files with more detailed descriptions of the corresponding data and analysis procedures.

---

## Reproducing the manuscript results

The repository is organized so that individual figures and supplementary analyses can be reproduced independently.

| Manuscript output | Directory | Main script | Input data |
|---|---|---|---|
| Figure 1 | `Figure1/` | `figure1_analysis.py` | `figure1_input.nc` |
| Appendix Figure 1 | `Figure1/` | `figure1_analysis.py` | `figure1_input.nc` |
| Appendix Table 1 | `Figure1/` | `figure1_analysis.py` | `figure1_input.nc` |
| Figure 2 | `Figure2/` | `figure2_analysis.py` | `figure2_input.nc` |
| Figure 3 | `Figure3/` | `figure3_analysis.py` | `figure3_input.nc` |
| Appendix Figure 2 | `Figure3/` | `figure3_analysis.py` | `figure3_input.nc` |
| Appendix Figure 4 | `Figure3/` | `figure3_analysis.py` | `figure3_input.nc` |
| Appendix Table 2 | `Figure3/` | `figure3_analysis.py` | `figure3_input.nc` |
| Appendix Figure 3 | `Appendix_Figure3/` | `appendix_figure3_picontrol_leadlag.py` | Processed CMIP6 piControl time series |
| Figure 4 | `Figure4/` | `figure4_analysis.py` | Processed historical and future climate indices |
| DL-PDO training | `DL_PDO/` | `dl_pdo_train.py` | CMIP6-derived and observational climate-index time series |

---

## Figure 1

The `Figure1/` directory contains the processed input data and analysis code used to reproduce:

- **Figure 1**
- **Appendix Figure 1**
- **Appendix Table 1**

The analysis uses processed observational and ocean-reanalysis time series, including thermocline-related and atmospheric predictors together with the subsequent OND Niño-3.4 SST anomaly.

The processed inputs are provided in:

```text
Figure1/figure1_input.nc
```

See `Figure1/README.md` for additional information.

---

## Figure 2

The `Figure2/` directory contains the processed input data and analysis code used to reproduce **Figure 2**.

The processed inputs used in the published analysis are provided in:

```text
Figure2/figure2_input.nc
```

See `Figure2/README.md` for additional information.

---

## Figure 3 and associated Appendix analyses

The `Figure3/` directory contains the processed climate indices and analysis code used to reproduce:

- **Figure 3**
- **Appendix Figure 2**
- **Appendix Figure 4**
- **Appendix Table 2**

These analyses examine the relationships among low-frequency climate variability, the AMO and PDO, and Pacific thermocline variability, including the combined AMO–PDO index used in the manuscript.

The processed inputs are provided in:

```text
Figure3/figure3_input.nc
```

See `Figure3/README.md` for additional information.

---

## Appendix Figure 3

The `Appendix_Figure3/` directory contains the analysis code and processed CMIP6 pre-industrial control (piControl) time series used to reproduce **Appendix Figure 3**.

This analysis provides a model-based assessment of the lead–lag relationships examined in the manuscript using long CMIP6 piControl simulations.

Processed model time series are supplied so that the published analysis can be reproduced without downloading and preprocessing the complete CMIP6 piControl gridded datasets.

See `Appendix_Figure3/README.md` for additional information.

---

## Figure 4

The `Figure4/` directory contains the analysis code and processed historical and future climate-index data used to reproduce **Figure 4**.

The analysis combines:

- historical PDO;
- historical AMO;
- the near-term DL-PDO prediction;
- three future AMO scenarios;
- historical global warming influence (GWI); and
- the future GWI trajectory and associated uncertainty.

The future PDO prediction is generated using the DL-PDO framework documented in the `DL_PDO/` directory.

The three future AMO trajectories used directly in Figure 4 are provided as processed data. The auxiliary script:

```text
Figure4/amo_three_scenarios.py
```

is provided to document and reproduce how the three future AMO scenarios were constructed.

Running this auxiliary script is not required to reproduce Figure 4 because the resulting future AMO trajectories are already supplied in the repository.

See `Figure4/README.md` for additional information.

---

## DL-PDO model

The `DL_PDO/` directory contains the resources used for the deep-learning PDO prediction framework developed in the manuscript.

The DL-PDO framework uses five climate predictors:

- Atlantic Multidecadal Oscillation (AMO)
- Pacific Decadal Oscillation (PDO)
- Arctic Oscillation (AO)
- Ocean heat content (OHC)
- Sea surface height anomaly (SSHA)

The neural network uses a 10-year history of these predictors to predict PDO evolution over the subsequent 10 years.

The model is first pretrained using CMIP6 simulations and subsequently fine-tuned using observational data. The final prediction is based on a 20-member ensemble constructed using bootstrap resampling of the CMIP6 training models.

The `DL_PDO/` directory provides:

- processed CMIP6-derived climate indices;
- processed observational climate indices;
- the complete DL-PDO training script;
- selected Optuna hyperparameters;
- ensemble-member bootstrap metadata;
- lead-dependent calibration information; and
- the 20 trained neural-network ensemble members used in the manuscript.

The trained models are provided to preserve the model states used for the published prediction and to avoid requiring users to repeat the computationally intensive training procedure solely to inspect the final ensemble.

See `DL_PDO/README.md` for detailed training and reproducibility information.

---

## Processed data and reproducibility

This repository primarily provides the **processed inputs used directly in the published analyses**.

For example, processed climate indices and time series are supplied rather than redistributing the complete original gridded observational, reanalysis, and climate-model datasets.

This design allows the statistical analyses and figure-generation procedures reported in the manuscript to be reproduced directly while keeping the repository compact and avoiding unnecessary redistribution of large third-party datasets.

The basic reproducibility structure is:

```text
Original observational / reanalysis / CMIP6 data
                         │
                         │  upstream processing
                         ▼
              Processed input data
                         │
                         │  supplied in this repository
                         ▼
                 Analysis scripts
                         │
                         ▼
          Figures / Appendix results
```

For the DL-PDO framework, processed climate-index time series are additionally supplied so that the neural-network training workflow can be repeated without reconstructing all indices from the original gridded climate fields.

---

## Original data sources

The processed data in this repository were derived from the observational, reanalysis, and climate-model datasets described in the manuscript.

The complete original gridded datasets are not redistributed in this repository.

The original datasets remain subject to the terms and conditions of their respective data providers.

Full dataset descriptions, references, temporal coverage, processing procedures, and access information are provided in the manuscript and its **Data Availability** statement.

---

## Software requirements

The analyses were implemented primarily in Python.

The principal packages used across the repository include:

- NumPy
- pandas
- xarray
- SciPy
- statsmodels
- scikit-learn
- Matplotlib
- netCDF4
- TensorFlow
- Optuna

Not all packages are required for every analysis.

The conventional statistical analyses in `Figure1/`, `Figure2/`, `Figure3/`, `Figure4/`, and `Appendix_Figure3/` can be run independently of TensorFlow.

TensorFlow and Optuna are required for the DL-PDO training workflow.

A CUDA-compatible GPU is recommended for reproducing the complete 20-member DL-PDO training procedure.

---

## General usage

Clone or download the repository and enter the directory corresponding to the analysis of interest.

For example, to reproduce Figure 1:

```bash
cd Figure1
python figure1_analysis.py --input figure1_input.nc
```

To reproduce Figure 3:

```bash
cd Figure3
python figure3_analysis.py --input figure3_input.nc
```

To reproduce Figure 4:

```bash
cd Figure4
python figure4_analysis.py
```

To repeat the DL-PDO training workflow:

```bash
cd DL_PDO
python dl_pdo_train.py
```

More detailed instructions are provided in the `README.md` file within each directory.

---

## Reproducibility of neural-network training

Neural-network optimization may exhibit small numerical differences across hardware platforms, TensorFlow versions, CUDA versions, and GPU configurations.

Consequently, independently retrained DL-PDO model weights are not expected to be identical bit-for-bit in all computational environments.

To document the exact trained ensemble used for the results reported in the manuscript, the 20 final trained neural-network models and their associated ensemble metadata are provided in the `DL_PDO/` directory.

---

## Citation

If you use the code or processed data provided in this repository, please cite the associated manuscript.

The full citation will be added following publication.

---

## License

Please refer to the repository license for terms governing reuse of the code provided here.

The original observational, reanalysis, and climate-model datasets are governed by the licenses and terms of use of their respective data providers.

---

## Contact

Questions regarding the code, processed data, or reproduction of the analyses can be directed to the corresponding author of the manuscript.
