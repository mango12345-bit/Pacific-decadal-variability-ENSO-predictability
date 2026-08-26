# Figure 2

This directory contains the processed input data and analysis code used to reproduce **Figure 2** of the manuscript.

---

## Files

### `figure2_input.nc`

This NetCDF file contains the processed time series used as direct inputs to the analyses presented in Figure 2.

The processed data are provided so that the published analysis can be reproduced directly without requiring users to repeat the upstream processing of the original gridded observational and reanalysis datasets.

### `figure2_analysis.py`

This script performs the analyses and generates Figure 2 from `figure2_input.nc`.

---

## Reproducing the analysis

From this directory, run:

```bash
python figure2_analysis.py --input figure2_input.nc
```

The reproduced figure and corresponding source-data files are written to the output directory.

---

## Software requirements

The analysis was implemented in Python. The principal required packages are:

- NumPy
- pandas
- xarray
- SciPy
- Matplotlib
- netCDF4

Additional package requirements, if applicable, are specified in the repository-level environment information.

---

## Data provenance

`figure2_input.nc` contains the processed time series used in the published analysis. These data were derived from the observational and/or reanalysis datasets described in the manuscript.

The processed data are provided here to facilitate direct reproduction of Figure 2. The original datasets remain subject to the terms and conditions of their respective data providers.

Full descriptions of the original datasets, their references, and availability information are provided in the manuscript and its Data Availability statement.

---

## Notes

The definitions of the indices, spatial domains, temporal averaging, detrending, standardization, and statistical procedures follow those described in the Methods of the manuscript.

This directory is intended to reproduce the published analysis from the processed inputs rather than to reproduce each upstream dataset from its original gridded fields.
