# Figure 2

This directory contains the processed input data and analysis code used to reproduce **Figure 2** of the manuscript.

## Files

### `figure2_input.nc`

This NetCDF file contains the processed time series used as direct inputs to the analyses presented in Figure 2.

The processed data are provided so that the published analysis can be reproduced directly without requiring users to repeat the upstream processing of the original gridded observational and reanalysis datasets.

### `figure2_analysis.py`

This script performs the analyses and generates Figure 2 from `figure2_input.nc`.

## Reproducing the analysis

From this directory, run:

```bash
python figure2_analysis.py --input figure2_input.nc
