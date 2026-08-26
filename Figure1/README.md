# Figure 1

This directory contains the processed input data and analysis code used to reproduce **Figure 1**, **Appendix Figure 1**, and **Appendix Table 1** of the manuscript.

## Files

### `figure1_input.nc`

This NetCDF file contains the processed annual/seasonal time series used as direct inputs to the statistical analyses.

The data are provided for the ocean reanalysis products used in the manuscript:

- SODA–GODAS
- ORA-20C
- GECCO3

The principal variables include:

- `H` — spring (MAM) thermocline-related predictor used in the ENSO regression framework.
- `H_gradient` — spring (MAM) zonal thermocline-gradient predictor.
- `tau_x` — spring (MAM) zonal wind-stress predictor.
- `tau_y` — spring (MAM) meridional wind-stress predictor.
- `nino34_ond` — subsequent October–December (OND) Niño-3.4 SST anomaly.
- `TCD_G` — zonal thermocline-depth gradient.

The predictor time series were processed consistently with the procedures described in the Methods of the manuscript. The Niño-3.4 target is based on HadISST1.

The processed time series used in the published analysis are provided directly in this repository. Therefore, access to the original gridded ocean reanalysis products is not required to reproduce Figure 1 and its associated statistical analyses.

### `figure1_analysis.py`

This script performs the statistical analyses using `figure1_input.nc`, including the 21-year moving-window regression analysis and associated diagnostics.

The script reproduces:

- **Figure 1** — temporal evolution of the ENSO regression relationships and associated thermocline variability.
- **Appendix Figure 1** — additional diagnostics associated with the moving-window regression analysis.
- **Appendix Table 1** — statistical results associated with the regression analysis.

## Reproducing the analysis

From this directory, run:

```bash
python figure1_analysis.py --input figure1_input.nc
