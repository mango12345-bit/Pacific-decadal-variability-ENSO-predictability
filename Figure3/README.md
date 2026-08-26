# Figure 3

This directory contains the processed input data and analysis code used to reproduce **Figure 3** and the associated supplementary analyses in the manuscript.

## Files

### `figure3_input.nc`

This NetCDF file contains the processed time series used as direct inputs to the analyses presented in Figure 3 and the associated Appendix analyses.

The principal variables include the low-frequency climate indices and thermocline-related quantities used to examine the relationship between Pacific thermocline variability and large-scale decadal climate variability.

The input data include the AMO and PDO indices used in the manuscript, together with the thermocline-depth-gradient time series and other processed quantities required by the analysis.

The processed time series are provided directly in this repository so that the published statistical analyses can be reproduced without requiring users to reconstruct all indices from the original gridded climate datasets.

### `figure3_analysis.py`

This script performs the statistical analyses associated with Figure 3, including the construction and evaluation of the combined AMO–PDO index (APJPT), regression analyses, lead–lag relationships, partial-correlation analyses, and robustness tests.

The script reproduces:

- **Figure 3** — relationships between low-frequency Pacific thermocline variability and the combined influence of the AMO and PDO.
- **Appendix Figure 2** — lead–lag relationships used to examine the temporal relationship among the relevant low-frequency climate indices and thermocline variability.
- **Appendix Figure 4** — additional robustness and statistical diagnostics associated with the Figure 3 analysis.
- **Appendix Table 2** — statistical results associated with the regression and partial-correlation analyses.

## APJPT index

The analysis uses the combined AMO–PDO index defined in the manuscript as APJPT.

APJPT is calculated from the processed AMO and PDO time series within `figure3_analysis.py`. The individual AMO and PDO indices are provided in `figure3_input.nc`, allowing the combined index and subsequent statistical analyses to be reproduced directly from the supplied input data.

The precise definition and physical interpretation of APJPT follow those given in the Methods section of the manuscript.

## Reproducing the analysis

From this directory, run:

```bash
python figure3_analysis.py --input figure3_input.nc
