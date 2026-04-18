# Winter-Chill Attribution and CMIP6 Projections of ENSO-Driven Olive Yield Collapse on the Hyper-Arid Peruvian Coast — Reproducibility Archive

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19640135.svg)](https://doi.org/10.5281/zenodo.19640135)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the code, intermediate data, figures and LaTeX manuscript source supporting:

> Quille-Mamani J., Huanuqueño-Murillo J., Quispe-Tito D., Huayna-Felipe G.,
> Espinoza-Molina J., Acosta-Caipa K., Pérez-Cubas H.S., Ingol-Blanco E.,
> Ramos-Fernández L., Pino-Vargas E. (2026). *Winter-Chill Attribution and CMIP6
> Projections of ENSO-Driven Olive Yield Collapse on the Hyper-Arid Peruvian Coast.*
> **Agronomy** (MDPI).

## Key findings reproduced by this archive

- Year-level log-OLS yield model with mean chill-window and fruit-growth temperatures:
  **R²_log = 0.65**, chill slope **β = −0.82**.
- Chill slope robust across three independent tests:
  one-sided permutation **p = 0.036**; closed-form Bayesian posterior with
  **99.8 %** of mass below zero (Savage–Dickey **BF₁₀ = 15.9**); parcel-year
  mixed model **p < 10⁻¹⁴**.
- Counterfactual restoration of chill-window temperature to its non-failure
  climatology recovers the full 2016 and 2024 collapses; restoring fruit-growth
  temperature recovers nothing.
- CMIP6 delta-method projections identify a chill-collapse threshold at
  **ΔT_winter ≈ +1.25 °C**: SSP1-2.6 alone yields **~−52 %** mid-century,
  SSP5-8.5 reaches **−89 %** by 2051–2070.

## Data availability

- `00_raw_data/meteo_daily_2015_2025.csv` — in-situ daily T_min / T_max / RH,
  La Yarada, Tacna (2015–2025).
- `00_raw_data/s2_indices_per_parcel.csv` — Sentinel-2 NDVI / NDRE / CIre per parcel.
- `00_raw_data/oni.data` — NOAA PSL Oceanic Niño Index.
- `00_raw_data/Parcelas.*` — parcel polygon shapefile.
- **Parcel-level annual yields (`yield_parcels_2016_2024.csv`) are NOT redistributed**
  because they are commercially sensitive; the aggregated year-level dataset
  (`03_modeling/outputs/tables/year_level_dataset.csv`) is included and is sufficient
  to reproduce every number, table and figure in the manuscript. Raw parcel-level
  yields are available from the corresponding authors upon reasonable request and
  with written permission from the grower.

## Reproducing the analysis

```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# 1. Preprocessing (chill portions, heat indices, phenology aggregation, VPD)
python 01_preprocessing/scripts/01_chill_portions.py
python 01_preprocessing/scripts/02_heat_stress_indices.py
python 01_preprocessing/scripts/03_s2_phenology_aggregation.py
python 01_preprocessing/scripts/04_build_feature_matrix.py

# 2. Exploratory data analysis
python 02_eda/scripts/*.py

# 3. Primary model, robustness and Bayesian sensitivity
python 03_modeling/scripts/07_final_primary_log.py
python 03_modeling/scripts/06_robustness_and_loyo_variants.py
python 03_modeling/scripts/10_bayesian_sensitivity.py

# 4. CMIP6 delta-method projections
python 04_projections/scripts/01_cmip6_delta_projections.py
```

Key output: `03_modeling/outputs/tables/FINAL_MODEL_SUMMARY.txt` reproduces the
primary log-OLS fit (R²_log = 0.65, β_chill = −0.82).

## Directory layout

```
00_raw_data/            in-situ met, Sentinel-2 indices, ONI, parcel shapefile
01_preprocessing/       chill portions, heat indices, phenology windows, VPD
02_eda/                 exploratory plots and descriptive statistics
03_modeling/            primary log-OLS, bootstrap, LOYO, counterfactual, Bayesian, mixed model
04_projections/         CMIP6 delta-method, warming sweep, risk surface
05_manuscript/          LaTeX source, figures, tables and built PDFs
```

## License

Code: MIT. Data files (CSV, shapefile, PNG) and the LaTeX manuscript source:
CC-BY-4.0. See `LICENSE`.

## Citation

If you use this archive, please cite the Agronomy paper above and this
archive via its Zenodo DOI: [10.5281/zenodo.19640135](https://doi.org/10.5281/zenodo.19640135).

BibTeX entry:

```bibtex
@misc{quillemamani2026zenodo,
  author    = {Quille-Mamani, J. and Huanuqueño-Murillo, J. and Quispe-Tito, D. and
               Huayna-Felipe, G. and Espinoza-Molina, J. and Acosta-Caipa, K. and
               Pérez-Cubas, H.S. and Ingol-Blanco, E. and Ramos-Fernández, L. and
               Pino-Vargas, E.},
  title     = {Winter-Chill Attribution and {CMIP6} Projections of {ENSO}-Driven Olive Yield
               Collapse on the Hyper-Arid {Peruvian} Coast --- Reproducibility Archive},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v1.0.0},
  doi       = {10.5281/zenodo.19640135},
  url       = {https://doi.org/10.5281/zenodo.19640135}
}
```
