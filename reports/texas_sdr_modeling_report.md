# Texas SDR Modeling Report for Groundwater Prospectivity

## Purpose

The Texas SDR dataset is useful as an open-data proof of concept for a Kazakhstan groundwater heatmap workflow. It provides point observations of drilled wells, coordinates, reported water levels, well tests, lithology intervals, water-bearing strata, construction details, and usage labels. In Kazakhstan, the same trained workflow should not rely on Texas-only administrative labels; it should learn relationships between well outcomes and transferable spatial predictors such as DEM derivatives, distance to drainage, geological units, lineaments/fault proxies, Sentinel/Landsat indices, climate, soil, and terrain position.

Parsing note: SDR text files are pipe-delimited (`|`). A small number of rows contain unescaped delimiter characters inside free-text fields, so the EDA reader uses tolerant parsing and skips malformed rows for this proof-of-concept pass. For production ingestion, those rows should be repaired with a table-specific parser if their text fields are needed.

## SDR Tables That Matter Most

- `WellData.txt`: primary one-row-per-report table. Use for well id, coordinates, county, elevation, proposed use, drilling dates, pump depth, and QA flags.
- `WellLevels.txt`: water-level measurements. Best target source for depth-to-water heatmaps.
- `WellTest.txt`: yield, drawdown, hours, and derived specific capacity. Best target source for productivity heatmaps.
- `WellStrata.txt`: reported water-bearing intervals and water type. Useful for aquifer presence and fresh-water labels.
- `WellLithology.txt`: interval descriptions. Useful for extracting coarse geology proxies such as sand/gravel/clay/limestone thickness.
- `WellBoreHole.txt`, `WellCasing.txt`, `WellFilter.txt`, `WellSealRange.txt`, `WellCompletion.txt`, `WellDrillingMethod.txt`: useful technical covariates and QA signals, but many are partly outcomes of drilling design rather than pre-drilling predictors.
- `Plug*.txt`: useful mainly for excluding plugged/abandoned wells or studying failure/closure patterns; lower priority for first prospectivity model.

## Recommended Target Variables

| Priority | Target | Field in clean parquet | Modeling type | Heatmap meaning | Notes |
|---|---|---|---|---|---|
| 1 | Well yield | `yield_gpm_max` or `yield_gpm_median` | Regression | Expected production rate | Directly useful for groundwater productivity. Use log transform or quantile bins because yield is usually skewed. |
| 2 | High-yield class | `target_high_yield_20gpm` | Classification | Probability of productive well | Good first XGBoost target. Threshold should later be tuned by use case: domestic, livestock, irrigation. |
| 3 | Depth to water | `static_water_level_ft_first` | Regression | Shallower/deeper groundwater surface | Useful for drilling cost and accessibility. Needs care because measurement date and pumping conditions vary. |
| 4 | Shallow water class | `target_shallow_water_le_100ft` | Classification | Probability water is reachable within 100 ft | Good for heatmap communication, but threshold must be geologically and economically justified. |
| 5 | Fresh-water presence | `target_fresh_water_present` | Classification | Probability of fresh water in reported strata | Useful in arid/saline regions. Texas text labels are noisy, so treat this as weak supervision. |
| 6 | Fresh-water thickness | `fresh_water_thickness_ft` | Regression | Total reported fresh interval thickness | Potentially powerful but derived from driller text and interval reporting conventions. |
| 7 | Specific capacity | `specific_capacity_gpm_per_ft_median` | Regression | Productivity normalized by drawdown | Hydrogeologically strong target, but coverage is lower because drawdown is often missing/non-numeric. |

## Recommended Clean Dataset Columns

Keep one row per `WellReportTrackingNumber`. The current script writes this to `data/processed/texas_sdr_wells.parquet`.

Core identifiers and location:

- `well_id`, `WellReportTrackingNumber`, `latitude`, `longitude`, `CoordDDLat`, `CoordDDLong`, `HorizontalDatumType`, `County`, `has_valid_coordinates`.

Temporal and administrative fields:

- `DateSubmitted`, `DrillingStartDate`, `DrillingEndDate`, `drilling_duration_days`, `TypeOfWork`, `ProposedUse`.

Candidate targets:

- `static_water_level_ft_first`, `static_water_level_ft_min`, `static_water_level_ft_median`, `yield_gpm_max`, `yield_gpm_median`, `drawdown_ft_median`, `specific_capacity_gpm_per_ft_median`, `fresh_water_interval_count`, `fresh_water_thickness_ft`, `target_high_yield_20gpm`, `target_shallow_water_le_100ft`, `target_fresh_water_present`.

Transferable geologic/well-derived features:

- `Elevation`, `PumpDepth`, `water_bearing_interval_count`, `water_bearing_max_bottom_ft`, `saline_or_bad_water_interval_count`, `borehole_max_bottom_ft`, `casing_max_bottom_ft`, `filter_total_thickness_ft`, `seal_total_thickness_ft`, `lithology_total_logged_thickness_ft`, and all `lith_*_thickness_ft` / `lith_*_interval_count` columns.

Construction/design fields to use cautiously:

- `SealMethod`, `SurfaceCompletion`, `PumpType`, `borehole_completion_first`, `drilling_method_first`, casing/filter/seal interval aggregates. These may leak human design choices made after a promising drilling decision, so prefer them for diagnostics or auxiliary models, not final Kazakhstan inference unless equivalent pre-drilling information exists.

Fields to avoid as model features for Kazakhstan transfer:

- Owner/driller names, addresses, license numbers, comments, exact county labels, and post-drilling completion details that would not exist at inference time in Kazakhstan. These can create leakage or Texas-specific memorization.

## Target Availability in Current Clean Dataset

| field                               |   non_null_rows |   coverage_pct |
|:------------------------------------|----------------:|---------------:|
| static_water_level_ft_first         |          374654 |          53.21 |
| yield_gpm_max                       |          297908 |          42.31 |
| yield_gpm_median                    |          297908 |          42.31 |
| specific_capacity_gpm_per_ft_median |          109099 |          15.5  |
| fresh_water_interval_count          |          253172 |          35.96 |
| fresh_water_thickness_ft            |          253172 |          35.96 |
| target_high_yield_20gpm             |          297908 |          42.31 |
| target_fresh_water_present          |          253172 |          35.96 |
| target_shallow_water_le_100ft       |          374654 |          53.21 |

## XGBoost Dataset Strategy

For the first proof of concept, train separate models rather than one overloaded target:

1. `target_high_yield_20gpm` classification for a productivity probability heatmap.
2. `yield_gpm_max` regression for expected yield, evaluated with log-transformed yield.
3. `static_water_level_ft_first` regression for depth-to-water, after filtering implausible depths and wells with coordinate issues.
4. `target_fresh_water_present` classification as a weak-label salinity/freshness layer.

For Kazakhstan inference, create a prediction grid, extract the same transferable raster/vector predictors at each grid cell, and apply the Texas-trained model as a proof-of-concept analogue. Treat the output as a methodological demonstration, not a calibrated Kazakhstan groundwater map until local well validation is available.
