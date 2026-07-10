# 20 GPM Yield Classification Dataset

Output parquet: `data/processed/texas_sdr_yield20_classification_last5y.parquet`
Time window: `2022-2026` based on `anchor_date`.

## Target Definition

`target_yield_ge_20gpm_int` is the classification label:

- `1`: `yield_gpm_max >= 20`
- `0`: `yield_gpm_max < 20`

Rows with missing yield, non-positive yield, invalid coordinates, missing anchor date, known location errors, or anchor years outside the selected 5-year window are excluded.

## Why This File Is Lean

This file intentionally avoids post-drilling construction and test fields as default model features. Pump depth, casing depth, filter/screen intervals, borehole depth, drawdown, lithology logs, and water-bearing strata are measured or decided during/after drilling. They are useful for QA and hydrogeological interpretation, but they can leak information that would not be available when predicting groundwater potential in Kazakhstan.

Use this file mainly to upload points to GEE, extract pre-drilling spatial predictors, and train XGBoost/RandomForest on those extracted predictors.

Safe columns included:

- identifiers: `well_id`, `WellReportTrackingNumber`
- location: `latitude`, `longitude`, `County`
- timing: `DateSubmitted`, `DrillingStartDate`, `DrillingEndDate`, `anchor_date`, `anchor_year`
- metadata for stratification/filtering: `ProposedUse`, `TypeOfWork`
- target source fields: `yield_gpm_max`, `yield_gpm_median`, `well_test_count`
- labels: `target_yield_ge_20gpm`, `target_yield_ge_20gpm_int`

## Label Distribution

| label   |   count |   pct |
|:--------|--------:|------:|
| True    |   30371 | 64.55 |
| False   |   16676 | 35.45 |

## Time Window Summary

| window        |   year_min |   year_max |   rows |   high_yield_rows |   low_yield_rows |   high_yield_pct |
|:--------------|-----------:|-----------:|-------:|------------------:|-----------------:|-----------------:|
| overall       |       2022 |       2026 |  47047 |             30371 |            16676 |            64.55 |
| last_10_years |       2022 |       2026 |  47047 |             30371 |            16676 |            64.55 |
| last_5_years  |       2022 |       2026 |  47047 |             30371 |            16676 |            64.55 |

## Yearly Summary

|   anchor_year |   rows |   high_yield_rows |   median_yield_gpm |   high_yield_pct |
|--------------:|-------:|------------------:|-------------------:|-----------------:|
|          2022 |  12870 |              7889 |                 25 |            61.3  |
|          2023 |  11005 |              7040 |                 25 |            63.97 |
|          2024 |   9968 |              6532 |                 25 |            65.53 |
|          2025 |   9568 |              6448 |                 30 |            67.39 |
|          2026 |   3636 |              2462 |                 30 |            67.71 |

## Recommended ML Usage

For the first real XGBoost/RandomForest dataset, join this parquet with GEE-derived predictors using `well_id` or point geometry. Use only predictors that are available before drilling/inference:

- Sentinel-1 VV/VH seasonal and annual statistics
- Sentinel-2/Landsat vegetation, wetness, bare-soil, water, and thermal features
- DEM terrain derivatives
- climate and water-balance variables
- hydrology distances and flow accumulation
- soil and geology layers

Use `County`, `anchor_year`, or spatial grid blocks for validation splits, not as final Kazakhstan-transfer predictors.
