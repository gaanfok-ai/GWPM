# GEE Feature Extraction Pipeline

## Purpose

This pipeline extracts pre-drilling spatial predictors for the last-5-years Texas SDR 20 gpm classification dataset. The extracted table is intended for local XGBoost/RandomForest training.

Input labels:

```text
data/processed/texas_sdr_yield20_classification_last5y.parquet
```

Smoke-test output:

```text
data/features/gee_yield20_smoke_10.parquet
reports/gee_feature_smoke_summary.json
```

## Security

The script uses:

```text
gee-key.json
```

The key file is not printed and is ignored by `.gitignore`.

## Leakage Rules

For each well:

```text
feature_start_date = anchor_date - 365 days
feature_end_date = anchor_date
```

No imagery or climate data after `anchor_date` is used.

The script extracts only predictors that can also be generated for unknown Kazakhstan grid cells:

- Sentinel-1 SAR
- Sentinel-2 optical indices
- SRTM DEM terrain
- JRC/MERIT hydrology proxies
- TerraClimate climate and water-balance summaries

It does not use pump depth, casing, borehole depth, drawdown, lithology logs from the well, strata from the well, driller fields, owner fields, or comments.

## Current Smoke-Test Feature Groups

The smoke-test version extracts a compact set first:

- Sentinel-1: VV, VH, incidence angle, VV minus VH
- Sentinel-2: blue, green, red, NIR, SWIR1, SWIR2, NDVI, NDMI, MNDWI, NBR, BSI
- DEM: elevation, slope, aspect sine/cosine, TPI at 500 m
- Hydrology: surface-water occurrence, seasonality, upstream area
- Climate: precipitation, PET, AET, climatic deficit, min/max temperature, aridity

The default reducer calculates mean, median, and standard deviation inside a 500 m buffer.

## Run Smoke Test

```bash
gpm/bin/python scripts/extract_gee_features_smoke.py --sample-size 10
```

## Current Blocker

The first run reached Google Cloud but failed during Earth Engine initialization because the service account lacks permission to use the configured project.

Required IAM fix:

```text
Grant the service account roles/serviceusage.serviceUsageConsumer
```

The Earth Engine API must also be enabled and the project must be registered for Earth Engine.

After IAM propagation, rerun the smoke test.

## Scale-Up Plan

After the 10-sample smoke test succeeds:

1. Increase to 100 samples.
2. Validate non-null feature coverage.
3. Extract all 47,047 last-5-years training wells in batches by anchor year or quarter.
4. Save feature parquet partitions by year.
5. Train local XGBoost/RandomForest.
6. Build a small Kazakhstan grid.
7. Extract the exact same feature columns for the grid.
8. Run local model inference and write prediction parquet/GeoPackage.

