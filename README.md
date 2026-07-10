# GWPM

Groundwater Prospectivity Modeling proof of concept.

This project investigates how groundwater-potential heatmaps for Kazakhstan can
be built from remotely sensed and environmental predictors. Because open,
well-level groundwater labels for Kazakhstan are limited, the proof of concept
uses the Texas Submitted Drillers Reports (SDR) dataset as an open analogue:
labeled Texas wells are used for training, Google Earth Engine is used for
pre-drilling geospatial feature extraction, and tree-based models are used to
predict well-yield potential.

The first target is a binary well-productivity label:

```text
target_yield_ge_20gpm_int = 1 if reported yield >= 20 gallons/minute
target_yield_ge_20gpm_int = 0 if reported yield < 20 gallons/minute
```

## Repository Layout

```text
src/gpm/        Reusable Python package code
scripts/        Command-line entrypoints for data, features, and models
reports/        Markdown reports, metrics, figures, and presentation files
data/           Local generated parquet outputs; large files ignored by git
SDRDownload/    Local raw Texas SDR export; ignored by git
```

Important files:

```text
scripts/create_yield20_classification_dataset.py
scripts/extract_gee_features_smoke.py
scripts/merge_feature_batches.py
src/gpm/gee_features.py
src/gpm/gee_auth.py
src/gpm/batch_merge.py
scripts/train_xgboost_yield20.py
scripts/evaluate_spatial_generalization.py
```

## Environment

Install `uv`, then create and activate the Python 3.13 environment:

```bash
uv venv gpm --python 3.13
source gpm/bin/activate
uv pip install -r requirements.txt
```

The virtual environment folder is intentionally ignored by git.

Optional editable package install:

```bash
uv pip install -e .
```

## Secrets

Earth Engine authentication uses:

```text
gee-key.json
```

This file is ignored by git. Do not commit or print its contents. The extractor
also supports local `earthengine authenticate` credentials when no service
account key is present.

Expected service account setup:

```text
SERVICE_ACCOUNT = gee-extractor@sar-check-499610.iam.gserviceaccount.com
JSON_KEY_PATH = ./gee-key.json
```

## Local Data

Place the Texas SDR export in:

```text
SDRDownload/
```

The raw export and generated parquet datasets are intentionally excluded from
Git because they are large/local artifacts. Keep only documentation and code in
the repository.

## Pipeline

### 1. Explore SDR Tables

```bash
python scripts/explore_sdr.py
python scripts/modeling_readiness_eda.py
```

Outputs include general SDR profiles, target availability summaries, and
modeling-readiness reports under `reports/`.

### 2. Create Last-5-Year Yield Classification Labels

```bash
python scripts/create_yield20_classification_dataset.py
```

Output:

```text
data/processed/texas_sdr_yield20_classification_last5y.parquet
```

This file intentionally contains only identifiers, location, dates, yield fields
needed to define the target, and the binary label. It avoids post-drilling
construction fields to prevent leakage.

### 3. Extract GEE Features

Smoke test:

```bash
python scripts/extract_gee_features_smoke.py \
  --sample-size 10 \
  --output-path data/features/gee_yield20_smoke_10.parquet \
  --summary-path reports/gee_feature_smoke_summary.json
```

Batch extraction example:

```bash
python scripts/extract_gee_features_smoke.py \
  --start-index 0 \
  --sample-size 500 \
  --output-path data/features/gee_yield20_batch_000000_000500.parquet \
  --summary-path reports/gee_yield20_batch_000000_000500_summary.json
```

Progress is shown as one updating terminal line. Use `--progress-every 0` to
disable it.

Leakage rule:

```text
feature_start_date = anchor_date - 365 days
feature_end_date = anchor_date
```

No imagery or climate data after drilling is used.

### 4. Merge Feature Batches

```bash
python scripts/merge_feature_batches.py \
  --drop-source-file \
  --output-path data/features/gee_yield20_features_merged_5000.parquet \
  --summary-path reports/gee_yield20_features_merged_5000_summary.json
```

The merge is deterministic and deduplicates by `well_id`.

### 5. Train XGBoost

Smoke test:

```bash
python scripts/train_xgboost_yield20.py \
  --input-path data/features/gee_yield20_features_merged_5000.parquet \
  --cv-folds 2 \
  --n-iter 1 \
  --sample-size 300 \
  --output-prefix reports/yield20_xgboost_smoke \
  --predictions-path data/features/yield20_xgboost_smoke_predictions.parquet
```

Main experiment:

```bash
python scripts/train_xgboost_yield20.py \
  --input-path data/features/gee_yield20_features_merged_5000.parquet \
  --cv-folds 5 \
  --n-iter 20 \
  --n-jobs -1 \
  --output-prefix reports/yield20_xgboost_5000 \
  --predictions-path data/features/yield20_xgboost_5000_holdout_predictions.parquet
```

Spatial generalization check:

```bash
python scripts/evaluate_spatial_generalization.py \
  --input-path data/features/gee_yield20_features_merged_5000.parquet \
  --output-path reports/yield20_spatial_generalization_5000.json
```

Feature-importance plots:

```bash
python scripts/plot_feature_importance.py \
  --builtin-path reports/yield20_xgboost_5000_feature_importance_builtin.csv \
  --permutation-path reports/yield20_xgboost_5000_feature_importance_permutation.csv \
  --output-dir reports/figures/yield20_xgboost_5000
```

## Current Feature Sources

The current extractor uses:

```text
COPERNICUS/S1_GRD                 Sentinel-1 SAR
COPERNICUS/S2_SR_HARMONIZED       Sentinel-2 optical surface reflectance
USGS/SRTMGL1_003                  SRTM DEM
JRC/GSW1_4/GlobalSurfaceWater     surface-water occurrence/seasonality
MERIT/Hydro/v1_0_1                drainage/upstream area
IDAHO_EPSCOR/TERRACLIMATE         climate and water balance
```

Feature definitions and physical interpretation are documented in:

```text
reports/current_gee_feature_dictionary.md
```

Additional project reports:

```text
reports/feature_extraction_research_plan.md
reports/gee_groundwater_dataset_plan.md
reports/groundwater_poc_presentation_report.md
reports/sdr_modeling_readiness_report.md
reports/yield20_classification_dataset.md
```

## Modeling Notes

For the first model, train only on GEE-derived predictors and the target label.
Do not use these as model features:

```text
yield_gpm_max
yield_gpm_median
well_test_count
pump depth
casing/filter/borehole depths
drawdown
specific capacity
lithology logs from the drilled well
water-bearing strata from the drilled well
owner/driller/address/comment fields
```

Use spatial validation rather than only random splitting. County or spatial block
splits are more honest because nearby wells share geology, climate, land use,
and reporting practices.

## Current Baseline Result

On the first merged 5,000-example feature dataset, the tuned XGBoost baseline
uses 93 GEE-derived predictors. The random holdout result is approximately:

```text
ROC-AUC: 0.840
PR-AUC:  0.891
F1:      0.819
```

Spatial-block validation is lower, which is expected and more realistic for
transfer to Kazakhstan. Treat the Texas result as proof-of-concept evidence, not
as a deployable Kazakhstan model without local calibration and field validation.

## GitHub Setup

For a fresh repository:

```bash
git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/gaanfok-ai/GWPM.git
git push -u origin main
```

Before pushing, verify that local secrets and large artifacts are not staged:

```bash
git status --short
git check-ignore gee-key.json gpm data/features/gee_yield20_features_merged_5000.parquet
```
