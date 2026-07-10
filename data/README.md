# Data Directory

This folder stores local generated data artifacts.

Expected local outputs:

```text
processed/texas_sdr_wells.parquet
processed/texas_sdr_yield20_classification_last5y.parquet
features/gee_yield20_batch_*.parquet
features/gee_yield20_features_merged_5000.parquet
```

Parquet/CSV/JSON data files are ignored by git because they can be large and are
reproducible from scripts plus the raw SDR export.

