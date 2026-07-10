# Scripts

Command-line entrypoints for the project.

Recommended order:

```bash
python scripts/explore_sdr.py
python scripts/modeling_readiness_eda.py
python scripts/create_yield20_classification_dataset.py
python scripts/extract_gee_features_smoke.py --sample-size 10
python scripts/merge_feature_batches.py --drop-source-file
python scripts/train_xgboost_yield20.py --n-iter 20
python scripts/evaluate_spatial_generalization.py
python scripts/plot_feature_importance.py
python scripts/build_project_presentation.py
```

The reusable implementation lives in `src/gpm`; scripts should remain thin.
