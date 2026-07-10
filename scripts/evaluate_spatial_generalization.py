from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gpm.modeling import (
    TARGET_COLUMN,
    classification_metrics,
    feature_columns,
    load_training_table,
    make_pipeline_from_params,
    split_xy,
)
from gpm.paths import FEATURE_DIR, REPORT_DIR


DEFAULT_INPUT_PATH = FEATURE_DIR / "gee_yield20_features_merged_5000.parquet"
DEFAULT_TUNED_METRICS_PATH = REPORT_DIR / "yield20_xgboost_metrics.json"
DEFAULT_OUTPUT_PATH = REPORT_DIR / "yield20_spatial_generalization.json"


def spatial_block_id(df: pd.DataFrame, block_size_degrees: float) -> pd.Series:
    """Build coarse geographic groups from lat/lon for spatial cross-validation."""
    lat_block = np.floor(df["latitude"].astype(float) / block_size_degrees).astype(int)
    lon_block = np.floor(df["longitude"].astype(float) / block_size_degrees).astype(int)
    return lat_block.astype(str) + "_" + lon_block.astype(str)


def load_best_params(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    metrics = json.loads(path.read_text(encoding="utf-8"))
    return metrics.get("best_params")


def summarize_groups(groups: pd.Series) -> dict[str, object]:
    counts = groups.value_counts()
    return {
        "group_count": int(counts.size),
        "min_group_size": int(counts.min()),
        "median_group_size": float(counts.median()),
        "max_group_size": int(counts.max()),
        "top_10_group_sizes": {str(k): int(v) for k, v in counts.head(10).items()},
    }


def evaluate_cv(df: pd.DataFrame, groups: pd.Series | None, folds: int, best_params: dict[str, object] | None, random_state: int) -> dict[str, object]:
    features = feature_columns(df)
    x, y = split_xy(df, features)
    estimator = make_pipeline_from_params(random_state=random_state, best_params=best_params)

    if groups is None:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
        y_proba = cross_val_predict(estimator, x, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        split_type = "stratified_random_cv"
    else:
        group_count = groups.nunique()
        if group_count < folds:
            raise ValueError(f"Need at least {folds} spatial groups, found {group_count}")
        cv = GroupKFold(n_splits=folds)
        y_proba = cross_val_predict(
            estimator,
            x,
            y,
            cv=cv,
            groups=groups,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        split_type = "spatial_group_cv"

    metrics = classification_metrics(y, y_proba)
    metrics["split_type"] = split_type
    metrics["folds"] = folds
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate random vs spatial generalization for yield20 XGBoost.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--tuned-metrics-path", type=Path, default=DEFAULT_TUNED_METRICS_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--block-size-degrees", type=float, default=1.0)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--ignore-tuned-params",
        action="store_true",
        help="Use default XGBoost params instead of best params from tuned metrics JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_training_table(args.input_path, limit_rows=args.limit_rows)
    groups = spatial_block_id(df, args.block_size_degrees)
    best_params = None if args.ignore_tuned_params else load_best_params(args.tuned_metrics_path)

    random_cv = evaluate_cv(
        df=df,
        groups=None,
        folds=args.folds,
        best_params=best_params,
        random_state=args.random_state,
    )
    spatial_cv = evaluate_cv(
        df=df,
        groups=groups,
        folds=args.folds,
        best_params=best_params,
        random_state=args.random_state,
    )

    report = {
        "rows": int(len(df)),
        "target_distribution": {
            str(key): int(value) for key, value in df[TARGET_COLUMN].value_counts().sort_index().items()
        },
        "feature_count": len(feature_columns(df)),
        "block_size_degrees": args.block_size_degrees,
        "approx_block_size_km": round(args.block_size_degrees * 111.0, 1),
        "spatial_group_summary": summarize_groups(groups),
        "used_tuned_params": best_params is not None,
        "random_cv": random_cv,
        "spatial_cv": spatial_cv,
        "generalization_gap": {
            "roc_auc_random_minus_spatial": random_cv["roc_auc"] - spatial_cv["roc_auc"],
            "pr_auc_random_minus_spatial": random_cv["pr_auc"] - spatial_cv["pr_auc"],
            "f1_random_minus_spatial": random_cv["f1"] - spatial_cv["f1"],
        },
        "interpretation": (
            "If spatial CV is much lower than random CV, the model is learning local geographic "
            "structure that does not transfer well to unseen regions. That is expected in groundwater "
            "problems and should guide feature engineering and validation."
        ),
    }
    args.output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Rows: {report['rows']}")
    print(f"Spatial groups: {report['spatial_group_summary']['group_count']}")
    print(f"Random CV ROC-AUC: {random_cv['roc_auc']:.4f}")
    print(f"Spatial CV ROC-AUC: {spatial_cv['roc_auc']:.4f}")
    print(f"Random CV PR-AUC: {random_cv['pr_auc']:.4f}")
    print(f"Spatial CV PR-AUC: {spatial_cv['pr_auc']:.4f}")
    print(f"Wrote {args.output_path}")


if __name__ == "__main__":
    main()

