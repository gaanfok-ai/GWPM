from __future__ import annotations

import argparse
import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gpm.modeling import (
    XGBoostTrainingConfig,
    load_training_table,
    save_feature_importance,
    save_training_artifacts,
    train_tuned_xgboost_with_artifacts,
)
from gpm.paths import FEATURE_DIR, REPORT_DIR


DEFAULT_INPUT_PATH = FEATURE_DIR / "gee_yield20_features_merged_5000.parquet"
DEFAULT_MODEL_PATH = Path("models") / "yield20_xgboost.joblib"
DEFAULT_METRICS_PATH = REPORT_DIR / "yield20_xgboost_metrics.json"
DEFAULT_PREDICTIONS_PATH = FEATURE_DIR / "yield20_xgboost_holdout_predictions.parquet"
DEFAULT_BUILTIN_IMPORTANCE_PATH = REPORT_DIR / "yield20_xgboost_feature_importance_builtin.csv"
DEFAULT_PERMUTATION_IMPORTANCE_PATH = REPORT_DIR / "yield20_xgboost_feature_importance_permutation.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tuned XGBoost classifier for 20 gpm yield target.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--predictions-path", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--builtin-importance-path", type=Path, default=DEFAULT_BUILTIN_IMPORTANCE_PATH)
    parser.add_argument("--permutation-importance-path", type=Path, default=DEFAULT_PERMUTATION_IMPORTANCE_PATH)
    parser.add_argument("--permutation-repeats", type=int, default=5)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = XGBoostTrainingConfig(
        test_size=args.test_size,
        random_state=args.random_state,
        cv_folds=args.cv_folds,
        n_iter=args.n_iter,
        n_jobs=args.n_jobs,
    )

    df = load_training_table(args.input_path, limit_rows=args.limit_rows)
    model, metrics, predictions, builtin_importance, permutation_importance = train_tuned_xgboost_with_artifacts(
        df,
        config,
        permutation_repeats=args.permutation_repeats,
    )
    save_training_artifacts(
        model=model,
        metrics=metrics,
        predictions=predictions,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        predictions_path=args.predictions_path,
    )
    save_feature_importance(
        builtin_importance=builtin_importance,
        permutation_importance_df=permutation_importance,
        builtin_path=args.builtin_importance_path,
        permutation_path=args.permutation_importance_path,
    )

    print(f"Rows: {metrics['rows']}")
    print(f"Features: {metrics['feature_count']}")
    print(f"Best CV {config.scoring}: {metrics['best_cv_score']:.4f}")
    print(f"Holdout ROC-AUC: {metrics['holdout_roc_auc']:.4f}")
    print(f"Holdout PR-AUC: {metrics['holdout_pr_auc']:.4f}")
    print(f"Holdout F1: {metrics['holdout_f1']:.4f}")
    print(f"Wrote model: {args.model_path}")
    print(f"Wrote metrics: {args.metrics_path}")
    print(f"Wrote predictions: {args.predictions_path}")
    print(f"Wrote built-in feature importance: {args.builtin_importance_path}")
    print(f"Wrote permutation feature importance: {args.permutation_importance_path}")


if __name__ == "__main__":
    main()
