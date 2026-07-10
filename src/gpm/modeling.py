from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier


TARGET_COLUMN = "target_yield_ge_20gpm_int"

# These columns are identifiers, geometry/date metadata, target construction fields,
# QA counters, or direct target values. They should not be used as model predictors.
NON_FEATURE_COLUMNS = {
    "well_id",
    "latitude",
    "longitude",
    "anchor_date",
    "feature_start_date",
    "feature_end_date",
    "buffer_radius_m",
    TARGET_COLUMN,
    "yield_gpm_max",
    "source_file",
}


@dataclass(frozen=True)
class XGBoostTrainingConfig:
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 3
    n_iter: int = 20
    n_jobs: int = -1
    scoring: str = "average_precision"


def load_training_table(path: Path, limit_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    if limit_rows is not None and limit_rows > 0:
        df = df.head(limit_rows).copy()
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in df.columns:
        if column in NON_FEATURE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            columns.append(column)
    if not columns:
        raise ValueError("No numeric feature columns were found.")
    return columns


def split_xy(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    x = df[features].replace([np.inf, -np.inf], np.nan)
    y = df[TARGET_COLUMN].astype(int)
    return x, y


def make_estimator(random_state: int) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_state,
        n_estimators=300,
    )


def parameter_distributions() -> dict[str, list[Any]]:
    return {
        "model__n_estimators": [150, 250, 400, 600],
        "model__max_depth": [2, 3, 4, 5, 6],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.08, 0.12],
        "model__subsample": [0.7, 0.85, 1.0],
        "model__colsample_bytree": [0.7, 0.85, 1.0],
        "model__min_child_weight": [1, 3, 5, 10],
        "model__gamma": [0, 0.1, 0.5, 1.0],
        "model__reg_alpha": [0, 0.01, 0.1, 1.0],
        "model__reg_lambda": [0.5, 1.0, 2.0, 5.0],
    }


def make_pipeline(random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", make_estimator(random_state=random_state)),
        ]
    )


def make_pipeline_from_params(random_state: int, best_params: dict[str, Any] | None = None) -> Pipeline:
    """Create the training pipeline and optionally apply RandomizedSearchCV-style params."""
    pipeline = make_pipeline(random_state)
    if best_params:
        pipeline.set_params(**best_params)
    return pipeline


def classification_metrics(y_true: pd.Series | np.ndarray, y_proba: np.ndarray) -> dict[str, Any]:
    y_pred = (y_proba >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def train_tuned_xgboost(
    df: pd.DataFrame,
    config: XGBoostTrainingConfig,
) -> tuple[RandomizedSearchCV, dict[str, Any], pd.DataFrame]:
    features = feature_columns(df)
    x, y = split_xy(df, features)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config.test_size,
        stratify=y,
        random_state=config.random_state,
    )

    search = RandomizedSearchCV(
        estimator=make_pipeline(config.random_state),
        param_distributions=parameter_distributions(),
        n_iter=config.n_iter,
        scoring=config.scoring,
        cv=StratifiedKFold(
            n_splits=config.cv_folds,
            shuffle=True,
            random_state=config.random_state,
        ),
        n_jobs=config.n_jobs,
        verbose=1,
        random_state=config.random_state,
        refit=True,
    )
    search.fit(x_train, y_train)

    y_proba = search.predict_proba(x_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics: dict[str, Any] = {
        "config": asdict(config),
        "rows": int(len(df)),
        "feature_count": len(features),
        "features": features,
        "target_distribution": {
            str(key): int(value) for key, value in y.value_counts().sort_index().items()
        },
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "best_cv_score": float(search.best_score_),
        "best_params": search.best_params_,
        "holdout_roc_auc": float(roc_auc_score(y_test, y_proba)),
        "holdout_pr_auc": float(average_precision_score(y_test, y_proba)),
        "holdout_accuracy": float(accuracy_score(y_test, y_pred)),
        "holdout_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "holdout_f1": float(f1_score(y_test, y_pred)),
        "holdout_precision": float(precision_score(y_test, y_pred)),
        "holdout_recall": float(recall_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    predictions = pd.DataFrame(
        {
            "well_id": df.loc[x_test.index, "well_id"].to_numpy()
            if "well_id" in df.columns
            else x_test.index.to_numpy(),
            "y_true": y_test.to_numpy(),
            "y_pred": y_pred,
            "y_proba": y_proba,
        }
    )
    return search, metrics, predictions


def xgboost_feature_importance(model: RandomizedSearchCV, features: list[str]) -> pd.DataFrame:
    """Return XGBoost built-in importances for the refit best estimator."""
    booster = model.best_estimator_.named_steps["model"].get_booster()
    rows = []
    for importance_type in ["gain", "weight", "cover", "total_gain", "total_cover"]:
        scores = booster.get_score(importance_type=importance_type)
        for feature_key, value in scores.items():
            # XGBoost receives numpy arrays from the imputer, so features are f0, f1, ...
            if feature_key.startswith("f") and feature_key[1:].isdigit():
                index = int(feature_key[1:])
                feature = features[index] if index < len(features) else feature_key
            else:
                feature = feature_key
            rows.append(
                {
                    "feature": feature,
                    "importance_type": importance_type,
                    "importance": float(value),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["feature", "importance_type", "importance", "importance_normalized"])
    importance = pd.DataFrame(rows)
    totals = importance.groupby("importance_type")["importance"].transform("sum")
    importance["importance_normalized"] = importance["importance"] / totals.replace(0, np.nan)
    return importance.sort_values(["importance_type", "importance"], ascending=[True, False])


def holdout_permutation_importance(
    model: RandomizedSearchCV,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int,
    n_repeats: int = 5,
    scoring: str = "average_precision",
) -> pd.DataFrame:
    """Measure holdout importance by shuffling each feature and scoring degradation."""
    result = permutation_importance(
        model.best_estimator_,
        x_test,
        y_test,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )
    return (
        pd.DataFrame(
            {
                "feature": x_test.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
                "scoring": scoring,
                "n_repeats": n_repeats,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def train_tuned_xgboost_with_artifacts(
    df: pd.DataFrame,
    config: XGBoostTrainingConfig,
    permutation_repeats: int = 5,
) -> tuple[RandomizedSearchCV, dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train, evaluate, and return model artifacts including feature importance."""
    features = feature_columns(df)
    x, y = split_xy(df, features)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config.test_size,
        stratify=y,
        random_state=config.random_state,
    )

    search = RandomizedSearchCV(
        estimator=make_pipeline(config.random_state),
        param_distributions=parameter_distributions(),
        n_iter=config.n_iter,
        scoring=config.scoring,
        cv=StratifiedKFold(
            n_splits=config.cv_folds,
            shuffle=True,
            random_state=config.random_state,
        ),
        n_jobs=config.n_jobs,
        verbose=1,
        random_state=config.random_state,
        refit=True,
    )
    search.fit(x_train, y_train)

    y_proba = search.predict_proba(x_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    metrics: dict[str, Any] = {
        "config": asdict(config),
        "rows": int(len(df)),
        "feature_count": len(features),
        "features": features,
        "target_distribution": {
            str(key): int(value) for key, value in y.value_counts().sort_index().items()
        },
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "best_cv_score": float(search.best_score_),
        "best_params": search.best_params_,
        "holdout_roc_auc": float(roc_auc_score(y_test, y_proba)),
        "holdout_pr_auc": float(average_precision_score(y_test, y_proba)),
        "holdout_accuracy": float(accuracy_score(y_test, y_pred)),
        "holdout_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "holdout_f1": float(f1_score(y_test, y_pred)),
        "holdout_precision": float(precision_score(y_test, y_pred)),
        "holdout_recall": float(recall_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }
    predictions = pd.DataFrame(
        {
            "well_id": df.loc[x_test.index, "well_id"].to_numpy()
            if "well_id" in df.columns
            else x_test.index.to_numpy(),
            "y_true": y_test.to_numpy(),
            "y_pred": y_pred,
            "y_proba": y_proba,
        }
    )
    builtin_importance = xgboost_feature_importance(search, features)
    permutation = holdout_permutation_importance(
        search,
        x_test,
        y_test,
        random_state=config.random_state,
        n_repeats=permutation_repeats,
        scoring=config.scoring,
    )
    return search, metrics, predictions, builtin_importance, permutation


def save_training_artifacts(
    model: RandomizedSearchCV,
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    model_path: Path,
    metrics_path: Path,
    predictions_path: Path,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    predictions.to_parquet(predictions_path, index=False)


def save_feature_importance(
    builtin_importance: pd.DataFrame,
    permutation_importance_df: pd.DataFrame,
    builtin_path: Path,
    permutation_path: Path,
) -> None:
    builtin_path.parent.mkdir(parents=True, exist_ok=True)
    permutation_path.parent.mkdir(parents=True, exist_ok=True)
    builtin_importance.to_csv(builtin_path, index=False)
    permutation_importance_df.to_csv(permutation_path, index=False)
