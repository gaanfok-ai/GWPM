from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_BUILTIN_PATH = Path("reports/yield20_xgboost_feature_importance_builtin.csv")
DEFAULT_PERMUTATION_PATH = Path("reports/yield20_xgboost_feature_importance_permutation.csv")
DEFAULT_OUTPUT_DIR = Path("reports/figures")


def make_bar_plot(
    data: pd.DataFrame,
    value_column: str,
    title: str,
    output_base: Path,
    top_n: int,
) -> None:
    plot_data = data.sort_values(value_column, ascending=False).head(top_n).copy()
    plot_data = plot_data.sort_values(value_column, ascending=True)

    height = max(5, top_n * 0.35)
    fig, ax = plt.subplots(figsize=(10, height))
    ax.barh(plot_data["feature"], plot_data[value_column], color="#2f6f8f")
    ax.set_title(title)
    ax.set_xlabel(value_column.replace("_", " "))
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_builtin_importance(path: Path, output_dir: Path, top_n: int) -> list[Path]:
    importance = pd.read_csv(path)
    written: list[Path] = []
    for importance_type in ["gain", "total_gain", "weight", "cover"]:
        subset = importance[importance["importance_type"] == importance_type]
        if subset.empty:
            continue
        output_base = output_dir / f"feature_importance_xgb_{importance_type}_top{top_n}"
        make_bar_plot(
            data=subset,
            value_column="importance",
            title=f"XGBoost Feature Importance ({importance_type})",
            output_base=output_base,
            top_n=top_n,
        )
        written.extend([output_base.with_suffix(".png"), output_base.with_suffix(".pdf")])
    return written


def plot_permutation_importance(path: Path, output_dir: Path, top_n: int) -> list[Path]:
    importance = pd.read_csv(path)
    output_base = output_dir / f"feature_importance_permutation_top{top_n}"
    make_bar_plot(
        data=importance,
        value_column="importance_mean",
        title="Holdout Permutation Feature Importance",
        output_base=output_base,
        top_n=top_n,
    )
    return [output_base.with_suffix(".png"), output_base.with_suffix(".pdf")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot XGBoost feature importance CSV files.")
    parser.add_argument("--builtin-path", type=Path, default=DEFAULT_BUILTIN_PATH)
    parser.add_argument("--permutation-path", type=Path, default=DEFAULT_PERMUTATION_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written: list[Path] = []
    if args.builtin_path.exists():
        written.extend(plot_builtin_importance(args.builtin_path, args.output_dir, args.top_n))
    else:
        print(f"Skipping missing built-in importance file: {args.builtin_path}")

    if args.permutation_path.exists():
        written.extend(plot_permutation_importance(args.permutation_path, args.output_dir, args.top_n))
    else:
        print(f"Skipping missing permutation importance file: {args.permutation_path}")

    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
