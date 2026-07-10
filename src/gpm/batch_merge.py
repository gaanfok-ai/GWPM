from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from gpm.paths import display_path


def read_batch(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "well_id" not in df.columns:
        raise ValueError(f"{path} has no well_id column")
    df = df.copy()
    df["source_file"] = path.name
    return df


def merge_feature_batches(
    input_dir: Path,
    pattern: str,
    drop_source_file: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {input_dir / pattern}")

    combined = pd.concat([read_batch(path) for path in files], ignore_index=True)
    duplicate_rows = combined.loc[combined.duplicated(subset=["well_id"], keep=False)]
    deduped = combined.drop_duplicates(subset=["well_id"], keep="first").reset_index(drop=True)

    summary: dict[str, Any] = {
        "input_dir": display_path(input_dir),
        "pattern": pattern,
        "input_files": [display_path(path) for path in files],
        "input_file_count": len(files),
        "raw_rows": int(len(combined)),
        "unique_well_id_rows": int(deduped["well_id"].nunique()),
        "merged_rows": int(len(deduped)),
        "duplicate_rows_removed": int(len(combined) - len(deduped)),
        "duplicated_well_id_count": int(duplicate_rows["well_id"].nunique()),
        "target_distribution": {
            str(key): int(value)
            for key, value in deduped["target_yield_ge_20gpm_int"].value_counts(dropna=False).sort_index().items()
        }
        if "target_yield_ge_20gpm_int" in deduped.columns
        else {},
        "min_anchor_date": str(deduped["anchor_date"].min()) if "anchor_date" in deduped.columns else None,
        "max_anchor_date": str(deduped["anchor_date"].max()) if "anchor_date" in deduped.columns else None,
    }

    if not duplicate_rows.empty:
        examples = duplicate_rows.sort_values(["well_id", "source_file"])
        summary["duplicate_examples_first_50"] = (
            examples.groupby("well_id")["source_file"].apply(list).head(50).to_dict()
        )

    if drop_source_file and "source_file" in deduped.columns:
        deduped = deduped.drop(columns=["source_file"])
        summary["source_file_dropped"] = True
    else:
        summary["source_file_dropped"] = False
    summary["columns"] = list(deduped.columns)
    return deduped, summary

