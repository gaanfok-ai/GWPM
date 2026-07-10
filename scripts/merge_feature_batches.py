from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gpm.batch_merge import merge_feature_batches
from gpm.paths import FEATURE_DIR, REPORT_DIR, display_path


DEFAULT_OUTPUT_PATH = FEATURE_DIR / "gee_yield20_features_merged_5000.parquet"
DEFAULT_SUMMARY_PATH = REPORT_DIR / "gee_yield20_features_merged_5000_summary.json"
DEFAULT_PATTERN = "gee_yield20_batch_*.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge GEE feature batch parquet files.")
    parser.add_argument("--input-dir", type=Path, default=FEATURE_DIR)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument(
        "--drop-source-file",
        action="store_true",
        help="Drop source_file from the final parquet after deduplication.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)

    merged, summary = merge_feature_batches(
        input_dir=args.input_dir,
        pattern=args.pattern,
        drop_source_file=args.drop_source_file,
    )
    merged.to_parquet(args.output_path, index=False)
    summary["output_path"] = display_path(args.output_path)
    args.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {args.output_path}")
    print(f"Wrote {args.summary_path}")
    print(f"Input files: {summary['input_file_count']}")
    print(f"Raw rows: {summary['raw_rows']}")
    print(f"Merged rows: {summary['merged_rows']}")
    print(f"Duplicate rows removed: {summary['duplicate_rows_removed']}")


if __name__ == "__main__":
    main()

