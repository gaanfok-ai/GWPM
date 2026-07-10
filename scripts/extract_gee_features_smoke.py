from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gpm.gee_auth import initialize_earth_engine, load_service_account_metadata
from gpm.gee_features import (
    ExtractionConfig,
    extract_features_for_sample,
    load_samples,
    validate_feature_table,
)
from gpm.paths import DEFAULT_GEE_KEY_PATH, FEATURE_DIR, PROCESSED_DIR, REPORT_DIR, display_path
from gpm.progress import ProgressPrinter


DEFAULT_LABELS_PATH = PROCESSED_DIR / "texas_sdr_yield20_classification_last5y.parquet"
DEFAULT_OUTPUT_PATH = FEATURE_DIR / "gee_yield20_smoke_10.parquet"
DEFAULT_SUMMARY_PATH = REPORT_DIR / "gee_feature_smoke_summary.json"
DEFAULT_SAMPLE_SIZE = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract GEE features for yield20 well samples.")
    parser.add_argument("--key-path", type=Path, default=DEFAULT_GEE_KEY_PATH)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of rows to extract. Use 0 or a negative value for all rows after start-index.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based row offset after sorting labels by anchor_date and well_id.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Read an existing output parquet and write validation summary without calling GEE.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Update one in-place progress line every N rows. Use 0 to disable progress output.",
    )
    return parser.parse_args()


def write_summary(features: pd.DataFrame, args: argparse.Namespace, extra: dict[str, object]) -> None:
    summary = validate_feature_table(features)
    summary["output_path"] = display_path(args.output_path)
    summary["labels_path"] = display_path(args.labels_path)
    summary.update(extra)
    args.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {args.summary_path}")
    print(f"Rows: {len(features)}")
    print(f"Feature columns: {len(summary['feature_columns'])}")


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        features = pd.read_parquet(args.output_path)
        write_summary(features, args, {"summary_only": True})
        return

    auth = load_service_account_metadata(args.key_path)
    try:
        ee = initialize_earth_engine(auth)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from None

    config = ExtractionConfig()
    samples = load_samples(args.labels_path, args.sample_size, args.start_index)
    if samples.empty:
        raise SystemExit("No samples selected. Check --sample-size and --start-index.")

    rows = []
    progress = ProgressPrinter(total=len(samples), update_every=args.progress_every)
    if args.progress_every > 0:
        progress.print(processed=0, current_id=None)
    for position, (_, sample) in enumerate(samples.iterrows(), start=1):
        rows.append(extract_features_for_sample(ee, sample, config))
        progress.maybe_update(position, str(sample["well_id"]))

    features = pd.DataFrame(rows)
    features.to_parquet(args.output_path, index=False)
    print(f"Wrote {args.output_path}")
    write_summary(
        features,
        args,
        {
            "service_account_project_id_present": bool(auth.project_id),
            "service_account_email_present": bool(auth.service_account_email),
        },
    )


if __name__ == "__main__":
    main()

