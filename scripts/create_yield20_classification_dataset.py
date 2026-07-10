from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "processed" / "texas_sdr_wells.parquet"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "texas_sdr_yield20_classification_last5y.parquet"
REPORT_PATH = ROOT / "reports" / "yield20_classification_dataset.md"

YIELD_THRESHOLD_GPM = 20.0
YEARS_BACK = 5


def choose_anchor_date(df: pd.DataFrame) -> pd.Series:
    return df["DrillingEndDate"].combine_first(df["DrillingStartDate"]).combine_first(df["DateSubmitted"])


def make_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    data = df.copy()
    data["anchor_date"] = choose_anchor_date(data)
    data["anchor_year"] = data["anchor_date"].dt.year
    max_year = int(data["anchor_year"].dropna().max())
    min_year = max_year - YEARS_BACK + 1

    # Keep only rows that have enough information for supervised target modeling.
    mask = (
        data["well_id"].notna()
        & data["latitude"].notna()
        & data["longitude"].notna()
        & data["has_valid_coordinates"].fillna(False)
        & data["anchor_date"].notna()
        & data["yield_gpm_max"].notna()
        & data["yield_gpm_max"].gt(0)
        & data["anchor_year"].ge(min_year)
        & data["anchor_year"].le(max_year)
        & (data["KnownLocationError"].fillna("No").str.lower() != "yes")
    )
    data = data.loc[mask].copy()

    data["target_yield_ge_20gpm"] = data["yield_gpm_max"] >= YIELD_THRESHOLD_GPM
    data["target_yield_ge_20gpm_int"] = data["target_yield_ge_20gpm"].astype("int8")
    data["yield_threshold_gpm"] = YIELD_THRESHOLD_GPM

    # Keep leakage-safe columns for GEE extraction and model splitting.
    # Do not include construction details such as pump depth, casing, screen/filter,
    # borehole depth, test type, drawdown, or lithology logs in this target file.
    columns = [
        "well_id",
        "WellReportTrackingNumber",
        "latitude",
        "longitude",
        "County",
        "ProposedUse",
        "TypeOfWork",
        "DateSubmitted",
        "DrillingStartDate",
        "DrillingEndDate",
        "anchor_date",
        "anchor_year",
        "yield_gpm_max",
        "yield_gpm_median",
        "well_test_count",
        "yield_threshold_gpm",
        "target_yield_ge_20gpm",
        "target_yield_ge_20gpm_int",
    ]
    out = data[columns].sort_values(["anchor_date", "well_id"]).reset_index(drop=True)
    return out, min_year, max_year


def write_report(dataset: pd.DataFrame, min_year: int, max_year: int) -> None:
    label_counts = (
        dataset["target_yield_ge_20gpm"]
        .value_counts(dropna=False)
        .rename_axis("label")
        .reset_index(name="count")
    )
    label_counts["pct"] = (label_counts["count"] / len(dataset) * 100).round(2)

    year_summary = (
        dataset.groupby("anchor_year", dropna=True)
        .agg(
            rows=("well_id", "count"),
            high_yield_rows=("target_yield_ge_20gpm_int", "sum"),
            median_yield_gpm=("yield_gpm_max", "median"),
        )
        .reset_index()
    )
    year_summary["high_yield_pct"] = (
        year_summary["high_yield_rows"] / year_summary["rows"] * 100
    ).round(2)

    latest_year = int(dataset["anchor_year"].max())
    windows = {
        "overall": dataset,
        "last_10_years": dataset[dataset["anchor_year"] >= latest_year - 9],
        "last_5_years": dataset[dataset["anchor_year"] >= latest_year - 4],
    }
    window_rows = []
    for name, frame in windows.items():
        high = int(frame["target_yield_ge_20gpm_int"].sum())
        total = len(frame)
        window_rows.append(
            {
                "window": name,
                "year_min": int(frame["anchor_year"].min()),
                "year_max": int(frame["anchor_year"].max()),
                "rows": total,
                "high_yield_rows": high,
                "low_yield_rows": total - high,
                "high_yield_pct": round(high / total * 100, 2) if total else 0,
            }
        )
    window_summary = pd.DataFrame(window_rows)

    REPORT_PATH.write_text(
        "\n".join(
            [
                "# 20 GPM Yield Classification Dataset",
                "",
                f"Output parquet: `{OUTPUT_PATH.relative_to(ROOT)}`",
                f"Time window: `{min_year}-{max_year}` based on `anchor_date`.",
                "",
                "## Target Definition",
                "",
                "`target_yield_ge_20gpm_int` is the classification label:",
                "",
                "- `1`: `yield_gpm_max >= 20`",
                "- `0`: `yield_gpm_max < 20`",
                "",
                "Rows with missing yield, non-positive yield, invalid coordinates, missing anchor date, known location errors, or anchor years outside the selected 5-year window are excluded.",
                "",
                "## Why This File Is Lean",
                "",
                "This file intentionally avoids post-drilling construction and test fields as default model features. Pump depth, casing depth, filter/screen intervals, borehole depth, drawdown, lithology logs, and water-bearing strata are measured or decided during/after drilling. They are useful for QA and hydrogeological interpretation, but they can leak information that would not be available when predicting groundwater potential in Kazakhstan.",
                "",
                "Use this file mainly to upload points to GEE, extract pre-drilling spatial predictors, and train XGBoost/RandomForest on those extracted predictors.",
                "",
                "Safe columns included:",
                "",
                "- identifiers: `well_id`, `WellReportTrackingNumber`",
                "- location: `latitude`, `longitude`, `County`",
                "- timing: `DateSubmitted`, `DrillingStartDate`, `DrillingEndDate`, `anchor_date`, `anchor_year`",
                "- metadata for stratification/filtering: `ProposedUse`, `TypeOfWork`",
                "- target source fields: `yield_gpm_max`, `yield_gpm_median`, `well_test_count`",
                "- labels: `target_yield_ge_20gpm`, `target_yield_ge_20gpm_int`",
                "",
                "## Label Distribution",
                "",
                label_counts.to_markdown(index=False),
                "",
                "## Time Window Summary",
                "",
                window_summary.to_markdown(index=False),
                "",
                "## Yearly Summary",
                "",
                year_summary.tail(30).to_markdown(index=False),
                "",
                "## Recommended ML Usage",
                "",
                "For the first real XGBoost/RandomForest dataset, join this parquet with GEE-derived predictors using `well_id` or point geometry. Use only predictors that are available before drilling/inference:",
                "",
                "- Sentinel-1 VV/VH seasonal and annual statistics",
                "- Sentinel-2/Landsat vegetation, wetness, bare-soil, water, and thermal features",
                "- DEM terrain derivatives",
                "- climate and water-balance variables",
                "- hydrology distances and flow accumulation",
                "- soil and geology layers",
                "",
                "Use `County`, `anchor_year`, or spatial grid blocks for validation splits, not as final Kazakhstan-transfer predictors.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INPUT_PATH)
    dataset, min_year, max_year = make_dataset(df)
    dataset.to_parquet(OUTPUT_PATH, index=False)
    write_report(dataset, min_year, max_year)

    print(f"Wrote {OUTPUT_PATH}")
    print(f"Rows: {len(dataset)}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
