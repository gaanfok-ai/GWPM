from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "processed" / "texas_sdr_wells.parquet"
REPORT_DIR = ROOT / "reports"
SUMMARY_DIR = REPORT_DIR / "modeling_readiness"

TARGETS = {
    "high_yield_20gpm": "target_high_yield_20gpm",
    "yield_gpm_max": "yield_gpm_max",
    "static_water_level_ft": "static_water_level_ft_first",
    "shallow_water_le_100ft": "target_shallow_water_le_100ft",
    "fresh_water_present": "target_fresh_water_present",
    "specific_capacity": "specific_capacity_gpm_per_ft_median",
}

IMPORTANT_FEATURES = [
    "County",
    "ProposedUse",
    "TypeOfWork",
    "Elevation",
    "PumpDepth",
    "water_bearing_interval_count",
    "lithology_count",
    "lith_sand_thickness_ft",
    "lith_gravel_thickness_ft",
    "lith_clay_thickness_ft",
    "borehole_max_bottom_ft",
    "filter_total_thickness_ft",
    "drilling_method_first",
]


def pct(value: float) -> float:
    return round(float(value) * 100, 2)


def normalize_bool(series: pd.Series) -> pd.Series:
    return series.astype("boolean")


def choose_anchor_date(df: pd.DataFrame) -> pd.Series:
    return df["DrillingEndDate"].combine_first(df["DrillingStartDate"]).combine_first(df["DateSubmitted"])


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["anchor_date"] = choose_anchor_date(df)
    df["anchor_year"] = df["anchor_date"].dt.year
    max_year = int(df["anchor_year"].dropna().max())
    df["time_window"] = "overall"
    df.loc[df["anchor_year"] >= max_year - 4, "time_window"] = "last_5_years"
    df.loc[df["anchor_year"] >= max_year - 9, "within_last_10_years"] = True
    df["within_last_10_years"] = df["within_last_10_years"].fillna(False)
    df["within_last_5_years"] = df["anchor_year"] >= max_year - 4
    df["within_last_10_years"] = df["anchor_year"] >= max_year - 9
    return df


def target_non_null(df: pd.DataFrame, target_col: str) -> pd.Series:
    return df[target_col].notna()


def base_clean_mask(df: pd.DataFrame, target_col: str) -> pd.Series:
    return (
        df["has_valid_coordinates"].fillna(False)
        & df["anchor_date"].notna()
        & target_non_null(df, target_col)
        & (df["KnownLocationError"].fillna("No").str.lower() != "yes")
    )


def target_clean_mask(df: pd.DataFrame, target_col: str) -> pd.Series:
    mask = base_clean_mask(df, target_col)
    if target_col in ["yield_gpm_max", "specific_capacity_gpm_per_ft_median"]:
        mask &= df[target_col].gt(0).fillna(False)
    if target_col == "static_water_level_ft_first":
        mask &= df[target_col].between(0, 2000).fillna(False)
    return mask


def strict_feature_mask(df: pd.DataFrame) -> pd.Series:
    return (
        df["ProposedUse"].notna()
        & df["Elevation"].notna()
        & df["water_bearing_interval_count"].notna()
        & df["lithology_count"].notna()
    )


def write_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(SUMMARY_DIR / name, index=False)


def summarize_clean_samples(df: pd.DataFrame, max_year: int) -> pd.DataFrame:
    windows = {
        "overall": pd.Series(True, index=df.index),
        "last_10_years": df["anchor_year"] >= max_year - 9,
        "last_5_years": df["anchor_year"] >= max_year - 4,
    }
    rows: list[dict[str, object]] = []
    for target_name, target_col in TARGETS.items():
        for window_name, window_mask in windows.items():
            target_mask = target_clean_mask(df, target_col)
            base = df[window_mask]
            target_df = df[window_mask & target_mask]
            strict_df = df[window_mask & target_mask & strict_feature_mask(df)]
            rows.append(
                {
                    "target": target_name,
                    "target_column": target_col,
                    "window": window_name,
                    "year_min": int(base["anchor_year"].min()) if base["anchor_year"].notna().any() else None,
                    "year_max": int(base["anchor_year"].max()) if base["anchor_year"].notna().any() else None,
                    "all_rows": len(base),
                    "target_non_null_rows": int((window_mask & df[target_col].notna()).sum()),
                    "clean_target_rows": len(target_df),
                    "strict_model_rows": len(strict_df),
                    "valid_coordinate_pct_in_window": pct(base["has_valid_coordinates"].fillna(False).mean()),
                    "strict_model_pct_of_window": pct(len(strict_df) / len(base)) if len(base) else 0,
                }
            )
    return pd.DataFrame(rows)


def summarize_binary_target(df: pd.DataFrame, target_col: str, clean_mask: pd.Series) -> pd.DataFrame:
    data = df[clean_mask].copy()
    data[target_col] = normalize_bool(data[target_col])
    counts = data[target_col].value_counts(dropna=False).rename_axis("label").reset_index(name="count")
    counts["pct"] = counts["count"] / counts["count"].sum() * 100
    counts["pct"] = counts["pct"].round(2)
    return counts


def summarize_binary_by_window(df: pd.DataFrame, target_col: str, clean_mask: pd.Series, max_year: int) -> pd.DataFrame:
    windows = {
        "overall": pd.Series(True, index=df.index),
        "last_10_years": df["anchor_year"] >= max_year - 9,
        "last_5_years": df["anchor_year"] >= max_year - 4,
    }
    rows: list[dict[str, object]] = []
    for window_name, window_mask in windows.items():
        data = df[clean_mask & window_mask].copy()
        data[target_col] = normalize_bool(data[target_col])
        counts = data[target_col].value_counts(dropna=False)
        total = int(counts.sum())
        for label, count in counts.items():
            rows.append(
                {
                    "window": window_name,
                    "label": str(label),
                    "count": int(count),
                    "pct": round(count / total * 100, 2) if total else 0,
                }
            )
    return pd.DataFrame(rows)


def summarize_numeric_target(df: pd.DataFrame, target_col: str, clean_mask: pd.Series) -> pd.DataFrame:
    values = pd.to_numeric(df.loc[clean_mask, target_col], errors="coerce").dropna()
    summary = values.describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    return summary.rename("value").reset_index().rename(columns={"index": "stat"})


def summarize_numeric_bins(
    df: pd.DataFrame,
    target_col: str,
    clean_mask: pd.Series,
    bins: list[float],
    labels: list[str],
) -> pd.DataFrame:
    values = pd.to_numeric(df.loc[clean_mask, target_col], errors="coerce").dropna()
    binned = pd.cut(values, bins=bins, labels=labels, include_lowest=True, right=False)
    out = binned.value_counts(sort=False).rename_axis("bin").reset_index(name="count")
    out["pct"] = (out["count"] / out["count"].sum() * 100).round(2)
    return out


def summarize_by_year(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target_name, target_col in TARGETS.items():
        clean = target_clean_mask(df, target_col)
        yearly = (
            df.assign(clean_target=clean)
            .groupby("anchor_year", dropna=True)
            .agg(
                all_rows=("well_id", "count"),
                target_non_null=(target_col, lambda s: int(s.notna().sum())),
                clean_target_rows=("clean_target", "sum"),
                valid_coordinates=("has_valid_coordinates", "sum"),
            )
            .reset_index()
        )
        yearly["target"] = target_name
        rows.append(yearly)
    return pd.concat(rows, ignore_index=True)


def summarize_locations(df: pd.DataFrame, clean_mask: pd.Series, target_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = df[clean_mask].copy()
    data["lon_bin_1deg"] = data["longitude"].astype(float).round(0)
    data["lat_bin_1deg"] = data["latitude"].astype(float).round(0)

    by_county = (
        data.groupby("County", dropna=False)
        .agg(
            rows=("well_id", "count"),
            target_median=(target_col, "median") if pd.api.types.is_numeric_dtype(data[target_col]) else (target_col, "count"),
            lat_min=("latitude", "min"),
            lat_max=("latitude", "max"),
            lon_min=("longitude", "min"),
            lon_max=("longitude", "max"),
        )
        .reset_index()
        .sort_values("rows", ascending=False)
    )

    by_grid = (
        data.groupby(["lat_bin_1deg", "lon_bin_1deg"], dropna=False)
        .agg(rows=("well_id", "count"))
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    return by_county, by_grid


def summarize_feature_completeness(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in IMPORTANT_FEATURES:
        rows.append(
            {
                "feature": feature,
                "non_null_rows": int(df[feature].notna().sum()),
                "coverage_pct": pct(df[feature].notna().mean()),
                "unique_values": int(df[feature].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows).sort_values(["coverage_pct", "feature"], ascending=[False, True])


def summarize_coordinate_quality(df: pd.DataFrame) -> pd.DataFrame:
    coord_present = df["latitude"].notna() & df["longitude"].notna()
    coord_pairs = df.loc[coord_present, ["latitude", "longitude"]].astype(float)
    rounded_3 = coord_pairs.round(3)
    rounded_4 = coord_pairs.round(4)
    duplicated_exact = coord_pairs.duplicated(keep=False)
    return pd.DataFrame(
        [
            {"metric": "total_rows", "value": len(df)},
            {"metric": "coordinate_present_rows", "value": int(coord_present.sum())},
            {"metric": "valid_texas_bbox_rows", "value": int(df["has_valid_coordinates"].fillna(False).sum())},
            {"metric": "unique_exact_coordinate_pairs", "value": int(coord_pairs.drop_duplicates().shape[0])},
            {"metric": "rows_with_duplicated_exact_coordinate_pair", "value": int(duplicated_exact.sum())},
            {"metric": "unique_coordinate_pairs_rounded_4_decimals", "value": int(rounded_4.drop_duplicates().shape[0])},
            {"metric": "unique_coordinate_pairs_rounded_3_decimals", "value": int(rounded_3.drop_duplicates().shape[0])},
            {"metric": "rounded_whole_degree_coordinate_rows", "value": int(((coord_pairs % 1).abs() < 1e-9).all(axis=1).sum())},
        ]
    )


def summarize_categories(df: pd.DataFrame, column: str, clean_mask: pd.Series, top_n: int = 30) -> pd.DataFrame:
    data = df[clean_mask]
    counts = data[column].fillna("<missing>").value_counts().head(top_n).reset_index()
    counts.columns = [column, "count"]
    counts["pct"] = (counts["count"] / len(data) * 100).round(2) if len(data) else 0
    return counts


def write_report(
    df: pd.DataFrame,
    clean_samples: pd.DataFrame,
    high_yield_dist: pd.DataFrame,
    yield_summary: pd.DataFrame,
    water_level_summary: pd.DataFrame,
    feature_completeness: pd.DataFrame,
    top_counties: pd.DataFrame,
    top_grid_cells: pd.DataFrame,
    coordinate_quality: pd.DataFrame,
    high_yield_by_window: pd.DataFrame,
    yield_bins: pd.DataFrame,
    water_level_bins: pd.DataFrame,
    proposed_use_counts: pd.DataFrame,
    drilling_method_counts: pd.DataFrame,
    max_year: int,
) -> None:
    last_5_start = max_year - 4
    last_10_start = max_year - 9
    report = REPORT_DIR / "sdr_modeling_readiness_report.md"
    report.write_text(
        "\n".join(
            [
                "# SDR Modeling Readiness Report",
                "",
                "This report is focused on modeling usefulness, not only table inventory. It answers how many clean samples exist for target modeling, how recent they are, where they are, and which fields are usable for XGBoost/GEE feature extraction.",
                "",
                "## Clean Sample Definition",
                "",
                "A `clean_target_rows` sample has valid Texas coordinates, a usable anchor date, no known location error, and a non-null target. Numeric targets additionally require plausible positive or bounded values. A `strict_model_rows` sample also has `ProposedUse`, `Elevation`, water-bearing strata, and lithology aggregates.",
                "",
                f"Latest anchor year in the SDR-derived dataset: `{max_year}`.",
                f"`last_5_years` means `{last_5_start}-{max_year}`.",
                f"`last_10_years` means `{last_10_start}-{max_year}`.",
                "",
                "## Clean Samples by Target and Time Window",
                "",
                clean_samples.to_markdown(index=False),
                "",
                "## Main Target Interpretation",
                "",
                "`target_high_yield_20gpm` is the recommended first target. It is `True` when `yield_gpm_max >= 20`, `False` when reported yield is below 20, and null when yield is missing. Use only non-null rows for this model.",
                "",
                "### High-Yield Label Distribution",
                "",
                high_yield_dist.to_markdown(index=False),
                "",
                "### High-Yield Label Distribution by Window",
                "",
                high_yield_by_window.to_markdown(index=False),
                "",
                "### Yield Distribution",
                "",
                yield_summary.to_markdown(index=False),
                "",
                "### Yield Bins",
                "",
                yield_bins.to_markdown(index=False),
                "",
                "### Static Water Level Distribution",
                "",
                water_level_summary.to_markdown(index=False),
                "",
                "### Static Water Level Bins",
                "",
                water_level_bins.to_markdown(index=False),
                "",
                "## Coordinate Quality",
                "",
                coordinate_quality.to_markdown(index=False),
                "",
                "## Feature Completeness",
                "",
                feature_completeness.to_markdown(index=False),
                "",
                "## Top Counties for High-Yield Model Samples",
                "",
                top_counties.head(25).to_markdown(index=False),
                "",
                "## Top 1-Degree Grid Cells for High-Yield Model Samples",
                "",
                top_grid_cells.head(25).to_markdown(index=False),
                "",
                "## Important Categorical Distributions for High-Yield Samples",
                "",
                "### Proposed Use",
                "",
                proposed_use_counts.to_markdown(index=False),
                "",
                "### Drilling Method",
                "",
                drilling_method_counts.to_markdown(index=False),
                "",
                "## Practical Conclusions",
                "",
                "- The first XGBoost dataset should use `target_high_yield_20gpm` because it has a large labeled set and gives a clear heatmap probability.",
                "- Keep `yield_gpm_max` as a second regression target using `log1p(yield_gpm_max)`.",
                "- Keep `static_water_level_ft_first` as a separate depth-to-water regression target.",
                "- For GEE extraction, prioritize wells with valid coordinates and anchor dates. Use the anchor date to extract previous-12-month Sentinel/Landsat/climate features.",
                "- Use spatial validation by county or grid block. Random validation will likely overestimate performance because wells are spatially clustered.",
                "- Do not treat missing target rows as negative samples. Missing yield, missing water level, or missing strata means unknown, not bad.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATASET_PATH)
    df = add_time_columns(df)
    max_year = int(df["anchor_year"].dropna().max())

    clean_samples = summarize_clean_samples(df, max_year)
    write_csv(clean_samples, "clean_samples_by_target_window.csv")

    by_year = summarize_by_year(df)
    write_csv(by_year, "target_counts_by_year.csv")

    feature_completeness = summarize_feature_completeness(df)
    write_csv(feature_completeness, "important_feature_completeness.csv")

    high_yield_clean = target_clean_mask(df, "target_high_yield_20gpm")
    high_yield_dist = summarize_binary_target(df, "target_high_yield_20gpm", high_yield_clean)
    write_csv(high_yield_dist, "high_yield_20gpm_distribution.csv")
    high_yield_by_window = summarize_binary_by_window(df, "target_high_yield_20gpm", high_yield_clean, max_year)
    write_csv(high_yield_by_window, "high_yield_20gpm_distribution_by_window.csv")

    shallow_clean = target_clean_mask(df, "target_shallow_water_le_100ft")
    write_csv(
        summarize_binary_target(df, "target_shallow_water_le_100ft", shallow_clean),
        "shallow_water_distribution.csv",
    )

    fresh_clean = target_clean_mask(df, "target_fresh_water_present")
    write_csv(
        summarize_binary_target(df, "target_fresh_water_present", fresh_clean),
        "fresh_water_distribution.csv",
    )

    yield_summary = summarize_numeric_target(df, "yield_gpm_max", target_clean_mask(df, "yield_gpm_max"))
    write_csv(yield_summary, "yield_gpm_max_distribution.csv")
    yield_bins = summarize_numeric_bins(
        df,
        "yield_gpm_max",
        target_clean_mask(df, "yield_gpm_max"),
        bins=[0, 5, 10, 20, 50, 100, 500, float("inf")],
        labels=["0-5", "5-10", "10-20", "20-50", "50-100", "100-500", "500+"],
    )
    write_csv(yield_bins, "yield_gpm_max_bins.csv")

    water_level_summary = summarize_numeric_target(
        df,
        "static_water_level_ft_first",
        target_clean_mask(df, "static_water_level_ft_first"),
    )
    write_csv(water_level_summary, "static_water_level_distribution.csv")
    water_level_bins = summarize_numeric_bins(
        df,
        "static_water_level_ft_first",
        target_clean_mask(df, "static_water_level_ft_first"),
        bins=[0, 25, 50, 100, 200, 500, 1000, float("inf")],
        labels=["0-25", "25-50", "50-100", "100-200", "200-500", "500-1000", "1000+"],
    )
    write_csv(water_level_bins, "static_water_level_bins.csv")

    specific_capacity_summary = summarize_numeric_target(
        df,
        "specific_capacity_gpm_per_ft_median",
        target_clean_mask(df, "specific_capacity_gpm_per_ft_median"),
    )
    write_csv(specific_capacity_summary, "specific_capacity_distribution.csv")

    top_counties, location_grid = summarize_locations(df, high_yield_clean, "yield_gpm_max")
    write_csv(top_counties, "high_yield_samples_by_county.csv")
    write_csv(location_grid, "high_yield_samples_by_1deg_grid.csv")

    coordinate_quality = summarize_coordinate_quality(df)
    write_csv(coordinate_quality, "coordinate_quality.csv")

    for column in ["ProposedUse", "TypeOfWork", "test_type_first", "drilling_method_first", "water_type_first"]:
        write_csv(summarize_categories(df, column, high_yield_clean), f"high_yield_samples_{column.lower()}_counts.csv")
    proposed_use_counts = summarize_categories(df, "ProposedUse", high_yield_clean)
    drilling_method_counts = summarize_categories(df, "drilling_method_first", high_yield_clean)

    write_report(
        df=df,
        clean_samples=clean_samples,
        high_yield_dist=high_yield_dist,
        yield_summary=yield_summary,
        water_level_summary=water_level_summary,
        feature_completeness=feature_completeness,
        top_counties=top_counties,
        top_grid_cells=location_grid,
        coordinate_quality=coordinate_quality,
        high_yield_by_window=high_yield_by_window,
        yield_bins=yield_bins,
        water_level_bins=water_level_bins,
        proposed_use_counts=proposed_use_counts,
        drilling_method_counts=drilling_method_counts,
        max_year=max_year,
    )

    print(f"Wrote modeling readiness report to {REPORT_DIR / 'sdr_modeling_readiness_report.md'}")
    print(f"Wrote modeling readiness CSV summaries to {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
