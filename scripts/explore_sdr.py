from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "SDRDownload"
REPORT_DIR = ROOT / "reports"
SUMMARY_DIR = REPORT_DIR / "data_summaries"
PROCESSED_DIR = ROOT / "data" / "processed"


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="replace") as file:
        return file.readline().strip("\n").split("|")


def count_lines(path: Path) -> int:
    with path.open("rb") as file:
        return sum(1 for _ in file)


def load_text_table(
    path: Path,
    usecols: list[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="|",
        encoding="latin1",
        dtype="string",
        usecols=usecols,
        nrows=nrows,
        na_values=["", "NA", "N/A", "null", "NULL"],
        keep_default_na=True,
        low_memory=False,
        on_bad_lines="skip",
    )


def summarize_table(path: Path) -> dict[str, object]:
    columns = read_header(path)
    rows = max(count_lines(path) - 1, 0)
    sample = load_text_table(path, nrows=10000)
    non_null = sample.notna().sum().sort_values(ascending=True)
    return {
        "table": path.stem,
        "rows": rows,
        "columns": len(columns),
        "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
        "column_names": ", ".join(columns),
        "sparsest_columns_in_sample": "; ".join(
            f"{column}={int(value)}" for column, value in non_null.head(8).items()
        ),
    }


def numeric_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    converted = df[columns].apply(pd.to_numeric, errors="coerce")
    return converted.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T


def value_counts(df: pd.DataFrame, column: str, top_n: int = 25) -> pd.DataFrame:
    return (
        df[column]
        .fillna("<missing>")
        .value_counts(dropna=False)
        .head(top_n)
        .rename_axis(column)
        .reset_index(name="count")
    )


def to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def read_required(path: Path, columns: list[str]) -> pd.DataFrame:
    available = set(read_header(path))
    missing = sorted(set(columns) - available)
    if missing:
        raise ValueError(f"{path.name} is missing expected columns: {missing}")
    return load_text_table(path, usecols=columns)


def add_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        df[column] = to_number(df[column])
    return df


def interval_features(
    df: pd.DataFrame,
    id_col: str,
    top_col: str = "TopDepth",
    bottom_col: str = "BottomDepth",
    prefix: str = "interval",
) -> pd.DataFrame:
    data = df.copy()
    data = add_numeric_columns(data, [top_col, bottom_col])
    data[f"{prefix}_thickness_ft"] = data[bottom_col] - data[top_col]
    grouped = data.groupby(id_col, dropna=False)
    return grouped.agg(
        **{
            f"{prefix}_count": (bottom_col, "count"),
            f"{prefix}_max_bottom_ft": (bottom_col, "max"),
            f"{prefix}_total_thickness_ft": (f"{prefix}_thickness_ft", "sum"),
        }
    ).reset_index()


def first_numeric_by_well(
    df: pd.DataFrame,
    id_col: str,
    value_col: str,
    prefix: str,
    date_col: str | None = None,
) -> pd.DataFrame:
    data = df.copy()
    data[value_col] = to_number(data[value_col])
    data = data.dropna(subset=[id_col, value_col])
    if date_col is not None:
        data[date_col] = pd.to_datetime(data[date_col], errors="coerce", format="%Y-%m-%d")
        data = data.sort_values([id_col, date_col], na_position="last")
    grouped = data.groupby(id_col, dropna=False)
    out = grouped[value_col].agg(["first", "min", "max", "median", "count"]).reset_index()
    out = out.rename(
        columns={
            "first": f"{prefix}_first",
            "min": f"{prefix}_min",
            "max": f"{prefix}_max",
            "median": f"{prefix}_median",
            "count": f"{prefix}_count",
        }
    )
    return out


def lithology_features() -> pd.DataFrame:
    lithology_path = DATA_DIR / "WellLithology.txt"
    terms = {
        "sand": "sand",
        "gravel": "gravel",
        "clay": "clay",
        "limestone": "limestone",
        "shale": "shale",
        "caliche": "caliche",
        "silt": "silt",
        "rock": "rock",
        "water": "water",
    }
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        lithology_path,
        sep="|",
        encoding="latin1",
        dtype="string",
        usecols=["WellReportTrackingNumber", "TopDepth", "BottomDepth", "LithologyDescription"],
        chunksize=250000,
        na_values=["", "NA", "N/A"],
        keep_default_na=True,
        on_bad_lines="skip",
    ):
        chunk = add_numeric_columns(chunk, ["TopDepth", "BottomDepth"])
        chunk["thickness_ft"] = chunk["BottomDepth"] - chunk["TopDepth"]
        desc = chunk["LithologyDescription"].fillna("").str.lower()
        grouped_input = pd.DataFrame({"WellReportTrackingNumber": chunk["WellReportTrackingNumber"]})
        grouped_input["lithology_count"] = 1
        grouped_input["lithology_total_logged_thickness_ft"] = chunk["thickness_ft"].clip(lower=0)
        for name, pattern in terms.items():
            mask = desc.str.contains(pattern, regex=False, na=False)
            grouped_input[f"lith_{name}_interval_count"] = mask.astype("int16")
            grouped_input[f"lith_{name}_thickness_ft"] = chunk["thickness_ft"].where(mask, 0).clip(lower=0)
        chunks.append(grouped_input.groupby("WellReportTrackingNumber", dropna=False).sum().reset_index())
    return pd.concat(chunks).groupby("WellReportTrackingNumber", dropna=False).sum().reset_index()


def build_clean_dataset() -> pd.DataFrame:
    well_cols = [
        "WellReportTrackingNumber",
        "DateSubmitted",
        "County",
        "Elevation",
        "CoordDDLat",
        "CoordDDLong",
        "HorizontalDatumType",
        "TypeOfWork",
        "ProposedUse",
        "DrillingStartDate",
        "DrillingEndDate",
        "SealMethod",
        "SurfaceCompletion",
        "PumpType",
        "PumpDepth",
        "ChemicalAnalysis",
        "InjuriousWater",
        "KnownLocationError",
        "PluggedWithin48Hrs",
    ]
    dataset = read_required(DATA_DIR / "WellData.txt", well_cols)
    dataset = add_numeric_columns(dataset, ["Elevation", "CoordDDLat", "CoordDDLong", "PumpDepth"])
    for column in ["DateSubmitted", "DrillingStartDate", "DrillingEndDate"]:
        dataset[column] = pd.to_datetime(dataset[column], errors="coerce", format="%Y-%m-%d")
    dataset["drilling_duration_days"] = (
        dataset["DrillingEndDate"] - dataset["DrillingStartDate"]
    ).dt.days
    dataset["well_id"] = dataset["WellReportTrackingNumber"]
    dataset["latitude"] = dataset["CoordDDLat"]
    dataset["longitude"] = dataset["CoordDDLong"]

    levels = read_required(
        DATA_DIR / "WellLevels.txt",
        [
            "WellReportTrackingNumber",
            "Measurement",
            "MeasurementDate",
            "ArtesianFlow",
            "MeasurementMethod",
        ],
    )
    dataset = dataset.merge(
        first_numeric_by_well(
            levels,
            "WellReportTrackingNumber",
            "Measurement",
            "static_water_level_ft",
            "MeasurementDate",
        ),
        on="WellReportTrackingNumber",
        how="left",
    )

    test = read_required(
        DATA_DIR / "WellTest.txt",
        ["WellReportTrackingNumber", "TestType", "Yield", "Drawdown", "Hours"],
    )
    for column in ["Yield", "Drawdown", "Hours"]:
        test[column] = to_number(test[column])
    test["specific_capacity_gpm_per_ft"] = test["Yield"] / test["Drawdown"].where(test["Drawdown"] > 0)
    test_agg = test.groupby("WellReportTrackingNumber", dropna=False).agg(
        yield_gpm_max=("Yield", "max"),
        yield_gpm_median=("Yield", "median"),
        drawdown_ft_median=("Drawdown", "median"),
        test_hours_median=("Hours", "median"),
        specific_capacity_gpm_per_ft_median=("specific_capacity_gpm_per_ft", "median"),
        well_test_count=("Yield", "count"),
        test_type_first=("TestType", "first"),
    ).reset_index()
    dataset = dataset.merge(test_agg, on="WellReportTrackingNumber", how="left")

    strata = read_required(
        DATA_DIR / "WellStrata.txt",
        ["WellReportTrackingNumber", "TopDepth", "BottomDepth", "WaterType"],
    )
    strata = add_numeric_columns(strata, ["TopDepth", "BottomDepth"])
    strata["strata_thickness_ft"] = (strata["BottomDepth"] - strata["TopDepth"]).clip(lower=0)
    water_type = strata["WaterType"].fillna("").str.lower()
    strata["fresh_water_interval_count"] = water_type.str.contains("fresh", regex=False).astype("int16")
    strata["saline_or_bad_water_interval_count"] = water_type.str.contains(
        "salt|saline|brackish|sulfur|sulphur|bad", regex=True
    ).astype("int16")
    strata["fresh_water_thickness_ft"] = strata["strata_thickness_ft"].where(
        strata["fresh_water_interval_count"] == 1, 0
    )
    strata_agg = strata.groupby("WellReportTrackingNumber", dropna=False).agg(
        water_bearing_interval_count=("BottomDepth", "count"),
        water_bearing_max_bottom_ft=("BottomDepth", "max"),
        fresh_water_interval_count=("fresh_water_interval_count", "sum"),
        fresh_water_thickness_ft=("fresh_water_thickness_ft", "sum"),
        saline_or_bad_water_interval_count=("saline_or_bad_water_interval_count", "sum"),
        water_type_first=("WaterType", "first"),
    ).reset_index()
    dataset = dataset.merge(strata_agg, on="WellReportTrackingNumber", how="left")

    for file_name, prefix in [
        ("WellBoreHole.txt", "borehole"),
        ("WellCasing.txt", "casing"),
        ("WellFilter.txt", "filter"),
        ("WellSealRange.txt", "seal"),
    ]:
        path = DATA_DIR / file_name
        cols = ["WellReportTrackingNumber", "TopDepth", "BottomDepth"]
        dataset = dataset.merge(
            interval_features(read_required(path, cols), "WellReportTrackingNumber", prefix=prefix),
            on="WellReportTrackingNumber",
            how="left",
        )

    completion = read_required(
        DATA_DIR / "WellCompletion.txt",
        ["WellReportTrackingNumber", "BoreholeCompletion"],
    )
    completion_agg = completion.groupby("WellReportTrackingNumber", dropna=False).agg(
        borehole_completion_first=("BoreholeCompletion", "first"),
        borehole_completion_count=("BoreholeCompletion", "count"),
    ).reset_index()
    dataset = dataset.merge(completion_agg, on="WellReportTrackingNumber", how="left")

    drilling = read_required(
        DATA_DIR / "WellDrillingMethod.txt",
        ["WellReportTrackingNumber", "DrillingMethod"],
    )
    drilling_agg = drilling.groupby("WellReportTrackingNumber", dropna=False).agg(
        drilling_method_first=("DrillingMethod", "first"),
        drilling_method_count=("DrillingMethod", "count"),
    ).reset_index()
    dataset = dataset.merge(drilling_agg, on="WellReportTrackingNumber", how="left")

    dataset = dataset.merge(lithology_features(), on="WellReportTrackingNumber", how="left")

    dataset["has_valid_coordinates"] = dataset["latitude"].between(25, 37) & dataset["longitude"].between(-107, -93)
    dataset["target_has_water_level"] = dataset["static_water_level_ft_first"].notna()
    dataset["target_has_yield"] = dataset["yield_gpm_max"].notna()
    dataset["target_high_yield_20gpm"] = dataset["yield_gpm_max"] >= 20
    dataset["target_fresh_water_present"] = (dataset["fresh_water_interval_count"] > 0).where(
        dataset["water_bearing_interval_count"].notna()
    )
    dataset["target_shallow_water_le_100ft"] = dataset["static_water_level_ft_first"] <= 100

    return dataset


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    SUMMARY_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    text_paths = sorted(DATA_DIR.glob("*.txt"))
    table_summary = pd.DataFrame(summarize_table(path) for path in text_paths)
    table_summary.to_csv(SUMMARY_DIR / "table_summary.csv", index=False)

    well_cols = [
        "WellReportTrackingNumber",
        "DateSubmitted",
        "County",
        "Elevation",
        "CoordDDLat",
        "CoordDDLong",
        "HorizontalDatumType",
        "TypeOfWork",
        "ProposedUse",
        "DrillingStartDate",
        "DrillingEndDate",
        "PumpDepth",
        "ChemicalAnalysis",
        "InjuriousWater",
        "KnownLocationError",
        "PluggedWithin48Hrs",
    ]
    well = load_text_table(DATA_DIR / "WellData.txt", usecols=well_cols)
    well_numeric_cols = ["Elevation", "CoordDDLat", "CoordDDLong", "PumpDepth"]
    numeric_summary(well, well_numeric_cols).to_csv(SUMMARY_DIR / "well_numeric_summary.csv")
    for column in ["County", "ProposedUse", "TypeOfWork", "HorizontalDatumType"]:
        value_counts(well, column).to_csv(SUMMARY_DIR / f"well_{column.lower()}_counts.csv", index=False)

    levels = load_text_table(DATA_DIR / "WellLevels.txt")
    levels["Measurement_num"] = pd.to_numeric(levels["Measurement"], errors="coerce")
    levels["MeasurementDate_dt"] = pd.to_datetime(
        levels["MeasurementDate"], errors="coerce", format="%Y-%m-%d"
    )
    numeric_summary(levels, ["Measurement_num"]).to_csv(SUMMARY_DIR / "levels_summary.csv")

    test = load_text_table(DATA_DIR / "WellTest.txt")
    for column in ["Yield", "Drawdown", "Hours"]:
        test[f"{column}_num"] = pd.to_numeric(test[column], errors="coerce")
    numeric_summary(test, ["Yield_num", "Drawdown_num", "Hours_num"]).to_csv(
        SUMMARY_DIR / "welltest_summary.csv"
    )
    value_counts(test, "TestType").to_csv(SUMMARY_DIR / "welltest_type_counts.csv", index=False)

    strata = load_text_table(DATA_DIR / "WellStrata.txt")
    for column in ["TopDepth", "BottomDepth"]:
        strata[f"{column}_num"] = pd.to_numeric(strata[column], errors="coerce")
    numeric_summary(strata, ["TopDepth_num", "BottomDepth_num"]).to_csv(
        SUMMARY_DIR / "strata_depth_summary.csv"
    )
    value_counts(strata, "WaterType").to_csv(SUMMARY_DIR / "strata_water_type_counts.csv", index=False)

    # Lithology is large, so summarize in chunks.
    lithology_path = DATA_DIR / "WellLithology.txt"
    lith_rows = 0
    lith_top_terms: dict[str, int] = {}
    lith_depths: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        lithology_path,
        sep="|",
        encoding="latin1",
        dtype="string",
        chunksize=250000,
        na_values=["", "NA", "N/A"],
        keep_default_na=True,
        on_bad_lines="skip",
    ):
        lith_rows += len(chunk)
        desc = chunk["LithologyDescription"].fillna("").str.lower()
        for term in [
            "sand",
            "gravel",
            "clay",
            "limestone",
            "shale",
            "caliche",
            "water",
            "rock",
            "silt",
        ]:
            lith_top_terms[term] = lith_top_terms.get(term, 0) + int(desc.str.contains(term).sum())
        lith_depths.append(
            chunk[["TopDepth", "BottomDepth"]].apply(pd.to_numeric, errors="coerce")
        )
    pd.DataFrame(
        [{"term": term, "rows_containing_term": count} for term, count in lith_top_terms.items()]
    ).sort_values("rows_containing_term", ascending=False).to_csv(
        SUMMARY_DIR / "lithology_term_counts.csv", index=False
    )
    numeric_summary(pd.concat(lith_depths, ignore_index=True), ["TopDepth", "BottomDepth"]).to_csv(
        SUMMARY_DIR / "lithology_depth_summary.csv"
    )

    workbook = DATA_DIR / "ReadMe" / "SDRDownloadColumnDescriptions.xlsx"
    column_descriptions = pd.read_excel(workbook, sheet_name=None)
    workbook_summary = pd.DataFrame(
        {
            "sheet": sheet,
            "rows": len(frame),
            "columns": len(frame.columns),
            "column_names": ", ".join(str(column) for column in frame.columns),
        }
        for sheet, frame in column_descriptions.items()
    )
    workbook_summary.to_csv(SUMMARY_DIR / "column_description_workbook_summary.csv", index=False)

    profile_md = REPORT_DIR / "sdr_data_profile.md"
    profile_md.write_text(
        "\n".join(
            [
                "# Texas SDR Data Profile",
                "",
                "Generated by `scripts/explore_sdr.py`.",
                "",
                "## Table Inventory",
                "",
                table_summary[["table", "rows", "columns", "size_mb"]].to_markdown(index=False),
                "",
                "## Key Field Summaries",
                "",
                "### Well Coordinates and Pump Depth",
                "",
                pd.read_csv(SUMMARY_DIR / "well_numeric_summary.csv").to_markdown(index=False),
                "",
                "### Water Level Measurement",
                "",
                pd.read_csv(SUMMARY_DIR / "levels_summary.csv").to_markdown(index=False),
                "",
                "### Well Test Yield / Drawdown / Hours",
                "",
                pd.read_csv(SUMMARY_DIR / "welltest_summary.csv").to_markdown(index=False),
                "",
                "### Lithology Term Counts",
                "",
                pd.read_csv(SUMMARY_DIR / "lithology_term_counts.csv").to_markdown(index=False),
                "",
                "## Workbook Summary",
                "",
                workbook_summary.to_markdown(index=False),
                "",
            ]
        ),
        encoding="utf-8",
    )

    clean_dataset = build_clean_dataset()
    dataset_path = PROCESSED_DIR / "texas_sdr_wells.parquet"
    clean_dataset.to_parquet(dataset_path, index=False)

    target_columns = [
        "static_water_level_ft_first",
        "yield_gpm_max",
        "yield_gpm_median",
        "specific_capacity_gpm_per_ft_median",
        "fresh_water_interval_count",
        "fresh_water_thickness_ft",
        "target_high_yield_20gpm",
        "target_fresh_water_present",
        "target_shallow_water_le_100ft",
    ]
    target_availability = pd.DataFrame(
        {
            "field": target_columns,
            "non_null_rows": [int(clean_dataset[column].notna().sum()) for column in target_columns],
            "coverage_pct": [
                round(float(clean_dataset[column].notna().mean() * 100), 2) for column in target_columns
            ],
        }
    )
    target_availability.to_csv(SUMMARY_DIR / "target_availability.csv", index=False)

    useful_report = REPORT_DIR / "texas_sdr_modeling_report.md"
    useful_report.write_text(
        "\n".join(
            [
                "# Texas SDR Modeling Report for Groundwater Prospectivity",
                "",
                "## Purpose",
                "",
                "The Texas SDR dataset is useful as an open-data proof of concept for a Kazakhstan groundwater heatmap workflow. It provides point observations of drilled wells, coordinates, reported water levels, well tests, lithology intervals, water-bearing strata, construction details, and usage labels. In Kazakhstan, the same trained workflow should not rely on Texas-only administrative labels; it should learn relationships between well outcomes and transferable spatial predictors such as DEM derivatives, distance to drainage, geological units, lineaments/fault proxies, Sentinel/Landsat indices, climate, soil, and terrain position.",
                "",
                "Parsing note: SDR text files are pipe-delimited (`|`). A small number of rows contain unescaped delimiter characters inside free-text fields, so the EDA reader uses tolerant parsing and skips malformed rows for this proof-of-concept pass. For production ingestion, those rows should be repaired with a table-specific parser if their text fields are needed.",
                "",
                "## SDR Tables That Matter Most",
                "",
                "- `WellData.txt`: primary one-row-per-report table. Use for well id, coordinates, county, elevation, proposed use, drilling dates, pump depth, and QA flags.",
                "- `WellLevels.txt`: water-level measurements. Best target source for depth-to-water heatmaps.",
                "- `WellTest.txt`: yield, drawdown, hours, and derived specific capacity. Best target source for productivity heatmaps.",
                "- `WellStrata.txt`: reported water-bearing intervals and water type. Useful for aquifer presence and fresh-water labels.",
                "- `WellLithology.txt`: interval descriptions. Useful for extracting coarse geology proxies such as sand/gravel/clay/limestone thickness.",
                "- `WellBoreHole.txt`, `WellCasing.txt`, `WellFilter.txt`, `WellSealRange.txt`, `WellCompletion.txt`, `WellDrillingMethod.txt`: useful technical covariates and QA signals, but many are partly outcomes of drilling design rather than pre-drilling predictors.",
                "- `Plug*.txt`: useful mainly for excluding plugged/abandoned wells or studying failure/closure patterns; lower priority for first prospectivity model.",
                "",
                "## Recommended Target Variables",
                "",
                "| Priority | Target | Field in clean parquet | Modeling type | Heatmap meaning | Notes |",
                "|---|---|---|---|---|---|",
                "| 1 | Well yield | `yield_gpm_max` or `yield_gpm_median` | Regression | Expected production rate | Directly useful for groundwater productivity. Use log transform or quantile bins because yield is usually skewed. |",
                "| 2 | High-yield class | `target_high_yield_20gpm` | Classification | Probability of productive well | Good first XGBoost target. Threshold should later be tuned by use case: domestic, livestock, irrigation. |",
                "| 3 | Depth to water | `static_water_level_ft_first` | Regression | Shallower/deeper groundwater surface | Useful for drilling cost and accessibility. Needs care because measurement date and pumping conditions vary. |",
                "| 4 | Shallow water class | `target_shallow_water_le_100ft` | Classification | Probability water is reachable within 100 ft | Good for heatmap communication, but threshold must be geologically and economically justified. |",
                "| 5 | Fresh-water presence | `target_fresh_water_present` | Classification | Probability of fresh water in reported strata | Useful in arid/saline regions. Texas text labels are noisy, so treat this as weak supervision. |",
                "| 6 | Fresh-water thickness | `fresh_water_thickness_ft` | Regression | Total reported fresh interval thickness | Potentially powerful but derived from driller text and interval reporting conventions. |",
                "| 7 | Specific capacity | `specific_capacity_gpm_per_ft_median` | Regression | Productivity normalized by drawdown | Hydrogeologically strong target, but coverage is lower because drawdown is often missing/non-numeric. |",
                "",
                "## Recommended Clean Dataset Columns",
                "",
                "Keep one row per `WellReportTrackingNumber`. The current script writes this to `data/processed/texas_sdr_wells.parquet`.",
                "",
                "Core identifiers and location:",
                "",
                "- `well_id`, `WellReportTrackingNumber`, `latitude`, `longitude`, `CoordDDLat`, `CoordDDLong`, `HorizontalDatumType`, `County`, `has_valid_coordinates`.",
                "",
                "Temporal and administrative fields:",
                "",
                "- `DateSubmitted`, `DrillingStartDate`, `DrillingEndDate`, `drilling_duration_days`, `TypeOfWork`, `ProposedUse`.",
                "",
                "Candidate targets:",
                "",
                "- `static_water_level_ft_first`, `static_water_level_ft_min`, `static_water_level_ft_median`, `yield_gpm_max`, `yield_gpm_median`, `drawdown_ft_median`, `specific_capacity_gpm_per_ft_median`, `fresh_water_interval_count`, `fresh_water_thickness_ft`, `target_high_yield_20gpm`, `target_shallow_water_le_100ft`, `target_fresh_water_present`.",
                "",
                "Transferable geologic/well-derived features:",
                "",
                "- `Elevation`, `PumpDepth`, `water_bearing_interval_count`, `water_bearing_max_bottom_ft`, `saline_or_bad_water_interval_count`, `borehole_max_bottom_ft`, `casing_max_bottom_ft`, `filter_total_thickness_ft`, `seal_total_thickness_ft`, `lithology_total_logged_thickness_ft`, and all `lith_*_thickness_ft` / `lith_*_interval_count` columns.",
                "",
                "Construction/design fields to use cautiously:",
                "",
                "- `SealMethod`, `SurfaceCompletion`, `PumpType`, `borehole_completion_first`, `drilling_method_first`, casing/filter/seal interval aggregates. These may leak human design choices made after a promising drilling decision, so prefer them for diagnostics or auxiliary models, not final Kazakhstan inference unless equivalent pre-drilling information exists.",
                "",
                "Fields to avoid as model features for Kazakhstan transfer:",
                "",
                "- Owner/driller names, addresses, license numbers, comments, exact county labels, and post-drilling completion details that would not exist at inference time in Kazakhstan. These can create leakage or Texas-specific memorization.",
                "",
                "## Target Availability in Current Clean Dataset",
                "",
                target_availability.to_markdown(index=False),
                "",
                "## XGBoost Dataset Strategy",
                "",
                "For the first proof of concept, train separate models rather than one overloaded target:",
                "",
                "1. `target_high_yield_20gpm` classification for a productivity probability heatmap.",
                "2. `yield_gpm_max` regression for expected yield, evaluated with log-transformed yield.",
                "3. `static_water_level_ft_first` regression for depth-to-water, after filtering implausible depths and wells with coordinate issues.",
                "4. `target_fresh_water_present` classification as a weak-label salinity/freshness layer.",
                "",
                "For Kazakhstan inference, create a prediction grid, extract the same transferable raster/vector predictors at each grid cell, and apply the Texas-trained model as a proof-of-concept analogue. Treat the output as a methodological demonstration, not a calibrated Kazakhstan groundwater map until local well validation is available.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote {profile_md}")
    print(f"Wrote CSV summaries to {SUMMARY_DIR}")
    print(f"Wrote clean dataset to {dataset_path}")
    print(f"Wrote modeling report to {useful_report}")


if __name__ == "__main__":
    main()
