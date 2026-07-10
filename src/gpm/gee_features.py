from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


PRE_DRILL_LOOKBACK_DAYS = 365
BUFFER_RADIUS_METERS = 500
EXPORT_SCALE_METERS = 30
SENTINEL2_CLOUDY_PIXEL_PERCENTAGE_MAX = 60
SENTINEL1_EDGE_NOISE_DB_MIN = -35


@dataclass(frozen=True)
class ExtractionConfig:
    """Spatial and temporal settings for leakage-safe feature extraction."""

    lookback_days: int = PRE_DRILL_LOOKBACK_DAYS
    buffer_radius_m: int = BUFFER_RADIUS_METERS
    scale_m: int = EXPORT_SCALE_METERS
    s2_cloudy_pixel_pct_max: int = SENTINEL2_CLOUDY_PIXEL_PERCENTAGE_MAX
    s1_edge_noise_db_min: int = SENTINEL1_EDGE_NOISE_DB_MIN


def load_samples(labels_path: Path, sample_size: int, start_index: int) -> pd.DataFrame:
    labels = pd.read_parquet(labels_path)
    required = [
        "well_id",
        "latitude",
        "longitude",
        "anchor_date",
        "target_yield_ge_20gpm_int",
        "yield_gpm_max",
    ]
    missing = sorted(set(required) - set(labels.columns))
    if missing:
        raise ValueError(f"{labels_path} is missing required columns: {missing}")

    labels = labels.dropna(subset=required).copy()
    labels["anchor_date"] = pd.to_datetime(labels["anchor_date"])
    labels = labels.sort_values(["anchor_date", "well_id"]).reset_index(drop=True)
    if start_index < 0:
        raise ValueError("--start-index must be >= 0")
    labels = labels.iloc[start_index:]
    if sample_size > 0:
        labels = labels.head(sample_size)
    return labels.reset_index(drop=True)


def iso_date(value: pd.Timestamp | datetime | date) -> str:
    return pd.Timestamp(value).date().isoformat()


def date_window(anchor_date: pd.Timestamp, config: ExtractionConfig) -> tuple[str, str]:
    end_date = pd.Timestamp(anchor_date).date()
    start_date = end_date - timedelta(days=config.lookback_days)
    return start_date.isoformat(), end_date.isoformat()


def safe_number(value: Any) -> float | int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def point_buffer(ee: Any, latitude: float, longitude: float, config: ExtractionConfig) -> Any:
    point = ee.Geometry.Point([float(longitude), float(latitude)])
    return point.buffer(config.buffer_radius_m)


def reduce_image_at_buffer(
    ee: Any,
    image: Any,
    geometry: Any,
    config: ExtractionConfig,
) -> dict[str, Any]:
    reducer = (
        ee.Reducer.mean()
        .combine(ee.Reducer.median(), sharedInputs=True)
        .combine(ee.Reducer.stdDev(), sharedInputs=True)
    )
    values = image.reduceRegion(
        reducer=reducer,
        geometry=geometry,
        scale=config.scale_m,
        bestEffort=True,
        maxPixels=1_000_000,
        tileScale=4,
    )
    return values.getInfo()


def add_sentinel2_indices(ee: Any, image: Any) -> Any:
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("s2_ndvi")
    ndmi = image.normalizedDifference(["B8", "B11"]).rename("s2_ndmi")
    mndwi = image.normalizedDifference(["B3", "B11"]).rename("s2_mndwi")
    nbr = image.normalizedDifference(["B8", "B12"]).rename("s2_nbr")
    bsi = image.expression(
        "((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue))",
        {
            "swir1": image.select("B11"),
            "red": image.select("B4"),
            "nir": image.select("B8"),
            "blue": image.select("B2"),
        },
    ).rename("s2_bsi")
    return image.addBands([ndvi, ndmi, mndwi, nbr, bsi])


def mask_sentinel2_sr(ee: Any, image: Any) -> Any:
    scl = image.select("SCL")
    # Mask saturated/defective, cloud shadow, clouds, cirrus, and snow.
    good = (
        scl.neq(1)
        .And(scl.neq(3))
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )
    reflectance_bands = ["B2", "B3", "B4", "B8", "B11", "B12"]
    scaled = image.select(reflectance_bands).multiply(0.0001)
    return image.addBands(scaled, overwrite=True).updateMask(good)


def sentinel2_features(
    ee: Any,
    geometry: Any,
    start_date: str,
    end_date: str,
    config: ExtractionConfig,
) -> tuple[dict[str, Any], int]:
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", config.s2_cloudy_pixel_pct_max))
        .map(lambda image: add_sentinel2_indices(ee, mask_sentinel2_sr(ee, image)))
    )
    count = int(collection.size().getInfo())
    if count == 0:
        return {}, count

    bands = ["B2", "B3", "B4", "B8", "B11", "B12", "s2_ndvi", "s2_ndmi", "s2_mndwi", "s2_nbr", "s2_bsi"]
    composite = collection.select(bands).median()
    renamed_bands = [
        f"{band.lower()}_median_12m" if band.lower().startswith("s2_") else f"s2_{band.lower()}_median_12m"
        for band in bands
    ]
    return reduce_image_at_buffer(ee, composite.rename(renamed_bands), geometry, config), count


def sentinel1_features(
    ee: Any,
    geometry: Any,
    start_date: str,
    end_date: str,
    config: ExtractionConfig,
) -> tuple[dict[str, Any], int]:
    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH", "angle"])
        .map(lambda image: image.updateMask(image.select("VV").gt(config.s1_edge_noise_db_min)))
    )
    count = int(collection.size().getInfo())
    if count == 0:
        return {}, count

    composite = collection.median()
    vv = composite.select("VV").rename("s1_vv_median_12m")
    vh = composite.select("VH").rename("s1_vh_median_12m")
    angle = composite.select("angle").rename("s1_angle_median_12m")
    vv_minus_vh = vv.subtract(vh).rename("s1_vv_minus_vh_median_12m")
    return reduce_image_at_buffer(ee, vv.addBands([vh, angle, vv_minus_vh]), geometry, config), count


def dem_features(ee: Any, geometry: Any, config: ExtractionConfig) -> dict[str, Any]:
    dem = ee.Image("USGS/SRTMGL1_003").select("elevation").rename("dem_elevation")
    terrain = ee.Terrain.products(dem)
    slope = terrain.select("slope").rename("dem_slope")
    aspect = terrain.select("aspect")
    aspect_sin = aspect.multiply(3.141592653589793 / 180).sin().rename("dem_aspect_sin")
    aspect_cos = aspect.multiply(3.141592653589793 / 180).cos().rename("dem_aspect_cos")
    local_mean = dem.focal_mean(radius=config.buffer_radius_m, units="meters")
    tpi = dem.subtract(local_mean).rename("dem_tpi_500m")
    return reduce_image_at_buffer(ee, dem.addBands([slope, aspect_sin, aspect_cos, tpi]), geometry, config)


def hydrology_features(ee: Any, geometry: Any, config: ExtractionConfig) -> dict[str, Any]:
    water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select(
        ["occurrence", "seasonality"],
        ["hydro_surface_water_occurrence", "hydro_surface_water_seasonality"],
    )
    merit = ee.Image("MERIT/Hydro/v1_0_1").select("upa").rename("hydro_upstream_area")
    return reduce_image_at_buffer(ee, water.addBands(merit), geometry, config)


def climate_features(
    ee: Any,
    geometry: Any,
    start_date: str,
    end_date: str,
    config: ExtractionConfig,
) -> tuple[dict[str, Any], int]:
    collection = (
        ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE")
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
    )
    count = int(collection.size().getInfo())
    if count == 0:
        return {}, count

    pr_sum = collection.select("pr").sum().rename("clim_pr_sum_12m")
    pet_sum = collection.select("pet").sum().multiply(0.1).rename("clim_pet_sum_12m")
    aet_sum = collection.select("aet").sum().multiply(0.1).rename("clim_aet_sum_12m")
    def_sum = collection.select("def").sum().multiply(0.1).rename("clim_def_sum_12m")
    tmmn_mean = collection.select("tmmn").mean().multiply(0.1).rename("clim_tmin_mean_12m")
    tmmx_mean = collection.select("tmmx").mean().multiply(0.1).rename("clim_tmax_mean_12m")
    aridity = pr_sum.divide(pet_sum.max(0.001)).rename("clim_aridity_pr_pet_12m")
    image = pr_sum.addBands([pet_sum, aet_sum, def_sum, tmmn_mean, tmmx_mean, aridity])
    return reduce_image_at_buffer(ee, image, geometry, config), count


def extract_features_for_sample(ee: Any, sample: pd.Series, config: ExtractionConfig) -> dict[str, Any]:
    start_date, end_date = date_window(sample["anchor_date"], config)
    geometry = point_buffer(ee, sample["latitude"], sample["longitude"], config)

    row: dict[str, Any] = {
        "well_id": sample["well_id"],
        "latitude": float(sample["latitude"]),
        "longitude": float(sample["longitude"]),
        "anchor_date": iso_date(sample["anchor_date"]),
        "feature_start_date": start_date,
        "feature_end_date": end_date,
        "buffer_radius_m": config.buffer_radius_m,
        "target_yield_ge_20gpm_int": int(sample["target_yield_ge_20gpm_int"]),
        "yield_gpm_max": float(sample["yield_gpm_max"]),
    }

    s2, s2_count = sentinel2_features(ee, geometry, start_date, end_date, config)
    s1, s1_count = sentinel1_features(ee, geometry, start_date, end_date, config)
    clim, clim_count = climate_features(ee, geometry, start_date, end_date, config)
    row["qa_sentinel2_image_count"] = s2_count
    row["qa_sentinel1_image_count"] = s1_count
    row["qa_terraclimate_month_count"] = clim_count

    groups = [dem_features(ee, geometry, config), hydrology_features(ee, geometry, config), s1, s2, clim]
    for feature_group in groups:
        row.update({key: safe_number(value) for key, value in feature_group.items()})
    return row


def validate_feature_table(features: pd.DataFrame) -> dict[str, Any]:
    non_feature = {
        "well_id",
        "latitude",
        "longitude",
        "anchor_date",
        "feature_start_date",
        "feature_end_date",
        "buffer_radius_m",
        "target_yield_ge_20gpm_int",
        "yield_gpm_max",
    }
    feature_columns = [column for column in features.columns if column not in non_feature]
    return {
        "rows": int(len(features)),
        "columns": int(len(features.columns)),
        "feature_columns": feature_columns,
        "feature_non_null_counts": {column: int(features[column].notna().sum()) for column in feature_columns},
        "target_distribution": {
            str(key): int(value)
            for key, value in features["target_yield_ge_20gpm_int"].value_counts().sort_index().items()
        },
        "min_anchor_date": str(features["anchor_date"].min()),
        "max_anchor_date": str(features["anchor_date"].max()),
        "all_feature_windows_end_on_or_before_anchor": bool(
            (pd.to_datetime(features["feature_end_date"]) <= pd.to_datetime(features["anchor_date"])).all()
        ),
    }

