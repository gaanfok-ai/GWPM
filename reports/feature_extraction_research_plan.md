# Groundwater Prospectivity Feature Extraction Research Plan

## Objective

Build a proof-of-concept groundwater prospectivity model using Texas SDR wells from the last 5 years as labels, then transfer the same feature pipeline to a Kazakhstan prediction grid.

The first target is:

```text
target_yield_ge_20gpm_int
1 = yield_gpm_max >= 20 gpm
0 = yield_gpm_max < 20 gpm
```

The first modeling file is:

```text
data/processed/texas_sdr_yield20_classification_last5y.parquet
```

The model should not use post-drilling construction information as predictors. The feature set should come from remotely sensed, climatic, hydrological, geological, soil, and terrain data that can also be extracted over Kazakhstan before drilling.

## General Strategy

State-of-practice groundwater ML papers commonly use large feature stacks, often tens to more than 100 predictors. That is reasonable here, but we should build them in controlled groups:

1. Static terrain features.
2. Hydrology and drainage features.
3. SAR features.
4. Optical spectral features.
5. Climate and water-balance features.
6. Soil and geology features.
7. Land-cover and human/sampling-bias diagnostics.

The first robust dataset should have approximately 80-120 features. XGBoost/RandomForest can handle this well, but we must avoid leakage and use spatial validation.

## Sampling Units

### Training Samples

Use one row per well point:

```text
well_id
latitude
longitude
anchor_date
target_yield_ge_20gpm_int
GEE/external features
```

The anchor date is selected from:

```text
DrillingEndDate -> DrillingStartDate -> DateSubmitted
```

### Prediction Samples

Use a regular grid over the Kazakhstan target area.

Recommended grid:

```text
500 m x 500 m
```

This grid is fine enough for Sentinel/DEM information but coarse enough to reduce coordinate uncertainty and local noise.

Optional comparison grids:

```text
250 m grid for local studies
1000 m grid for regional/national heatmaps
```

## Temporal Design

For Texas training wells:

```text
feature window = anchor_date - 365 days to anchor_date
```

For static data such as DEM, geology, soil, and hydrology, no temporal filtering is needed.

For Kazakhstan inference:

```text
recent composite = 2021-2025 median or seasonal median
climate normal = 1991-2020 or longest available normal
```

The important principle: do not use imagery or climate information after a well drilling date during Texas training if the model is intended to simulate pre-drilling prospectivity.

## Spatial Extraction Scales

For each point or grid cell, extract summaries at multiple scales:

```text
250 m radius
500 m radius
1000 m radius
3000 m radius for climate/hydrology/geology context
```

Recommended first version:

```text
500 m and 1000 m only
```

For raster values, extract:

```text
mean
median
standard deviation
10th percentile
90th percentile
```

Do not compute every statistic for every band at every radius in version 1. That creates too many correlated features. Start with median/mean/std for most rasters, then expand only if useful.

## Feature Group 1: DEM and Terrain

Primary sources:

- GEE: `USGS/SRTMGL1_003`
- Optional external: MERIT DEM, FABDEM, Copernicus DEM

Purpose:

Terrain is one of the most transferable feature groups. It captures valleys, recharge zones, slopes, discharge zones, plains, piedmonts, and basin morphology.

Features to extract:

```text
dem_elevation
dem_slope
dem_aspect_sin
dem_aspect_cos
dem_hillshade
dem_roughness_500m
dem_roughness_1000m
dem_local_relief_500m
dem_local_relief_1000m
dem_tpi_250m
dem_tpi_500m
dem_tpi_1000m
dem_tpi_3000m
dem_tri_500m
dem_tri_1000m
dem_curvature_general
dem_curvature_profile
dem_curvature_plan
dem_valley_bottom_flatness_proxy
dem_relative_elevation_1000m
```

Approximate count:

```text
15-20 features
```

Extraction method:

In GEE, compute terrain derivatives using `ee.Terrain` and neighborhood reducers. For TPI and local relief, subtract focal mean or local min/max from elevation using circular kernels.

## Feature Group 2: Hydrology and Drainage

Primary sources:

- GEE: `MERIT/Hydro/v1_0_1`
- GEE: `WWF/HydroSHEDS/15ACC`
- GEE: `WWF/HydroSHEDS/15DIR`
- GEE: `JRC/GSW1_4/GlobalSurfaceWater`

Purpose:

Groundwater productivity is often related to drainage networks, floodplains, alluvial valleys, recharge pathways, and surface-water persistence.

Features to extract:

```text
hydro_flow_accumulation
hydro_log_flow_accumulation
hydro_distance_to_stream_1
hydro_distance_to_stream_2
hydro_distance_to_major_stream
hydro_distance_to_permanent_water
hydro_distance_to_seasonal_water
hydro_surface_water_occurrence
hydro_surface_water_seasonality
hydro_surface_water_recurrence
hydro_height_above_nearest_drainage
hydro_drainage_density_1000m
hydro_drainage_density_3000m
hydro_floodplain_proxy
hydro_distance_to_valley_axis
```

Approximate count:

```text
12-16 features
```

Extraction method:

Threshold flow accumulation to define streams. Create distance rasters using fast distance transform. For surface water, use JRC occurrence/seasonality bands. HAND may need to be precomputed externally if GEE implementation becomes expensive.

## Feature Group 3: Sentinel-1 SAR

Primary source:

- GEE: `COPERNICUS/S1_GRD`

Purpose:

SAR can indicate surface roughness, moisture, vegetation structure, floodplain wetness, irrigation, and persistent damp surfaces. It is not a direct groundwater sensor, but it is useful in arid/semi-arid terrain.

Filters:

```text
instrumentMode = IW
polarizations include VV and VH
orbitProperties_pass = ASCENDING and DESCENDING handled separately or combined carefully
remove extreme edge/noise values
```

Temporal windows:

```text
previous_12_months
spring
summer
autumn
dry season
wet season
```

Features to extract:

```text
s1_vv_median_12m
s1_vh_median_12m
s1_vv_mean_12m
s1_vh_mean_12m
s1_vv_std_12m
s1_vh_std_12m
s1_vv_p10_12m
s1_vv_p90_12m
s1_vh_p10_12m
s1_vh_p90_12m
s1_vv_minus_vh_median_12m
s1_vv_div_vh_median_12m
s1_vv_spring_median
s1_vh_spring_median
s1_vv_summer_median
s1_vh_summer_median
s1_vv_autumn_median
s1_vh_autumn_median
s1_vv_seasonal_amplitude
s1_vh_seasonal_amplitude
s1_vv_texture_500m
s1_vh_texture_500m
```

Approximate count:

```text
18-24 features
```

Extraction method:

In GEE, filter Sentinel-1 by date window around each well. Build median and percentile composites. Extract values at 500 m and/or 1000 m buffers. For texture, compute local standard deviation or GLCM texture on annual median VV/VH if computationally feasible.

## Feature Group 4: Sentinel-2 Optical

Primary source:

- GEE: `COPERNICUS/S2_SR_HARMONIZED`

Purpose:

Optical data captures vegetation, surface water, soil brightness, salinity/wetness proxies, and land condition. In dry regions, persistent dry-season vegetation and wetness anomalies can be useful groundwater indicators.

Preprocessing:

```text
use surface reflectance
mask clouds
mask cloud shadows
mask snow where relevant
create seasonal composites
```

Features to extract:

Raw spectral:

```text
s2_blue_median
s2_green_median
s2_red_median
s2_nir_median
s2_swir1_median
s2_swir2_median
```

Indices:

```text
s2_ndvi_median
s2_ndvi_max
s2_ndvi_min
s2_ndvi_amplitude
s2_evi_median
s2_savi_median
s2_msavi_median
s2_ndwi_median
s2_mndwi_median
s2_ndmi_median
s2_nbr_median
s2_bsi_median
s2_brightness_median
s2_greeness_proxy
s2_wetness_proxy
```

Seasonal features:

```text
s2_ndvi_spring_median
s2_ndvi_summer_median
s2_ndvi_autumn_median
s2_ndmi_spring_median
s2_ndmi_summer_median
s2_mndwi_summer_median
s2_dryseason_ndvi_median
s2_dryseason_ndmi_median
```

Approximate count:

```text
25-30 features
```

Extraction method:

In GEE, create cloud-masked seasonal and annual median composites. Compute indices per image before compositing where possible. Extract median/mean over 500 m or 1000 m radius.

## Feature Group 5: Landsat Optical and Thermal

Primary sources:

- GEE: `LANDSAT/LC08/C02/T1_L2`
- GEE: `LANDSAT/LC09/C02/T1_L2`
- Optional history: Landsat 5/7 Collection 2 Level 2

Purpose:

Landsat provides long-term optical and thermal behavior. Thermal features can help identify cooler moist/vegetated zones, irrigation, evapotranspiration patterns, and terrain heat differences.

Features to extract:

```text
landsat_ndvi_median_5y
landsat_ndvi_persistence_5y
landsat_ndmi_median_5y
landsat_mndwi_median_5y
landsat_bsi_median_5y
landsat_lst_median_5y
landsat_lst_summer_median
landsat_lst_summer_min
landsat_lst_summer_max
landsat_lst_amplitude
landsat_dryseason_ndvi_median
landsat_dryseason_lst_median
```

Approximate count:

```text
10-14 features
```

Extraction method:

Use Level 2 surface reflectance and surface temperature bands. Build recent 5-year composites for Kazakhstan. For Texas training, use either previous 12 months or previous 5 years before drilling.

## Feature Group 6: Climate and Water Balance

Primary sources:

- GEE: `IDAHO_EPSCOR/TERRACLIMATE`
- GEE: `ECMWF/ERA5_LAND/MONTHLY_AGGR`
- GEE: `UCSB-CHG/CHIRPS/DAILY`

Purpose:

Climate controls recharge, aridity, snowmelt, evapotranspiration, soil moisture, and water availability. It is essential for transfer to Kazakhstan.

Features to extract:

```text
clim_pr_prev12m
clim_pr_spring_prev12m
clim_pr_summer_prev12m
clim_pr_winter_prev12m
clim_pr_5y_mean
clim_pr_longterm_normal
clim_pr_recent_anomaly
clim_pet_prev12m
clim_aet_prev12m
clim_water_deficit_prev12m
clim_aridity_index
clim_soil_moisture_prev12m
clim_runoff_prev12m
clim_snow_water_equivalent
clim_vpd_prev12m
clim_tmin_mean
clim_tmax_mean
clim_temperature_range
```

Approximate count:

```text
16-20 features
```

Extraction method:

Use monthly climate products. Aggregate sums for precipitation/runoff and means for temperature, PET, AET, VPD, soil moisture. For Kazakhstan, include long-term normals and recent anomalies.

## Feature Group 7: Soil

Primary sources:

- GEE OpenLandMap soil layers
- External SoilGrids if needed

Purpose:

Soil controls infiltration, runoff, recharge, surface moisture, vegetation, and water retention.

Features to extract:

```text
soil_texture_class
soil_sand_pct
soil_silt_pct
soil_clay_pct
soil_bulk_density
soil_organic_carbon
soil_water_capacity
soil_depth_to_bedrock
soil_permeability_proxy
soil_hydrologic_group
```

Approximate count:

```text
8-12 features
```

Extraction method:

Use categorical soil texture directly or one-hot encode major classes. For depth-specific rasters, use topsoil and subsoil summaries separately if available.

## Feature Group 8: Geology and Structure

Primary sources:

- OneGeology where accessible
- national geological maps
- GLiM global lithology, if downloaded
- open hydrogeological maps where available
- derived lineaments from DEM/SAR

Purpose:

Geology is one of the strongest controls on groundwater. Remote sensing can identify surface conditions, but geology controls storage, permeability, aquifer type, and transmissivity.

Features to extract:

```text
geo_lithology_class
geo_lithology_group
geo_unconsolidated_sediment_flag
geo_carbonate_flag
geo_sandstone_flag
geo_crystalline_bedrock_flag
geo_volcanic_flag
geo_permeability_class
geo_distance_to_fault
geo_distance_to_lithologic_boundary
geo_lineament_density_1000m
geo_lineament_density_3000m
geo_distance_to_alluvium
geo_basin_fill_flag
```

Approximate count:

```text
12-16 features
```

Extraction method:

If geology is vector-based, rasterize to the prediction grid. Compute distance transforms to faults, lithologic contacts, and alluvial units. For categorical lithology, use one-hot encoding or target-independent grouping into hydrogeological classes.

## Feature Group 9: Land Cover and Human Bias Diagnostics

Primary sources:

- GEE: ESA WorldCover
- GEE: Dynamic World
- Optional: OpenStreetMap roads/settlements externally

Purpose:

Land cover can be useful, but human features can introduce sampling bias. Wells are drilled where people live and work, not only where groundwater exists.

Features to extract:

```text
lc_class
lc_cropland_fraction_500m
lc_grassland_fraction_500m
lc_shrubland_fraction_500m
lc_tree_fraction_500m
lc_bare_fraction_500m
lc_urban_fraction_500m
lc_water_fraction_500m
lc_irrigation_proxy
distance_to_road_optional
distance_to_settlement_optional
```

Approximate count:

```text
8-12 features
```

Extraction method:

Use class fractions within buffers. Keep roads/settlements optional and separate. They are useful for bias analysis but risky as final transfer predictors.

## Initial Feature Count Target

Recommended first feature stack:

```text
DEM/Terrain:        18
Hydrology:          14
Sentinel-1 SAR:     20
Sentinel-2 Optical: 26
Landsat/Thermal:    12
Climate:            18
Soil:               10
Geology:            14
Land cover:          8
Total:             140
```

This is slightly above 100 features. That is acceptable for XGBoost, but we should expect strong feature correlation. After training, reduce to a smaller stable set based on:

```text
spatial cross-validation performance
SHAP importance
permutation importance
correlation pruning
domain interpretability
```

For a lighter first run:

```text
DEM/Terrain:        12
Hydrology:          10
Sentinel-1 SAR:     12
Sentinel-2 Optical: 18
Landsat/Thermal:     8
Climate:            12
Soil:                6
Geology:             8
Land cover:          4
Total:              90
```

## Extraction Workflow

### Step 1: Prepare Training Points

Input:

```text
data/processed/texas_sdr_yield20_classification_last5y.parquet
```

Convert to a geospatial point table:

```text
well_id
latitude
longitude
anchor_date
target_yield_ge_20gpm_int
```

Upload to GEE as an asset or stage through Google Cloud Storage.

### Step 2: Build Static Raster Stack

Create rasters for:

```text
DEM derivatives
hydrology distances
soil
geology
land cover
```

These do not depend on anchor date.

### Step 3: Build Dynamic Temporal Composites

For each well, use anchor-date-based extraction:

```text
start = anchor_date - 365 days
end = anchor_date
```

Extract:

```text
Sentinel-1 previous 12 months
Sentinel-2 previous 12 months
Landsat previous 12 months or previous 5 years
climate previous 12 months
```

For implementation efficiency, group wells by year or quarter and build composites per time group rather than per individual well.

### Step 4: Extract Features

For point training:

```text
sampleRegions for point/cell values
reduceRegions over 500 m and 1000 m buffers for neighborhood statistics
```

For Kazakhstan grid:

```text
create 500 m grid
extract same feature names per cell
export grid feature table
apply trained model
```

### Step 5: Export

Export feature tables as:

```text
Parquet preferred
CSV acceptable for GEE export
```

Expected training table:

```text
well_id
target_yield_ge_20gpm_int
anchor_date
latitude
longitude
feature_001...
feature_100...
```

Expected prediction table:

```text
grid_cell_id
latitude
longitude
feature_001...
feature_100...
```

## Leakage Rules

Do not use these as default predictors:

```text
yield_gpm_max
yield_gpm_median
well_test_count
drawdown_ft_median
specific_capacity_gpm_per_ft_median
PumpDepth
borehole depth
casing depth
filter/screen intervals
seal intervals
lithology logs from the drilled well
water-bearing intervals from the drilled well
driller/company/owner/address/license fields
comments
```

Use these only for:

```text
target construction
quality control
hydrogeological interpretation
secondary diagnostic models
```

Potentially safe but use carefully:

```text
County
ProposedUse
TypeOfWork
anchor_year
```

These are useful for filtering, grouping, and validation splits, but not ideal final transfer predictors for Kazakhstan.

## Validation Plan

Use spatial validation, not only random split.

Recommended:

```text
GroupKFold by county
spatial block cross-validation using 50-100 km grid blocks
hold out entire aquifer/geologic regions if possible
```

Metrics:

```text
ROC-AUC
PR-AUC
F1 at selected threshold
balanced accuracy
calibration curve
Brier score
```

Why:

Random split will overestimate performance because nearby wells share geology, climate, land use, and reporting practices.

## First Practical Version

For version 1, extract about 90 features:

```text
DEM/Terrain: 12
Hydrology: 10
Sentinel-1: 12
Sentinel-2: 18
Landsat/Thermal: 8
Climate: 12
Soil: 6
Geology: 8
Land cover: 4
```

Use:

```text
500 m buffer statistics
previous 12 months for dynamic imagery/climate
static DEM/geology/soil/hydrology
spatial cross-validation
```

After the first result, expand toward 120-140 features if the model is stable and the extraction pipeline is reliable.

