# Google Earth Engine Groundwater Dataset Plan

## Short Answer

Google Earth Engine (GEE) is a good strategy for this project. It is especially strong for the proof-of-concept stage because the same feature extraction logic can be applied to Texas well points and later to a Kazakhstan prediction grid. The main rule is: targets come from wells, predictors come from spatial data that would also be available before drilling in Kazakhstan.

This means the Texas SDR dataset should provide labels such as yield, high-yield class, depth to water, or fresh-water presence. GEE and other free sources should provide predictors such as SAR backscatter, optical vegetation/water indices, DEM derivatives, climate, hydrology, soils, geology, and distance-to-feature metrics.

## What `target_high_yield_20gpm` Means

Yes: `target_high_yield_20gpm` means a binary classification label:

- `1` / `True`: the well has reported yield `>= 20 gallons per minute`.
- `0` / `False`: the well has reported yield `< 20 gallons per minute`.
- `null`: the well has no usable yield record.

This target comes from `WellTest.txt`, mainly the `Yield` field aggregated per `WellReportTrackingNumber`.

Why 20 gpm? It is not a universal hydrogeological constant. It is a practical first threshold for a proof-of-concept productivity map. It separates weak/domestic-scale wells from more useful productive wells in a simple way. Later, we should tune the threshold by purpose:

- Domestic/household: maybe `>= 5` or `>= 10 gpm`.
- Livestock/rural supply: maybe `>= 10` or `>= 20 gpm`.
- Irrigation/commercial: maybe `>= 50`, `>= 100`, or local percentile classes.

For XGBoost, this is a clean first target because classification is often more stable than raw yield regression. The model output can be interpreted as `P(high-yield well | spatial predictors)`, which is naturally a heatmap.

## Recommended Target Variables

### 1. High-Yield Classification

Field: `target_high_yield_20gpm`

Definition:

```text
yield_gpm_max >= 20 -> 1
yield_gpm_max < 20  -> 0
missing yield       -> null / excluded from this model
```

Use this first. It is interpretable, robust, and maps directly to groundwater productivity probability. For Texas training, generate the label from well tests. For Kazakhstan inference, predict this probability over a grid.

Important: do not train on wells with missing yield as zero. Missing yield does not mean low yield.

### 2. Yield Regression

Field: `yield_gpm_max` or `yield_gpm_median`

Use this after the classification model. Raw yield is useful but usually skewed: many low-yield wells and a few very high-yield wells. Train on:

```text
log1p(yield_gpm_max)
```

Then back-transform predictions if needed. This gives a map of expected production potential, but it will be noisier than the binary target.

### 3. Depth-to-Water Regression

Field: `static_water_level_ft_first` or `static_water_level_ft_median`

Use this to map likely water-table depth or drilling accessibility. It is valuable for cost and feasibility, not just water presence.

Careful: water-level values can vary by season, pumping status, measurement method, and well construction. Filter impossible values and keep measurement date as metadata.

### 4. Shallow-Water Classification

Field: `target_shallow_water_le_100ft`

Definition:

```text
static_water_level_ft_first <= 100 -> 1
static_water_level_ft_first > 100  -> 0
missing level                      -> null / excluded
```

This is useful for a second communication-friendly heatmap: probability of shallow groundwater. The `100 ft` threshold is only a starting point. It should be tuned for Kazakhstan drilling economics.

### 5. Fresh-Water Presence Classification

Field: `target_fresh_water_present`

Definition:

```text
fresh_water_interval_count > 0 -> 1
fresh_water_interval_count = 0 with strata record -> 0
no strata record -> null
```

This target is useful in arid/saline regions, including parts of Kazakhstan. Treat it as weak supervision because it depends on driller text and water-type reporting consistency.

### 6. Specific Capacity Regression

Field: `specific_capacity_gpm_per_ft_median`

Definition:

```text
yield_gpm / drawdown_ft
```

This is hydrogeologically strong because it normalizes yield by drawdown. However, coverage is lower because drawdown is often missing or recorded as non-numeric. Use this as an advanced target, not the first target.

## Recommended First Modeling Sequence

1. Train `target_high_yield_20gpm` classification.
2. Train `log1p(yield_gpm_max)` regression.
3. Train `static_water_level_ft_first` regression.
4. Train `target_fresh_water_present` classification.
5. Compare feature importance and spatial validation results.
6. Use Kazakhstan grid inference only with predictors available in Kazakhstan.

## Training Dataset Geometry

### Texas Training Rows

Use one row per well report or one row per spatial cell.

For the first model, use one row per well:

```text
row = one WellReportTrackingNumber
geometry = well point
target = well-derived label
features = GEE predictors sampled at or around the point
```

Later, deduplicate nearby wells into cells to reduce spatial clustering bias:

```text
row = grid cell containing one or more wells
target = max/median target within cell
features = cell-level GEE summaries
```

### Grid Cell Size

Use a multi-scale design:

- `250 m` grid: good for Sentinel-1/Sentinel-2 local signal and terrain.
- `500 m` grid: recommended first production grid. More stable, less noisy, still useful for regional exploration.
- `1000 m` grid: good for coarse climate/geology/hydrology layers and regional Kazakhstan heatmaps.

Recommended first choice:

```text
500 m x 500 m grid cells
cell area = 0.25 km2
```

Why 500 m? It is a practical compromise. Sentinel bands are 10-30 m, DEM is often 30 m, climate is much coarser, and well locations may have uncertainty. A 500 m cell reduces point-location noise while preserving useful local variation.

### Feature Extraction Radius

For each well or grid-cell centroid, extract both point/cell values and neighborhood summaries:

- `100 m`: very local terrain/land-cover signal.
- `250 m`: local drainage and vegetation context.
- `500 m`: recommended default.
- `1000 m`: broader recharge and geomorphology context.
- `3000 m` or `5000 m`: climate/hydrogeology context, especially in arid basins.

For each radius, calculate:

```text
mean, median, stdDev, min, max, percentile_10, percentile_90
```

Do not extract too many combinations at first. Start with `500 m` and `1000 m`, then expand.

## Time Windows

The target date should be tied to drilling or measurement date where available.

For each Texas well:

- Use `DrillingEndDate` as the primary anchor date.
- Use `MeasurementDate` for water-level models if available.
- Use `DateSubmitted` only as fallback.

Recommended temporal windows:

- `same-year`: January 1 to December 31 of the drilling year.
- `previous 12 months`: more physically sensible for recharge/vegetation/climate before drilling.
- `previous 3 years`: stable mean conditions.
- `long-term climatology`: 10-20 year climate normals where possible.

For Kazakhstan inference, choose a target prediction year:

- Start with `2023-2025` composite if using Sentinel-era predictors.
- For long-term groundwater prospectivity, use multi-year medians rather than one wet/dry year.

Recommended first model:

```text
Texas training:
  optical/SAR features = median over previous 12 months before drilling
  climate features = previous 12 months + previous 5-year mean
  DEM/geology/soil/hydrology = static

Kazakhstan inference:
  optical/SAR features = median over 2021-2025 or latest complete 3-5 years
  climate features = 1991-2020 normal + recent 5-year anomalies
```

## Predictor Sources and Features

## Sentinel-1 SAR

GEE dataset:

- `COPERNICUS/S1_GRD`

Official notes: Sentinel-1 GRD is C-band SAR, includes VV/VH or HH/HV polarizations, has an angle band, and GEE provides calibrated, terrain-corrected, log-scaled dB values. The catalog lists availability from 2014 onward and a 6-day revisit interval.

Use:

- Moisture proxy.
- Surface roughness.
- Vegetation/structure proxy.
- Flooding or temporary water influence.
- All-weather predictor where optical imagery is cloudy or snow-affected.

Filters:

```text
instrumentMode = IW
polarizations include VV and VH
orbit pass separated or controlled: ASCENDING, DESCENDING
remove edge noise / extreme low values
```

Features to extract:

- `VV_mean`, `VV_median`, `VV_std`.
- `VH_mean`, `VH_median`, `VH_std`.
- `VV_minus_VH`.
- `VV_div_VH` or `VV/VH` in linear scale.
- seasonal medians: spring, summer, autumn.
- annual amplitude: wet-season median minus dry-season median.
- percentiles: `VV_p10`, `VV_p90`, `VH_p10`, `VH_p90`.
- local texture if feasible: neighborhood variance of VV/VH.

Suggested radii:

- `250 m`, `500 m`, `1000 m`.

Why useful for groundwater:

SAR does not see groundwater directly, but it can detect surface moisture, riparian vegetation structure, irrigated/agricultural patterns, floodplain wetness, and geomorphic surfaces related to shallow groundwater.

## Sentinel-2 Optical

GEE dataset:

- `COPERNICUS/S2_SR_HARMONIZED`

Use:

- Vegetation vigor.
- Surface water.
- soil/mineral brightness.
- wetness and salinity proxies.
- land cover and irrigated vegetation.

Preprocessing:

- Use surface reflectance.
- Mask clouds, cloud shadows, and snow where needed.
- Build seasonal composites, not single images.

Bands/features:

- Raw median bands: `B2`, `B3`, `B4`, `B8`, `B11`, `B12`.
- NDVI: `(NIR - Red) / (NIR + Red)`.
- NDWI/MNDWI: water and wetness indicators.
- NDMI: `(NIR - SWIR1) / (NIR + SWIR1)`.
- NBR: `(NIR - SWIR2) / (NIR + SWIR2)`.
- SAVI/MSAVI: vegetation adjusted for sparse cover.
- Bare soil index.
- brightness/greenness/wetness tasseled-cap-like features if implemented.
- seasonal NDVI max, median, min, amplitude.
- frequency of high NDVI in dry season.
- distance to persistent surface water from water mask.

Suggested windows:

- spring green-up.
- summer dry period.
- autumn.
- annual median.
- previous 12 months before drilling for Texas.

Why useful for groundwater:

In arid/semi-arid zones, anomalously persistent vegetation or riparian greenness can indicate shallow groundwater, seepage, irrigation, or favorable recharge zones. SWIR indices help separate dry soil, wet soil, and vegetation moisture.

## Landsat

GEE datasets:

- `LANDSAT/LC08/C02/T1_L2`
- `LANDSAT/LC09/C02/T1_L2`
- optionally `LANDSAT/LE07/C02/T1_L2` and `LANDSAT/LT05/C02/T1_L2` for long historical context.

Use:

- Long-term optical history.
- Pre-Sentinel temporal features.
- Thermal features from Landsat surface temperature.

Features:

- NDVI, NDWI/MNDWI, NDMI, NBR.
- land surface temperature median/mean.
- long-term vegetation persistence.
- dry-season greenness anomaly.
- change metrics across decades.

Why use Landsat if Sentinel-2 exists?

Sentinel-2 has higher spatial resolution, but Landsat has a much longer archive. For groundwater, long-term vegetation persistence and thermal behavior can be more informative than one recent high-resolution image.

## DEM and Terrain

GEE datasets:

- `USGS/SRTMGL1_003` for 30 m SRTM.
- Consider MERIT DEM or FABDEM outside GEE if better hydrologic correction is needed.

Features:

- elevation.
- slope.
- aspect encoded as `sin(aspect)` and `cos(aspect)`.
- curvature: profile, planform, general curvature.
- topographic position index at 250 m, 500 m, 1000 m, 3000 m.
- terrain ruggedness index.
- valley-bottom flatness proxy.
- relative elevation above nearest drainage.
- topographic wetness index: `ln(flow_accumulation / tan(slope))`.
- local relief within 500 m and 1000 m.

Why useful:

Groundwater occurrence is strongly controlled by topographic position, recharge zones, discharge zones, valley bottoms, alluvial plains, piedmonts, and basin margins. DEM features are some of the most transferable predictors from Texas to Kazakhstan.

## Hydrology

GEE datasets:

- `WWF/HydroSHEDS/15ACC`
- `WWF/HydroSHEDS/15DIR`
- `MERIT/Hydro/v1_0_1`
- `JRC/GSW1_4/GlobalSurfaceWater`

Official notes: HydroSHEDS provides global hydrographic information such as drainage directions and flow accumulation based on SRTM, but quality is lower above 60 degrees north. This matters for northern Kazakhstan. JRC Global Surface Water provides mapped surface-water history.

Features:

- flow accumulation.
- drainage direction class.
- distance to river/stream.
- distance to high-flow-accumulation pixels.
- distance to persistent surface water.
- surface water occurrence.
- surface water seasonality.
- distance to floodplain/alluvial corridor proxy.
- upstream catchment area.
- HAND: height above nearest drainage, if generated.

Suggested distance features:

- nearest stream distance at several stream thresholds.
- nearest permanent water distance.
- nearest seasonal water distance.
- log distance transforms.

Why useful:

Recharge and shallow groundwater are often tied to river valleys, ephemeral channels, alluvial fans, floodplains, and endorheic basin margins.

## Climate and Water Balance

GEE datasets:

- `IDAHO_EPSCOR/TERRACLIMATE`
- `ECMWF/ERA5_LAND/MONTHLY_AGGR`
- `UCSB-CHG/CHIRPS/DAILY`

Official notes: TerraClimate is monthly, global, and includes precipitation, temperature, evapotranspiration, runoff, soil moisture, snow-water equivalent, vapor pressure deficit, and drought variables. ERA5-Land monthly aggregates are global from 1950 to near-real-time. CHIRPS provides daily precipitation.

Features:

- annual precipitation.
- seasonal precipitation: winter/spring/summer/autumn.
- precipitation previous 3, 6, 12 months before drilling.
- long-term precipitation normal.
- precipitation anomaly: recent period minus long-term normal.
- potential evapotranspiration.
- actual evapotranspiration.
- water deficit.
- aridity index: `precipitation / PET`.
- runoff.
- soil moisture.
- snow water equivalent.
- vapor pressure deficit.
- temperature min/max/mean.

Why useful:

Climate controls recharge potential. In Kazakhstan, snowmelt, spring precipitation, aridity, and evapotranspiration are likely essential predictors.

## Soil

GEE datasets:

- OpenLandMap soil texture and related soil properties.
- SoilGrids via ISRIC API or downloaded rasters if needed.

Features:

- texture class.
- sand/silt/clay percentage by depth where available.
- bulk density.
- organic carbon.
- soil water capacity.
- depth to bedrock if available.
- permeability/infiltration proxy.

Why useful:

Soil controls infiltration, runoff, recharge, and surface moisture persistence. For groundwater models, soil features are especially useful when combined with slope and climate.

## Geology and Structure

Possible sources:

- OneGeology WMS/WFS where available.
- national geological survey layers where open.
- GLiM global lithological map if locally downloaded.
- Hydrogeological maps from national or international portals if available.
- fault/lineament layers from geological maps or derived lineaments from DEM/SAR.

Features:

- lithology class.
- generalized permeability class.
- unconsolidated sediment vs bedrock.
- carbonate/limestone presence.
- volcanic/crystalline/sedimentary class.
- distance to fault.
- distance to lithological boundary.
- lineament density within 1 km, 3 km, 5 km.
- distance to alluvial deposits.
- basin-fill/alluvial plain indicator.

Why useful:

Geology is often the strongest control on groundwater storage and transmissivity. Satellite data can help, but without geology the model may confuse green vegetation with water availability.

## Land Cover and Human Activity

GEE datasets:

- ESA WorldCover.
- Dynamic World.
- MODIS land cover if coarser long-term context is needed.

Features:

- land-cover class.
- cropland fraction.
- grassland/shrubland fraction.
- urban fraction.
- irrigated vegetation proxy from dry-season NDVI.
- distance to agriculture.
- distance to settlement/roads if using OpenStreetMap externally.

Why useful:

Human activity affects well placement and observed yield. This is both useful and dangerous: it can introduce sampling bias. Use land-cover features, but keep human-access features separate so you can test whether the model is learning geology or simply where people drill.

## Sampling Design

### Positive and Negative Labels

For yield classification:

- Positive: wells with `yield_gpm_max >= threshold`.
- Negative: wells with reported yield below threshold.
- Exclude missing yield.

Do not use random locations without wells as negative groundwater labels. A random location without a well is unlabeled, not necessarily dry.

### Spatial Bias Control

Texas wells are not randomly distributed. They cluster near farms, roads, towns, and private land. Use:

- spatial train/test split by county, HUC basin, or large grid block.
- remove duplicate or near-duplicate wells.
- limit overrepresented areas.
- compare random split vs spatial split. Spatial split is the honest score.

### Time Leakage Control

For training, extract predictors only before or around the drilling/measurement date. Do not use future imagery after the well was drilled unless building a static geology-style model.

Recommended:

```text
For each well:
  anchor_date = DrillingEndDate
  feature_window = anchor_date - 365 days to anchor_date
```

For long-term static models:

```text
Use multi-year medians, but label the model as long-term prospectivity, not date-specific prediction.
```

## Proposed Dataset Schema

### Core Columns

```text
well_id
latitude
longitude
geometry
anchor_date
target_name
target_value
split_group
```

### Target Columns

```text
yield_gpm_max
yield_gpm_median
target_high_yield_20gpm
static_water_level_ft_first
target_shallow_water_le_100ft
target_fresh_water_present
specific_capacity_gpm_per_ft_median
```

### Predictor Columns

Use clear prefixes:

```text
s1_vv_median_500m_prev12m
s1_vh_median_500m_prev12m
s1_vv_vh_diff_500m_prev12m
s2_ndvi_median_500m_prev12m
s2_ndmi_median_500m_prev12m
landsat_lst_median_1000m_prev5y
dem_elevation
dem_slope
dem_tpi_1000m
hydro_dist_stream_m
hydro_flow_accumulation
clim_pr_prev12m
clim_pet_prev12m
clim_aridity_1991_2020
soil_texture_class
geo_lithology_class
geo_fault_distance_m
```

## Texas-to-Kazakhstan Transfer Plan

### Phase 1: Texas Well-Point Dataset

1. Use `data/processed/texas_sdr_wells.parquet`.
2. Filter to valid coordinates.
3. Select one target, starting with `target_high_yield_20gpm`.
4. Upload well points to GEE as an asset or read from Cloud Storage.
5. Extract predictors for each point using `sampleRegions` or buffered reductions.
6. Export table to Parquet/CSV.
7. Train XGBoost locally.
8. Evaluate with spatial cross-validation.

### Phase 2: Texas Grid Dataset

1. Build 500 m grid over Texas or target aquifer regions.
2. Assign well labels to cells.
3. Aggregate multiple wells per cell:
   - high-yield label: max or majority.
   - yield regression: median/max.
   - depth-to-water: median.
4. Extract GEE predictors per cell.
5. Compare cell model against point model.

### Phase 3: Kazakhstan Prediction Grid

1. Select a Kazakhstan study area.
2. Build 500 m or 1 km grid.
3. Extract the exact same predictor columns.
4. Apply trained model.
5. Produce heatmap:
   - probability high-yield.
   - expected yield.
   - probability shallow water.
   - uncertainty / out-of-domain score.

### Phase 4: Validation and Calibration

1. Add any Kazakhstan well observations that become available.
2. Validate rank ordering: do predicted high-prospect zones contain known wells/springs/oases?
3. Calibrate thresholds locally.
4. Retrain or fine-tune with Kazakhstan labels when possible.

## Important Modeling Warnings

- Texas-trained predictions in Kazakhstan are proof-of-concept, not final groundwater truth.
- A no-well location is not a dry-well label.
- Construction details such as pump depth and casing depth are not available before drilling, so they should not be used for Kazakhstan inference unless building a diagnostic Texas-only model.
- County, owner, driller, and address fields should not be used as features.
- Use spatial cross-validation; random train/test split will overestimate performance.
- Keep target models separate. Do not mix yield, depth-to-water, and fresh-water presence into one target.

## Source Notes

- Sentinel-1 GRD in GEE: `COPERNICUS/S1_GRD`, C-band SAR, VV/VH/HH/HV polarizations, angle band, calibrated and terrain-corrected dB values. Catalog: <https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD>
- Sentinel-2 SR Harmonized in GEE: `COPERNICUS/S2_SR_HARMONIZED`. Catalog: <https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED>
- Landsat Collection 2 Level 2 in GEE: `LANDSAT/LC08/C02/T1_L2`, `LANDSAT/LC09/C02/T1_L2`. Landsat 9 catalog: <https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC09_C02_T1_L2>
- SRTM 30 m DEM in GEE: `USGS/SRTMGL1_003`. Catalog: <https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003>
- TerraClimate in GEE: `IDAHO_EPSCOR/TERRACLIMATE`. Catalog: <https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_TERRACLIMATE>
- ERA5-Land monthly in GEE: `ECMWF/ERA5_LAND/MONTHLY_AGGR`. Catalog: <https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_MONTHLY_AGGR>
- CHIRPS daily precipitation in GEE: `UCSB-CHG/CHIRPS/DAILY`. Catalog: <https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY>
- HydroSHEDS flow products in GEE: `WWF/HydroSHEDS/15ACC`, `WWF/HydroSHEDS/15DIR`. Catalogs: <https://developers.google.com/earth-engine/datasets/catalog/WWF_HydroSHEDS_15ACC>, <https://developers.google.com/earth-engine/datasets/catalog/WWF_HydroSHEDS_15DIR>
- MERIT Hydro in GEE: `MERIT/Hydro/v1_0_1`. Catalog: <https://developers.google.com/earth-engine/datasets/catalog/MERIT_Hydro_v1_0_1>
- JRC Global Surface Water in GEE: `JRC/GSW1_4/GlobalSurfaceWater`. Catalog: <https://developers.google.com/earth-engine/datasets/catalog/JRC_GSW1_4_GlobalSurfaceWater>
- OpenLandMap soil layers are available in the GEE catalog; SoilGrids can be used externally if finer soil attributes are needed. Example texture catalog: <https://developers.google.com/earth-engine/datasets/catalog/OpenLandMap_SOL_SOL_TEXTURE-CLASS_USDA-TT_M_v02>
