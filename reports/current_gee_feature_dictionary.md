# Current GEE Feature Dictionary and Rationale

## Scope

This report documents the features currently extracted by:

```text
scripts/extract_gee_features_smoke.py
```

Current output examples:

```text
data/features/gee_yield20_smoke_10.parquet
data/features/gee_yield20_batch_000000_000100.parquet
```

The current extractor produces:

```text
102 total columns
93 feature / QA columns
```

The target is **not** extracted from GEE. It comes from the Texas SDR well tests:

```text
target_yield_ge_20gpm_int
1 = yield_gpm_max >= 20 gpm
0 = yield_gpm_max < 20 gpm
```

## No-Leakage Design

For every well, the feature extraction window is:

```text
feature_start_date = anchor_date - 365 days
feature_end_date = anchor_date
```

where:

```text
anchor_date = DrillingEndDate -> DrillingStartDate -> DateSubmitted
```

This is important because we are pretending we are standing at the drilling date and asking: *what could we know before drilling?*

The script does **not** use:

- pump depth
- casing depth
- borehole depth
- filter/screen intervals
- drawdown
- specific capacity
- lithology logs from the drilled well
- strata/water-bearing intervals from the drilled well
- owner, driller, address, license, or comments

Those would either define the target or leak information observed after drilling.

## Spatial Extraction Design

For each well coordinate:

```text
geometry = 500 m buffer around the well point
scale = 30 m
reducers = mean, median, standard deviation
```

So a feature such as:

```text
s2_ndvi_median_12m_mean
```

means:

```text
1. Build a Sentinel-2 NDVI median composite over the 365 days before drilling.
2. Take all pixels inside a 500 m radius around the well.
3. Calculate the mean value inside that buffer.
```

Why 500 m?

- Well coordinates can be imperfect.
- A single 10-30 m pixel may not represent the hydrogeological setting.
- Groundwater productivity is controlled by local terrain, vegetation, drainage, and geologic context, not only the exact well point.
- 500 m is small enough for local prospectivity but large enough to reduce pixel noise.

Why 365 days?

- Captures seasonal conditions before drilling.
- Avoids using future information.
- Gives Sentinel-1/Sentinel-2 enough observations to build stable composites.
- Captures annual water-balance conditions from climate data.

## QA / Count Features

### `qa_sentinel2_image_count`

Source:

```text
COPERNICUS/S2_SR_HARMONIZED
```

Meaning:

Number of Sentinel-2 images available in the pre-drilling 365-day window after spatial/date filtering and cloud-percentage filtering.

Why useful:

This is a quality-control feature, not a hydrological feature. If image count is low, optical features may be less reliable due to cloud cover, snow, missing acquisitions, or filtering.

### `qa_sentinel1_image_count`

Source:

```text
COPERNICUS/S1_GRD
```

Meaning:

Number of Sentinel-1 SAR images available in the pre-drilling 365-day window after filtering to IW mode and VV/VH dual polarization.

Why useful:

SAR is all-weather, so low counts may indicate data availability or filtering issues. This helps diagnose feature reliability.

### `qa_terraclimate_month_count`

Source:

```text
IDAHO_EPSCOR/TERRACLIMATE
```

Meaning:

Number of monthly TerraClimate records in the 365-day pre-drilling window. Expected value is usually around 12.

Why useful:

Confirms climate aggregation completeness.

## Sentinel-1 SAR Features

API / GEE collection:

```text
COPERNICUS/S1_GRD
```

Official catalog:

```text
https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD
```

Current filters:

```text
instrumentMode = IW
polarizations include VV and VH
date = anchor_date - 365 days to anchor_date
edge/noise mask: VV > -35 dB
```

Current bands:

```text
s1_vv_median_12m
s1_vh_median_12m
s1_angle_median_12m
s1_vv_minus_vh_median_12m
```

For each band, the extracted columns are:

```text
*_mean
*_median
*_stdDev
```

So the actual features are:

```text
s1_vv_median_12m_mean
s1_vv_median_12m_median
s1_vv_median_12m_stdDev
s1_vh_median_12m_mean
s1_vh_median_12m_median
s1_vh_median_12m_stdDev
s1_angle_median_12m_mean
s1_angle_median_12m_median
s1_angle_median_12m_stdDev
s1_vv_minus_vh_median_12m_mean
s1_vv_minus_vh_median_12m_median
s1_vv_minus_vh_median_12m_stdDev
```

### Physics

Sentinel-1 is an active radar system. It sends microwave energy toward Earth and measures the backscattered signal. Unlike optical imagery, SAR does not depend on sunlight and can observe through most clouds.

The GEE Sentinel-1 GRD product is C-band SAR. GEE provides calibrated, terrain-corrected backscatter in decibels. The collection includes VV and VH polarizations plus an incidence angle band.

### `VV`

VV means vertical transmit and vertical receive.

What it represents:

- surface roughness
- soil moisture influence
- built structures
- open bare ground response
- vegetation structure, especially where canopy is not too dense

Why useful for groundwater:

Groundwater is not directly visible to SAR, but shallow groundwater can influence soil moisture, riparian vegetation, floodplain wetness, irrigation, and surface roughness patterns.

### `VH`

VH means vertical transmit and horizontal receive.

What it represents:

- volume scattering
- vegetation structure
- roughness and canopy complexity

Why useful:

Areas with persistent vegetation or wetter floodplain vegetation can have different VH behavior. In arid/semi-arid regions, this may indirectly indicate shallow groundwater or favorable hydrogeomorphic zones.

### `VV - VH`

What it represents:

Contrast between co-polarized and cross-polarized backscatter.

Why useful:

This can help separate bare/rough surfaces from vegetated or structurally complex surfaces. It is often more stable than using raw VV and VH alone.

### `angle`

What it represents:

Approximate radar incidence angle.

Why useful:

Backscatter depends partly on viewing geometry. Including angle helps the model understand acquisition geometry effects rather than confusing them with hydrological signal.

## Sentinel-2 Optical Features

API / GEE collection:

```text
COPERNICUS/S2_SR_HARMONIZED
```

Official catalog:

```text
https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
```

Current filters:

```text
date = anchor_date - 365 days to anchor_date
CLOUDY_PIXEL_PERCENTAGE <= 60
cloud/shadow/snow classes masked with SCL
surface reflectance scaled by 0.0001
```

Current raw bands:

```text
B2  blue
B3  green
B4  red
B8  near infrared
B11 shortwave infrared 1
B12 shortwave infrared 2
```

Current indices:

```text
NDVI
NDMI
MNDWI
NBR
BSI
```

For each Sentinel-2 band/index, the script first creates a 365-day median composite, then extracts:

```text
*_mean
*_median
*_stdDev
```

### Optical Remote Sensing Physics

Sentinel-2 is a passive optical sensor. It measures reflected sunlight in visible, near-infrared, and shortwave-infrared wavelengths. Different materials absorb and reflect wavelengths differently:

- green vegetation strongly reflects NIR and absorbs red light
- water absorbs strongly in NIR/SWIR
- wet soil and dry soil differ strongly in SWIR
- bare soil, salts, and minerals affect visible/SWIR brightness

For groundwater mapping, optical data does not see groundwater directly. It captures surface expressions of groundwater:

- persistent vegetation
- riparian greenness
- wet soil
- seepage/wetland zones
- surface water
- salinity or bare soil patterns
- irrigated vegetation

### Raw Reflectance Bands

Actual features:

```text
s2_b2_median_12m_mean
s2_b2_median_12m_median
s2_b2_median_12m_stdDev
s2_b3_median_12m_mean
s2_b3_median_12m_median
s2_b3_median_12m_stdDev
s2_b4_median_12m_mean
s2_b4_median_12m_median
s2_b4_median_12m_stdDev
s2_b8_median_12m_mean
s2_b8_median_12m_median
s2_b8_median_12m_stdDev
s2_b11_median_12m_mean
s2_b11_median_12m_median
s2_b11_median_12m_stdDev
s2_b12_median_12m_mean
s2_b12_median_12m_median
s2_b12_median_12m_stdDev
```

Meaning:

- `B2`, `B3`, `B4`: visible color and brightness.
- `B8`: vegetation vigor and biomass response.
- `B11`, `B12`: moisture, dry soil, mineral/bare ground, burn/rock/soil response.

Why useful:

These raw bands allow the model to learn spectral patterns that may not be fully captured by hand-crafted indices.

### NDVI

Formula:

```text
(B8 - B4) / (B8 + B4)
```

Actual features:

```text
s2_ndvi_median_12m_mean
s2_ndvi_median_12m_median
s2_ndvi_median_12m_stdDev
```

What it represents:

Vegetation greenness and vigor.

Why useful for groundwater:

In dry regions, persistent vegetation can indicate shallow groundwater, seepage, riparian corridors, or irrigation. It is not proof of groundwater, but it is a strong surface proxy.

### NDMI

Formula:

```text
(B8 - B11) / (B8 + B11)
```

Actual features:

```text
s2_ndmi_median_12m_mean
s2_ndmi_median_12m_median
s2_ndmi_median_12m_stdDev
```

What it represents:

Vegetation and surface moisture using NIR and SWIR.

Why useful:

SWIR is sensitive to water content in vegetation and soil. NDMI can help identify moist vegetation or wetter surface conditions.

### MNDWI

Formula:

```text
(B3 - B11) / (B3 + B11)
```

Actual features:

```text
s2_mndwi_median_12m_mean
s2_mndwi_median_12m_median
s2_mndwi_median_12m_stdDev
```

What it represents:

Open water and wet surface signal.

Why useful:

Groundwater prospects often relate to floodplains, river corridors, springs, wetlands, playas, and shallow water-table zones. MNDWI helps capture surface water influence.

### NBR

Formula:

```text
(B8 - B12) / (B8 + B12)
```

Actual features:

```text
s2_nbr_median_12m_mean
s2_nbr_median_12m_median
s2_nbr_median_12m_stdDev
```

What it represents:

NIR/SWIR2 contrast. Commonly used for burn severity, but also useful as a vegetation/moisture/soil contrast index.

Why useful:

Can help distinguish vegetated, dry, exposed, burned, or rocky surfaces that influence recharge and infiltration.

### BSI

Formula:

```text
((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2))
```

Actual features:

```text
s2_bsi_median_12m_mean
s2_bsi_median_12m_median
s2_bsi_median_12m_stdDev
```

What it represents:

Bare soil brightness / exposed ground proxy.

Why useful:

Bare soil, alluvial fans, dry riverbeds, playas, and sparsely vegetated recharge areas can be relevant to groundwater. BSI helps separate exposed soil/rock from vegetation and water.

## DEM and Terrain Features

API / GEE collection:

```text
USGS/SRTMGL1_003
```

Official catalog:

```text
https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003
```

Current features:

```text
dem_elevation
dem_slope
dem_aspect_sin
dem_aspect_cos
dem_tpi_500m
```

For each, the extracted columns are:

```text
*_mean
*_median
*_stdDev
```

### DEM Physics / Meaning

SRTM is a radar-derived digital elevation model. It represents land-surface elevation. Terrain controls groundwater through recharge, runoff, erosion, sediment accumulation, drainage convergence, and valley formation.

### Elevation

Actual features:

```text
dem_elevation_mean
dem_elevation_median
dem_elevation_stdDev
```

What it represents:

Height above sea level within the 500 m buffer.

Why useful:

Elevation influences climate, drainage, geomorphology, aquifer type, and hydraulic gradients.

### Slope

Actual features:

```text
dem_slope_mean
dem_slope_median
dem_slope_stdDev
```

What it represents:

Terrain steepness.

Why useful:

Gentle slopes often favor infiltration and sediment accumulation. Steep slopes often favor runoff and thinner regolith, though fractured mountain aquifers can be exceptions.

### Aspect Sine and Cosine

Actual features:

```text
dem_aspect_sin_mean
dem_aspect_sin_median
dem_aspect_sin_stdDev
dem_aspect_cos_mean
dem_aspect_cos_median
dem_aspect_cos_stdDev
```

What it represents:

Slope direction encoded cyclically. Using sine/cosine avoids the 0/360-degree discontinuity.

Why useful:

Aspect affects solar exposure, snow persistence, evapotranspiration, vegetation, and soil moisture. This is especially relevant for Kazakhstan where snowmelt and exposure can matter.

### TPI 500 m

Formula:

```text
dem_tpi_500m = elevation - local mean elevation within 500 m
```

Actual features:

```text
dem_tpi_500m_mean
dem_tpi_500m_median
dem_tpi_500m_stdDev
```

What it represents:

Topographic position. Negative values indicate local lows/valleys; positive values indicate ridges or local highs.

Why useful:

Valley bottoms, depressions, and lower landscape positions often concentrate water, sediment, and recharge/discharge processes.

## Hydrology Features

APIs / GEE collections:

```text
JRC/GSW1_4/GlobalSurfaceWater
MERIT/Hydro/v1_0_1
```

Official catalogs:

```text
https://developers.google.com/earth-engine/datasets/catalog/JRC_GSW1_4_GlobalSurfaceWater
https://developers.google.com/earth-engine/datasets/catalog/MERIT_Hydro_v1_0_1
```

Current features:

```text
hydro_surface_water_occurrence
hydro_surface_water_seasonality
hydro_upstream_area
```

For each, the extracted columns are:

```text
*_mean
*_median
*_stdDev
```

### JRC Surface Water Occurrence

Actual features:

```text
hydro_surface_water_occurrence_mean
hydro_surface_water_occurrence_median
hydro_surface_water_occurrence_stdDev
```

What it represents:

Frequency with which surface water has been observed historically.

Why useful:

Persistent or recurring surface water can indicate rivers, reservoirs, wetlands, floodplain zones, playas, or shallow groundwater discharge areas. In the current output, this feature is sometimes null because JRC masks areas where water was never detected.

### JRC Surface Water Seasonality

Actual features:

```text
hydro_surface_water_seasonality_mean
hydro_surface_water_seasonality_median
hydro_surface_water_seasonality_stdDev
```

What it represents:

Number of months per year water is typically present.

Why useful:

Separates permanent water from seasonal/ephemeral water. Seasonal water can indicate floodplain recharge, ephemeral channels, wetlands, or depressions.

### MERIT Upstream Area

Actual features:

```text
hydro_upstream_area_mean
hydro_upstream_area_median
hydro_upstream_area_stdDev
```

What it represents:

Upstream drainage area / contributing area from MERIT Hydro.

Why useful:

Large upstream area usually means proximity to drainage networks or flow-accumulation corridors. These areas can correspond to alluvial aquifers, valley fills, recharge pathways, or river-connected groundwater systems.

## TerraClimate Features

API / GEE collection:

```text
IDAHO_EPSCOR/TERRACLIMATE
```

Official catalog:

```text
https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_TERRACLIMATE
```

Current time window:

```text
anchor_date - 365 days to anchor_date
monthly data, usually 12 records
```

Current climate bands:

```text
pr
pet
aet
def
tmmn
tmmx
derived aridity = pr / pet
```

For each derived climate image, the extracted columns are:

```text
*_mean
*_median
*_stdDev
```

### Climate Physics / Meaning

Climate data are not satellite-only. TerraClimate combines climate observations and climatological data to estimate monthly water-balance variables. These features represent recharge potential, aridity, evaporative demand, vegetation stress, and water availability.

### Precipitation Sum

Actual features:

```text
clim_pr_sum_12m_mean
clim_pr_sum_12m_median
clim_pr_sum_12m_stdDev
```

What it represents:

Total precipitation during the 365-day pre-drilling period.

Why useful:

Precipitation is a primary control on recharge. Higher rainfall does not automatically mean higher well yield, but it affects water availability, vegetation, runoff, and shallow aquifer replenishment.

### Potential Evapotranspiration Sum

Actual features:

```text
clim_pet_sum_12m_mean
clim_pet_sum_12m_median
clim_pet_sum_12m_stdDev
```

What it represents:

Atmospheric demand for water, summed over the pre-drilling year.

Why useful:

High PET means water is more likely to be lost to evaporation/transpiration before recharging aquifers. It helps separate wet climates from dry/high-demand climates.

### Actual Evapotranspiration Sum

Actual features:

```text
clim_aet_sum_12m_mean
clim_aet_sum_12m_median
clim_aet_sum_12m_stdDev
```

What it represents:

Estimated water actually returned to the atmosphere by evaporation and plant transpiration.

Why useful:

High AET can indicate available water and vegetation use. In dry areas, AET can be constrained by water availability and therefore indirectly reveal wetter zones.

### Climatic Deficit Sum

Actual features:

```text
clim_def_sum_12m_mean
clim_def_sum_12m_median
clim_def_sum_12m_stdDev
```

What it represents:

Water deficit: atmospheric demand not met by available water.

Why useful:

High deficit indicates aridity/stress. This is important for mapping regions where recharge is limited or vegetation survives only in groundwater-supported zones.

### Minimum Temperature Mean

Actual features:

```text
clim_tmin_mean_12m_mean
clim_tmin_mean_12m_median
clim_tmin_mean_12m_stdDev
```

What it represents:

Mean minimum temperature over the pre-drilling year.

Why useful:

Temperature affects snow persistence, evapotranspiration, freeze/thaw, vegetation, and water balance.

### Maximum Temperature Mean

Actual features:

```text
clim_tmax_mean_12m_mean
clim_tmax_mean_12m_median
clim_tmax_mean_12m_stdDev
```

What it represents:

Mean maximum temperature over the pre-drilling year.

Why useful:

High maximum temperature increases evaporative demand and can reduce effective recharge. It also helps distinguish climatic zones.

### Aridity Ratio

Formula:

```text
clim_aridity_pr_pet_12m = precipitation_sum / potential_evapotranspiration_sum
```

Actual features:

```text
clim_aridity_pr_pet_12m_mean
clim_aridity_pr_pet_12m_median
clim_aridity_pr_pet_12m_stdDev
```

What it represents:

Water supply relative to atmospheric water demand.

Why useful:

This is one of the most interpretable climate predictors. Low values indicate arid conditions; higher values indicate more favorable recharge potential.

## Why Mean, Median, and Standard Deviation?

For every raster band, the script extracts:

```text
mean
median
stdDev
```

### Mean

Represents the average condition around the well.

Useful for:

- general climate/terrain/vegetation setting
- broad local context

### Median

Represents the robust central condition.

Useful because:

- less sensitive to outlier pixels
- more stable near mixed land cover
- better for noisy SAR/optical composites

### Standard Deviation

Represents local heterogeneity.

Useful because:

- high stdDev may indicate edges, channels, mixed terrain, riparian corridors, or variable vegetation
- hydrogeologically important zones are often transitional, not uniform

## Current Feature Counts by Group

Approximate current counts:

```text
QA counts:          3
Sentinel-1 SAR:    12
Sentinel-2 optical:33
DEM terrain:       15
Hydrology:          9
TerraClimate:      21
Total feature/QA:  93
```

## Current Limitations

1. The current extractor is row-by-row and uses `getInfo()`. This is fine for smoke tests and small batches but slow for the full 47k dataset.
2. Hydrology JRC water occurrence/seasonality can be null in places where water was never mapped. This should be treated carefully, possibly filled with 0 after confirming mask semantics.
3. Sentinel-2 cloud masking is simple. Later versions can use a stronger cloud-probability product.
4. Sentinel-1 ascending/descending orbits are currently combined. Later versions may separate them.
5. Geology, soil, Landsat thermal, lineaments, and distance-to-stream features are not yet included in the current extractor, although they are planned.
6. Some features are indirect proxies. The model should be interpreted as groundwater prospectivity, not direct groundwater observation.

## Source Catalog Links

- Sentinel-1 GRD: <https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD>
- Sentinel-2 SR Harmonized: <https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED>
- SRTM 30 m DEM: <https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003>
- JRC Global Surface Water: <https://developers.google.com/earth-engine/datasets/catalog/JRC_GSW1_4_GlobalSurfaceWater>
- MERIT Hydro: <https://developers.google.com/earth-engine/datasets/catalog/MERIT_Hydro_v1_0_1>
- TerraClimate: <https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_TERRACLIMATE>

