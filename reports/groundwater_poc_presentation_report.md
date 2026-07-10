# Groundwater Prospectivity Mapping for Kazakhstan

## Presentation Purpose

This presentation summarizes the completed proof of concept: open Texas well
reports provide labels, Google Earth Engine (GEE) provides leakage-controlled
environmental predictors, and XGBoost learns a first groundwater-yield
classification model. The intended Kazakhstan output is a grid-cell
prospectivity heatmap that prioritizes field investigation.

The map must be interpreted as a ranking of candidate locations. Satellite data
do not directly detect a productive aquifer, and the model does not replace
geophysics, hydrochemistry, test drilling, or hydrogeological judgment.

## 1. Why the Problem Matters in Kazakhstan

Kazakhstan has a large, climatically diverse territory, substantial arid and
semi-arid areas, highly seasonal runoff, and important dependence on
transboundary surface water. UNDP reports that more than 44% of river flow forms
outside Kazakhstan and cites scenarios in which the country could face a major
water shortfall by 2040. These pressures make diversified, evidence-based water
planning increasingly important.

Groundwater is relevant because it can support drinking-water supply,
agriculture and resilience where surface water is seasonal or geographically
distant. In January 2026, the Government of Kazakhstan reported more than 4,000
explored groundwater deposits with approved operational reserves of about
15.7 km3/year, while also emphasizing that a substantial part of the resource
remains insufficiently studied.

The practical question is therefore not "Can a satellite see groundwater?" It
is: **Where should limited survey and drilling budgets be focused first?**

## 2. How Satellite and Environmental Data Help

Groundwater is inferred through multiple indirect indicators:

| Segment | Main current/planned features | Hydrogeological relevance |
|---|---|---|
| DEM and terrain | elevation, slope, aspect, TPI, relief and curvature | Controls runoff concentration, infiltration opportunity, valleys and discharge positions |
| Sentinel-1 SAR | VV, VH, VV-VH, incidence angle, seasonal statistics | C-band microwave backscatter responds to surface roughness, moisture and vegetation structure; it works through clouds and without sunlight |
| Sentinel-2 / Landsat | visible, NIR and SWIR bands; NDVI, NDMI, MNDWI, BSI, NBR | Indicates vegetation condition, wetness, bare soil, surface water and land-surface patterns |
| Climate | precipitation, PET, AET, climatic deficit, aridity, Tmin and Tmax | Represents water supply, atmospheric demand and recharge constraints |
| Hydrology | upstream area, drainage, surface-water occurrence and seasonality | Represents flow pathways, floodplains, valley systems and surface-groundwater interaction |
| Geology and soils | lithology, faults, permeability proxies, soil texture and depth | Constrains aquifer storage, transmissivity, fractures and infiltration |

The current implementation contains **93 model and quality-control features**:
3 QA, 15 terrain, 9 hydrology, 12 Sentinel-1, 33 Sentinel-2 and 21 climate
features. Geology, faults and soils are priority additions because they are
fundamental controls and may improve transfer across regions.

## 3. Data Collection Methodology

### Why Google Earth Engine

GEE is preferable to downloading every source independently for this proof of
concept because it offers:

- a co-located catalog of satellite, terrain, hydrology and climate data;
- common geospatial operations for filtering, masking, compositing and zonal
  statistics;
- server-side processing close to the source data;
- reproducible Python calls that can be applied to both Texas points and a
  Kazakhstan grid;
- standard collection identifiers and preprocessing metadata.

Direct APIs remain useful for datasets that are absent from GEE, especially
national geology, boreholes and hydrogeological maps. The mature system should
therefore use GEE as the raster-processing backbone and external APIs/files for
specialist geological evidence.

### Spatial and Temporal Unit

For each Texas well, the current extractor:

1. Selects an anchor date from drilling end, drilling start, then submission
   date.
2. Uses only the previous **365 days** of dynamic satellite and climate data.
3. Builds filtered or masked annual composites.
4. Summarizes pixels inside a **500 m radius** at a nominal **30 m scale**.
5. Stores mean, median and standard deviation for most variables.

The annual window captures seasonality and enough observations for stable
composites while preventing post-drilling leakage. The 500 m buffer reduces
sensitivity to coordinate error and represents local hydro-environmental
context rather than one possibly noisy pixel.

### Texas Labels

Kazakhstan currently lacks an equally accessible, consistently structured open
well-yield label set for the proof of concept. Texas was selected because the
Texas Water Development Board Submitted Drillers Reports database provides well
coordinates, drilling dates and reported test yield at substantial scale.

The current experimental table contains 5,000 deduplicated wells sampled from
the recent five-year label dataset. The binary target is:

```text
target_yield_ge_20gpm_int = 1 when maximum reported yield >= 20 gpm
target_yield_ge_20gpm_int = 0 when maximum reported yield < 20 gpm
```

Yield, pump tests, construction details, drilled-well lithology and other
post-drilling fields are excluded from model inputs. They would leak the outcome
or describe evidence unavailable when scoring an unknown location.

## 4. Model, Results and Interpretation

XGBoost was selected because gradient-boosted decision trees are effective for
moderate-sized tabular datasets with nonlinear relationships, interacting
environmental controls, different numeric scales, missing values and correlated
predictors. It also provides regularization and feature-subsampling controls.

The pipeline uses median imputation followed by an XGBoost classifier. A
randomized search evaluates 20 parameter combinations with five-fold
cross-validation and optimizes average precision. Tuned parameters include tree
count, depth, learning rate, row/feature subsampling, minimum child weight,
gamma, L1 and L2 regularization.

Current 1,000-row random holdout results are:

| Metric | Score |
|---|---:|
| Accuracy | 0.768 |
| Balanced accuracy | 0.743 |
| ROC-AUC | 0.840 |
| PR-AUC | 0.891 |
| F1 | 0.819 |

These are promising proof-of-concept results, not evidence of Kazakhstan
performance. Five-fold random CV produced ROC-AUC 0.859, while 1-degree spatial
group CV produced 0.771. The 0.089 gap indicates that nearby wells share spatial
structure and that random validation is optimistic for geographic transfer.

## 5. Kazakhstan Caveats

The main limitation is domain shift. Texas and Kazakhstan differ in aquifer
types, lithology, structural geology, climate and snow regime, land use,
well-construction practice, test methods and sampling patterns. The 20 gpm
threshold is an operational Texas proof-of-concept label, not automatically the
right Kazakhstan target.

Before operational use:

- define a Kazakhstan-relevant yield or success target;
- add national geology, fault, soil and aquifer evidence;
- compare feature distributions and flag extrapolation;
- validate using spatially separated Kazakhstan wells;
- recalibrate probabilities with local labels;
- report model uncertainty and data-quality masks;
- verify high-ranked sites through hydrogeology, geophysics and drilling.

## 6. Further Development

### Improve XGBoost

1. Use nested spatial cross-validation for unbiased tuning and evaluation.
2. Expand beyond the first chronological slice to geographically balanced data.
3. Add multi-radius and seasonal predictors selectively.
4. Add geology, faults, soils, snow and longer-term climate normals.
5. Tune with Bayesian optimization and compare LightGBM, CatBoost and RF.
6. Calibrate probabilities and optimize a threshold for field-survey cost.
7. Use SHAP, permutation importance and ablation studies to debug reliance on
   unstable features.
8. Build ensembles and quantify prediction disagreement.

### Computer Vision and Deep Learning

CNNs or vision transformers can process spatial image patches rather than only
buffer summaries. Multi-channel patches can include Sentinel-1 VV/VH,
Sentinel-2 bands or indices, DEM derivatives, drainage and geology. These models
may learn lineaments, drainage texture, landforms, floodplain shape and spatial
context that mean/median statistics discard.

A practical route is multimodal fusion:

1. Train a CNN or pretrained geospatial encoder on image patches.
2. Export a compact embedding for each well or grid cell.
3. Concatenate the embedding with climate, geology and tabular features.
4. Compare XGBoost on embeddings, a neural late-fusion model and the original
   tabular baseline using identical spatial folds.

Deep learning should follow broader label collection. With only 5,000 spatially
clustered examples, a strong regularized tree baseline is more defensible than
an unconstrained CNN trained from scratch.

## 7. Intended Final Product

The Kazakhstan inference layer should use a regular grid, initially 500 m x
500 m for a small study territory. Every cell receives the same feature schema
as the training wells and a model score such as `P(yield >= operational
threshold)`. The deliverable should include:

- prospectivity score;
- selected decision class or priority tier;
- feature-availability and QA flags;
- out-of-distribution score;
- model/ensemble uncertainty;
- versioned feature and model metadata.

The heatmap is then used to prioritize diverse, high-value field locations. New
local observations should be added through active learning, gradually turning
the Texas proof of concept into a Kazakhstan-calibrated decision-support model.

## References

1. [UNDP Kazakhstan: The climate change impact on water resources in Kazakhstan](https://www.undp.org/kazakhstan/stories/climate-change-impact-water-resources-kazakhstan)
2. [Government of Kazakhstan: A Concept for the Integrated Use of Groundwater Is Being Developed](https://primeminister.kz/en/news/a-concept-for-the-integrated-use-of-groundwater-is-being-developed-in-kazakhstan-30953)
3. [Texas Water Development Board: Locating a Water Well Report](https://www.twdb.texas.gov/groundwater/data/locwwrep.asp)
4. [Google Earth Engine Data Catalog](https://developers.google.com/earth-engine/datasets/)
5. [Google Earth Engine Sentinel-2 documentation](https://developers.google.com/earth-engine/datasets/catalog/sentinel-2/)
6. [Chen and Guestrin (2016): XGBoost](https://doi.org/10.1145/2939672.2939785)

Project evidence is recorded in `reports/current_gee_feature_dictionary.md`,
`reports/yield20_xgboost_metrics.json`, and
`reports/yield20_spatial_generalization_5000.json`.
