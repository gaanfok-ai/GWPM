# SDR Modeling Readiness Report

This report is focused on modeling usefulness, not only table inventory. It answers how many clean samples exist for target modeling, how recent they are, where they are, and which fields are usable for XGBoost/GEE feature extraction.

## Clean Sample Definition

A `clean_target_rows` sample has valid Texas coordinates, a usable anchor date, no known location error, and a non-null target. Numeric targets additionally require plausible positive or bounded values. A `strict_model_rows` sample also has `ProposedUse`, `Elevation`, water-bearing strata, and lithology aggregates.

Latest anchor year in the SDR-derived dataset: `2026`.
`last_5_years` means `2022-2026`.
`last_10_years` means `2017-2026`.

## Clean Samples by Target and Time Window

| target                 | target_column                       | window        |   year_min |   year_max |   all_rows |   target_non_null_rows |   clean_target_rows |   strict_model_rows |   valid_coordinate_pct_in_window |   strict_model_pct_of_window |
|:-----------------------|:------------------------------------|:--------------|-----------:|-----------:|-----------:|-----------------------:|--------------------:|--------------------:|---------------------------------:|-----------------------------:|
| high_yield_20gpm       | target_high_yield_20gpm             | overall       |       1901 |       2026 |     704078 |                 297908 |              296782 |               48398 |                              100 |                         6.87 |
| high_yield_20gpm       | target_high_yield_20gpm             | last_10_years |       2017 |       2026 |     268514 |                 101919 |              101919 |               27267 |                              100 |                        10.15 |
| high_yield_20gpm       | target_high_yield_20gpm             | last_5_years  |       2022 |       2026 |     123176 |                  47453 |               47453 |               13519 |                              100 |                        10.98 |
| yield_gpm_max          | yield_gpm_max                       | overall       |       1901 |       2026 |     704078 |                 297908 |              292144 |               48021 |                              100 |                         6.82 |
| yield_gpm_max          | yield_gpm_max                       | last_10_years |       2017 |       2026 |     268514 |                 101919 |              100937 |               27118 |                              100 |                        10.1  |
| yield_gpm_max          | yield_gpm_max                       | last_5_years  |       2022 |       2026 |     123176 |                  47453 |               47047 |               13404 |                              100 |                        10.88 |
| static_water_level_ft  | static_water_level_ft_first         | overall       |       1901 |       2026 |     704078 |                 374654 |              373233 |               56564 |                              100 |                         8.03 |
| static_water_level_ft  | static_water_level_ft_first         | last_10_years |       2017 |       2026 |     268514 |                 131008 |              130997 |               31182 |                              100 |                        11.61 |
| static_water_level_ft  | static_water_level_ft_first         | last_5_years  |       2022 |       2026 |     123176 |                  62102 |               62098 |               14770 |                              100 |                        11.99 |
| shallow_water_le_100ft | target_shallow_water_le_100ft       | overall       |       1901 |       2026 |     704078 |                 374654 |              373263 |               56571 |                              100 |                         8.03 |
| shallow_water_le_100ft | target_shallow_water_le_100ft       | last_10_years |       2017 |       2026 |     268514 |                 131008 |              131007 |               31188 |                              100 |                        11.62 |
| shallow_water_le_100ft | target_shallow_water_le_100ft       | last_5_years  |       2022 |       2026 |     123176 |                  62102 |               62101 |               14771 |                              100 |                        11.99 |
| fresh_water_present    | target_fresh_water_present          | overall       |       1901 |       2026 |     704078 |                 253172 |              252492 |               67893 |                              100 |                         9.64 |
| fresh_water_present    | target_fresh_water_present          | last_10_years |       2017 |       2026 |     268514 |                 104054 |              104054 |               36735 |                              100 |                        13.68 |
| fresh_water_present    | target_fresh_water_present          | last_5_years  |       2022 |       2026 |     123176 |                  48846 |               48846 |               17717 |                              100 |                        14.38 |
| specific_capacity      | specific_capacity_gpm_per_ft_median | overall       |       1901 |       2026 |     704078 |                 109099 |              108397 |               17937 |                              100 |                         2.55 |
| specific_capacity      | specific_capacity_gpm_per_ft_median | last_10_years |       2017 |       2026 |     268514 |                  31429 |               31297 |                8510 |                              100 |                         3.17 |
| specific_capacity      | specific_capacity_gpm_per_ft_median | last_5_years  |       2022 |       2026 |     123176 |                  14570 |               14464 |                4312 |                              100 |                         3.5  |

## Main Target Interpretation

`target_high_yield_20gpm` is the recommended first target. It is `True` when `yield_gpm_max >= 20`, `False` when reported yield is below 20, and null when yield is missing. Use only non-null rows for this model.

### High-Yield Label Distribution

| label   |   count |   pct |
|:--------|--------:|------:|
| True    |  197335 | 66.49 |
| False   |   99447 | 33.51 |

### High-Yield Label Distribution by Window

| window        | label   |   count |   pct |
|:--------------|:--------|--------:|------:|
| overall       | True    |  197335 | 66.49 |
| overall       | False   |   99447 | 33.51 |
| last_10_years | True    |   65332 | 64.1  |
| last_10_years | False   |   36587 | 35.9  |
| last_5_years  | True    |   30371 | 64    |
| last_5_years  | False   |   17082 | 36    |

### Yield Distribution

| stat   |           value |
|:-------|----------------:|
| count  | 292144          |
| mean   |     67.7346     |
| std    |   4074.07       |
| min    |      0.01       |
| 1%     |      1.5        |
| 5%     |      5          |
| 10%    |     10          |
| 25%    |     15          |
| 50%    |     30          |
| 75%    |     60          |
| 90%    |    100          |
| 95%    |    150          |
| 99%    |    700          |
| max    |      2.2001e+06 |

### Yield Bins

| bin     |   count |   pct |
|:--------|--------:|------:|
| 0-5     |   11697 |  4    |
| 5-10    |   15935 |  5.45 |
| 10-20   |   67177 | 22.99 |
| 20-50   |   94446 | 32.33 |
| 50-100  |   69660 | 23.84 |
| 100-500 |   28447 |  9.74 |
| 500+    |    4782 |  1.64 |

### Static Water Level Distribution

| stat   |      value |
|:-------|-----------:|
| count  | 373233     |
| mean   |    121.099 |
| std    |    121.614 |
| min    |      0     |
| 1%     |      5     |
| 5%     |     12     |
| 10%    |     18     |
| 25%    |     40     |
| 50%    |     80     |
| 75%    |    160     |
| 90%    |    283     |
| 95%    |    370     |
| 99%    |    555     |
| max    |   1989     |

### Static Water Level Bins

| bin      |   count |   pct |
|:---------|--------:|------:|
| 0-25     |   56078 | 15.02 |
| 25-50    |   61068 | 16.36 |
| 50-100   |   98366 | 26.36 |
| 100-200  |   86043 | 23.05 |
| 200-500  |   65530 | 17.56 |
| 500-1000 |    5896 |  1.58 |
| 1000+    |     252 |  0.07 |

## Coordinate Quality

| metric                                     |   value |
|:-------------------------------------------|--------:|
| total_rows                                 |  704078 |
| coordinate_present_rows                    |  704078 |
| valid_texas_bbox_rows                      |  704076 |
| unique_exact_coordinate_pairs              |  578504 |
| rows_with_duplicated_exact_coordinate_pair |  167212 |
| unique_coordinate_pairs_rounded_4_decimals |  571619 |
| unique_coordinate_pairs_rounded_3_decimals |  493519 |
| rounded_whole_degree_coordinate_rows       |       2 |

## Feature Completeness

| feature                      |   non_null_rows |   coverage_pct |   unique_values |
|:-----------------------------|----------------:|---------------:|----------------:|
| County                       |          704077 |         100    |             254 |
| ProposedUse                  |          704077 |         100    |              16 |
| TypeOfWork                   |          704060 |         100    |               7 |
| drilling_method_first        |          703976 |          99.99 |              12 |
| borehole_max_bottom_ft       |          703665 |          99.94 |            4175 |
| lith_clay_thickness_ft       |          700302 |          99.46 |            3641 |
| lith_gravel_thickness_ft     |          700302 |          99.46 |            1679 |
| lith_sand_thickness_ft       |          700302 |          99.46 |            3955 |
| lithology_count              |          700302 |          99.46 |             105 |
| filter_total_thickness_ft    |          314370 |          44.65 |            2141 |
| water_bearing_interval_count |          253172 |          35.96 |              14 |
| PumpDepth                    |          216592 |          30.76 |            1205 |
| Elevation                    |          140416 |          19.94 |            4879 |

## Top Counties for High-Yield Model Samples

| County     |   rows |   target_median |   lat_min |   lat_max |   lon_min |   lon_max |
|:-----------|-------:|----------------:|----------:|----------:|----------:|----------:|
| Parker     |  11800 |              15 |   30.9086 |   33.7258 |  -98.1339 |  -97.0833 |
| Montgomery |  10785 |              40 |   29.8014 |   33.2839 |  -96.7942 |  -94.9975 |
| Harris     |   9490 |              40 |   28.9089 |   33.05   |  -99.5372 |  -94.2983 |
| Brazoria   |   9124 |              60 |   28.6422 |   31.4025 |  -97.4264 |  -94.6658 |
| Tarrant    |   5447 |              15 |   32.0403 |   33.1278 |  -98.0211 |  -96.1122 |
| Fort Bend  |   5005 |              60 |   29.2628 |   30.2342 |  -96.9581 |  -95.0328 |
| Midland    |   4931 |              18 |   31.0264 |   32.9631 | -103.976  | -101.644  |
| Wise       |   4516 |              15 |   32.0347 |   33.5117 |  -98.1775 |  -97.3453 |
| Washington |   4127 |              40 |   29.8417 |   30.8478 |  -96.8383 |  -95.6947 |
| Ector      |   4061 |              15 |   31.4819 |   32.8286 | -103.279  | -101.984  |
| Randall    |   4041 |              14 |   34.7333 |   35.2431 | -102.669  | -101.037  |
| Burnet     |   3900 |              20 |   30.4206 |   31.5317 |  -98.9739 |  -97.2158 |
| Lavaca     |   3824 |              50 |   28.1311 |   29.9125 |  -97.9569 |  -96.5394 |
| Gillespie  |   3694 |              20 |   30.0983 |   30.6306 |  -99.8783 |  -95.7581 |
| Tom Green  |   3592 |              20 |   31.0864 |   31.6978 | -101.316  | -100.111  |
| DeWitt     |   3227 |              60 |   28.0444 |   29.9881 |  -97.9514 |  -96.6333 |
| Waller     |   3199 |              50 |   29.0775 |   30.8292 |  -96.3978 |  -94.975  |
| Fayette    |   3178 |              45 |   29.0667 |   30.6831 |  -97.9833 |  -96.0481 |
| Colorado   |   3059 |              40 |   29.295  |   29.9381 |  -97.6956 |  -95.5878 |
| Austin     |   3015 |              40 |   28.9242 |   30.3708 |  -97.3839 |  -95.14   |
| Travis     |   3010 |              25 |   29.1372 |   30.6347 |  -98.6306 |  -97.0625 |
| Victoria   |   2952 |              50 |   28.2092 |   30.5933 |  -98.7647 |  -96.3197 |
| Erath      |   2849 |              20 |   31.2261 |   33.0733 |  -98.7342 |  -97.1294 |
| Llano      |   2757 |               5 |   30.49   |   30.9239 |  -98.9767 |  -98.0167 |
| Harrison   |   2727 |              55 |   32.3167 |   32.8956 |  -96.4342 |  -94.0433 |

## Top 1-Degree Grid Cells for High-Yield Model Samples

|   lat_bin_1deg |   lon_bin_1deg |   rows |
|---------------:|---------------:|-------:|
|             30 |            -96 |  23905 |
|             33 |            -98 |  20814 |
|             30 |            -95 |  15015 |
|             32 |           -102 |  12479 |
|             30 |            -97 |  11552 |
|             30 |            -98 |  11086 |
|             29 |            -97 |  10706 |
|             30 |            -99 |  10235 |
|             29 |            -96 |   9325 |
|             31 |            -98 |   8502 |
|             33 |            -97 |   8057 |
|             32 |            -98 |   7957 |
|             29 |            -98 |   7904 |
|             32 |            -95 |   7833 |
|             31 |            -96 |   7499 |
|             29 |            -95 |   7350 |
|             28 |            -98 |   6157 |
|             32 |            -96 |   5720 |
|             35 |           -102 |   5644 |
|             32 |            -94 |   5349 |
|             31 |            -99 |   4428 |
|             33 |            -95 |   4424 |
|             31 |            -95 |   4382 |
|             32 |           -101 |   4212 |
|             31 |           -100 |   4071 |

## Important Categorical Distributions for High-Yield Samples

### Proposed Use

| ProposedUse               |   count |   pct |
|:--------------------------|--------:|------:|
| Domestic                  |  199091 | 67.08 |
| Irrigation                |   28322 |  9.54 |
| Stock                     |   25125 |  8.47 |
| Rig Supply                |   23002 |  7.75 |
| Industrial                |    6911 |  2.33 |
| Public Supply             |    5094 |  1.72 |
| Fracking Supply           |    3585 |  1.21 |
| Test Well                 |    2805 |  0.95 |
| Monitor                   |     992 |  0.33 |
| Other                     |     976 |  0.33 |
| De-watering               |     596 |  0.2  |
| Unknown                   |     108 |  0.04 |
| Injection                 |      75 |  0.03 |
| Closed-Loop Geothermal    |      59 |  0.02 |
| Environmental Soil Boring |      36 |  0.01 |
| Extraction                |       5 |  0    |

### Drilling Method

| drilling_method_first   |   count |   pct |
|:------------------------|--------:|------:|
| Mud (Hydraulic) Rotary  |  181606 | 61.19 |
| Air Rotary              |   92457 | 31.15 |
| Air Hammer              |   16935 |  5.71 |
| Reverse Circulation     |    2433 |  0.82 |
| Cable Tool              |    1312 |  0.44 |
| Other                   |    1101 |  0.37 |
| Jetted                  |     375 |  0.13 |
| Bored                   |     259 |  0.09 |
| Hollow Stem Auger       |     132 |  0.04 |
| Driven                  |      69 |  0.02 |
| Unknown                 |      65 |  0.02 |
| <missing>               |      22 |  0.01 |
| Direct Push             |      16 |  0.01 |

## Practical Conclusions

- The first XGBoost dataset should use `target_high_yield_20gpm` because it has a large labeled set and gives a clear heatmap probability.
- Keep `yield_gpm_max` as a second regression target using `log1p(yield_gpm_max)`.
- Keep `static_water_level_ft_first` as a separate depth-to-water regression target.
- For GEE extraction, prioritize wells with valid coordinates and anchor dates. Use the anchor date to extract previous-12-month Sentinel/Landsat/climate features.
- Use spatial validation by county or grid block. Random validation will likely overestimate performance because wells are spatially clustered.
- Do not treat missing target rows as negative samples. Missing yield, missing water level, or missing strata means unknown, not bad.
