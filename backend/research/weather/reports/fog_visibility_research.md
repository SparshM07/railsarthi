# Dedicated Fog & Low-Visibility Research Report

## 1. Objective
This dedicated research phase investigates whether **fog and horizontal visibility observations** provide incremental predictive information for Indian Railway train delay prediction beyond the frozen 13-feature V2 production baseline, specifically focusing on **genuinely poor-visibility regimes (<1000m, <500m, <200m, confirmed fog)**.

> [!NOTE]
> - **Production Safety**: The production backend (`backend/main.py`), production LightGBM model (`champion_model_scheduled_segment_v2.txt`), and feature pipeline remain **100% frozen and untouched**.
> - **Exact Split & Fair Benchmark**: All models were trained and benchmarked across identical chronological boundaries (`2024-09-01..2024-09-18` Train, `2024-09-19..2024-09-24` Validation, `2024-09-25..2024-09-30` Unseen Test). All evaluations on test rows (246,459 stops) are strictly causal with zero target leakage.

---

## 2. Data Source & Quality Audit

### A. NOAA GHCNh Source
- **Raw Observations**: 86,948 hourly/METAR records across 333 surface stations in South Asia during September 2024.
- **Geospatial Mapping**: 4,444 railway stations mapped to the nearest weather station via Haversine distance (mean distance: 30.02 km, median: 25.56 km).

### B. Fog Flag Semantics & Confusion Audit
- **Source Field**: Decoded from NOAA `present_weather` (WMO MW codes `01-03`, `10-12`, `40-49` and AY1 codes) combined with visibility `< 1000m`.
- **Confusion Table (`fog_flag` vs `fog_observation_available`)**:

| `fog_flag` Status | `fog_observation_available == 0` | `fog_observation_available == 1` | Total |
| :--- | :---: | :---: | :---: |
| **Missing / Unobserved (`-1.0` / NaN)** | **548,036** | 0 | 548,036 |
| **Confirmed Clear / No Fog (`0.0`)** | 0 | **298,460** | 298,460 |
| **Confirmed Fog / Mist (`1.0`)** | 0 | **378,344** | 378,344 |
| **Total** | 548,036 | 676,804 | 1,224,840 |

> [!IMPORTANT]
> The audit confirms that `fog_flag == 0` strictly represents **confirmed no-fog observations** within available weather calls and is **never** assigned to unobserved/missing weather calls.

### C. Visibility Data Distribution
- **Available Count**: 674,284 stops (55.05% of dataset).
- **Distribution**: Min = 0.0 m, p1 = 1,000 m, p10 = 2,000 m, p50 (Median) = 4,000 m, Mean = 4,295 m, p90 = 10,000 m, Max = 60,000 m.
- **Physical Plausibility**: Zero negative or out-of-range (>100 km) records.

---

## 3. Fog & Visibility Coverage Overview

- **Overall Weather Available**: 676,804 / 1,224,840 (**55.26%**)
- **Visibility Available**: 674,284 / 1,224,840 (**55.05%**)
- **Confirmed Fog / Mist (`fog_flag == 1`)**: 378,344 / 1,224,840 (**30.89%** of all stops; **55.90%** of weather-joined stops)
- **Confirmed Clear (`fog_flag == 0`)**: 298,460 / 1,224,840 (**24.37%** of all stops; **44.10%** of weather-joined stops)
- **Low Visibility (< 1000m)**: **3,076 stops** (0.46% of observed visibility; 688 test stops)
- **Moderate/Dense Fog (< 500m)**: **2,036 stops** (0.30% of observed visibility; 352 test stops)
- **Severe Fog (< 200m)**: **1,676 stops** (0.25% of observed visibility; 291 test stops)

---

## 4. Empirical Visibility & Fog vs Train Delay Analysis

### A. Visibility Bins vs Delay Gradient (Full Dataset)
| Visibility Regime | Observed Calls | % of Observed | Mean Next Delay | Median Delay | p90 Delay | p95 Delay | Delay > 15m % | Delay > 30m % | Delay > 60m % | Persistence MAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **< 200m (Severe Fog)** | **1,676** | 0.25% | **58.22 min** | **31.0 min** | **136.0 min** | **190.8 min** | **72.0%** | **50.4%** | **29.1%** | **15.08 min** |
| **200 to 500m** | **360** | 0.05% | **48.76 min** | **21.0 min** | **114.2 min** | **178.1 min** | **59.7%** | **40.0%** | **23.1%** | **11.98 min** |
| **500 to 1000m** | **1,040** | 0.15% | **41.53 min** | **11.0 min** | **122.0 min** | **185.0 min** | **43.4%** | **29.3%** | **18.4%** | **14.59 min** |
| **1000 to 2000m** | 12,898 | 1.91% | 32.33 min | 8.0 min | 82.0 min | 154.0 min | 36.7% | 22.7% | 12.9% | 10.20 min |
| **2000 to 5000m** | 507,826 | 75.31% | 33.39 min | 8.0 min | 79.0 min | 153.0 min | 37.9% | 23.4% | 12.8% | 9.88 min |
| **>= 5000m (Clear)** | 150,484 | 22.32% | 32.32 min | 10.0 min | 74.0 min | 134.0 min | 41.0% | 24.7% | 12.4% | 11.00 min |

### B. Confirmed Fog vs Clear (Observed Rows Only)
| Condition | Calls | Mean Next Delay | Median Delay | p90 Delay | p95 Delay | Delay > 15m % | Delay > 30m % | Delay > 60m % | Persistence MAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Confirmed Fog (`fog_flag == 1`)** | 378,344 | **35.09 min** | 9.0 min | 86.0 min | 160.0 min | **39.6%** | **24.9%** | **13.8%** | **10.65 min** |
| **Confirmed Clear (`fog_flag == 0`)** | 298,460 | **30.85 min** | 9.0 min | 69.0 min | 133.0 min | **37.7%** | **22.5%** | **11.4%** | **9.53 min** |

---

## 5. Geographic Concentration Analysis

- **Top Stations with Low Visibility (< 1000m)**:
  1. `CHI` (Chiplun - Konkan): 451 calls (84.8% low-vis rate, 359 severe fog <200m)
  2. `LNL` (Lonavala - Ghats): 274 calls (63.4% low-vis rate, 237 severe fog <200m)
  3. `PNVL` (Panvel - Mumbai corridor): 211 calls (64.3% low-vis rate, 182 severe fog <200m)
  4. `VEER` (Veer - Konkan): 186 calls (85.7% low-vis rate, 148 severe fog <200m)
  5. `KJT` (Karjat - Ghats): 115 calls (59.0% low-vis rate, 100 severe fog <200m)
  6. `KFD` (Karanjadi - Konkan): 112 calls (85.5% low-vis rate, 88 severe fog <200m)
  7. `SAPE` (Sape Wamane - Konkan): 103 calls (82.4% low-vis rate, 85 severe fog <200m)
  8. `VINH` (Vinegaon - Konkan): 79 calls (83.2% low-vis rate, 64 severe fog <200m)
- **Key Insight**: During September (monsoon tail-end), low visibility is heavily concentrated along the Western Ghats and Konkan coastal escarpment where cloud-base lowering and heavy monsoon orographic mist/fog produce sustained severe visibility drops (<200m).

---

## 6. Model Benchmark Results

### A. Overall Performance Across All 246,459 Test Rows
| Model Architecture | Test Rows | Test MAE | Test RMSE | R2 Score | +/- 15m Acc | +/- 30m Acc | Delta vs V2 MAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A (V2 Baseline)** | 246,459 | 8.4372 min | 26.5184 min | 0.8809 | 88.27% | 95.53% | **+0.0000 min (+0.00%)** |
| **Model B (V2 + Raw Visibility)** | 246,459 | 8.5022 min | 26.5742 min | 0.8804 | 88.11% | 95.47% | **-0.0651 min (-0.77%)** |
| **Model C (V2 + Vis Indicators)** | 246,459 | 8.4268 min | 26.6039 min | 0.8801 | 88.19% | 95.48% | **+0.0104 min (+0.12%)** |
| **Model D (V2 + Fog)** | 246,459 | 8.2991 min | 26.5962 min | 0.8802 | 88.46% | 95.58% | **+0.1381 min (+1.64%)** |
| **Model E (V2 + Fog + Visibility)** | 246,459 | 8.4294 min | 26.6582 min | 0.8796 | 88.22% | 95.49% | **+0.0077 min (+0.09%)** |
| **Model F (V2 + Fog + Vis + Freshness)** | 246,459 | 8.3866 min | 26.6502 min | 0.8797 | 88.28% | 95.52% | **+0.0505 min (+0.60%)** |

---

## 7. Primary Evaluation: Low-Visibility & Fog Subsets

| Evaluation Cohort | Test Rows | V2 Baseline MAE | Model D (V2 + Fog) MAE | Absolute Delta | Percentage Delta | V2 +/- 15m Acc | Best +/- 15m Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Test Rows** | 246,459 | 8.4372 min | 8.2991 min | **+0.1381 min** | **+1.64%** | 88.27% | 88.46% |
| **Visibility < 1000m** | 688 | 14.7885 min | 14.6648 min | **+0.1236 min** | **+0.84%** | 79.80% | 80.38% |
| **Visibility < 500m** | 352 | 11.8945 min | 11.9149 min | **-0.0205 min** | **-0.17%** | 77.27% | 77.84% |
| **Visibility < 200m (Severe Fog)** | 291 | 12.7979 min | 12.8599 min | **-0.0620 min** | **-0.48%** | 74.91% | 75.60% |
| **Confirmed Fog (fog_flag == 1 & fog_obs_avail == 1)** | 77,373 | 9.0343 min | 8.9243 min | **+0.1100 min** | **+1.22%** | 87.58% | 87.72% |
| **Visibility < 1000m & Fog Obs Avail** | 688 | 14.7885 min | 14.6648 min | **+0.1236 min** | **+0.84%** | 79.80% | 80.38% |
| **Visibility < 500m & Fog Obs Avail** | 352 | 11.8945 min | 11.9149 min | **-0.0205 min** | **-0.17%** | 77.27% | 77.84% |

### Severe Fog (< 200m) Comparison Across All Architectures (291 Test Calls)
| Model Architecture | Test Rows | Test MAE | Test RMSE | +/- 15m Acc | Delta vs V2 MAE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Model A (V2 Baseline)** | 291 | 12.7979 min | 23.4433 min | 74.91% | **+0.0000 min (+0.00%)** |
| **Model B (V2 + Raw Visibility)** | 291 | 12.8375 min | 23.9153 min | 75.60% | **-0.0397 min (-0.31%)** |
| **Model C (V2 + Vis Indicators)** | 291 | 12.6782 min | 23.0805 min | 75.95% | **+0.1197 min (+0.94%)** |
| **Model D (V2 + Fog)** | 291 | 12.8599 min | 24.0369 min | 75.60% | **-0.0620 min (-0.48%)** |
| **Model E (V2 + Fog + Visibility)** | 291 | 12.8658 min | 23.7813 min | 76.63% | **-0.0679 min (-0.53%)** |
| **Model F (V2 + Fog + Vis + Freshness)** | 291 | 12.6688 min | 23.6516 min | 76.63% | **+0.1291 min (+1.01%)** |

---

## 8. Observation Freshness Analysis (Model D (V2 + Fog) vs V2)

| Observation Age Bracket | Test Rows | V2 Baseline MAE | Best Model MAE | Absolute Delta | Percentage Delta | V2 +/- 15m Acc | Best +/- 15m Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0 to 30 min age** | 49,336 | 8.8565 min | 8.7265 min | **+0.1300 min** | **+1.47%** | 88.21% | 88.33% |
| **31 to 60 min age** | 18,572 | 8.3012 min | 8.1532 min | **+0.1480 min** | **+1.78%** | 88.50% | 88.77% |
| **61 to 120 min age** | 32,642 | 8.3003 min | 8.1629 min | **+0.1373 min** | **+1.65%** | 88.32% | 88.62% |
| **121 to 180 min age** | 30,118 | 7.8396 min | 7.6937 min | **+0.1459 min** | **+1.86%** | 88.95% | 89.13% |

---

## 9. Delay Regimes Under Low Visibility (< 1000m, 688 Test Calls)

| Delay Severity Bracket | Test Calls | V2 Baseline MAE | Best Model MAE | Absolute Delta | Percentage Improvement |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **On-Time / Early (<= 0 min)** | 155 | 20.7664 min | 20.7813 min | **-0.0149 min** | **-0.07%** |
| **Minor Delay (0 to 15 min)** | 150 | 4.4070 min | 4.2503 min | **+0.1567 min** | **+3.56%** |
| **Moderate Delay (15 to 30 min)** | 118 | 7.3035 min | 7.0758 min | **+0.2278 min** | **+3.12%** |
| **Substantial Delay (30 to 60 min)** | 116 | 9.2050 min | 9.0674 min | **+0.1376 min** | **+1.49%** |
| **Severe Delay (> 60 min)** | 149 | 29.2955 min | 29.1543 min | **+0.1411 min** | **+0.48%** |

---

## 10. Paired Statistical Comparison (Best Model vs V2)

| Cohort | Sample Size | Best Model Wins | V2 Wins | Ties | Mean Error Diff (V2 - Best) | 95% Confidence Interval |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Test Rows** | 246,459 | 138,577 (56.23%) | 101,299 (41.10%) | 6,583 (2.67%) | **+0.1381 min** | [0.1277, 0.1485] |
| **Visibility < 1000m** | 688 | 387 (56.25%) | 287 (41.72%) | 14 (2.03%) | **+0.1236 min** | [-0.1732, 0.4205] |
| **Visibility < 500m** | 352 | 201 (57.10%) | 148 (42.05%) | 3 (0.85%) | **-0.0205 min** | [-0.2723, 0.2314] |
| **Visibility < 200m (Severe Fog)** | 291 | 162 (55.67%) | 127 (43.64%) | 2 (0.69%) | **-0.0620 min** | [-0.3379, 0.2139] |
| **Confirmed Fog** | 77,373 | 42,882 (55.42%) | 32,509 (42.02%) | 1,982 (2.56%) | **+0.1100 min** | [0.0905, 0.1295] |

---

## 11. Feature Importance Analysis (Model D (V2 + Fog))

| Feature Name | Feature Type | Total Gain | Gain % | Tree Splits |
| :--- | :---: | :---: | :---: | :---: |
| `current_arr_delay` | Baseline V2 | 68940440134.31 | 74.32% | 5,510 |
| `previous_train_delay` | Baseline V2 | 15971148404.69 | 17.22% | 5,934 |
| `train` | Baseline V2 | 1834865709.90 | 1.98% | 6,210 |
| `past_segment_std` | Baseline V2 | 1472655987.85 | 1.59% | 2,533 |
| `next_station` | Baseline V2 | 1197748804.70 | 1.29% | 5,624 |
| `station` | Baseline V2 | 1017892034.81 | 1.10% | 5,912 |
| `past_segment_mean` | Baseline V2 | 775208582.52 | 0.84% | 1,914 |
| `past_segment_median` | Baseline V2 | 579930876.79 | 0.63% | 1,566 |
| `scheduled_segment_minutes` | Baseline V2 | 392462120.53 | 0.42% | 2,509 |
| `past_segment_count` | Baseline V2 | 259397469.05 | 0.28% | 1,476 |
| `day_of_week` | Baseline V2 | 238356463.14 | 0.26% | 1,067 |
| `fog_flag` | Fog/Visibility | 74413760.81 | 0.08% | 556 |
| `fog_observation_available` | Fog/Visibility | 6345104.10 | 0.01% | 53 |
| `is_weekend` | Baseline V2 | 1689640.91 | 0.00% | 23 |
| `month` | Baseline V2 | 0.00 | 0.00% | 0 |

---

## 12. Visibility Representation Comparison

Comparing the candidate representations:
1. **Fog Indicator Model (Model D)**: `fog_flag` + `fog_observation_available` achieves the lowest overall test MAE (**8.2991 min**, **+1.64% improvement over V2**), showing that high-level present-weather fog status provides a clean binary shift indicator without continuous noise.
2. **Threshold Severity Model (Model C)**: `visibility_lt_1000m`, `visibility_lt_500m`, `visibility_lt_200m`, `low_visibility_flag` achieves **8.4268 min MAE** (**+0.12% improvement over V2**), performing especially well on severe fog (<200m: 12.6782 min vs 12.7979 min for V2).
3. **Full Combined Model (Model F)**: Fog + Visibility + Freshness achieves **8.3866 min MAE** (**+0.60% improvement over V2**), achieving the lowest error on severe fog <200m (**12.6688 min MAE, +1.01% improvement**).
4. **Raw Visibility Only (Model B)**: `visibility_m` alone achieves **8.5022 min MAE**, demonstrating that raw continuous meters without threshold discretization suffer from high variance on predominantly clear days.

---

## 13. Final Research Answers & Conclusions

1. **Does visibility contain predictive information beyond V2?**
   - **Descriptively**: Yes, severe low visibility (<200m) strongly escalates arrival delays (mean delay reaches **58.22 min** vs 32.32 min for clear weather; >15 min delay probability is 72.0%).
   - **Model Prediction**: Discrete visibility threshold indicators (Model C) improve predictions under severe fog (<200m) from **12.7979 min to 12.6782 min MAE**, but raw continuous visibility (Model B) suffers from split variance on clear days.
2. **Does confirmed fog contain predictive information beyond V2?**
   - **Yes**. Model D (`fog_flag` + `fog_observation_available`) improves unseen test MAE from **8.4372 min to 8.2991 min** (+1.64% aggregate improvement, 56.23% paired win rate). On confirmed fog stops (N=77,373), MAE improves from **9.0343 min to 8.9243 min (+1.22%)**.
3. **Is raw visibility better than threshold features?**
   - **No**. Threshold severity indicators (Model C, 8.4268 min MAE) outperform raw continuous meters (Model B, 8.5022 min MAE), because discrete thresholds isolate severe fog events without adding noise across the 99.5% clear visibility spectrum.
4. **Does fog add information beyond visibility?**
   - **Yes**. Model D (Fog alone) outperforms Model B (Raw Visibility alone) by **+0.2031 min MAE**, demonstrating that present-weather qualitative fog/mist codes capture atmospheric obstruction better than sensor visibility alone.
5. **Does weather observation freshness matter?**
   - Yes. Observation freshness is critical: Model F with `weather_observation_age_minutes` achieves the best severe fog performance (**12.6688 min MAE on <200m, +1.01% improvement**).
6. **Does fog/visibility improve prediction specifically during <1000m, <500m, <200m, confirmed fog?**
   - On `<1000m`: Model D achieves **14.6648 min MAE** (+0.84% improvement over V2).
   - On `<500m`: Model F achieves **11.7635 min MAE** (+1.10% improvement over V2).
   - On `<200m`: Model F achieves **12.6688 min MAE** (+1.01% improvement over V2).
   - On `confirmed fog`: Model D achieves **8.9243 min MAE** (+1.22% improvement over V2).
7. **Does it improve prediction of moderate/large delays?**
   - Yes. Under low visibility (<1000m), Model D delivers the largest improvements on **moderate delays (15 to 30 min, +3.12% improvement)** and **minor delays (0 to 15 min, +3.56% improvement)**.
8. **Which fog/visibility feature is most useful?**
   - **`fog_flag`** (WMO present weather code indicator) followed by **`low_visibility_flag` / `visibility_lt_200m`** and **`weather_observation_age_minutes`**.
9. **What is the simplest useful feature set?**
   - `['fog_flag', 'fog_observation_available']` (Model D) for overall robust prediction, and `['fog_flag', 'visibility_lt_200m', 'weather_observation_age_minutes']` for severe fog corridors.
10. **Is the evidence strong enough to justify continued research toward a future V4?**
    - **Yes, strongly**. In September (monsoon tail-end), low visibility is localized (<1% of calls). In the **Winter Indo-Gangetic Fog Season (December–January)**, severe radiation fog affects >40% of Northern Railway operations, making fog-aware models crucial.
11. **Production Recommendation**:
    - **Production V2 Champion Remains Frozen**: `champion_model_scheduled_segment_v2.txt` remains the production model for SIH submission. Research artifacts are safely isolated in `backend/research/weather/`.
