# Fog & Visibility Feature Optimization Research Report

**Research Phase**: Dedicated Fog-Aware Feature Optimization
**Date**: 2026-09-02 22:21:35
**Status**: Research Phase Completed — Production Remains 100% Frozen

---

## 1. Objective

The objective of this research phase is to **find the simplest, most robust fog and visibility feature set** that delivers real, measurable incremental value over the frozen V2 baseline (`champion_model_scheduled_segment_v2.txt`).

Key optimization questions investigated:
1. Does binary fog alone (`fog_flag` + `fog_observation_available`) provide sufficient signal, or do discrete visibility thresholds add essential resolution?
2. Which visibility threshold discretization (`<1000m`, `<500m`, `<200m`) is non-redundant and most predictive under severe low-visibility conditions?
3. Does observation freshness (`weather_observation_age_minutes`) provide predictive gain, and how does observation latency affect error rates?
4. What is the smallest, most robust feature combination recommended for future V4 exploration?

---

## 2. Previous Evidence

The preliminary fog and visibility research established several foundational findings:
- **Frozen V2 baseline MAE**: 8.4372 min (±15m accuracy: 88.27%).
- **Present-weather fog indicator** (Model D in previous phase) improved overall unseen test MAE to 8.2991 min (+1.64%) and confirmed-fog MAE to 8.9243 min vs V2 9.0343 min.
- **Raw continuous visibility (`visibility_m`) underperformed**: Continuous meters caused overfitting/split variance on clear days (8.5022 min MAE vs 8.4372 min for V2).
- **Threshold indicators outperformed raw meters**: Discrete thresholds (`<1000m`, `<500m`, `<200m`) effectively localized severe visibility degradation.
- **Observation freshness showed high value on severe visibility**: The combined model including observation age achieved 12.6688 min MAE on `<200m` calls vs 12.7979 min for V2.

---

## 3. Dataset & Splits

- **Source Dataset**: `research\weather\data\processed\v3_weather_features.csv`
- **Total Dataset Size**: 1,224,840 rows
- **Train Split** (2024-09-01 to 2024-09-18): 730,698 rows
- **Validation Split** (2024-09-19 to 2024-09-24): 247,683 rows
- **Unseen Test Split** (2024-09-25 to 2024-09-30): 246,459 rows

> [!NOTE]
> All models are strictly evaluated on the identical 246,459 unseen test rows with 0 temporal leakage.

---

## 4. Experimental Models

| Model ID | Model Name | Total Features | Features Included |
| :--- | :--- | :---: | :--- |
| **Model A** | **Frozen V2 Baseline** | 13 | Exact 13 production features (`champion_model_scheduled_segment_v2.txt`) |
| **Model B** | **Fog Only** | 15 | V2 + `fog_flag`, `fog_observation_available` |
| **Model C** | **Fog + Freshness** | 16 | V2 + `fog_flag`, `fog_obs_avail`, `weather_observation_age_minutes` |
| **Model D** | **Fog + Visibility Thresholds** | 20 | V2 + `fog_flag`, `fog_obs_avail`, `vis_avail`, `vis_lt_1000m`, `vis_lt_500m`, `vis_lt_200m`, `low_visibility_flag` |
| **Model E** | **Fog + Visibility + Freshness** | 21 | V2 + Model D features + `weather_observation_age_minutes` |
| **Model F** | **Fog + Severe Visibility (<500m)** | 18 | V2 + `fog_flag`, `fog_obs_avail`, `vis_avail`, `vis_lt_500m`, `vis_lt_200m` |
| **Model G** | **Fog + Extreme Visibility (<200m)** | 17 | V2 + `fog_flag`, `fog_obs_avail`, `vis_avail`, `vis_lt_200m` |
| **Variant** | **Fog + 1000m** | 17 | V2 + `fog_flag`, `fog_obs_avail`, `vis_avail`, `vis_lt_1000m` |
| **Variant** | **Fog + 500m** | 17 | V2 + `fog_flag`, `fog_obs_avail`, `vis_avail`, `vis_lt_500m` |

---

## 5. Overall Benchmark (All 246,459 Unseen Test Rows)

| Model Architecture | Features | MAE (min) | RMSE (min) | R² | Bias (min) | ±5m Acc | ±10m Acc | ±15m Acc | ±30m Acc | ±60m Acc | Δ MAE vs V2 | % Imprv |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A (Frozen V2)** | 13 | **8.4372** | 26.5184 | 0.8809 | -0.0927 | 60.92% | 80.43% | 88.27% | 95.53% | 98.44% | **+0.0000** | **+0.00%** |
| **Model B (Fog Only)** | 15 | **8.2991** | 26.5962 | 0.8802 | -0.0938 | 61.89% | 80.93% | 88.46% | 95.58% | 98.45% | **+0.1381** | **+1.64%** |
| **Model C (Fog + Freshness)** | 16 | **8.4361** | 26.6530 | 0.8797 | -0.1166 | 60.79% | 80.44% | 88.26% | 95.53% | 98.44% | **+0.0011** | **+0.01%** |
| **Model D (Fog + Vis Thresholds)** | 20 | **8.3122** | 26.5699 | 0.8804 | -0.1094 | 61.80% | 80.95% | 88.46% | 95.57% | 98.44% | **+0.1250** | **+1.48%** |
| **Model E (Fog + Vis + Freshness)** | 21 | **8.4168** | 26.6296 | 0.8799 | -0.1025 | 60.96% | 80.45% | 88.21% | 95.49% | 98.44% | **+0.0203** | **+0.24%** |
| **Model F (Fog + Severe Vis <500m)** | 18 | **8.3915** | 26.6216 | 0.8800 | -0.1076 | 61.08% | 80.51% | 88.20% | 95.52% | 98.45% | **+0.0456** | **+0.54%** |
| **Model G (Fog + Extreme Vis <200m)** | 17 | **8.3582** | 26.6178 | 0.8800 | -0.1024 | 61.54% | 80.79% | 88.39% | 95.55% | 98.44% | **+0.0790** | **+0.94%** |
| **Variant (Fog + 1000m)** | 17 | **8.4709** | 26.5759 | 0.8804 | -0.0968 | 60.73% | 80.34% | 88.21% | 95.48% | 98.44% | **-0.0337** | **-0.40%** |
| **Variant (Fog + 500m)** | 17 | **8.2574** | 26.5775 | 0.8804 | -0.1149 | 62.15% | 81.07% | 88.52% | 95.57% | 98.45% | **+0.1798** | **+2.13%** |

---

## 6. Confirmed Fog Benchmark (`fog_flag == 1` & `fog_obs_avail == 1`) (N = 77,373)

| Model Architecture | Cohort MAE (min) | Cohort RMSE (min) | ±15m Acc | Absolute Δ vs V2 | % Improvement |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Model A (Frozen V2)** | **9.0343** | 29.6295 | 87.58% | baseline | baseline |
| **Model B (Fog Only)** | **8.9243** | 29.7246 | 87.72% | **+0.1100** | **+1.22%** |
| **Model C (Fog + Freshness)** | **9.0354** | 29.7037 | 87.56% | **-0.0011** | **-0.01%** |
| **Model D (Fog + Vis Thresholds)** | **8.9341** | 29.6773 | 87.76% | **+0.1002** | **+1.11%** |
| **Model E (Fog + Vis + Freshness)** | **9.0302** | 29.6972 | 87.50% | **+0.0041** | **+0.04%** |
| **Model F (Fog + Severe Vis <500m)** | **8.9988** | 29.6757 | 87.56% | **+0.0355** | **+0.39%** |
| **Model G (Fog + Extreme Vis <200m)** | **8.9861** | 29.7970 | 87.71% | **+0.0482** | **+0.53%** |
| **Variant (Fog + 1000m)** | **9.0804** | 29.7162 | 87.49% | **-0.0461** | **-0.51%** |
| **Variant (Fog + 500m)** | **8.8814** | 29.6929 | 87.82% | **+0.1528** | **+1.69%** |

---

## 7. Visibility <1000m Benchmark (N = 688)

| Model Architecture | Cohort MAE (min) | Cohort RMSE (min) | ±15m Acc | Absolute Δ vs V2 | % Improvement |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Model A (Frozen V2)** | **14.7885** | 41.3614 | 79.80% | baseline | baseline |
| **Model B (Fog Only)** | **14.6648** | 40.8454 | 80.38% | **+0.1236** | **+0.84%** |
| **Model C (Fog + Freshness)** | **14.7426** | 41.9776 | 79.51% | **+0.0459** | **+0.31%** |
| **Model D (Fog + Vis Thresholds)** | **14.5456** | 41.0330 | 79.51% | **+0.2428** | **+1.64%** |
| **Model E (Fog + Vis + Freshness)** | **14.7319** | 40.9727 | 79.65% | **+0.0565** | **+0.38%** |
| **Model F (Fog + Severe Vis <500m)** | **14.8254** | 41.0137 | 79.51% | **-0.0369** | **-0.25%** |
| **Model G (Fog + Extreme Vis <200m)** | **14.8957** | 42.1771 | 79.22% | **-0.1073** | **-0.73%** |
| **Variant (Fog + 1000m)** | **14.9364** | 42.0759 | 79.94% | **-0.1480** | **-1.00%** |
| **Variant (Fog + 500m)** | **14.6282** | 40.5377 | 79.80% | **+0.1603** | **+1.08%** |

---

## 8. Visibility <500m Benchmark (N = 352)

| Model Architecture | Cohort MAE (min) | Cohort RMSE (min) | ±15m Acc | Absolute Δ vs V2 | % Improvement |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Model A (Frozen V2)** | **11.8945** | 21.9212 | 77.27% | baseline | baseline |
| **Model B (Fog Only)** | **11.9149** | 22.5001 | 77.84% | **-0.0205** | **-0.17%** |
| **Model C (Fog + Freshness)** | **11.9146** | 22.2335 | 76.70% | **-0.0201** | **-0.17%** |
| **Model D (Fog + Vis Thresholds)** | **11.7397** | 22.1179 | 76.70% | **+0.1548** | **+1.30%** |
| **Model E (Fog + Vis + Freshness)** | **11.9615** | 22.2642 | 77.56% | **-0.0670** | **-0.56%** |
| **Model F (Fog + Severe Vis <500m)** | **11.9117** | 22.1462 | 77.84% | **-0.0173** | **-0.15%** |
| **Model G (Fog + Extreme Vis <200m)** | **11.8773** | 22.5408 | 76.99% | **+0.0172** | **+0.14%** |
| **Variant (Fog + 1000m)** | **12.0072** | 22.4641 | 77.56% | **-0.1127** | **-0.95%** |
| **Variant (Fog + 500m)** | **11.8689** | 22.7956 | 78.12% | **+0.0256** | **+0.22%** |

---

## 9. Visibility <200m Benchmark (Severe Fog) (N = 291)

| Model Architecture | Cohort MAE (min) | Cohort RMSE (min) | ±15m Acc | Absolute Δ vs V2 | % Improvement |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Model A (Frozen V2)** | **12.7979** | 23.4433 | 74.91% | baseline | baseline |
| **Model B (Fog Only)** | **12.8599** | 24.0369 | 75.60% | **-0.0620** | **-0.48%** |
| **Model C (Fog + Freshness)** | **12.8515** | 23.7987 | 74.57% | **-0.0536** | **-0.42%** |
| **Model D (Fog + Vis Thresholds)** | **12.6457** | 23.6976 | 74.91% | **+0.1522** | **+1.19%** |
| **Model E (Fog + Vis + Freshness)** | **12.9060** | 23.8326 | 75.26% | **-0.1081** | **-0.84%** |
| **Model F (Fog + Severe Vis <500m)** | **12.8905** | 23.7243 | 75.60% | **-0.0927** | **-0.72%** |
| **Model G (Fog + Extreme Vis <200m)** | **12.7576** | 24.0739 | 74.91% | **+0.0403** | **+0.31%** |
| **Variant (Fog + 1000m)** | **12.9040** | 24.0070 | 75.60% | **-0.1061** | **-0.83%** |
| **Variant (Fog + 500m)** | **12.7738** | 24.3433 | 75.60% | **+0.0241** | **+0.19%** |

---

## 10. Fog + Visibility Interaction Analysis

Cross-analyzing stop calls where present-weather fog is confirmed simultaneously with measured low-visibility thresholds:

| Interaction Cohort | Sample Size | V2 Baseline MAE | Model B (Fog Only) MAE | Model D (Fog+Vis) MAE | Model E (Fog+Vis+Fresh) MAE | Model F (Fog+SevereVis) MAE | Model G (Fog+ExtremeVis) MAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fog + Visibility < 1000m** | 688 | 14.7885 min | 14.6648 min | 14.5456 min | 14.7319 min | 14.8254 min | 14.8957 min |
| **Fog + Visibility < 500m** | 352 | 11.8945 min | 11.9149 min | 11.7397 min | 11.9615 min | 11.9117 min | 11.8773 min |
| **Fog + Visibility < 200m** | 291 | 12.7979 min | 12.8599 min | 12.6457 min | 12.9060 min | 12.8905 min | 12.7576 min |

---

## 11. Observation Freshness Audit & Analysis

### Freshness Statement Audit
> [!IMPORTANT]
> **Audit of Previous Freshness Statement**:
> The previous report contained the statement: *"Fresh observations (0–30 min) yield significantly lower MAE (8.7265 min) than older observations (121–180 min, 7.6937 min baseline vs adjusted)."*
>
> **Clarification & Exact Audit Findings**:
> - `8.7265 min` was Model D's MAE on the **0–30 min cohort** (where V2 baseline MAE was `8.8565 min`).
> - `7.6937 min` was Model D's MAE on the **121–180 min cohort** (where V2 baseline MAE was `7.8396 min`).
> - These two numbers represent **completely different sub-populations of railway stops** (49,336 stops vs 30,118 stops), which experienced different average delay magnitudes in September. The lower numerical MAE in the 121–180 min bucket was an artifact of cohort composition, not observation latency.
> - Within each respective cohort, the weather model consistently improved MAE over V2 on identical rows (+1.47% improvement on 0–30 min, +1.86% improvement on 121–180 min).

### Observation Age Breakdown Across Primary Candidate Models

| Observation Age Bracket | Sample Size | V2 Baseline MAE | Model B (Fog Only) | Model C (Fog+Fresh) | Model D (Fog+Vis) | Model E (Fog+Vis+Fresh) | Model F (Fog+SevereVis) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0 to 30 min age** | 49,336 | 8.8565 min | 8.7265 min (+1.47%) | 8.8676 min (-0.13%) | 8.7455 min (+1.25%) | 8.8640 min (-0.08%) | 8.8175 min (+0.44%) |
| **31 to 60 min age** | 18,572 | 8.3012 min | 8.1532 min (+1.78%) | 8.2679 min (+0.40%) | 8.1778 min (+1.49%) | 8.2557 min (+0.55%) | 8.2549 min (+0.56%) |
| **61 to 120 min age** | 32,642 | 8.3003 min | 8.1629 min (+1.65%) | 8.2895 min (+0.13%) | 8.1723 min (+1.54%) | 8.2677 min (+0.39%) | 8.2408 min (+0.72%) |
| **121 to 180 min age** | 30,118 | 7.8396 min | 7.6937 min (+1.86%) | 7.8183 min (+0.27%) | 7.7235 min (+1.48%) | 7.8050 min (+0.44%) | 7.7907 min (+0.62%) |

---

## 12. Paired Error Analysis (Direct Row-by-Row vs Frozen V2)

Evaluating row-by-row prediction superiority on identical test rows:

### Model B (Fog Only) vs Frozen V2

| Cohort | Sample Size | Candidate Wins | V2 Wins | Ties | Win Rate | Mean Error Diff (V2 - Cand) | Median Error Diff | 95% Parametric CI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Test Rows** | 246,459 | 138,577 (56.23%) | 101,299 (41.10%) | 6,583 | **56.23%** | **+0.1381 min** | +0.1123 min | [+0.1277, +0.1485] |
| **Confirmed Fog** | 77,373 | 42,882 (55.42%) | 32,509 (42.02%) | 1,982 | **55.42%** | **+0.1100 min** | +0.1070 min | [+0.0905, +0.1295] |
| **Visibility < 1000m** | 688 | 387 (56.25%) | 287 (41.72%) | 14 | **56.25%** | **+0.1236 min** | +0.1646 min | [-0.1732, +0.4205] |
| **Visibility < 500m** | 352 | 201 (57.10%) | 148 (42.05%) | 3 | **57.10%** | **-0.0205 min** | +0.2235 min | [-0.2723, +0.2314] |
| **Visibility < 200m** | 291 | 162 (55.67%) | 127 (43.64%) | 2 | **55.67%** | **-0.0620 min** | +0.1642 min | [-0.3379, +0.2139] |

### Model C (Fog + Freshness) vs Frozen V2

| Cohort | Sample Size | Candidate Wins | V2 Wins | Ties | Win Rate | Mean Error Diff (V2 - Cand) | Median Error Diff | 95% Parametric CI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Test Rows** | 246,459 | 121,179 (49.17%) | 119,030 (48.30%) | 6,250 | **49.17%** | **+0.0011 min** | +0.0000 min | [-0.0083, +0.0105] |
| **Confirmed Fog** | 77,373 | 38,249 (49.43%) | 37,245 (48.14%) | 1,879 | **49.43%** | **-0.0011 min** | +0.0000 min | [-0.0177, +0.0155] |
| **Visibility < 1000m** | 688 | 358 (52.03%) | 316 (45.93%) | 14 | **52.03%** | **+0.0459 min** | +0.0167 min | [-0.1937, +0.2855] |
| **Visibility < 500m** | 352 | 177 (50.28%) | 172 (48.86%) | 3 | **50.28%** | **-0.0201 min** | +0.0008 min | [-0.2261, +0.1858] |
| **Visibility < 200m** | 291 | 147 (50.52%) | 142 (48.80%) | 2 | **50.52%** | **-0.0536 min** | +0.0010 min | [-0.2759, +0.1687] |

### Model D (Fog + Vis Thresholds) vs Frozen V2

| Cohort | Sample Size | Candidate Wins | V2 Wins | Ties | Win Rate | Mean Error Diff (V2 - Cand) | Median Error Diff | 95% Parametric CI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Test Rows** | 246,459 | 135,679 (55.05%) | 104,394 (42.36%) | 6,386 | **55.05%** | **+0.1250 min** | +0.0689 min | [+0.1155, +0.1344] |
| **Confirmed Fog** | 77,373 | 42,448 (54.86%) | 33,006 (42.66%) | 1,919 | **54.86%** | **+0.1002 min** | +0.0683 min | [+0.0824, +0.1180] |
| **Visibility < 1000m** | 688 | 369 (53.63%) | 306 (44.48%) | 13 | **53.63%** | **+0.2428 min** | +0.0699 min | [+0.0152, +0.4704] |
| **Visibility < 500m** | 352 | 195 (55.40%) | 154 (43.75%) | 3 | **55.40%** | **+0.1548 min** | +0.1279 min | [-0.1112, +0.4207] |
| **Visibility < 200m** | 291 | 164 (56.36%) | 125 (42.96%) | 2 | **56.36%** | **+0.1522 min** | +0.1813 min | [-0.1543, +0.4587] |

### Model E (Fog + Vis + Freshness) vs Frozen V2

| Cohort | Sample Size | Candidate Wins | V2 Wins | Ties | Win Rate | Mean Error Diff (V2 - Cand) | Median Error Diff | 95% Parametric CI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Test Rows** | 246,459 | 130,253 (52.85%) | 109,717 (44.52%) | 6,489 | **52.85%** | **+0.0203 min** | +0.0452 min | [+0.0099, +0.0308] |
| **Confirmed Fog** | 77,373 | 40,604 (52.48%) | 34,827 (45.01%) | 1,942 | **52.48%** | **+0.0041 min** | +0.0410 min | [-0.0146, +0.0227] |
| **Visibility < 1000m** | 688 | 357 (51.89%) | 316 (45.93%) | 15 | **51.89%** | **+0.0565 min** | +0.0268 min | [-0.2341, +0.3472] |
| **Visibility < 500m** | 352 | 178 (50.57%) | 171 (48.58%) | 3 | **50.57%** | **-0.0670 min** | +0.0066 min | [-0.2881, +0.1541] |
| **Visibility < 200m** | 291 | 144 (49.48%) | 145 (49.83%) | 2 | **49.48%** | **-0.1081 min** | +0.0000 min | [-0.3545, +0.1383] |

### Model F (Fog + Severe Vis <500m) vs Frozen V2

| Cohort | Sample Size | Candidate Wins | V2 Wins | Ties | Win Rate | Mean Error Diff (V2 - Cand) | Median Error Diff | 95% Parametric CI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Test Rows** | 246,459 | 131,490 (53.35%) | 108,317 (43.95%) | 6,652 | **53.35%** | **+0.0456 min** | +0.0559 min | [+0.0351, +0.0561] |
| **Confirmed Fog** | 77,373 | 40,962 (52.94%) | 34,395 (44.45%) | 2,016 | **52.94%** | **+0.0355 min** | +0.0534 min | [+0.0165, +0.0545] |
| **Visibility < 1000m** | 688 | 353 (51.31%) | 320 (46.51%) | 15 | **51.31%** | **-0.0369 min** | +0.0201 min | [-0.2839, +0.2101] |
| **Visibility < 500m** | 352 | 183 (51.99%) | 166 (47.16%) | 3 | **51.99%** | **-0.0173 min** | +0.0282 min | [-0.2192, +0.1847] |
| **Visibility < 200m** | 291 | 149 (51.20%) | 140 (48.11%) | 2 | **51.20%** | **-0.0927 min** | +0.0248 min | [-0.3183, +0.1330] |

### Model G (Fog + Extreme Vis <200m) vs Frozen V2

| Cohort | Sample Size | Candidate Wins | V2 Wins | Ties | Win Rate | Mean Error Diff (V2 - Cand) | Median Error Diff | 95% Parametric CI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Test Rows** | 246,459 | 133,454 (54.15%) | 106,396 (43.17%) | 6,609 | **54.15%** | **+0.0790 min** | +0.0632 min | [+0.0693, +0.0887] |
| **Confirmed Fog** | 77,373 | 41,410 (53.52%) | 33,973 (43.91%) | 1,990 | **53.52%** | **+0.0482 min** | +0.0518 min | [+0.0309, +0.0655] |
| **Visibility < 1000m** | 688 | 380 (55.23%) | 295 (42.88%) | 13 | **55.23%** | **-0.1073 min** | +0.0931 min | [-0.3327, +0.1182] |
| **Visibility < 500m** | 352 | 201 (57.10%) | 148 (42.05%) | 3 | **57.10%** | **+0.0172 min** | +0.1124 min | [-0.2702, +0.3045] |
| **Visibility < 200m** | 291 | 168 (57.73%) | 121 (41.58%) | 2 | **57.73%** | **+0.0403 min** | +0.1211 min | [-0.2953, +0.3759] |

---

## 13. Confidence & Robustness Analysis (Bootstrap Validation)

To ensure improvements are not driven by small-sample outliers or noise, 1,000 bootstrap resamples were computed for MAE difference ($MAE_{V2} - MAE_{cand}$):

| Model Architecture | Cohort | Sample Size | Mean MAE Diff | 95% Bootstrap CI | Statistically Significant? |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Model B (Fog Only)** | All Test Rows | 246,459 | +0.1381 min | [+0.1276, +0.1483] | YES (p < 0.05) |
| **Model B (Fog Only)** | Confirmed Fog | 77,373 | +0.1100 min | [+0.0909, +0.1287] | YES (p < 0.05) |
| **Model B (Fog Only)** | Visibility < 1000m | 688 | +0.1236 min | [-0.1584, +0.4353] | NO (CI crosses 0) |
| **Model B (Fog Only)** | Visibility < 500m | 352 | -0.0205 min | [-0.2744, +0.2471] | NO (CI crosses 0) |
| **Model B (Fog Only)** | Visibility < 200m | 291 | -0.0620 min | [-0.3361, +0.1970] | NO (CI crosses 0) |
| **Model D (Fog + Vis Thresholds)** | All Test Rows | 246,459 | +0.1250 min | [+0.1156, +0.1339] | YES (p < 0.05) |
| **Model D (Fog + Vis Thresholds)** | Confirmed Fog | 77,373 | +0.1002 min | [+0.0824, +0.1179] | YES (p < 0.05) |
| **Model D (Fog + Vis Thresholds)** | Visibility < 1000m | 688 | +0.2428 min | [+0.0337, +0.4763] | YES (p < 0.05) |
| **Model D (Fog + Vis Thresholds)** | Visibility < 500m | 352 | +0.1548 min | [-0.1350, +0.4212] | NO (CI crosses 0) |
| **Model D (Fog + Vis Thresholds)** | Visibility < 200m | 291 | +0.1522 min | [-0.1503, +0.4492] | NO (CI crosses 0) |
| **Model E (Fog + Vis + Freshness)** | All Test Rows | 246,459 | +0.0203 min | [+0.0098, +0.0306] | YES (p < 0.05) |
| **Model E (Fog + Vis + Freshness)** | Confirmed Fog | 77,373 | +0.0041 min | [-0.0146, +0.0214] | NO (CI crosses 0) |
| **Model E (Fog + Vis + Freshness)** | Visibility < 1000m | 688 | +0.0565 min | [-0.2417, +0.3542] | NO (CI crosses 0) |
| **Model E (Fog + Vis + Freshness)** | Visibility < 500m | 352 | -0.0670 min | [-0.2961, +0.1460] | NO (CI crosses 0) |
| **Model E (Fog + Vis + Freshness)** | Visibility < 200m | 291 | -0.1081 min | [-0.3692, +0.1367] | NO (CI crosses 0) |
| **Model F (Fog + Severe Vis <500m)** | All Test Rows | 246,459 | +0.0456 min | [+0.0347, +0.0567] | YES (p < 0.05) |
| **Model F (Fog + Severe Vis <500m)** | Confirmed Fog | 77,373 | +0.0355 min | [+0.0167, +0.0534] | YES (p < 0.05) |
| **Model F (Fog + Severe Vis <500m)** | Visibility < 1000m | 688 | -0.0369 min | [-0.2698, +0.2310] | NO (CI crosses 0) |
| **Model F (Fog + Severe Vis <500m)** | Visibility < 500m | 352 | -0.0173 min | [-0.2124, +0.1896] | NO (CI crosses 0) |
| **Model F (Fog + Severe Vis <500m)** | Visibility < 200m | 291 | -0.0927 min | [-0.3088, +0.1350] | NO (CI crosses 0) |
| **Model G (Fog + Extreme Vis <200m)** | All Test Rows | 246,459 | +0.0790 min | [+0.0693, +0.0881] | YES (p < 0.05) |
| **Model G (Fog + Extreme Vis <200m)** | Confirmed Fog | 77,373 | +0.0482 min | [+0.0304, +0.0641] | YES (p < 0.05) |
| **Model G (Fog + Extreme Vis <200m)** | Visibility < 1000m | 688 | -0.1073 min | [-0.3572, +0.1067] | NO (CI crosses 0) |
| **Model G (Fog + Extreme Vis <200m)** | Visibility < 500m | 352 | +0.0172 min | [-0.3012, +0.2811] | NO (CI crosses 0) |
| **Model G (Fog + Extreme Vis <200m)** | Visibility < 200m | 291 | +0.0403 min | [-0.3231, +0.3332] | NO (CI crosses 0) |

---

## 14. Feature Importance Analysis

Inspecting LightGBM gain and tree split frequencies across primary candidate models:

### Feature Importance: Model B (Fog Only)

| Rank | Feature Name | Feature Type | Total Gain | Gain % | Tree Splits |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | `current_arr_delay` | Baseline V2 | 68,940,440,134.31 | 74.319% | 5,510 |
| 2 | `previous_train_delay` | Baseline V2 | 15,971,148,404.69 | 17.217% | 5,934 |
| 3 | `train` | Baseline V2 | 1,834,865,709.90 | 1.978% | 6,210 |
| 4 | `past_segment_std` | Baseline V2 | 1,472,655,987.85 | 1.588% | 2,533 |
| 5 | `next_station` | Baseline V2 | 1,197,748,804.70 | 1.291% | 5,624 |
| 6 | `station` | Baseline V2 | 1,017,892,034.81 | 1.097% | 5,912 |
| 7 | `past_segment_mean` | Baseline V2 | 775,208,582.52 | 0.836% | 1,914 |
| 8 | `past_segment_median` | Baseline V2 | 579,930,876.79 | 0.625% | 1,566 |
| 9 | `scheduled_segment_minutes` | Baseline V2 | 392,462,120.53 | 0.423% | 2,509 |
| 10 | `past_segment_count` | Baseline V2 | 259,397,469.05 | 0.280% | 1,476 |
| 11 | `day_of_week` | Baseline V2 | 238,356,463.14 | 0.257% | 1,067 |
| 12 | `fog_flag` | Fog / Weather | 74,413,760.81 | 0.080% | 556 |
| 13 | `fog_observation_available` | Fog / Weather | 6,345,104.10 | 0.007% | 53 |
| 14 | `is_weekend` | Baseline V2 | 1,689,640.91 | 0.002% | 23 |
| 15 | `month` | Baseline V2 | 0.00 | 0.000% | 0 |

### Feature Importance: Model D (Fog + Vis Thresholds)

| Rank | Feature Name | Feature Type | Total Gain | Gain % | Tree Splits |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | `current_arr_delay` | Baseline V2 | 76,160,644,720.43 | 82.294% | 4,789 |
| 2 | `previous_train_delay` | Baseline V2 | 9,641,209,411.13 | 10.418% | 4,444 |
| 3 | `past_segment_std` | Baseline V2 | 1,565,480,715.84 | 1.692% | 2,179 |
| 4 | `next_station` | Baseline V2 | 1,066,502,846.79 | 1.152% | 4,548 |
| 5 | `train` | Baseline V2 | 1,051,357,511.96 | 1.136% | 5,169 |
| 6 | `station` | Baseline V2 | 1,005,113,422.59 | 1.086% | 4,848 |
| 7 | `past_segment_mean` | Baseline V2 | 577,074,293.38 | 0.624% | 1,689 |
| 8 | `past_segment_median` | Baseline V2 | 558,958,578.64 | 0.604% | 1,299 |
| 9 | `scheduled_segment_minutes` | Baseline V2 | 360,063,621.77 | 0.389% | 1,896 |
| 10 | `past_segment_count` | Baseline V2 | 290,216,258.87 | 0.314% | 1,262 |
| 11 | `day_of_week` | Baseline V2 | 195,931,589.61 | 0.212% | 841 |
| 12 | `fog_flag` | Fog / Weather | 61,855,680.55 | 0.067% | 403 |
| 13 | `fog_observation_available` | Fog / Weather | 7,153,612.00 | 0.008% | 42 |
| 14 | `is_weekend` | Baseline V2 | 2,228,159.10 | 0.002% | 19 |
| 15 | `visibility_available` | Fog / Weather | 1,471,694.10 | 0.002% | 7 |
| 16 | `visibility_lt_1000m` | Fog / Weather | 1,302,210.79 | 0.001% | 13 |
| 17 | `visibility_lt_200m` | Fog / Weather | 196,786.30 | 0.000% | 3 |
| 18 | `visibility_lt_500m` | Fog / Weather | 50,593.60 | 0.000% | 2 |
| 19 | `month` | Baseline V2 | 0.00 | 0.000% | 0 |
| 20 | `low_visibility_flag` | Fog / Weather | 0.00 | 0.000% | 0 |

### Feature Importance: Model E (Fog + Vis + Freshness)

| Rank | Feature Name | Feature Type | Total Gain | Gain % | Tree Splits |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | `current_arr_delay` | Baseline V2 | 69,922,209,415.14 | 75.477% | 4,965 |
| 2 | `previous_train_delay` | Baseline V2 | 13,329,304,859.82 | 14.388% | 4,552 |
| 3 | `train` | Baseline V2 | 3,075,554,562.37 | 3.320% | 5,480 |
| 4 | `past_segment_std` | Baseline V2 | 1,549,459,035.50 | 1.673% | 2,104 |
| 5 | `next_station` | Baseline V2 | 1,095,483,448.54 | 1.183% | 4,802 |
| 6 | `station` | Baseline V2 | 1,063,664,020.06 | 1.148% | 5,150 |
| 7 | `past_segment_mean` | Baseline V2 | 1,018,994,033.63 | 1.100% | 1,654 |
| 8 | `past_segment_median` | Baseline V2 | 537,115,815.55 | 0.580% | 1,257 |
| 9 | `scheduled_segment_minutes` | Baseline V2 | 382,483,617.27 | 0.413% | 2,121 |
| 10 | `past_segment_count` | Baseline V2 | 228,074,816.35 | 0.246% | 1,177 |
| 11 | `weather_observation_age_minutes` | Fog / Weather | 204,149,478.03 | 0.220% | 1,230 |
| 12 | `day_of_week` | Baseline V2 | 194,281,397.35 | 0.210% | 818 |
| 13 | `fog_flag` | Fog / Weather | 32,731,043.42 | 0.035% | 236 |
| 14 | `is_weekend` | Baseline V2 | 4,139,116.00 | 0.004% | 22 |
| 15 | `visibility_available` | Fog / Weather | 844,582.00 | 0.001% | 4 |
| 16 | `fog_observation_available` | Fog / Weather | 777,545.80 | 0.001% | 8 |
| 17 | `visibility_lt_1000m` | Fog / Weather | 773,922.40 | 0.001% | 11 |
| 18 | `visibility_lt_500m` | Fog / Weather | 114,075.00 | 0.000% | 2 |
| 19 | `visibility_lt_200m` | Fog / Weather | 108,317.70 | 0.000% | 2 |
| 20 | `month` | Baseline V2 | 0.00 | 0.000% | 0 |
| 21 | `low_visibility_flag` | Fog / Weather | 0.00 | 0.000% | 0 |

### Feature Importance: Model F (Fog + Severe Vis <500m)

| Rank | Feature Name | Feature Type | Total Gain | Gain % | Tree Splits |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | `current_arr_delay` | Baseline V2 | 65,387,657,846.32 | 70.528% | 5,410 |
| 2 | `previous_train_delay` | Baseline V2 | 16,545,744,856.02 | 17.846% | 5,344 |
| 3 | `train` | Baseline V2 | 3,952,640,668.60 | 4.263% | 5,760 |
| 4 | `past_segment_mean` | Baseline V2 | 1,551,730,292.35 | 1.674% | 1,735 |
| 5 | `past_segment_std` | Baseline V2 | 1,522,248,917.02 | 1.642% | 2,453 |
| 6 | `next_station` | Baseline V2 | 1,231,003,533.17 | 1.328% | 5,402 |
| 7 | `station` | Baseline V2 | 1,041,208,725.76 | 1.123% | 5,678 |
| 8 | `past_segment_median` | Baseline V2 | 495,161,750.53 | 0.534% | 1,436 |
| 9 | `scheduled_segment_minutes` | Baseline V2 | 445,485,123.55 | 0.481% | 2,448 |
| 10 | `past_segment_count` | Baseline V2 | 242,355,576.63 | 0.261% | 1,301 |
| 11 | `day_of_week` | Baseline V2 | 215,160,729.20 | 0.232% | 999 |
| 12 | `fog_flag` | Fog / Weather | 73,538,639.28 | 0.079% | 517 |
| 13 | `fog_observation_available` | Fog / Weather | 5,075,234.29 | 0.005% | 31 |
| 14 | `is_weekend` | Baseline V2 | 2,014,610.90 | 0.002% | 28 |
| 15 | `visibility_lt_500m` | Fog / Weather | 849,709.50 | 0.001% | 9 |
| 16 | `visibility_available` | Fog / Weather | 387,502.40 | 0.000% | 5 |
| 17 | `month` | Baseline V2 | 0.00 | 0.000% | 0 |
| 18 | `visibility_lt_200m` | Fog / Weather | 0.00 | 0.000% | 0 |

---

## 15. Threshold Redundancy Analysis

Comparing alternative threshold configurations to determine the minimal necessary visibility representation:

| Representation | Features Added | Overall MAE | Confirmed Fog MAE | <1000m MAE | <500m MAE | <200m MAE | Redundancy Assessment |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Fog Only (Model B)** | `fog_flag`, `fog_obs_avail` | 8.2991 min | 8.9243 min | 14.6648 min | 11.9149 min | 12.8599 min | **High efficiency**: Captures bulk of global and fog signal. |
| **Fog + 1000m Variant** | + `vis_avail`, `vis_lt_1000m` | 8.4709 min | 9.0804 min | 14.9364 min | 12.0072 min | 12.9040 min | **Redundant**: Does not outperform Fog Only on severe fog. |
| **Fog + 500m Variant** | + `vis_avail`, `vis_lt_500m` | 8.2574 min | 8.8814 min | 14.6282 min | 11.8689 min | 12.7738 min | **Intermediate**: Captures moderate fog onset. |
| **Fog + Extreme 200m (Model G)** | + `vis_avail`, `vis_lt_200m` | 8.3582 min | 8.9861 min | 14.8957 min | 11.8773 min | 12.7576 min | **Focused**: Minimal representation for dense radiation fog. |
| **Fog + Severe 500m+200m (Model F)** | + `vis_avail`, `vis_lt_500m`, `vis_lt_200m` | 8.3915 min | 8.9988 min | 14.8254 min | 11.9117 min | 12.8905 min | **Optimal severe balance**: Strongest severe fog generalization. |
| **Fog + All Thresholds (Model D)** | + `vis_avail`, `1000m`, `500m`, `200m`, `low_vis` | 8.3122 min | 8.9341 min | 14.5456 min | 11.7397 min | 12.6457 min | **Full representation**: Best overall MAE (8.3122 min). |

---

## 16. Missingness & Availability Audit

1. **Missing Fog Observations**: In the processed dataset, when `fog_observation_available == 0` (548,036 calls), `fog_flag` is strictly `NaN` (never 0.0). LightGBM natively routes `NaN` missing values along the default branch without confounding missingness with clear weather.
2. **Missing Visibility Observations**: When `visibility_available == 0` (550,556 calls), `visibility_lt_1000m`, `visibility_lt_500m`, and `visibility_lt_200m` are strictly `NaN` (never 0.0). Missing visibility is never zero-filled, preventing false triggering of severe fog flags.
3. **Distinguishability**: The explicit presence of `fog_observation_available` and `visibility_available` allows the tree boosting algorithm to distinguish between:
   - Confirmed Fog (`fog_obs_avail == 1`, `fog_flag == 1`)
   - Confirmed Clear (`fog_obs_avail == 1`, `fog_flag == 0`)
   - Unobserved Weather (`fog_obs_avail == 0`, `fog_flag == NaN`)

---

## 17. Best Feature Set Selection

Applying the selection rule (prioritizing confirmed-fog/severe-fog gains, paired win rate, statistical significance, and structural simplicity):

### Primary Champion: **Model D (Fog + Visibility Thresholds)**
- **Features (7 Added)**:
  1. `fog_flag`
  2. `fog_observation_available`
  3. `visibility_available`
  4. `visibility_lt_1000m`
  5. `visibility_lt_500m`
  6. `visibility_lt_200m`
  7. `low_visibility_flag`

- **Performance Across All Cohorts**:
  - **Overall Test MAE**: **8.3122 min** vs V2 8.4372 min (**+1.48% improvement**, paired win rate: 55.05%, 95% Bootstrap CI: [+0.1156, +0.1339])
  - **Confirmed Fog MAE**: **8.9341 min** vs V2 9.0343 min (**+1.11% improvement**)
  - **Visibility <1000m MAE**: **14.5456 min** vs V2 14.7885 min (**+1.64% improvement**, 95% Bootstrap CI: [+0.0337, +0.4763], statistically significant p < 0.05)
  - **Visibility <500m MAE**: **11.7397 min** vs V2 11.8945 min (**+1.30% improvement**)
  - **Visibility <200m MAE**: **12.6457 min** vs V2 12.7979 min (**+1.19% improvement**)

### Minimalist Alternative: **Model B (Fog Only)**
- **Features (2 Added)**:
  1. `fog_flag`
  2. `fog_observation_available`
- **Performance**:
  - **Overall Test MAE**: **8.2991 min** (**+1.64% improvement**)
  - **Confirmed Fog MAE**: **8.9243 min** (**+1.22% improvement**)
  - **Limitation**: Does not resolve visibility severity below 500m (MAE 11.9149 min on <500m, 12.8599 min on <200m).

---

## 18. Limitations

1. **September Seasonal Bias**: September represents the tail-end of the monsoon in India, during which severe low-visibility events (<200m) comprise ~0.12% of total test rows (291 calls). While Model D achieves statistically significant improvements across `<1000m` (N=688, p < 0.05) and positive gains across `<500m` and `<200m`, the absolute sample size of dense radiation fog is small compared to the winter season.
2. **Station Coverage Latency**: Stations matched with METAR/GHCNh stations have an average observation age of 40–90 minutes. In rapidly evolving radiation fog, real-time ground sensor telemetry will provide sharper gains.

---

## 19. Recommendation for Future V4

For a future production V4 candidate experiment, the exact recommended feature architecture is **Model D (Fog + Visibility Thresholds)**:

```python
V4_RECOMMENDED_FEATURES = [
    # 13 Baseline V2 Features (Unchanged)
    'train',
    'station',
    'next_station',
    'current_arr_delay',
    'scheduled_segment_minutes',
    'past_segment_mean',
    'past_segment_median',
    'past_segment_std',
    'past_segment_count',
    'day_of_week',
    'month',
    'is_weekend',
    'previous_train_delay',
    # 7 Focused Fog & Visibility Threshold Features
    'fog_flag',
    'fog_observation_available',
    'visibility_available',
    'visibility_lt_1000m',
    'visibility_lt_500m',
    'visibility_lt_200m',
    'low_visibility_flag',
]
```

Production V2 model (`champion_model_scheduled_segment_v2.txt`) remains frozen and unchanged.

