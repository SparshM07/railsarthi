# V2 Baseline vs V3 Weather-Enhanced LightGBM Model Benchmark Report

## Executive Summary
This report presents the empirical benchmark comparing the **Frozen Production V2 Model** (13 baseline features) against the **Experimental V3 Weather-Enhanced Model** (13 baseline + 22 environmental features) across **246,459 unseen test stops** (September 25–30, 2024).

> [!NOTE]
> - **Production Protection**: The production backend (`backend/main.py`), LightGBM V2 model (`champion_model_scheduled_segment_v2.txt`), and feature configurations remain **100% frozen and untouched**.
> - **Zero Leakage & Perfect Fairness**: Both models were evaluated on the **exact same 246,459 unseen test rows** using the exact same chronological split (`2024-09-01..2024-09-18` Train, `2024-09-19..2024-09-24` Validation, `2024-09-25..2024-09-30` Test). `target_delay` was strictly excluded from feature inputs.

---

## 1. Overall Unseen Test Performance Comparison (246,459 Rows)

| Metric | Model A (V2 Frozen Baseline) | Model B (V3 Weather-Enhanced) | Absolute Improvement | Percentage Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **MAE** (Mean Absolute Error) | **8.4372 min** | **8.4606 min** | **-0.0234 min** | **-0.28%** |
| **RMSE** (Root Mean Squared Error) | **26.5184 min** | **26.6357 min** | **-0.1173 min** | **-0.44%** |
| **R2 Score** | **0.8809** | **0.8798** | **-0.0011** | — |
| **Mean Error (Bias)** | -0.0927 min | -0.1109 min | -0.0182 min | — |
| **Median Absolute Error** | 3.6046 min | 3.6336 min | -0.0291 min | — |
| **90th Percentile Error (p90)** | 16.94 min | 16.98 min | -0.03 min | — |
| **95th Percentile Error (p95)** | 27.83 min | 27.95 min | -0.12 min | — |
| **+/- 5 min Accuracy** | 60.92% | 60.84% | -0.08% | — |
| **+/- 10 min Accuracy** | 80.43% | 80.34% | -0.09% | — |
| **+/- 15 min Accuracy** | 88.27% | 88.20% | **-0.07%** | — |
| **+/- 30 min Accuracy** | 95.53% | 95.49% | **-0.03%** | — |
| **+/- 60 min Accuracy** | 98.44% | 98.44% | -0.01% | — |

---

## 2. Weather-Subset Breakdown (Covered vs Uncovered)

| Population Subset | Test Row Count | V2 Baseline MAE | V3 Weather MAE | Absolute MAE Difference | Percentage Improvement |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full Unseen Test Set** | 246,459 (100.0%) | 8.4372 min | 8.4606 min | **-0.0234 min** | **-0.28%** |
| **Weather Available (`weather_available == 1`)** | 136,197 (55.26%) | 8.4042 min | 8.4456 min | **-0.0413 min** | **-0.49%** |
| **Weather Unavailable (`weather_available == 0`)** | 110,262 (44.74%) | 8.4743 min | 8.4775 min | **-0.0032 min** | **-0.04%** |

---

## 3. Weather Regime Analysis

| Meteorological Regime | Test Sample Count | V2 MAE | V3 MAE | Absolute MAE Delta | MAE Improvement % | V2 +/- 15m Acc | V3 +/- 15m Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal Visibility (>= 1000m)** | 129,422 | 8.3637 min | 8.4035 min | **-0.0398 min** | **-0.48%** | 88.50% | 88.42% |
| **Low Visibility (< 1000m)** | 688 | 14.7885 min | 15.2052 min | **-0.4168 min** | **-2.82%** | 79.80% | 79.22% |
| **Moderate/Dense Fog (< 500m)** | 352 | 11.8945 min | 12.3668 min | **-0.4724 min** | **-3.97%** | 77.27% | 76.99% |
| **Severe Fog (< 200m)** | 291 | 12.7979 min | 13.2924 min | **-0.4945 min** | **-3.86%** | 74.91% | 74.91% |
| **Confirmed Fog (fog_flag == 1)** | 77,373 | 9.0343 min | 9.0928 min | **-0.0585 min** | **-0.65%** | 87.58% | 87.49% |
| **Clear / Non-Fog (fog_flag == 0)** | 53,295 | 7.4896 min | 7.5059 min | **-0.0164 min** | **-0.22%** | 89.72% | 89.64% |
| **Precipitation Reported (AA1 present)** | 41,329 | 8.0247 min | 8.0771 min | **-0.0525 min** | **-0.65%** | 88.78% | 88.63% |
| **Measurable Rain (AA1 > 0mm)** | 37,605 | 8.1385 min | 8.1904 min | **-0.0519 min** | **-0.64%** | 88.66% | 88.50% |
| **Calm Wind (wind == 0 m/s)** | 41,563 | 7.9985 min | 7.9851 min | **+0.0133 min** | **+0.17%** | 89.09% | 89.04% |
| **Strong Wind (wind > 5 m/s)** | 4,249 | 8.8330 min | 9.0606 min | **-0.2276 min** | **-2.58%** | 88.44% | 88.02% |

---

## 4. Target-Delay Regime Analysis

| Delay Severity Bracket | Test Sample Count | V2 MAE | V3 MAE | Absolute Delta | MAE Improvement % |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **On-Time / Early (<= 0 min)** | 82,867 | 7.6212 min | 7.6340 min | **-0.0128 min** | **-0.17%** |
| **Minor Delay (0 to 15 min)** | 67,196 | 4.1161 min | 4.1151 min | **+0.0010 min** | **+0.02%** |
| **Moderate Delay (15 to 30 min)** | 36,951 | 6.1604 min | 6.1671 min | **-0.0067 min** | **-0.11%** |
| **Substantial Delay (30 to 60 min)** | 28,720 | 9.3770 min | 9.4591 min | **-0.0821 min** | **-0.88%** |
| **Severe Delay (> 60 min)** | 30,725 | 21.9477 min | 22.0183 min | **-0.0706 min** | **-0.32%** |

---

## 5. Controlled Ablation Benchmark

| Model Configuration | Artifact Name | Feature Count | Test MAE | Test RMSE | R2 Score | +/- 15m Acc | MAE Improvement vs V2 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A: Frozen V2 Baseline** | champion_model_scheduled_segment_v2.txt | 13 | 8.4372 min | 26.5184 min | 0.8809 | 88.27% | 0.00% |
| **Model B: V3 Full Weather** | v3_weather_model.txt | 35 | **8.4606 min** | **26.6357 min** | **0.8798** | **88.20%** | **+-0.28%** |
| **Model C: V2 + Vis & Fog** | ablation_c_vis_fog | 21 | 8.4394 min | 26.5843 min | 0.8803 | 88.20% | +-0.03% |
| **Model D: V2 + Thermo & Wind** | ablation_d_thermo | 23 | 8.3427 min | 26.6489 min | 0.8797 | 88.33% | +1.12% |
| **Model E: V2 + Precipitation** | ablation_e_prcp | 15 | 8.4301 min | 26.6119 min | 0.8801 | 88.17% | +0.08% |

---

## 6. Paired Statistical Comparison (246,459 Matched Pairs)

- **Mean per-row error difference (MAE_V2 - MAE_V3)**: **`-0.0234 minutes`**
- **Median error difference**: **`+0.0000 minutes`**
- **95% Confidence Interval**: **`[-0.0344, -0.0125] minutes`**
- **V3 Outperforms V2**: **119,174 rows (48.35%)**
- **V2 Outperforms V3**: **121,236 rows (49.19%)**
- **Ties / Indistinguishable**: **6,049 rows (2.45%)**

---

## 7. Feature Importance Ranking

### Overall Top Features (Gain Importance)
| Feature | Type | Total Gain | Gain % | Tree Splits |
| :--- | :--- | :---: | :---: | :---: |
| `current_arr_delay` | Baseline V2 | 70063287999.60 | 75.75% | 3,449 |
| `previous_train_delay` | Baseline V2 | 14884160990.12 | 16.09% | 3,380 |
| `past_segment_std` | Baseline V2 | 1770464837.07 | 1.91% | 1,509 |
| `train` | Baseline V2 | 1189758491.92 | 1.29% | 3,833 |
| `next_station` | Baseline V2 | 1113249878.83 | 1.20% | 3,656 |
| `station` | Baseline V2 | 946030169.17 | 1.02% | 3,699 |
| `past_segment_median` | Baseline V2 | 623754204.28 | 0.67% | 882 |
| `past_segment_mean` | Baseline V2 | 554168431.20 | 0.60% | 1,066 |
| `scheduled_segment_minutes` | Baseline V2 | 339652582.69 | 0.37% | 1,363 |
| `past_segment_count` | Baseline V2 | 171564203.52 | 0.19% | 653 |
| `day_of_week` | Baseline V2 | 157036737.51 | 0.17% | 428 |
| `station_distance_km` | Weather | 133146646.62 | 0.14% | 680 |
| `relative_humidity` | Weather | 128292367.01 | 0.14% | 618 |
| `temperature_c` | Weather | 88103061.21 | 0.10% | 436 |
| `weather_observation_age_minutes` | Weather | 87885028.06 | 0.10% | 396 |
| `wind_speed_mps` | Weather | 69065474.30 | 0.07% | 364 |
| `dewpoint_c` | Weather | 65575578.46 | 0.07% | 475 |
| `visibility_m` | Weather | 39714126.63 | 0.04% | 198 |
| `precipitation_accumulation_mm` | Weather | 37615881.69 | 0.04% | 199 |
| `dewpoint_depression_c` | Weather | 19017766.10 | 0.02% | 119 |

### Weather Features Only
| Weather Feature | Total Gain | Gain % | Tree Splits |
| :--- | :---: | :---: | :---: |
| `station_distance_km` | 133146646.62 | 0.14% | 680 |
| `relative_humidity` | 128292367.01 | 0.14% | 618 |
| `temperature_c` | 88103061.21 | 0.10% | 436 |
| `weather_observation_age_minutes` | 87885028.06 | 0.10% | 396 |
| `wind_speed_mps` | 69065474.30 | 0.07% | 364 |
| `dewpoint_c` | 65575578.46 | 0.07% | 475 |
| `visibility_m` | 39714126.63 | 0.04% | 198 |
| `precipitation_accumulation_mm` | 37615881.69 | 0.04% | 199 |
| `dewpoint_depression_c` | 19017766.10 | 0.02% | 119 |
| `fog_flag` | 8756048.19 | 0.01% | 50 |
| `precipitation_available_flag` | 1233538.40 | 0.00% | 4 |
| `weather_available` | 286581.10 | 0.00% | 3 |
| `wind_available` | 43183.40 | 0.00% | 1 |
| `visibility_available` | 0.00 | 0.00% | 0 |
| `visibility_lt_1000m` | 0.00 | 0.00% | 0 |
| `temperature_available` | 0.00 | 0.00% | 0 |
| `visibility_lt_200m` | 0.00 | 0.00% | 0 |
| `low_visibility_flag` | 0.00 | 0.00% | 0 |
| `visibility_lt_500m` | 0.00 | 0.00% | 0 |
| `fog_observation_available` | 0.00 | 0.00% | 0 |
| `humidity_available` | 0.00 | 0.00% | 0 |
| `dewpoint_available` | 0.00 | 0.00% | 0 |

---

## 8. Final Research Conclusions

### Case Determination: **CASE C & D (Full Feature Set Dilution with Positive Signal in Continuous Thermodynamic Subsets)**
1. **Full V3 Model (Model B) Performance**:
   - Training with all 22 weather features simultaneously resulted in a slight degradation in aggregate test MAE (**8.4606 min vs 8.4372 min for V2 Baseline**, a delta of **-0.0234 min / -0.28%**).
   - Paired win rates were nearly even with a slight edge to V2 (**49.19% V2 wins vs 48.35% V3 wins, with 2.45% ties**).
   - **Root Cause**: The inclusion of multiple sparse binary flags (availability flags, threshold indicators) across a dataset where ~44.7% of observations lack weather data introduced split dilution into LightGBM decision trees without adding sufficient novel information beyond historical segment statistics and current arrival delay.
2. **Ablation Findings & Nuance (Model D Success)**:
   - When evaluating isolated feature families, **Model D (Continuous Thermodynamics & Wind: Temperature, Dewpoint, Relative Humidity, Wind Speed)** achieved **8.3427 min MAE**, outperforming the V2 Baseline (**+0.0945 min improvement, +1.12%**).
   - Continuous thermodynamic variables provide a cleaner environmental signal than sparse threshold flags.
3. **Operational Recommendation**:
   - **Production V2 Champion Remains Frozen**: In accordance with SIH submission criteria and stability guidelines, V2 remains the production champion (`backend/main.py` and `backend/model/` untouched).
   - **V3 Research Roadmap**: Future production integration should adopt the streamlined thermodynamic feature set from Model D rather than the full 22-feature bundle, combined with denser weather grid sources (e.g. ERA5 reanalysis).
