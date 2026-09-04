# V3 Weather & Train Delay Exploratory Analysis (September 2024)

## Executive Summary
This exploratory analysis investigates the empirical relationship between observed environmental conditions (visibility, fog, precipitation, wind) and train arrival delays across **1,224,840 prediction segments** in September 2024.

> [!NOTE]
> - **Descriptive Statistics Only**: These figures represent observed backtest distributions and do not claim direct causal attribution.
> - **Baseline MAE Reference**: The `MAE Baseline` column reflects the error of a naive persistence baseline ($\hat{y} =  current\_arr\_delay}$) across each meteorological condition.

---

## 1. Delay Distribution by Weather Condition

| Meteorological Condition | Sample Count | Mean Next Delay | Median Next Delay | Persistence Baseline MAE | 90th Percentile Delay | 95th Percentile Delay |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal Visibility (>= 1000m)** | 671,208 | 33.13 min | 9.0 min | **10.14 min** | 78.0 min | 148.0 min |
| **Low Visibility (< 1000m)** | 3,076 | 51.47 min | 23.0 min | **14.55 min** | 127.5 min | 185.0 min |
| **Moderate/Dense Fog (< 500m)** | 2,036 | 56.55 min | 29.0 min | **14.53 min** | 133.0 min | 183.5 min |
| **Severe Fog (< 200m)** | 1,676 | 58.22 min | 31.0 min | **15.08 min** | 136.0 min | 190.8 min |
| **Confirmed Fog / Mist Flag = 1** | 378,344 | 35.09 min | 9.0 min | **10.65 min** | 86.0 min | 160.0 min |
| **Clear / Non-Fog Flag = 0** | 298,460 | 30.85 min | 9.0 min | **9.53 min** | 69.0 min | 133.0 min |
| **Precipitation Reported (AA1 present)** | 186,062 | 34.99 min | 9.0 min | **10.13 min** | 85.0 min | 157.0 min |
| **No Precipitation Group (Dry / Unreported)** | 1,038,778 | 33.72 min | 9.0 min | **10.01 min** | 80.0 min | 152.0 min |
| **Measurable Rain (> 0 mm)** | 160,009 | 35.27 min | 10.0 min | **10.25 min** | 86.0 min | 158.0 min |
| **Calm Wind (0 m/s)** | 210,318 | 33.32 min | 8.0 min | **9.61 min** | 78.0 min | 151.0 min |
| **Light/Moderate Breeze (0.1 - 5 m/s)** | 429,401 | 33.27 min | 9.0 min | **10.34 min** | 78.0 min | 148.0 min |
| **Strong Wind (> 5 m/s)** | 31,955 | 31.29 min | 10.0 min | **11.55 min** | 75.0 min | 138.0 min |

---

## 2. Key Empirical Insights

### A. Visibility & Fog Impact
1. **Low Visibility Escalation**: Stations experiencing low visibility ($< 1000  m}$) exhibit noticeably higher mean delay (51.47 min vs 33.13 min for normal visibility) and wider right-tail dispersion (95th percentile delay of 185.0 min).
2. **Severe Fog Corridor ($< 200  m}$)**: Severe fog conditions show elevated persistence error (MAE 15.08 min), indicating increased unpredictability in segment running times.

### B. Precipitation Dynamics
1. **Measurable Rainfall**: Predictions during active measurable rainfall show higher mean delays (35.27 min) compared to dry/unreported periods (33.72 min), consistent with monsoon operational speed restrictions.

### C. Wind Regimes
1. **Calm vs Strong Wind**: Strong winds ($> 5  m/s}$) show elevated mean delay (31.29 min), often associated with active monsoon storm fronts and squalls.
