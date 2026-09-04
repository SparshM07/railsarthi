# V3 Weather Feature Coverage & Quality Report

## Executive Summary
This report analyzes the coverage, observation age, and missingness characteristics of the newly engineered **V3 environmental features** merged with the **V2 railway baseline dataset** across **1,224,840 prediction stops** during September 2024.

> [!NOTE]
> - **Exact V2 Target & Features Preserved**: All 13 V2 baseline features and the exact next-station arrival delay target (`target_delay`) remain identical to production.
> - **Causal Integrity**: Weather observations are matched strictly in the backward direction (<= 180 minutes prior to train call). Future data is strictly zero.

---

## 1. Feature Availability Overview

| Feature | Description | Available Count | Coverage % |
| :--- | :--- | :---: | :---: |
| **`weather_available`** | Overall valid weather match (<= 180 min) | **676,804** | **55.26%** |
| **`visibility_m` / `visibility_available`** | Observed horizontal visibility | 674,284 | 55.05% |
| **`temperature_c` / `temperature_available`**| Air temperature (°C) | 675,962 | 55.19% |
| **`dewpoint_c` / `dewpoint_available`** | Dew point temperature (°C) | 676,426 | 55.23% |
| **`relative_humidity` / `humidity_available`**| Relative humidity (%) | 675,699 | 55.17% |
| **`wind_speed_mps` / `wind_available`** | Wind speed (m/s) | 671,674 | 54.84% |
| **`fog_observation_available`** | Confirmed fog/mist status | 676,804 | 55.26% |
| **`precipitation_available_flag`** | AA1 liquid precipitation report | 186,062 | 15.19% |

---

## 2. Spatial Distance Breakdown (Among Joined Observations)

Distance from railway station to nearest active weather station:

| Distance Bracket | Percentage of Joined Stops |
| :--- | :---: |
| **<= 25 km** | **49.34%** |
| **<= 50 km** | **78.91%** |
| **<= 100 km** | **99.28%** |

- **Minimum Distance**: `0.19 km`
- **Mean Distance**: `30.02 km`
- **Median Distance ($p_{50}$)**: `25.56 km`
- **90th Percentile ($p_{90}$)**: `63.15 km`
- **Maximum Distance**: `120.37 km`

---

## 3. Weather Observation Freshness / Age Breakdown

Elapsed time between train stop and weather observation timestamp:

| Observation Age Bracket | Percentage of Joined Stops | Operational Meaning |
| :--- | :---: | :--- |
| **0 to 30 minutes** | **37.88%** | Near-instantaneous / METAR airport reporting cadence. |
| **31 to 60 minutes** | **14.10%** | Hourly weather observation cycle. |
| **61 to 120 minutes** | **24.96%** | Intermediate synoptic cycle. |
| **121 to 180 minutes** | **23.06%** | Standard 3-hourly WMO synoptic observation cycle. |

- **Average Weather Age**: `69.4 minutes`
- **Median Weather Age ($p_{50}$)**: `57.0 minutes`
- **Maximum Permitted Weather Age**: `180.0 minutes`
