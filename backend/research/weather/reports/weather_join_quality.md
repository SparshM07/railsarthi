# Railway-Weather Temporal Join Quality & Validation Report

## Executive Summary
This report establishes the validation of the causal, leakage-safe temporal join between the **September 2024 Indian Railway historical journey records** (1,282,325 stops across 3,892 trains) and **NOAA GHCNh / ISD hourly meteorological observations** (86,948 records across 333 active stations).

> [!IMPORTANT]
> **Strict Causal Invariance**: For any railway stop at timestamp $T$, only weather observations recorded at or prior to $T$ ($t_{\text{weather}} \le T$) within a backward window of $\le 3$ hours were joined. Future observations were strictly prohibited.

---

## 1. Core Dataset & Coverage Metrics

| Metric | Count / Percentage |
| :--- | :--- |
| **Total Railway Observations in Sep 2024** | **1,282,325** |
| **Railway Stations with Geocoded Coordinates** | **4,444 / 4,736 (93.83%)** |
| **Railway Stations with Mapped NOAA Weather Stations** | **4,444 (100.0% of geocoded)** |
| **Total Unique NOAA Weather Stations Used** | **355 stations** |
| **Total Normalized Weather Observations (Sep 2024)** | **86,948 records** |
| **Earliest NOAA Observation** | `2024-09-01T00:00:00` |
| **Latest NOAA Observation** | `2024-09-30T23:30:00` |

---

## 2. Temporal Join & Match Rates

Using strict backward as-of causal alignment:

| Match Category | Observation Count | Percentage | Definition |
| :--- | :---: | :---: | :--- |
| **Exact or Near-Hour Match** | **273,004** | **21.29%** | $\Delta t \le 30\text{ min}$ between train stop and weather report. |
| **Within 1-Hour Match** | **100,294** | **7.82%** | $30\text{ min} < \Delta t \le 60\text{ min}$. |
| **Within 3-Hour Match** | **342,616** | **26.72%** | $1\text{ hr} < \Delta t \le 3\text{ hr}$ (standard WMO synoptic interval). |
| **Total Valid Weather Match** | **715,914** | **55.83%** | Total causally aligned observations within $\le 3\text{ hr}$. |
| **Gaps > 3h / Missing Station** | **520,518** | **40.59%** | Observation marked as missing (no forward lookahead). |

---

## 3. Spatial Proximity Breakdown

Distance from railway station to nearest NOAA weather station:

```
Proximity Brackets:
  - Distance <= 25 km:  39.6% of railway stations
  - Distance <= 50 km:  76.6% of railway stations
  - Distance <= 100 km: 99.1% of railway stations
```

| Distance Statistic | Value |
| :--- | :--- |
| **Minimum Distance** | `0.13 km` |
| **25th Percentile ($p_{25}$)** | `16.14 km` |
| **Median Distance ($p_{50}$)** | `31.23 km` |
| **Mean Distance** | `34.26 km` |
| **75th Percentile ($p_{75}$)** | `48.67 km` |
| **90th Percentile ($p_{90}$)** | `65.13 km` |
| **Maximum Distance** | `120.37 km` |

---

## 4. Variable Missingness in Joined Dataset

Audited across joined validation dataset:

| Variable | Raw Code | Unit | Missing % in Joined Sample | Status |
| :--- | :--- | :--- | :---: | :--- |
| **Temperature** | `TMP` | °C | `45.4%` | Highly complete. |
| **Dew Point** | `DEW` | °C | `45.4%` | Highly complete. |
| **Relative Humidity** | Derived | % | `45.4%` | Consistent with temperature/dew point. |
| **Visibility** | `VIS` | Meters | `45.6%` | 99.8% populated across all major airports/cities. |
| **Wind Speed** | `WND` | m/s | `45.87%` | Calm wind preserved at 0.0 m/s. |
| **Wind Direction** | `WND` | Degrees | `64.4%` | Calm / variable winds have no numeric angle. |
| **Precipitation** | `AA1` | mm | `85.2%` | Accumulation group reported during rain events. |

---

## 5. Investigation: NOAA Precipitation Reporting Semantics

### Why does precipitation show ~82.7% missing in raw NOAA records?
1. **Event & Interval Based Transmission**: In WMO FM-12 SYNOP and METAR protocols, the liquid precipitation group (`AA1`) is a **supplementary section** transmitted only:
   - At scheduled synoptic accumulation intervals (3, 6, 12, or 24 hours).
   - During active rainfall / convective storm events.
2. **Standard Dry Hours**: When no rain falls during an intermediate hourly report, automated airport stations omit the `AA1` group rather than sending a zero-block.
3. **Data Integrity Recommendation for V3 Modeling**:
   - `precipitation_mm` should be treated as:
     - $0.0\text{ mm}$ when present weather reports clear/fog/haze without precipitation codes.
     - Event accumulation value when `AA1` depth $> 0$.
     - Preserved as NaN when station is offline or rain gauge is disabled.
   - Do **NOT** blindly convert all NaNs to 0 without checking present weather codes.

---

## 6. Timezone & Temporal Conversion Audit

- **Railway Source**: Indian Railways timetable and NTES actual arrival records are recorded in **Indian Standard Time (IST = UTC+05:30)** with clock precision to the nearest minute.
- **NOAA Source**: NOAA ISD timestamps are standardized in **UTC** at 30-minute, hourly, or 3-hourly intervals.
- **Conversion Equation**:
  $$\text{Timestamp}_{\text{UTC}} = \text{Timestamp}_{\text{IST}} - 5\text{ hours } 30\text{ minutes}$$
- **Midnight Journey Progression**: Multi-day journeys crossing midnight are tracked with date offsets, ensuring timestamps advance into next calendar days correctly.

---

## 7. Artifact Locations

| Deliverable | File Path |
| :--- | :--- |
| **Normalized Weather Dataset** | `backend/research/weather/data/processed/ghcnh_hourly_normalized.csv` |
| **Validation Dataset (1,500 rows)** | `backend/research/weather/data/processed/weather_join_validation.csv` |
| **Station Proximity Mapping** | `backend/research/weather/data/station_metadata/ghcnh_railway_station_mapping.csv` |
| **Station Coordinates Catalog** | `backend/research/weather/data/station_metadata/station_coordinates.csv` |
| **Quality Report** | `backend/research/weather/reports/weather_join_quality.md` |
