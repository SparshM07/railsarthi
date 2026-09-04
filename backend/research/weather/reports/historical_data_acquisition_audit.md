# Historical Railway & Weather Data Acquisition Audit

**Research Phase**: Pre-Expansion Data Pipeline Audit  
**Date**: 2026-09-02  
**Status**: Research Audit Completed — Production Remains 100% Frozen  

---

## 1. Executive Summary

This audit establishes the exact data provenance, schemas, preprocessing transformations, temporal joining logic, and causality invariants of the September 2024 dataset used in the V2 champion and V3/V4 weather research experiments.

**Key Findings**:
1. **September Railway Dataset**: Derived from the station-level sequential dataset `train_routes_delays_Sep2024.csv` (1,282,325 raw stop records covering 3,892 trains and 4,736 stations across September 1–30, 2024).
2. **Prediction-Row Derivation**: After removing terminal station stops ($N = 57,485$ rows where no subsequent station exists), exactly **1,224,840 prediction rows** are produced with 100% non-null target values.
3. **NOAA GHCNh Weather Dataset**: Built by downloading hourly observations from 79 active Indian NOAA weather stations mapped to 200+ major railway hubs via Haversine distance proximity, causally matched backwards with zero future lookahead.
4. **Feasibility for Multi-Month Expansion (Oct 2024 – Jan 2025)**:
   - **NOAA Weather Data**: 100% accessible directly via the existing `acquire_ghcnh_weather.py` pipeline from NOAA NCEI Global Hourly HTTPS archives for 2024 and 2025.
   - **Railway Delay Data**: Requires station-level sequential stop records containing train ID, journey date, station sequence, scheduled arrival, and actual arrival/delay. Journey-level aggregate files (such as `ir_train.csv`) lack intermediate hop sequences and cannot be used.

---

## 2. September Railway Data Source

- **Original Raw File**: `train_routes_delays_Sep2024.csv`
- **Companion Archive**: `train_delays_Sep2024.json`, `train_routes_Sep2024.csv`, `stations_zones_mapping.json`, `IRN_edges.csv`
- **Dataset Origin**: Open Indian Railway delay and network tracking repository (NTES/RailYatri scraper dump for September 2024).
- **Physical Location in Environment**:
  `C:\Users\SPARSH MAURYA\Downloads\Indian-Railway-Network-and-Delays\Indian-Railway-Network-and-Delays\train_routes_delays_Sep2024.csv`
- **Raw File Size**: 84,928,205 bytes (~84.9 MB)
- **Total Records**: 1,282,325 rows
- **Unique Trains**: 3,892 trains
- **Unique Stations**: 4,736 station codes
- **Temporal Scope**: `2024-09-01` to `2024-09-30` (30 complete days)

---

## 3. September Data Acquisition

The dataset was acquired as a daily live tracking JSON dump (`train_delays_Sep2024.json`) structured as:
```json
{
  "12303": {
    "02-09-2024": {
      "HWH": ["08:00 AM", "08:01 AM", "00M", "08:00 AM", "08:01 AM", "00M"],
      "BWN": ["09:05 AM", "09:20 AM", "15M", "09:08 AM", "09:23 AM", "15M"],
      "DGR": ["09:57 AM", "10:00 AM", "03M", "09:59 AM", "10:02 AM", "03M"]
    }
  }
}
```
The raw array format `[sch_arr, act_arr, arr_delay, sch_dep, act_dep, dep_delay]` was flattened into CSV format with tabular columns:
`['train', 'date', 'station', 'sch_arr', 'act_arr', 'arr_delay', 'sch_dep', 'act_dep', 'dep_delay']`.

---

## 4. Railway Preprocessing Pipeline

The preprocessing logic is implemented in [`backend/feature_pipeline.py`](file:///d:/SIH-RAILWAY/backend/feature_pipeline.py) and replicated in [`backend/research/weather/scripts/build_v3_weather_features.py`](file:///d:/SIH-RAILWAY/backend/research/weather/scripts/build_v3_weather_features.py):

### A. Identifier Normalization
1. **Train Number Normalization** ([`normalize_train`](file:///d:/SIH-RAILWAY/backend/feature_pipeline.py#L38-L49)):
   - Strips whitespace.
   - Strips `.0` float string artifact (e.g. `"12303.0"` $\to$ `"12303"`).
   - Preserves non-numeric special train identifiers as clean strings.
2. **Station Code Normalization** ([`normalize_station`](file:///d:/SIH-RAILWAY/backend/feature_pipeline.py#L52-L56)):
   - Strips whitespace and forces uppercase (e.g. `" ndls "` $\to$ `"NDLS"`).
3. **Date Normalization**:
   - Parsed with `pd.to_datetime(df['date'], errors='coerce').dt.normalize()` to yield `YYYY-MM-DD`.

### B. Stable Journey Sequencing
- A monotonic index `source_order = np.arange(len(df))` is attached to preserve the true physical route sequence.
- Sorted by `["train", "date", "source_order"]` with a stable sort.
- **Critical Invariant**: Station codes are **never** sorted alphabetically, as alphabetical sorting destroys the sequential route topology.

---

## 5. Prediction-Row Construction

Each prediction instance represents a train currently at station $S_k$, predicting arrival delay at the immediate downstream station $S_{k+1}$:

1. **Next Station Formation**:
   - `df['next_station'] = df.groupby(['train', 'date'])['station'].shift(-1)`
2. **Next Timetable Arrival**:
   - `df['next_sch_arr'] = df.groupby(['train', 'date'])['sch_arr'].shift(-1)`
3. **Current Arrival Delay** ($S_k$):
   - `df['current_arr_delay'] = pd.to_numeric(df['arr_delay'], errors='coerce').fillna(0.0)`
4. **Previous Train Delay** ($S_{k-1}$):
   - `df['previous_train_delay'] = df.groupby(['train', 'date'])['current_arr_delay'].shift(1).fillna(0.0)`
   - At origin station ($k=0$), `previous_train_delay` is `0.0`.
5. **Scheduled Segment Duration** ($S_k \to S_{k+1}$):
   - Computed via [`scheduled_segment_from_arrivals(sch_arr, next_sch_arr)`](file:///d:/SIH-RAILWAY/backend/feature_pipeline.py#L109-L124).
   - Clock strings (`"08:00 AM"`, `"09:05 AM"`, or 24-hour `"20:00"`) converted to minutes from midnight.
   - Rollover rule: If `next_minutes < current_minutes`, `duration = (next_minutes - current_minutes) + 1440` (handles single midnight rollover).
6. **Calendar Features**:
   - `day_of_week = date.dt.dayofweek` (0=Monday, 6=Sunday)
   - `month = date.dt.month` (9 for September)
   - `is_weekend = (day_of_week >= 5).astype(int)`

---

## 6. Target Construction

- **Target Variable**: `target_delay`
- **Definition**: Exact arrival delay (in minutes) at the immediate next station $S_{k+1}$.
- **Formula**:
  `df['target_delay'] = df.groupby(['train', 'date'])['arr_delay'].shift(-1)`
- **Filtering**:
  - Terminal station rows (where `next_station` is `NaN`) have no downstream arrival and are removed.
  - Rows with non-null `next_station` and non-null `target_delay` form the final dataset.

---

## 7. Historical Segment Statistics

- **Source File**: `backend/model/segment_stats.csv` (17,317 pre-aggregated routes)
- **Features Extracted**:
  - `past_segment_mean`
  - `past_segment_median`
  - `past_segment_std`
  - `past_segment_count`
- **Segment Key**: `f"{station}->{next_station}"` (e.g. `"DR->CSMT"`, `"NDLS->CNB"`)
- **Missing Value Handling**:
  - If a station-hop pair is not present in `segment_stats.csv`, all 4 fields are assigned `0.0` (matching the production sentinel).

---

## 8. Timestamp & Date Rollover Handling

1. **Timetable Clock Parsing**:
   - Handled by [`scheduled_time_to_minutes()`](file:///d:/SIH-RAILWAY/backend/feature_pipeline.py#L59-L86) supporting:
     - 12-hour AM/PM: `"08:00 AM"`, `"11:50 PM"`
     - 24-hour: `"20:00"`, `"00:15"`
     - Numeric encodings: `830` $\to$ `510.0` min, `2350` $\to$ `1430.0` min
2. **Route Midnight Rollover**:
   - For multi-day journeys, `utc_sec` is tracked with monotonic day offset increments:
     `if prev_minutes >= 0 and clock_mins < prev_minutes - 180: day_offset += 1`
   - Converts Indian Standard Time (IST, UTC+5:30) to Unix epoch UTC seconds for weather joins.

---

## 9. September Row Counts at Each Stage

| Processing Stage | Rows Count | Change / Reason |
| :--- | :---: | :--- |
| **1. Raw CSV Records** | **1,282,325** | Raw input rows in `train_routes_delays_Sep2024.csv` |
| **2. Null Key Removal** | **1,282,325** | 0 null `train`, `station`, or `date` rows |
| **3. Terminal Hop Slicing** | **1,224,840** | **-57,485 rows** removed (terminal stops with no $S_{k+1}$) |
| **4. Final Prediction Dataset** | **1,224,840** | 100% valid target values ($S_k \to S_{k+1}$) |
| **- Chronological Train** (Sep 01–18) | **730,698** | 59.66% of prediction rows |
| **- Chronological Validation** (Sep 19–24) | **247,683** | 20.22% of prediction rows |
| **- Chronological Unseen Test** (Sep 25–30) | **246,459** | 20.12% of prediction rows |

---

## 10. NOAA Weather Acquisition Pipeline

Implemented in [`backend/research/weather/acquire_ghcnh_weather.py`](file:///d:/SIH-RAILWAY/backend/research/weather/acquire_ghcnh_weather.py):
1. **Station Geocoding**:
   - 4,736 railway stations mapped to coordinates using DataMeet master railway GeoJSON.
   - Result: `station_coordinates.csv` (4,400+ geocoded stations).
2. **NOAA Active Station Catalog**:
   - Loaded from NOAA ISD History (`https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv`).
   - Filtered for active Indian stations (WMO prefix 42/43/44) with valid 2024 observation coverage.
   - Result: `noaa_ghcnh_active_stations.csv` (79 active stations).
3. **Spatial Proximity Mapping**:
   - Computed pairwise Haversine distances between railway stations and NOAA stations.
   - Station matched to nearest NOAA station within 100 km.
   - Result: `ghcnh_railway_station_mapping.csv`.
4. **NOAA Global Hourly Download**:
   - Fetched hourly CSVs from NOAA NCEI Global Hourly HTTPS endpoint:
     `https://www.ncei.noaa.gov/data/global-hourly/access/2024/{station_id}.csv`

---

## 11. Weather Temporal Join & Causality

Implemented in [`backend/research/weather/build_weather_join.py`](file:///d:/SIH-RAILWAY/backend/research/weather/build_weather_join.py):
1. **UTC Timestamp Conversion**:
   - Railway stop scheduled/actual time in IST converted to UTC epoch seconds:
     $\text{UTC\_sec} = \text{base\_unix\_sec} + (\text{day\_offset} \times 86400) + (\text{clock\_mins} \times 60)$
2. **Strict Causal Backward Matching**:
   - Weather observations indexed by UTC timestamp:
     `pos = np.searchsorted(weather_ts_sec, train_utc_sec, side="right") - 1`
   - Guarantees: $\text{weather\_ts} \le \text{train\_prediction\_ts}$ with **zero future lookahead**.
3. **Maximum Latency Window**:
   - Observation valid if $\Delta t = \text{train\_utc\_sec} - \text{weather\_ts} \le 10,800\text{ sec}$ (3 hours).
   - If $\Delta t > 3\text{ hours}$ or station unmapped: `weather_available = 0`, weather features set to `NaN`.

---

## 12. Weather Feature Construction

Implemented in [`backend/research/weather/scripts/build_v3_weather_features.py`](file:///d:/SIH-RAILWAY/backend/research/weather/scripts/build_v3_weather_features.py):

| Weather Feature | Derivation / Logic | Missing Value Sentinel |
| :--- | :--- | :---: |
| `fog_flag` | 1.0 if NOAA present-weather codes include fog/mist ($01–03, 10–12, 40–49$), else 0.0 | `NaN` (when `fog_obs_avail == 0`) |
| `fog_observation_available` | 1 if present-weather observation exists, else 0 | 0 |
| `visibility_m` | NOAA VIS field parsed to meters | `NaN` (when `vis_avail == 0`) |
| `visibility_available` | 1 if visibility sensor recorded data, else 0 | 0 |
| `visibility_lt_1000m` | 1.0 if `visibility_m < 1000`, else 0.0 | `NaN` |
| `visibility_lt_500m` | 1.0 if `visibility_m < 500`, else 0.0 | `NaN` |
| `visibility_lt_200m` | 1.0 if `visibility_m < 200`, else 0.0 | `NaN` |
| `low_visibility_flag` | 1.0 if `visibility_lt_1000m == 1`, else 0.0 | `NaN` |
| `weather_observation_age_minutes` | $\Delta t / 60.0$ in minutes | `NaN` |

---

## 13. Required Fields for Future Railway Data

To reproduce the exact V2/V4 prediction dataset for October 2024 – January 2025, any future railway data source **must contain the following minimum fields**:

1. **`train`**: Train number/identifier (e.g. `"12303"`).
2. **`date`**: Journey departure date (e.g. `"2024-12-15"`).
3. **`station`**: Alpha station code (e.g. `"NDLS"`, `"CNB"`, `"HWH"`).
4. **`sch_arr`**: Scheduled arrival clock time (`"HH:MM AM/PM"` or `"HH:MM"`).
5. **`act_arr`**: Actual arrival clock time.
6. **`arr_delay`**: Numerical arrival delay in minutes (can be negative or 0.0).
7. **`sch_dep`**: Scheduled departure clock time.
8. **`act_dep`**: Actual departure clock time.
9. **`dep_delay`**: Numerical departure delay in minutes.
10. **Sequential Order**: Records must be ordered by sequential station halts along the train journey.

---

## 14. Potential Sources for Oct 2024 – Jan 2025

| Data Source | Content Type | Suitability for V4 Pipeline | Notes |
| :--- | :--- | :---: | :--- |
| **1. Indian-Railway-Network-and-Delays (Next Months Dump)** | Full sequential stop calls (`train_routes_delays_*.csv`) | **100% Compatible (Ideal)** | Exact same schema, zero transformation needed. |
| **2. NTES / RailYatri / WhereIsMyTrain Archival Scraping** | Station stop arrival/departure JSON logs | **100% Compatible** | Can be converted to CSV using existing `train_delays_*.json` parser. |
| **3. NOAA NCEI Global Hourly (2024 & 2025)** | Hourly weather observations (`2024/`, `2025/`) | **100% Compatible (Ready)** | Accessible immediately via `acquire_ghcnh_weather.py`. |
| **4. Journey-Level Dataset (`ir_train.csv`)** | Single end-to-end trip per train (`delay_minutes`) | **Incompatible (Do NOT use)** | Lacks intermediate station calls, hop sequence, and next-station target. |

---

## 15. Risks & Incompatibilities to Avoid

1. **Do NOT use end-to-end journey summary datasets**: Datasets with one row per train journey cannot produce intermediate next-station segment hops ($S_k \to S_{k+1}$) or compute arrival delays at intermediate junctions.
2. **Do NOT sort railway records by station code**: Sorting alphabetically destroys route sequence and produces invalid `next_station` pairs.
3. **Do NOT use forward-looking statistics**: Segment stats for multi-month evaluation must be static or strictly computed from prior chronological months.
4. **Do NOT impute missing weather as clear weather**: Missing observations must retain `fog_observation_available = 0` and `fog_flag = NaN`.

---

## 16. Recommended Multi-Month Acquisition Strategy

1. **Step 1: NOAA Weather Download (Immediate)**
   - Run `acquire_ghcnh_weather.py` pointing to NOAA NCEI 2024 and 2025 directories for October 2024, November 2024, December 2024, and January 2025.
2. **Step 2: Acquire Matching Sequential Railway Dumps**
   - Obtain monthly `train_routes_delays_{MonthYear}.csv` from the same NTES tracking source for October 2024, November 2024, December 2024, and January 2025.
3. **Step 3: Multi-Month Causal Weather Join**
   - Execute `build_weather_join.py` and `build_v3_weather_features.py` over the combined multi-month railway data.
4. **Step 4: Train on Autumn / Test on Peak Winter Fog**
   - Training: September – November 2024 (~3.6M rows)
   - Validation: December 1–15, 2024 (~600k rows)
   - Unseen Test: December 16, 2024 – January 31, 2025 (~1.8M rows during peak northern radiation fog).

---

## 17. Leakage Considerations for Multi-Month Pipeline

1. **Historical Segment Statistics**:
   - `past_segment_mean`, `median`, `std`, `count` can continue to use the static baseline `segment_stats.csv`, or be computed strictly using pre-December training splits.
2. **Observation Timestamp Alignment**:
   - Temporal joining uses `searchsorted(side="right") - 1`, ensuring $\text{weather\_ts} \le \text{prediction\_ts}$ across all months.

---

## 18. Production Safety Verification

- Production files remain 100% frozen:
  - `backend/main.py`
  - `backend/model/champion_model_scheduled_segment_v2.txt`
  - `backend/model/model_features_scheduled_segment_v2.json`
  - `backend/model/station_categories_scheduled_segment_v2.json`
  - `backend/model/segment_stats.csv`
- `git diff -- backend/main.py backend/model/` returned **0 changes**.

---

## Direct Audit Answers

### A. What exact railway data do we need?
We need **sequential station-level stop calls** with the schema:
`['train', 'date', 'station', 'sch_arr', 'act_arr', 'arr_delay', 'sch_dep', 'act_dep', 'dep_delay']` ordered chronologically by route halt sequence.

### B. Where can we obtain Oct 2024 – Jan 2025?
- **Weather Data**: Directly from NOAA NCEI Global Hourly public archives (`https://www.ncei.noaa.gov/data/global-hourly/access/2024/` and `2025/`).
- **Railway Data**: From the matching NTES / Indian Railway tracking repository dumps (or corresponding monthly Kaggle/GitHub releases).

### C. Can the existing pipeline process it without changing production?
**Yes, 100%**. The existing scripts [`acquire_ghcnh_weather.py`](file:///d:/SIH-RAILWAY/backend/research/weather/acquire_ghcnh_weather.py), [`build_weather_join.py`](file:///d:/SIH-RAILWAY/backend/research/weather/build_weather_join.py), and [`build_v3_weather_features.py`](file:///d:/SIH-RAILWAY/backend/research/weather/scripts/build_v3_weather_features.py) are parameterized and can process multi-month data entirely within `backend/research/weather/` without touching production files.

### D. What should we download first?
Download the **NOAA GHCNh hourly weather files for Oct 2024 – Jan 2025** for the 79 active Indian stations (publicly accessible via HTTPS), and locate the matching **October 2024 – January 2025 station-level railway delay datasets**.

### E. What should we NOT download/use?
Do **NOT** use trip-level / end-to-end journey datasets like `ir_train.csv` (which have only origin-to-destination overall delays and lack intermediate station hops).
