# Railway Dataset Source Investigation

**Research Phase**: Historical Railway Data Provenance & Post-September Availability  
**Date**: 2026-09-02  
**Status**: Investigation Completed — Production Baseline Remains 100% Frozen  

---

## 1. September Dataset

The baseline and weather research experiments rely on the September 2024 station-level sequential dataset:

- **Filename**: `train_routes_delays_Sep2024.csv`
- **Location in Environment**:
  `C:\Users\SPARSH MAURYA\Downloads\Indian-Railway-Network-and-Delays\Indian-Railway-Network-and-Delays\train_routes_delays_Sep2024.csv`
- **Dataset Metrics**:
  - **Raw Stop Records**: 1,282,325 rows
  - **Unique Trains**: 3,892 trains
  - **Unique Stations**: 4,736 station codes
  - **Date Range**: `2024-09-01` to `2024-09-30` (30 calendar days)
  - **Valid Prediction Segments**: Exactly **1,224,840 rows** ($S_k \to S_{k+1}$) after dropping 57,485 terminal stops.
- **Exact Column Schema**:
  ```python
  [
      "train",      # Train number identifier (e.g. "12303")
      "date",       # Journey date string (e.g. "2024-09-02")
      "station",    # Station uppercase code (e.g. "HWH", "BWN", "DGR")
      "sch_arr",    # Scheduled arrival clock string (e.g. "08:00 AM")
      "act_arr",    # Actual arrival clock string (e.g. "08:01 AM")
      "arr_delay",  # Numerical arrival delay in minutes (e.g. 0.0, 15.0)
      "sch_dep",    # Scheduled departure clock string (e.g. "08:00 AM")
      "act_dep",    # Actual departure clock string (e.g. "08:01 AM")
      "dep_delay",  # Numerical departure delay in minutes (e.g. 0.0, 15.0)
  ]
  ```

---

## 2. Confirmed Original Source

The September 2024 dataset package originated from:
- **Archive Name**: `Indian-Railway-Network-and-Delays.zip` (24,041,934 bytes)
- **Data Generator**: An Indian Railways NTES (National Train Enquiry System) live tracking scraper and network mapping compilation.
- **Companion Files in Archive**:
  1. `train_delays_Sep2024.json` (94,143,002 bytes): Raw station stop array logs: `{"12303": {"02-09-2024": {"HWH": ["08:00 AM", "08:01 AM", "00M", "08:00 AM", "08:01 AM", "00M"], ...}}}`
  2. `train_routes_Sep2024.csv` (5,353,376 bytes): Route halt sequences and station names.
  3. `stations_zones_mapping.json` (66,202 bytes): 4,735 railway station administrative zone mappings.
  4. `IRN_edges.csv` (135,055 bytes): Network graph adjacency edges.
- **Geographic & Weather Metadata Source**:
  - Railway station coordinates: DataMeet Open Indian Railways Master Dataset (`https://raw.githubusercontent.com/datameet/railways/master/stations.json`)
  - Weather observations: NOAA NCEI Global Hourly Data Access (`https://www.ncei.noaa.gov/data/global-hourly/access/2024/`)

---

## 3. Source/GitHub/Archive Evidence

Inspection of codebase references and external provenance:
1. **Extraction Notebook**: [`notebooks/ETA_MODEL26.ipynb`](file:///d:/SIH-RAILWAY/notebooks/ETA_MODEL26.ipynb#L19-L35) documents the initial extraction and validation of `train_routes_delays_Sep2024.csv` from `Indian-Railway-Network-and-Delays.zip`.
2. **Feature Engineering Pipeline**: [`backend/feature_pipeline.py`](file:///d:/SIH-RAILWAY/backend/feature_pipeline.py) and [`backend/train_scheduled_segment_v2.py`](file:///d:/SIH-RAILWAY/backend/train_scheduled_segment_v2.py) establish the causal segment hop mechanics and LightGBM V2 model training on this exact schema.
3. **DataMeet Master Repository**:
   - URL: `https://github.com/datameet/railways`
   - Role: Provides geocoded coordinates (`stations.json`) for 4,400+ Indian railway stations.
4. **NOAA ISD/GHCNh Global Hourly Repository**:
   - URL: `https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv`
   - Data Access URL (2024): `https://www.ncei.noaa.gov/data/global-hourly/access/2024/`
   - Data Access URL (2025): `https://www.ncei.noaa.gov/data/global-hourly/access/2025/`

---

## 4. October 2024 Availability

- **Status**: **UNAVAILABLE from the original single-month zip package**.
- **Evidence**:
  - The `Indian-Railway-Network-and-Delays.zip` package was published strictly as a September 2024 snapshot (`2024-09-01` to `2024-09-30`).
  - No file named `train_routes_delays_Oct2024.csv` or `train_delays_Oct2024.json` was packaged in the release.
- **NOAA Weather Status**: **AVAILABLE** (100% accessible via NOAA NCEI 2024 directory).

---

## 5. November 2024 Availability

- **Status**: **UNAVAILABLE from the original single-month zip package**.
- **NOAA Weather Status**: **AVAILABLE** via NOAA NCEI 2024 directory.

---

## 6. December 2024 Availability

- **Status**: **UNAVAILABLE from the original single-month zip package**.
- **NOAA Weather Status**: **AVAILABLE** via NOAA NCEI 2024 directory.
- **Context**: Marks the onset of the northern Indo-Gangetic winter radiation fog period.

---

## 7. January 2025 Availability

- **Status**: **UNAVAILABLE from the original single-month zip package**.
- **NOAA Weather Status**: **AVAILABLE** via NOAA NCEI 2025 directory (`https://www.ncei.noaa.gov/data/global-hourly/access/2025/`).
- **Context**: Peak dense-fog period (sub-200m visibility) across Northern Railway corridors.

---

## 8. Candidate Datasets Rejected

| Dataset Inspected | Location / Source | Rejection Reason |
| :--- | :--- | :--- |
| **`ir_train.csv`** | `C:\Users\SPARSH MAURYA\Downloads\indian-railways-predict-train-delay.zip` | **REJECTED (Incompatible Granularity)**: Contains only end-to-end journey summary rows (`distance_km`, `delay_minutes` at destination). Lacks intermediate station halts, hop sequences, and next-station arrival delays ($S_k \to S_{k+1}$). |
| **`Train_operation_data.csv`** | `C:\Users\SPARSH MAURYA\Downloads\delay-weather.zip` | **REJECTED (Non-Indian Network)**: European railway operations (Milan, Chiasso, Bolzano in Italy/Switzerland). |
| **Static `schedules.json`** | `https://github.com/datameet/railways` | **REJECTED as standalone delay source**: Contains scheduled timetable times, but lacks actual observed arrival/departure delays (`act_arr`, `arr_delay`). |

---

## 9. Compatibility Assessment

To be ingested by the existing [`backend/feature_pipeline.py`](file:///d:/SIH-RAILWAY/backend/feature_pipeline.py) and [`backend/research/weather/scripts/build_v3_weather_features.py`](file:///d:/SIH-RAILWAY/backend/research/weather/scripts/build_v3_weather_features.py) pipelines, any candidate dataset for Oct 2024 – Jan 2025 must satisfy:

1. **Station-Level Sequential Structure**: Each row must represent an individual station stop call along the train route sequence.
2. **Mandatory Fields**: `['train', 'date', 'station', 'sch_arr', 'act_arr', 'arr_delay', 'sch_dep', 'act_dep', 'dep_delay']`.
3. **Physical Journey Ordering**: Rows must follow physical progression ($S_0 \to S_1 \to \dots \to S_N$).
4. **Target Validity**: Must support downstream shift `df.groupby(['train', 'date'])['arr_delay'].shift(-1)` to construct `target_delay`.

---

## 10. Recommended Next Step

1. **NOAA GHCNh Weather Download**:
   - Download the hourly weather CSVs for October 2024 – January 2025 for the 79 active Indian stations using the existing [`backend/research/weather/acquire_ghcnh_weather.py`](file:///d:/SIH-RAILWAY/backend/research/weather/acquire_ghcnh_weather.py) pipeline.
2. **Source Matching Railway Station-Level Delay Logs**:
   - Obtain station-level NTES / Indian Railway tracking logs for October 2024, November 2024, December 2024, and January 2025 matching the verified September schema.
3. **Zero Production Risk**:
   - All multi-month joining and model benchmarking remains isolated strictly inside `backend/research/weather/`. Production model `champion_model_scheduled_segment_v2.txt` and `backend/main.py` remain 100% frozen.

---

## Final Verdict

- **OCTOBER 2024**: **UNAVAILABLE** in current download package (requires acquiring station-level sequential delay logs).
- **NOVEMBER 2024**: **UNAVAILABLE** in current download package.
- **DECEMBER 2024**: **UNAVAILABLE** in current download package.
- **JANUARY 2025**: **UNAVAILABLE** in current download package.

**RECOMMENDED NEXT ACTION**:
Acquire station-level sequential delay datasets (`train_routes_delays_*.csv` or `train_delays_*.json`) for October 2024 – January 2025 matching the September schema, while downloading the corresponding NOAA GHCNh weather observations for those months.
