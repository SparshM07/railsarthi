# Railway Dataset Availability Audit: October 2024 – January 2025

**Research Phase**: Pre-Acquisition Availability & Provenance Verification  
**Date**: 2026-09-02  
**Status**: Verification Completed — Production Baseline Remains 100% Frozen  

---

## 1. Original September 2024 Source

- **Original Archive**: `Indian-Railway-Network-and-Delays.zip` (24,041,934 bytes)
- **Local Location**: `C:\Users\SPARSH MAURYA\Downloads\Indian-Railway-Network-and-Delays\Indian-Railway-Network-and-Delays\`
- **Key Sequential Stop Dataset**: `train_routes_delays_Sep2024.csv` (84,928,205 bytes, 1,282,325 rows)
- **Companion Files**:
  - `train_delays_Sep2024.json` (94,143,002 bytes — raw station-level stop array logs)
  - `train_routes_Sep2024.csv` (5,353,376 bytes — route topology and station metadata)
  - `stations_zones_mapping.json` (66,202 bytes — 4,735 railway station administrative zone mappings)
  - `IRN_edges.csv` (135,055 bytes — graph network connectivity)
- **Temporal Coverage**: Strictly `2024-09-01` to `2024-09-30` (30 calendar days).
- **Exact Schema**:
  ```python
  [
      "train",      # e.g. "12303"
      "date",       # e.g. "2024-09-02"
      "station",    # e.g. "HWH", "BWN", "DGR"
      "sch_arr",    # e.g. "08:00 AM"
      "act_arr",    # e.g. "08:01 AM"
      "arr_delay",  # float minutes, e.g. 0.0, 15.0
      "sch_dep",    # e.g. "08:00 AM"
      "act_dep",    # e.g. "08:01 AM"
      "dep_delay",  # float minutes, e.g. 0.0, 15.0
  ]
  ```

---

## 2. October 2024 Availability Assessment

- **File Status**: **UNAVAILABLE** in the current repository and local system.
- **Source Inspection**: The `Indian-Railway-Network-and-Delays` package was released as a single-month dataset specifically capturing September 2024. No official companion file named `train_routes_delays_Oct2024.csv` or `train_delays_Oct2024.json` exists within the downloaded distribution.
- **Web / Public Index Search**: Comprehensive search across public repositories confirmed that the exact September dataset author did not bundle subsequent months in the same package.
- **NOAA Weather Status for October 2024**: **AVAILABLE** (100% accessible via NOAA NCEI Global Hourly HTTPS archive `https://www.ncei.noaa.gov/data/global-hourly/access/2024/`).

---

## 3. November 2024 Availability Assessment

- **File Status**: **UNAVAILABLE** in the current repository and local system.
- **NOAA Weather Status for November 2024**: **AVAILABLE** via NOAA NCEI Global Hourly 2024 archives.

---

## 4. December 2024 Availability Assessment

- **File Status**: **UNAVAILABLE** in the current repository and local system.
- **NOAA Weather Status for December 2024**: **AVAILABLE** via NOAA NCEI Global Hourly 2024 archives.
- **Significance**: December marks the onset of the northern radiation fog corridor across Punjab, Haryana, Uttar Pradesh, and Bihar.

---

## 5. January 2025 Availability Assessment

- **File Status**: **UNAVAILABLE** in the current repository and local system.
- **NOAA Weather Status for January 2025**: **AVAILABLE** via NOAA NCEI Global Hourly 2025 archive (`https://www.ncei.noaa.gov/data/global-hourly/access/2025/`).
- **Significance**: Peak winter dense-fog season (critical for validating severe-visibility threshold features).

---

## 6. Exact Filenames & Local Artifact Inventory

### Local Existing Files (September 2024 Only):
1. `train_routes_delays_Sep2024.csv` (84.9 MB) — Present
2. `train_delays_Sep2024.json` (94.1 MB) — Present
3. `train_routes_Sep2024.csv` (5.4 MB) — Present

### Required Target Filenames for Multi-Month Expansion:
1. `train_routes_delays_Oct2024.csv` (or JSON equivalent) — **Missing / To be acquired**
2. `train_routes_delays_Nov2024.csv` (or JSON equivalent) — **Missing / To be acquired**
3. `train_routes_delays_Dec2024.csv` (or JSON equivalent) — **Missing / To be acquired**
4. `train_routes_delays_Jan2025.csv` (or JSON equivalent) — **Missing / To be acquired**

---

## 7. Audit of Other Local Archives in Downloads

During the investigation, other candidate datasets in the local environment were inspected for potential multi-month coverage:

1. **`indian-railways-predict-train-delay.zip` (`ir_train.csv`, 337.2 MB)**:
   - **Inspection Result**: **INCOMPATIBLE**.
   - **Reason**: Contains journey-level end-to-end trips (`['journey_id', 'train_number', 'departure_date', 'distance_km', 'delay_minutes']`). Lacks intermediate station halts, sequential hop timestamps, and downstream next-station arrival delays ($S_k \to S_{k+1}$). Cannot be converted into prediction segments.
2. **`delay-weather.zip` (`Train_operation_data.csv`, 412.5 MB)**:
   - **Inspection Result**: **INCOMPATIBLE**.
   - **Reason**: Contains European railway operations (Italian/Swiss network: Milan, Chiasso, Bolzano), not Indian Railways.

---

## 8. Schema & Invariant Comparison

To be compatible with the existing [`backend/feature_pipeline.py`](file:///d:/SIH-RAILWAY/backend/feature_pipeline.py) and [`backend/research/weather/scripts/build_v3_weather_features.py`](file:///d:/SIH-RAILWAY/backend/research/weather/scripts/build_v3_weather_features.py) pipelines, future monthly files must satisfy the following strict contract:

| Field | Required Type | Semantic Requirement | September Status | Future Requirement |
| :--- | :--- | :--- | :---: | :---: |
| `train` | string / int | Train number identifier | Present | **Mandatory** |
| `date` | `YYYY-MM-DD` | Train journey start/halt date | Present | **Mandatory** |
| `station` | string | Station code (e.g. `NDLS`) | Present | **Mandatory** |
| `sch_arr` | string | Scheduled arrival time (`HH:MM AM/PM` or `HH:MM`) | Present | **Mandatory** |
| `act_arr` | string | Actual arrival time | Present | **Mandatory** |
| `arr_delay` | float | Arrival delay in minutes at current halt | Present | **Mandatory** |
| `sch_dep` | string | Scheduled departure time | Present | **Mandatory** |
| `act_dep` | string | Actual departure time | Present | **Mandatory** |
| `dep_delay` | float | Departure delay in minutes | Present | **Mandatory** |
| **Row Order** | sequential | Preserves train route sequence ($S_0 \to S_1 \to \dots \to S_N$) | Present | **Mandatory** |

---

## 9. Ordering & Spatial-Temporal Semantics

1. **Physical Journey Order**:
   - Ingested records must follow the sequential journey progression. Station codes must **never** be alphabetically sorted prior to hop derivation.
2. **Target Formation**:
   - `target_delay = df.groupby(['train', 'date'])['arr_delay'].shift(-1)`
   - Terminal stop rows (where `next_station` is `NaN`) are dropped.
3. **Causal Weather Join**:
   - Scheduled/actual timestamps in IST are mapped to UTC seconds and joined to NOAA hourly records via `np.searchsorted(weather_ts, train_ts, side="right") - 1` (guaranteeing $\text{weather\_ts} \le \text{prediction\_ts}$).

---

## 10. Compatibility Assessment

- **Pipeline Compatibility**: **100% Ready**. The existing scripts in [`backend/research/weather/`](file:///d:/SIH-RAILWAY/backend/research/weather/) require **zero code modifications** to process October 2024 – January 2025 once the matching station-level CSV/JSON files are supplied.
- **Production Safety**: The multi-month validation and training pipelines operate exclusively in `backend/research/weather/`. Production model `champion_model_scheduled_segment_v2.txt` and `backend/main.py` remain completely untouched.

---

## 11. Recommended Acquisition Strategy

1. **NOAA Weather Data (Ready for Immediate Download)**:
   - Run [`backend/research/weather/acquire_ghcnh_weather.py`](file:///d:/SIH-RAILWAY/backend/research/weather/acquire_ghcnh_weather.py) for the 79 Indian weather stations for months `2024-10`, `2024-11`, `2024-12`, and `2025-01`.
2. **Railway Stop Data (Acquisition Path)**:
   - Source the matching monthly NTES / Indian Railway tracking station-level dumps for October 2024, November 2024, December 2024, and January 2025.
   - If raw data is obtained in daily JSON format (`train_delays_{month}.json`), parse it into tabular CSV using the existing JSON-to-CSV parser.

---

## 12. Blockers

- **Direct Blocker**: The raw station-level railway dataset for October 2024 (and Nov 2024 – Jan 2025) is not currently stored locally in the environment.
- **No Pipeline Blocker**: All feature extraction, station mapping, temporal joining, and LightGBM model training logic are fully implemented and verified.

---

## Final Verdict

| Month | Dataset Availability Status |
| :--- | :--- |
| **OCTOBER 2024** | **UNAVAILABLE** (Requires downloading/sourcing matching station-level delay dump) |
| **NOVEMBER 2024** | **UNAVAILABLE** (Requires downloading/sourcing matching station-level delay dump) |
| **DECEMBER 2024** | **UNAVAILABLE** (Requires downloading/sourcing matching station-level delay dump) |
| **JANUARY 2025** | **UNAVAILABLE** (Requires downloading/sourcing matching station-level delay dump) |

---

## RECOMMENDED NEXT ACTION

**Source and provide the October 2024 – January 2025 station-level sequential railway delay dataset (`train_routes_delays_*.csv` or `train_delays_*.json`) matching the verified September schema, while downloading the corresponding NOAA GHCNh weather observations for those months.**
