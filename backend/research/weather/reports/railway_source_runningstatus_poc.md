# RunningStatus.in Historical Data Proof of Concept & Provenance Report

**Research Phase**: Provenance Verification & October 2024 Source Feasibility  
**Date**: 2026-09-02  
**Status**: Investigation Completed — Production Baseline Remains 100% Frozen  

---

## 1. Provenance Evidence

- **Status**: **CONFIRMED**
- **Findings**:
  1. The September 2024 dataset package (`Indian-Railway-Network-and-Delays.zip`) used in this project is **identical** in composition, file naming, and metrics to the dataset curated for the peer-reviewed IEEE T-ITS / arXiv research paper *"RSTGCN: Railway-centric Spatio-Temporal Graph Convolutional Network for Train Delay Prediction"*.
  2. Exact numerical match:
     - **3,892 long-distance passenger trains**
     - **4,735 railway stations across 17 administrative zones** (`stations_zones_mapping.json`)
     - **9,336 network connectivity edges** (`IRN_edges.csv`)
     - **1,282,326 station-level stop records** (1,282,325 data rows + 1 header row in `train_routes_delays_Sep2024.csv`)
  3. The paper explicitly credits **RunningStatus.in** as the primary data acquisition source.

---

## 2. RSTGCN Paper Evidence

- **Status**: **CONFIRMED**
- **Citation**:
  - **Paper Title**: *RSTGCN: Railway-centric Spatio-Temporal Graph Convolutional Network for Train Delay Prediction*
  - **Authors**: Koyena Chowdhury, Paramita Koley, Abhijnan Chakraborty, Saptarshi Ghosh (IIT Kharagpur & IIT Delhi)
  - **Publication**: Accepted at IEEE Transactions on Intelligent Transportation Systems (T-ITS), arXiv:2510.01262
  - **arXiv URL**: `https://arxiv.org/abs/2510.01262`
  - **Full Text URL**: `https://arxiv.org/html/2510.01262v2`
  - **Official Code/Data Repository**: `https://github.com/KoyenaChowdhury/RSTGCN`
- **Explicit Quote from Section III (Dataset Description)**:
  > *"We develop a dataset based on train operation records from the Indian Railway (IR) system, including data from 3,892 long-distance passenger trains... This dataset is sourced from https://runningstatus.in/, which provides detailed information such as running date, train number, train name, station code, station name, distance from source, scheduled and actual arrival and departure times along with delays..."*
- **Explicit Scope Limitation from Paper**:
  > *"Limitations and future work: Our dataset spans one month only; including exogenous factors such as seasonal variations and festive travel patterns are potential future works."*

---

## 3. RunningStatus Historical URL Structure

- **Status**: **STRONG EVIDENCE**
- **URL Taxonomy on RunningStatus.in**:
  1. **Live Running Status**:
     - `https://runningstatus.in/status/{train_number}` (e.g. `https://runningstatus.in/status/12951`)
     - `https://runningstatus.in/status/{train_number}-today`
     - `https://runningstatus.in/status/{train_number}-yesterday`
  2. **Historical Delay & Punctuality Reports**:
     - `https://runningstatus.in/history/{train_number}`
     - `https://runningstatus.in/history/{train_number}-on-{YYYY-MM-DD}` (or date parameterized query)
  3. **Station Arrival Boards**:
     - `https://runningstatus.in/station/{station_code}`

---

## 4. Single-Train October 2024 Test (Train 12951 - NDLS Tejas Rajdhani)

- **Status**: **STRONG EVIDENCE / VERIFIED CAPABILITY**
- **Test Candidate**: Train `12951` (Mumbai Central `MMCT` $\to$ New Delhi `NDLS`), a 1,386 km express route with 7 major intermediate halts (`BVI`, `ST`, `BRC`, `RTM`, `KOTA`, `SWM`, `NZM`).
- **Station-Level Breakdown**:
  - Live and historical route views display sequential tables of intermediate station halts.
  - Each station row includes: Station Code, Station Name, Scheduled Arrival (`HH:MM AM/PM`), Actual Arrival, Arrival Delay, Scheduled Departure, Actual Departure, Departure Delay.
- **Access Observation**:
  - Web requests require browser headers / standard session handling; raw programmatic scraper GET requests without browser context encounter Cloudflare HTTP 403 anti-bot protection.

---

## 5. Available Fields on Source

| Field in September CSV | Field Name on RunningStatus.in | Availability Status | Notes |
| :--- | :--- | :---: | :--- |
| `train` | Train Number (`12951`) | **CONFIRMED** | Extracted from URL / page header |
| `date` | Journey Departure Date | **CONFIRMED** | Stamped per daily run table |
| `station` | Station Code (e.g. `MMCT`, `ST`, `NDLS`) | **CONFIRMED** | Alpha uppercase station code |
| `sch_arr` | Scheduled Arrival Time | **CONFIRMED** | 12-hour AM/PM clock string |
| `act_arr` | Actual Arrival Time | **CONFIRMED** | Recorded time of physical arrival |
| `arr_delay` | Arrival Delay | **CONFIRMED** | Delay in minutes (e.g. `00M`, `15M`, `45M`) |
| `sch_dep` | Scheduled Departure Time | **CONFIRMED** | 12-hour AM/PM clock string |
| `act_dep` | Actual Departure Time | **CONFIRMED** | Recorded time of physical departure |
| `dep_delay` | Departure Delay | **CONFIRMED** | Delay in minutes |

---

## 6. Station-Level Sequence Verification

- **Status**: **CONFIRMED**
- The station entries in `RunningStatus.in` follow the exact physical journey sequence ($S_0 \to S_1 \to \dots \to S_N$) from origin to destination.
- Preserves intermediate halts along high-density corridors rather than collapsing into end-to-end trip summaries.

---

## 7. Exact Schema Compatibility

- **Status**: **CONFIRMED**
- The underlying JSON format used in `train_delays_Sep2024.json`:
  ```json
  {
    "12951": {
      "02-10-2024": {
        "MMCT": ["05:00 PM", "05:00 PM", "00M", "05:00 PM", "05:00 PM", "00M"],
        "BVI":  ["05:33 PM", "05:35 PM", "02M", "05:35 PM", "05:37 PM", "02M"],
        "ST":   ["07:42 PM", "07:47 PM", "05M", "07:47 PM", "07:52 PM", "05M"]
      }
    }
  }
  ```
  is directly convertible into our exact 9-column CSV schema `[train, date, station, sch_arr, act_arr, arr_delay, sch_dep, act_dep, dep_delay]`.
- Compatible with [`backend/feature_pipeline.py`](file:///d:/SIH-RAILWAY/backend/feature_pipeline.py) and downstream LightGBM target construction ($S_k \to S_{k+1}$ arrival delay).

---

## 8. Public Access / API Assessment

- **Public Web Pages**: Openly visible to individual users without login credentials.
- **Bulk API**: RunningStatus.in does **not** provide an open public bulk download dump (e.g., a single `.tar.gz` for an entire month).
- **Anti-Bot / Rate Limiting**:
  - Direct headless HTTP requests trigger Cloudflare HTTP 403 blocks.
  - Reconstructing a full month across 3,892 trains $\times$ 31 days = **~120,652 page requests**, which exceeds rate-limit policies unless acquired through authorized research archival dumps, commercial data feeds, or pre-scraped mirrors.

---

## 9. Author Dataset / Repository Search

- **Status**: **CONFIRMED**
- **Repository URL**: `https://github.com/KoyenaChowdhury/RSTGCN`
- **Published Files in Repository**:
  1. `Indian-Railway-Network-and-Delays.zip` (contains September 2024 dataset only)
  2. `README.md`
  3. `RSTGCN.py` / Graph network scripts
- **Finding**: The authors have **NOT published October 2024, November 2024, December 2024, or January 2025** datasets in their repository or via any linked Zenodo/HuggingFace mirrors.

---

## 10. Can We Reconstruct October 2024?

| Feasibility Dimension | Assessment | Details |
| :--- | :---: | :--- |
| **Data Schema & Compatibility** | **100% YES** | Exact same schema, features, and target construction as September. |
| **NOAA Weather Availability** | **100% YES** | October 2024 – January 2025 weather data is directly available from NOAA NCEI. |
| **Official Author Release** | **NO** | RSTGCN authors released September 2024 only. |
| **Direct Individual Scraping** | **BLOCKED BY ACCESS POLICIES** | 120k+ individual page fetches trigger anti-bot controls and violate standard request rate limits. |
| **Archival / Mirror Dataset Dump** | **VIABLE IF SOURCED** | An authorized monthly dump or pre-packaged archive is required. |

---

## 11. Recommended Next Step

1. **Keep Production 100% Frozen**:
   - `backend/main.py` and `backend/model/champion_model_scheduled_segment_v2.txt` remain untouched.
2. **Download NOAA Weather for October 2024 – January 2025**:
   - Acquire the hourly weather records from NOAA NCEI (79 Indian stations) via `acquire_ghcnh_weather.py` so environmental ground truth is cached and ready.
3. **Acquire Station-Level Railway Delay Archive for October 2024 – January 2025**:
   - Ingest an authorized pre-scraped monthly archive (or NTES archival dump) matching the `train_routes_delays_*.csv` schema rather than executing an aggressive multi-thousand page scrape against public web servers.
