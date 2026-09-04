# Railway Future Data Source Investigation

**Research Phase**: Multi-Month Data Sourcing & Acquisition Strategy  
**Date**: 2026-09-02  
**Status**: Investigation Completed — Production Baseline Remains 100% Frozen  

---

## 1. Objective

This investigation establishes a legitimate, practical, and ethically compliant strategy for acquiring station-level Indian Railway historical operational data for **October 2024, November 2024, December 2024, and January 2025**. 

The goal is to evaluate candidate V4 weather/fog models against severe winter radiation fog conditions without violating website access controls, terms of service, or making unverified assumptions.

---

## 2. September Dataset Provenance

- **Local File**: `train_routes_delays_Sep2024.csv` (84.9 MB, 1,282,325 rows)
- **Extracted From**: `Indian-Railway-Network-and-Delays.zip` (24,041,934 bytes)
- **Origin Paper**: *"RSTGCN: Railway-centric Spatio-Temporal Graph Convolutional Network for Train Delay Prediction"* (IEEE Transactions on Intelligent Transportation Systems / [arXiv:2510.01262](https://arxiv.org/abs/2510.01262))
- **Authors**: Koyena Chowdhury, Paramita Koley, Abhijnan Chakraborty, Saptarshi Ghosh (IIT Kharagpur & IIT Delhi)
- **Primary Scrape Source**: `https://runningstatus.in/`
- **Scope**: 3,892 trains, 4,735 stations across 17 zones, 9,336 network edges, 1,224,840 valid prediction segments ($S_k \to S_{k+1}$) across September 1–30, 2024.

---

## 3. Sources Investigated

1. **RSTGCN Authors & Repositories**: GitHub (`KoyenaChowdhury/RSTGCN`), arXiv preprints, author profile repositories, branches, and commits.
2. **RunningStatus.in**: Historical train route pages, API endpoints, sitemaps, robots policy, and access protection.
3. **Public Academic & ML Data Portals**: Kaggle, Hugging Face, Zenodo, Figshare, DataMeet, Semantic Scholar, GitHub public repositories.
4. **Government & Official Indian Railways Portals**: NTES (National Train Enquiry System), CRIS (Centre for Railway Information Systems), data.gov.in, Dataful.

---

## 4. Author / RSTGCN Investigation

- **Official GitHub Repo**: `https://github.com/KoyenaChowdhury/RSTGCN`
- **Repository Inspection**:
  - Contains a single branch (`main`) at commit `3725906` with files: `Indian-Railway-Network-and-Delays.zip`, `README.md`, and `RSTGCN.py`.
  - Remote inspection (`git ls-remote`) confirmed **zero secondary branches, tags, or pull requests**.
  - Author GitHub profile (`KoyenaChowdhury`) hosts only two public repositories: `RSTGCN` and `KoyenaChowdhury.github.io`.
- **Paper Statement on Scope**:
  - In Section VIII (Limitations & Future Work), the authors explicitly state:
    > *"Our dataset spans one month only; including exogenous factors such as seasonal variations and festive travel patterns are potential future works."*
  - **Status**: The authors **did not publish** October 2024, November 2024, December 2024, or January 2025 in their repository or any linked open mirrors.
- **Author Contact Option**:
  - Koyena Chowdhury (Lead Author, IIT Kharagpur): `koyenachowdhury@kgpian.iitkgp.ac.in`
  - Prof. Saptarshi Ghosh (Faculty Advisor, IIT Kharagpur): `saptarshi@cse.iitkgp.ac.in`
  - Prof. Abhijnan Chakraborty (Co-Author, IIT Delhi): `abhijnan@iitd.ac.in`
  - Reaching out directly for academic research collaboration regarding whether their automated collection scripts continued running into winter 2024–2025 is a legitimate acquisition path.

---

## 5. RunningStatus.in Investigation

- **Platform Nature**: A public web interface displaying live running status, timetable schedules, and historical performance charts for Indian Railways trains.
- **Data Granularity**: Web pages display station-level tables containing Scheduled Arrival, Actual Arrival, Arrival Delay, Scheduled Departure, Actual Departure, and Departure Delay for intermediate halts in journey order.
- **Access Assessment**:
  - **No Open Bulk Download**: RunningStatus.in does not provide downloadable monthly bulk dumps (`.csv.gz` or `.json.tar`).
  - **No Public API**: No documented unauthenticated public REST API exists for bulk historical retrieval.
  - **Cloudflare & Rate Limiting**: Programmatic HTTP GET requests trigger Cloudflare HTTP 403 anti-bot blocks.
  - **Scraping Feasibility**: To reconstruct Oct 2024 – Jan 2025 (123 days $\times$ 3,892 trains) would require **478,716 individual HTML page fetches**. Attempting this scale of automated scraping without authorization violates access policies, triggers IP blocks, and is unacceptable.

---

## 6. Public Dataset & Archive Search

| Platform | Query / Dataset Name | Finding | Compatibility Assessment |
| :--- | :--- | :--- | :--- |
| **Kaggle** | `naijilaji/indian-railways-train-delays-dataset-2025` | Aggregated delay statistics for ~1,900 train-station pairs (average delay, right-time %). | **REJECTED**: Summary aggregates only; lacks daily operational stop timestamps. |
| **Kaggle** | `indian-railways-predict-train-delay` (`ir_train.csv`) | 1.5M journey-level rows with origin/destination and total destination delay. | **REJECTED**: Lacks intermediate station halts, hop sequences, and next-station arrival delays. |
| **Kaggle** | `delay-weather.zip` (`Train_operation_data.csv`) | Station-level operations for Milan, Chiasso, Bolzano. | **REJECTED**: European railway network, not Indian Railways. |
| **Hugging Face** | `orailix/ride-*` & `faizmubeen/*` | Belgian railway benchmark and social media passenger complaint tweets. | **REJECTED**: Irrelevant to Indian Railways operational delays. |
| **Zenodo / Figshare** | `RSTGCN`, `Indian Railway Network` | Preprint metadata indexed; points to GitHub repository `KoyenaChowdhury/RSTGCN` (Sep 2024 only). | **NO ADDITIONAL MONTHS FOUND**. |
| **DataMeet** | `datameet/railways` | Master station geocodes (`stations.json`) and static timetables (`schedules.json`). | **PARTIAL**: Provides coordinates and schedules, but no actual delay timestamps. |

---

## 7. Government & Official Railway Sources

1. **NTES (National Train Enquiry System)** (`https://enquiry.indianrail.gov.in/`):
   - Authoritative live operational tracking system operated by CRIS.
   - Feeds downstream public apps and aggregators.
   - Does not offer unauthenticated bulk historical data exports for public download.
2. **data.gov.in (Open Government Data Portal India)**:
   - Publishes station directories, network line lengths, and annual/monthly punctuality indices aggregated by zone (e.g. Northern Railway punctuality %).
   - Does not publish transaction-level daily train halt logs.
3. **CRIS (Centre for Railway Information Systems)**:
   - Manages internal operational logs. Formal research data agreements through academic institutions are the designated channel for institutional research datasets.

---

## 8. Compatibility Matrix

To be compatible with [`backend/feature_pipeline.py`](file:///d:/SIH-RAILWAY/backend/feature_pipeline.py) and downstream LightGBM segment targets ($S_k \to S_{k+1}$), a candidate dataset must satisfy all structural criteria below:

| Structural Criterion | `train_routes_delays_Sep2024.csv` | RunningStatus Individual Page | Kaggle `ir_train.csv` | Kaggle `naijilaji-2025` |
| :--- | :---: | :---: | :---: | :---: |
| **Station-Level Records** | YES | YES | NO (Trip-level) | NO (Aggregate) |
| **Intermediate Halts Preserved** | YES | YES | NO | NO |
| **Physical Journey Order** | YES | YES | NO | NO |
| **Scheduled Arrival / Departure** | YES | YES | NO | NO |
| **Actual Arrival / Departure** | YES | YES | NO | NO |
| **Numerical Delay (Minutes)** | YES | YES | YES (Destination only) | YES (Average only) |
| **Next-Station Target Derivable** | YES | YES | NO | NO |
| **Bulk Dataset Available** | **YES (Local)** | **NO** | YES | YES |

---

## 9. Temporal Coverage Matrix

| Source / Repository | Sep 2024 | Oct 2024 | Nov 2024 | Dec 2024 | Jan 2025 | Overall Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **KoyenaChowdhury/RSTGCN** | **YES** | **NO** | **NO** | **NO** | **NO** | Single-month benchmark package. |
| **RunningStatus.in (Web Interface)** | **YES** | **PARTIAL** | **PARTIAL** | **PARTIAL** | **PARTIAL** | Individual queries visible; bulk download unavailable; scraping blocked. |
| **NOAA GHCNh Hourly Weather** | **YES** | **YES** | **YES** | **YES** | **YES** | **100% Available** via NOAA NCEI public HTTPS archives. |
| **Kaggle `ir_train.csv`** | NO | NO | NO | NO | NO | Incompatible schema (Trip summary). |
| **data.gov.in / CRIS Public** | NO | NO | NO | NO | NO | High-level zonal statistics only. |

---

## 10. Access & Authorization Assessment

| Source | Access Classification | Practical Feasibility |
| :--- | :--- | :--- |
| **September 2024 Existing Dataset** | **Open Public Download (Local)** | **100% Ready & Ingested** (1,224,840 prediction segments). |
| **NOAA GHCNh Weather (Oct 2024 – Jan 2025)** | **Open Public HTTPS Archive** | **100% Ready** via `acquire_ghcnh_weather.py`. |
| **RSTGCN Author Inquiry (Oct 2024 – Jan 2025)** | **Research-Accessible (Direct Contact)** | **High Potential**: Inquire if authors maintained scrapers during autumn/winter. |
| **Licensed / Partner API Feeds (RailRadar/ETrain)** | **Commercial / Partner API** | **Viable for production/research partner data feeds**. |
| **RunningStatus.in Large-Scale Scraping** | **Blocked / Restricted** | **Prohibited**: Violates rate limits and anti-bot policies (478k requests). |

---

## 11. Best Available Options (Ranked)

1. **Option 1 (Recommended Academic Route — Author Inquiry)**:
   - Contact the RSTGCN research authors at IIT Kharagpur (`koyenachowdhury@kgpian.iitkgp.ac.in`, `saptarshi@cse.iitkgp.ac.in`). Inquire whether they collected post-September 2024 station-level logs for ongoing research extensions.
2. **Option 2 (Pre-Ingest NOAA Winter Weather Data)**:
   - Download NOAA GHCNh weather observations for October 2024 – January 2025 (79 Indian stations) using the existing [`acquire_ghcnh_weather.py`](file:///d:/SIH-RAILWAY/backend/research/weather/acquire_ghcnh_weather.py) script. The environmental ground truth for fog events will be cached and ready for immediate pairing once railway data is obtained.
3. **Option 3 (Commercial / Partner Historical Ingestion)**:
   - Utilize partner data access (e.g. through RailRadar or commercial train tracking API partners) to obtain historical station-by-station operational logs.
4. **Option 4 (September Out-of-Sample Fog Robustness Deep-Dive)**:
   - Maximize rigorous evaluation on the existing 1,224,840-row September dataset: perform spatial cross-validation, route-cluster holdouts, severe-weather bootstrapping, and latency sensitivity analyses (as completed in our V4 validation report).

---

## 12. Recommended Next Step

1. **Keep Production 100% Frozen**: Zero changes to `backend/main.py`, `backend/model/champion_model_scheduled_segment_v2.txt`, or production ETA cascade.
2. **Download NOAA Weather for Winter Months**: Execute `acquire_ghcnh_weather.py` for October 2024 – January 2025 so weather ground truth is in place.
3. **Inquire with RSTGCN Authors / Sourcing Channels**: Pursue direct academic contact with the IIT Kharagpur research team to request multi-month operational data dumps.

---

## 13. Evidence & URLs

- **RSTGCN arXiv Paper**: `https://arxiv.org/abs/2510.01262`
- **RSTGCN Full Text**: `https://arxiv.org/html/2510.01262v2`
- **RSTGCN Official GitHub**: `https://github.com/KoyenaChowdhury/RSTGCN`
- **NOAA NCEI Global Hourly (2024)**: `https://www.ncei.noaa.gov/data/global-hourly/access/2024/`
- **NOAA NCEI Global Hourly (2025)**: `https://www.ncei.noaa.gov/data/global-hourly/access/2025/`
- **DataMeet Railway Master Stations**: `https://github.com/datameet/railways`
- **Kaggle Delays Dataset (Aji 2025)**: `https://www.kaggle.com/datasets/naijilaji/indian-railways-train-delays-dataset-2025`
- **Kaggle Predict Train Delay Competition**: `https://www.kaggle.com/competitions/133990/overview`

---

## 14. Production Safety Verification

```bash
git diff -- backend/main.py backend/model/
```
- **0 changes in production code or champion model artifacts**.
- Production model hash: `bbd06bc91ae20c9aee8366cb917589553effeb353c5e5442add08179db982c02` (verified unchanged).
