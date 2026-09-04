# NOAA Winter Weather Data Acquisition & Validation Report

**Research Phase**: Winter Meteorological Data Ingestion (Oct 2024 – Jan 2025)  
**Date**: 2026-09-02  
**Status**: Acquisition Completed — Production Baseline Remains 100% Frozen  

---

## 1. Objective

This report details the acquisition, normalization, and quality validation of NOAA Global Historical Climatology Network hourly (GHCNh / ISD) weather observations covering:
- **October 2024** (`2024-10-01 00:00 UTC` to `2024-10-31 23:59 UTC`)
- **November 2024** (`2024-11-01 00:00 UTC` to `2024-11-30 23:59 UTC`)
- **December 2024** (`2024-12-01 00:00 UTC` to `2024-12-31 23:59 UTC`)
- **January 2025** (`2025-01-01 00:00 UTC` to `2025-01-31 23:59 UTC`)

These data provide the high-resolution ground truth for winter radiation fog, dewpoint depression, and severe low-visibility events across the Indian Railway network.

---

## 2. NOAA Source & Endpoints

- **Data Provider**: NOAA National Centers for Environmental Information (NCEI)
- **Archive Catalog**: Global Hourly Integrated Surface Database (ISD / GHCNh)
- **Official Base URLs**:
  - 2024 Data: `https://www.ncei.noaa.gov/data/global-hourly/access/2024/{station_id}.csv`
  - 2025 Data: `https://www.ncei.noaa.gov/data/global-hourly/access/2025/{station_id}.csv`
- **Metadata Reference**: `https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv`

---

## 3. Station Set & Spatial Mapping

- **Target Network**: Preserves the exact 355 NOAA Indian/South Asian meteorological stations mapped to 4,444 Indian Railway stations via Haversine distance proximity in [`backend/research/weather/data/station_metadata/ghcnh_railway_station_mapping.csv`](file:///d:/SIH-RAILWAY/backend/research/weather/data/station_metadata/ghcnh_railway_station_mapping.csv).
- **Target Railway Coordinates**: DataMeet Indian Railways station geocode catalog (`stations.json`).
- **Station Network Performance**: 344/355 stations successfully downloaded for 2024 and 2025, with 329–334 stations actively reporting hourly observations across the winter months.

---

## 4. Acquisition Method

- **Acquisition Script**: [`backend/research/weather/scripts/acquire_winter_ghcnh.py`](file:///d:/SIH-RAILWAY/backend/research/weather/scripts/acquire_winter_ghcnh.py)
- **Raw Storage Location**: `backend/research/weather/data/raw/ghcnh/{station_id}_{YYYY}_{MM}.csv`
  - Existing September files (`{station_id}_2024_09.csv`) were strictly preserved without overwriting.
- **Normalized Output Files**:
  - `backend/research/weather/data/processed/ghcnh_hourly_normalized_oct2024.csv`
  - `backend/research/weather/data/processed/ghcnh_hourly_normalized_nov2024.csv`
  - `backend/research/weather/data/processed/ghcnh_hourly_normalized_dec2024.csv`
  - `backend/research/weather/data/processed/ghcnh_hourly_normalized_jan2025.csv`
  - Combined Winter Archive: `backend/research/weather/data/processed/ghcnh_hourly_normalized_winter2024_2025.csv` (327,208 rows)

---

## 5. October 2024 Acquisition Summary

- **Reporting Stations**: 334 / 355 stations
- **Total Hourly Observations**: 86,650 records
- **Variable Completeness**:
  - Temperature: 86,544 valid (99.88%)
  - Dew Point: 86,573 valid (99.91%)
  - Visibility: 86,474 valid (99.80%)
  - Wind Speed: 86,163 valid (99.44%)
  - Precipitation Availability: 9,970 valid (11.51%)
- **Severe Visibility Counts**:
  - $<1000$ m: 571 observations (0.66%)
  - $<500$ m: 130 observations (0.15%)
  - $<200$ m: 41 observations (0.05%)

---

## 6. November 2024 Acquisition Summary

- **Reporting Stations**: 330 / 355 stations
- **Total Hourly Observations**: 72,951 records
- **Variable Completeness**:
  - Temperature: 72,875 valid (99.90%)
  - Dew Point: 72,874 valid (99.89%)
  - Visibility: 72,798 valid (99.79%)
  - Wind Speed: 72,526 valid (99.42%)
  - Precipitation Availability: 4,161 valid (5.70%)
- **Severe Visibility Counts**:
  - $<1000$ m: 2,234 observations (3.06%) — *3.9x increase over October*
  - $<500$ m: 782 observations (1.07%)
  - $<200$ m: 368 observations (0.50%)

---

## 7. December 2024 Acquisition Summary

- **Reporting Stations**: 333 / 355 stations
- **Total Hourly Observations**: 82,339 records
- **Variable Completeness**:
  - Temperature: 82,272 valid (99.92%)
  - Dew Point: 82,274 valid (99.92%)
  - Visibility: 82,157 valid (99.78%)
  - Wind Speed: 81,857 valid (99.41%)
  - Precipitation Availability: 5,268 valid (6.40%)
- **Severe Visibility Counts**:
  - $<1000$ m: 3,175 observations (3.86%)
  - $<500$ m: 882 observations (1.07%)
  - $<200$ m: 360 observations (0.44%)

---

## 8. January 2025 Acquisition Summary

- **Reporting Stations**: 329 / 355 stations
- **Total Hourly Observations**: 85,268 records
- **Variable Completeness**:
  - Temperature: 85,188 valid (99.91%)
  - Dew Point: 85,123 valid (99.83%)
  - Visibility: 85,093 valid (99.79%)
  - Wind Speed: 84,777 valid (99.42%)
  - Precipitation Availability: 2,656 valid (3.11%)
- **Severe Visibility Counts (Peak Winter Radiation Fog)**:
  - $<1000$ m: **6,275 observations (7.36%)** — *17.5x higher than September*
  - $<500$ m: **3,014 observations (3.54%)** — *16.2x higher than September*
  - $<200$ m: **1,755 observations (2.06%)** — *12.6x higher than September*

---

## 9. Data Quality & Invariant Checks

1. **Timestamp Integrity**: 100% of timestamps are valid UTC epoch seconds (`timestamp_utc`), monotonically ordered within each station series with exact matching ISO strings (`YYYY-MM-DDTHH:MM:SSZ`).
2. **Deduplication**: 0 duplicate station-timestamp pairs across all monthly files.
3. **Physical Bound Verification**:
   - Visibility: Valid range [0 m, 70,000 m] with zero negative or unphysical values.
   - Temperature: Units in Celsius [$-10^\circ\text{C}, +45^\circ\text{C}$].
   - Wind Speed: Units in m/s [$0\text{ m/s}, 40\text{ m/s}$].
4. **Precipitation Availability**: Missing precipitation observations in NOAA records are accurately tracked as `precipitation_available = 0` (and `NaN` values), **never** falsely imputed as `0.0 mm`.

---

## 10. Fog & Visibility Distribution Statistics

| Month | Min (m) | p1 (m) | p5 (m) | p25 (m) | Median (m) | p75 (m) | p90 (m) | p99 (m) | Max (m) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sep 2024** | 0.0 | 1,000.0 | 2,000.0 | 3,000.0 | 4,000.0 | 5,000.0 | 7,000.0 | 10,000.0 | 60,000.0 |
| **Oct 2024** | 0.0 | 1,000.0 | 2,000.0 | 3,000.0 | 4,000.0 | 5,000.0 | 6,000.0 | 10,000.0 | 65,000.0 |
| **Nov 2024** | 0.0 | **400.0** | **1,000.0** | 2,000.0 | 4,000.0 | 4,000.0 | 6,000.0 | 10,000.0 | 60,000.0 |
| **Dec 2024** | 0.0 | **400.0** | **1,000.0** | 2,000.0 | 3,500.0 | 4,000.0 | 6,000.0 | 10,000.0 | 70,000.0 |
| **Jan 2025** | 0.0 | **50.0** | **500.0** | 2,000.0 | 3,500.0 | 4,000.0 | 6,000.0 | 10,000.0 | 50,000.0 |

*Key Takeaway*: In January 2025, the 1st percentile of visibility drops to **50 meters** and the 5th percentile to **500 meters**, capturing the intense northern radiation fog corridor that severely impacts Indian Railways operations.

---

## 11. Multi-Month Comparison Matrix

| Month | Stations | Records | Visibility Coverage | Temp Coverage | Wind Coverage | Precip Availability | Vis <1000m (%) | Vis <200m (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sep 2024** | 333 | 86,948 | 99.79% | 99.92% | 99.46% | 17.28% | 0.41% | 0.16% |
| **Oct 2024** | 334 | 86,650 | 99.80% | 99.88% | 99.44% | 11.51% | 0.66% | 0.05% |
| **Nov 2024** | 330 | 72,951 | 99.79% | 99.90% | 99.42% | 5.70% | 3.06% | 0.50% |
| **Dec 2024** | 333 | 82,339 | 99.78% | 99.92% | 99.41% | 6.40% | 3.86% | 0.44% |
| **Jan 2025** | 329 | 85,268 | 99.79% | 99.91% | 99.42% | 3.11% | **7.36%** | **2.06%** |
| **Winter Total** | **344** | **327,208** | **99.79%** | **99.90%** | **99.42%** | **6.66%** | **3.74%** | **0.77%** |

---

## 12. Missing Stations & Coverage Notes

- Of the 355 mapped candidate NOAA stations, 344 stations (96.9%) were successfully downloaded from NOAA NCEI.
- The 11 unretrieved stations (e.g. decommissioned military airstrips or temporary WMO stations) returned HTTP 404 from the NOAA archive for 2024/2025.
- Active reporting stations ranged from 329 (Jan 2025) to 334 (Oct 2024), identical to the 333 active stations in September 2024.

---

## 13. Reproducibility

To re-run or update the winter NOAA acquisition pipeline:
```bash
python backend/research/weather/scripts/acquire_winter_ghcnh.py
```
Outputs are written deterministically to `backend/research/weather/data/processed/` and `backend/research/weather/data/raw/ghcnh/`.

---

## 14. Production Safety Verification

```bash
git diff -- backend/main.py backend/model/
```
- **0 production changes**.
- Production champion model hash: `bbd06bc91ae20c9aee8366cb917589553effeb353c5e5442add08179db982c02` (100% frozen and untouched).
- All weather data, acquisition scripts, and reports remain strictly isolated under `backend/research/weather/`.
