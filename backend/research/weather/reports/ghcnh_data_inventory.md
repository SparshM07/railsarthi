# NOAA GHCNh Weather Data Inventory & Quality Report (September 2024)

## Executive Summary
This report summarizes the acquisition, geospatial linking, and quality audit of hourly meteorological observations from the **NOAA Global Historical Climatology Network - Hourly (GHCNh / ISD)** dataset for Indian Railway stations during **September 2024**.

> [!NOTE]
> - **Research Isolation**: All weather data and metadata are strictly isolated in `backend/research/weather/`. Production inference pipelines and V2 LightGBM model remain frozen and untouched.
> - **Data Fidelity**: Original NOAA measurements are preserved. No visibility, fog, or precipitation values were invented or zero-imputed.

---

## 1. Station Geospatial & Coverage Summary

| Metric | Value |
| :--- | :--- |
| **Total Railway Stations in Sep 2024 Dataset** | **4,736** |
| **Geocoded Railway Stations** | **4,444 (93.83%)** |
| **Total Candidate NOAA GHCNh Stations in Region** | **592** |
| **Unique Selected NOAA Weather Stations** | **355** |
| **Successfully Downloaded NOAA Stations (Sep 2024)** | **333** |
| **Total September 2024 Hourly Weather Observations** | **86,948** |

---

## 2. Railway-to-NOAA Proximity & Distance Distribution

For each geocoded railway station, the nearest active NOAA surface weather station was identified using the Haversine great-circle formula:

```
Proximity Thresholds:
  - Within 25 km:  39.6% of railway stations
  - Within 50 km:  76.6% of railway stations
  - Within 100 km: 99.1% of railway stations
```

| Statistic | Distance (km) |
| :--- | :--- |
| **Minimum Distance** | `0.13 km` |
| **Median Distance (p50)** | `31.23 km` |
| **Mean Distance** | `34.26 km` |
| **75th Percentile (p75)** | `48.67 km` |
| **90th Percentile (p90)** | `65.13 km` |
| **Maximum Distance** | `120.37 km` |

---

## 3. Meteorological Variable Identification & Missingness Audit

The raw NOAA ISD/GHCNh data files contain mandatory and supplementary weather fields parsed and converted into standard SI meteorological units:

| Variable | Raw NOAA Field | Decoded Unit | Valid Observations | Missing % | Notes |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Temperature** | `TMP` | Degrees Celsius (°C) | 86,879 | `0.08%` | Scaled by 0.1 from integer tenths of °C. |
| **Dew Point** | `DEW` | Degrees Celsius (°C) | 86,902 | `0.05%` | Scaled by 0.1 from integer tenths of °C. |
| **Relative Humidity** | Derived (`TMP`, `DEW`) | Percentage (%) | 86,854 | `0.11%` | Computed via Magnus-Tetens formula. |
| **Wind Speed** | `WND` (pos 4) | Meters / second (m/s) | 86,479 | `0.54%` | Scaled by 0.1 from tenths of m/s. |
| **Wind Direction** | `WND` (pos 1) | Degrees (0–360°) | 64,364 | `25.97%` | 999 = Missing/Calm. |
| **Visibility** | `VIS` (pos 1) | Meters (m) | 86,767 | `0.21%` | 999999 = Missing. |
| **Precipitation** | `AA1` / `PRCP` | Millimeters (mm) | 15,023 | `82.72%` | Reported during rain/monsoon events. |
| **Sea Level Pressure**| `SLP` | Hectopascals (hPa) | 36,666 | `57.83%` | Scaled by 0.1 from tenths of hPa. |

---

## 4. September 2024 Temporal Cadence & Observations

- **Temporal Coverage**: September 1, 2024 00:00 UTC to September 30, 2024 23:00 UTC.
- **Reporting Frequency**: 
  - Major international airport stations (e.g. VIDP New Delhi, VABB Mumbai, VECC Kolkata, VOCL Kozhikode) report **hourly or half-hourly (METAR/SPECI)** cadence (720–1,440 reports/month).
  - Regional IMD synoptic stations report at **3-hourly WMO standard intervals** (00, 03, 06, 09, 12, 15, 18, 21 UTC; ~240 reports/month).

---

## 5. File & Artifact Locations

| Artifact | Path | Description |
| :--- | :--- | :--- |
| **Station Coordinates** | `backend/research/weather/data/station_metadata/station_coordinates.csv` | Master geocoded railway station catalog. |
| **Active NOAA Stations**| `backend/research/weather/data/station_metadata/noaa_ghcnh_active_stations.csv` | Active NOAA GHCNh weather stations in South Asia. |
| **Station Mapping** | `backend/research/weather/data/station_metadata/ghcnh_railway_station_mapping.csv` | Nearest NOAA weather station mapping with distance in km. |
| **Raw Weather Data** | `backend/research/weather/data/raw/ghcnh/` | Downloaded raw NOAA CSV files for September 2024. |
| **Validation Sample** | `backend/research/weather/data/ghcnh_sample_september_2024.csv` | 300-row parsed validation sample for inspection. |
| **Inventory Report** | `backend/research/weather/reports/ghcnh_data_inventory.md` | This quality and metadata audit document. |
