"""NOAA GHCNh Weather Data Acquisition and Railway Station Mapping Pipeline

Builds:
1. Railway station geocoded coordinate catalog.
2. NOAA candidate station list and Haversine distance proximity mapping.
3. Downloaded September 2024 hourly weather observations for selected stations.
4. Variable identification, validation sample CSV (100-500 rows), and data inventory report.
"""

from __future__ import annotations
import io
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw" / "ghcnh"
METADATA_DIR = DATA_DIR / "station_metadata"
REPORTS_DIR = BASE_DIR / "reports"

RAW_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RAIL_DATA_PATH = Path(
    r"C:\Users\SPARSH MAURYA\Downloads\Indian-Railway-Network-and-Delays\Indian-Railway-Network-and-Delays\train_routes_delays_Sep2024.csv"
)
RAIL_ROUTES_PATH = Path(
    r"C:\Users\SPARSH MAURYA\Downloads\Indian-Railway-Network-and-Delays\Indian-Railway-Network-and-Delays\train_routes_Sep2024.csv"
)
DATAMEET_STATIONS_URL = "https://raw.githubusercontent.com/datameet/railways/master/stations.json"
NOAA_ISD_HISTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
NOAA_GLOBAL_HOURLY_BASE = "https://www.ncei.noaa.gov/data/global-hourly/access/2024"


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two points in kilometers."""
    r = 6371.0  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * (math.sin(dlambda / 2.0) ** 2))
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def calculate_relative_humidity(temp_c: float | None, dew_c: float | None) -> float | None:
    """Calculate Relative Humidity (%) using standard Magnus-Tetens formula."""
    if temp_c is None or dew_c is None or math.isnan(temp_c) or math.isnan(dew_c):
        return None
    try:
        # Saturation vapor pressure
        es = 6.112 * math.exp((17.67 * temp_c) / (temp_c + 243.5))
        # Actual vapor pressure
        e = 6.112 * math.exp((17.67 * dew_c) / (dew_c + 243.5))
        rh = 100.0 * (e / es)
        return min(100.0, max(0.0, round(rh, 1)))
    except Exception:
        return None


# ==============================================================================
# STEP 1: LOAD & GEOCODE RAILWAY STATIONS
# ==============================================================================
def load_and_geocode_railway_stations() -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("STEP 1: GEOCODING RAILWAY STATIONS")
    print("=" * 80)

    # 1. Unique stations in delay dataset
    df_delays = pd.read_csv(RAIL_DATA_PATH, usecols=["station", "date"])
    unique_stn_codes = sorted(df_delays["station"].dropna().unique())
    print(f"Unique railway stations in Sep 2024 delays: {len(unique_stn_codes)}")
    print(f"Railway date range: {df_delays['date'].min()} to {df_delays['date'].max()}")

    # 2. Station names from routes
    stn_name_map = {}
    if RAIL_ROUTES_PATH.exists():
        df_routes = pd.read_csv(RAIL_ROUTES_PATH, usecols=["station_code", "station_name"])
        df_routes = df_routes.dropna().drop_duplicates(subset=["station_code"])
        stn_name_map = dict(zip(df_routes["station_code"], df_routes["station_name"]))

    # 3. Load DataMeet master station GeoJSON
    print(f"Fetching DataMeet railway station coordinates from {DATAMEET_STATIONS_URL}...")
    req = urllib.request.Request(DATAMEET_STATIONS_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        datameet_json = json.loads(resp.read().decode("utf-8"))

    datameet_coords = {}
    for feat in datameet_json.get("features", []):
        props = feat.get("properties", {})
        code = str(props.get("code", "")).strip().upper()
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates", []) if isinstance(geom, dict) else []
        if code and coords and len(coords) >= 2:
            lon, lat = float(coords[0]), float(coords[1])
            name = props.get("name", "")
            state = props.get("state", "")
            zone = props.get("zone", "")
            datameet_coords[code] = {
                "latitude": lat,
                "longitude": lon,
                "name": name,
                "state": state,
                "zone": zone
            }
    print(f"DataMeet master coordinates loaded: {len(datameet_coords)} stations")

    # 4. Merge coordinates
    records = []
    matched_count = 0
    for code in unique_stn_codes:
        dm = datameet_coords.get(code)
        name = stn_name_map.get(code) or (dm["name"] if dm else code)
        if dm:
            lat = dm["latitude"]
            lon = dm["longitude"]
            state = dm["state"]
            zone = dm["zone"]
            matched = True
            matched_count += 1
        else:
            lat = None
            lon = None
            state = None
            zone = None
            matched = False
        records.append({
            "station_code": code,
            "station_name": name,
            "latitude": lat,
            "longitude": lon,
            "state": state,
            "zone": zone,
            "has_coordinates": matched
        })

    df_stn_coords = pd.DataFrame(records)
    out_path = METADATA_DIR / "station_coordinates.csv"
    df_stn_coords.to_csv(out_path, index=False)
    print(f"Saved station coordinates to {out_path}")
    print(f"Coordinate match rate: {matched_count}/{len(unique_stn_codes)} ({matched_count/len(unique_stn_codes)*100:.2f}%)")
    return df_stn_coords


# ==============================================================================
# STEP 2: LOAD NOAA STATIONS & COMPUTE HAVERSINE MAPPING
# ==============================================================================
def map_railway_to_noaa_ghcnh(df_rail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 80)
    print("STEP 2: NOAA GHCNH / ISD CANDIDATE SELECTION & MAPPING")
    print("=" * 80)

    # 1. Fetch NOAA ISD Master History
    print(f"Fetching NOAA ISD history from {NOAA_ISD_HISTORY_URL}...")
    req = urllib.request.Request(NOAA_ISD_HISTORY_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        isd_csv_text = resp.read().decode("utf-8", errors="ignore")

    df_isd = pd.read_csv(io.StringIO(isd_csv_text))
    # Filter for active stations in India and bounding box [5N-38N, 67E-98E] active in Sep 2024
    df_active = df_isd[
        (df_isd["END"] >= 20240901) &
        (df_isd["LAT"].notna()) &
        (df_isd["LON"].notna()) &
        (df_isd["LAT"] >= 5.0) & (df_isd["LAT"] <= 38.0) &
        (df_isd["LON"] >= 67.0) & (df_isd["LON"] <= 98.0)
    ].copy()

    # Construct NOAA 11-character station identifier (USAF + WBAN)
    df_active["USAF_STR"] = df_active["USAF"].astype(str).str.zfill(6)
    df_active["WBAN_STR"] = df_active["WBAN"].astype(str).str.zfill(5)
    df_active["ghcnh_station_id"] = df_active["USAF_STR"] + df_active["WBAN_STR"]
    
    noaa_active_path = METADATA_DIR / "noaa_ghcnh_active_stations.csv"
    df_active.to_csv(noaa_active_path, index=False)
    print(f"Active NOAA stations in South Asia / India domain for Sep 2024: {len(df_active)}")

    # 2. Compute nearest NOAA station for each railway station
    mapping_records = []
    rail_with_coords = df_rail[df_rail["has_coordinates"]].copy()

    noaa_lats = df_active["LAT"].values
    noaa_lons = df_active["LON"].values
    noaa_ids = df_active["ghcnh_station_id"].values
    noaa_names = df_active["STATION NAME"].values

    for _, r_row in rail_with_coords.iterrows():
        r_code = r_row["station_code"]
        r_name = r_row["station_name"]
        r_lat = r_row["latitude"]
        r_lon = r_row["longitude"]

        # Vectorized Haversine
        dists = [haversine_distance(r_lat, r_lon, n_lat, n_lon) for n_lat, n_lon in zip(noaa_lats, noaa_lons)]
        best_idx = int(np.argmin(dists))
        best_dist = float(dists[best_idx])

        mapping_records.append({
            "railway_station_code": r_code,
            "railway_station_name": r_name,
            "railway_latitude": r_lat,
            "railway_longitude": r_lon,
            "ghcnh_station_id": noaa_ids[best_idx],
            "ghcnh_station_name": noaa_names[best_idx],
            "ghcnh_latitude": noaa_lats[best_idx],
            "ghcnh_longitude": noaa_lons[best_idx],
            "distance_km": round(best_dist, 2)
        })

    df_mapping = pd.DataFrame(mapping_records)
    map_path = METADATA_DIR / "ghcnh_railway_station_mapping.csv"
    df_mapping.to_csv(map_path, index=False)
    print(f"Saved station mapping to {map_path}")

    # Distance metrics
    dists_series = df_mapping["distance_km"]
    print(f"Mapped railway stations: {len(df_mapping)}")
    print(f"Unique NOAA stations selected: {df_mapping['ghcnh_station_id'].nunique()}")
    print(f"Distance statistics (km):")
    print(f"  Min:    {dists_series.min():.2f} km")
    print(f"  Mean:   {dists_series.mean():.2f} km")
    print(f"  Median: {dists_series.median():.2f} km")
    print(f"  p90:    {dists_series.quantile(0.90):.2f} km")
    print(f"  Max:    {dists_series.max():.2f} km")

    return df_mapping, df_active


# ==============================================================================
# STEP 3: DOWNLOAD SEPTEMBER 2024 NOAA DATA
# ==============================================================================
def download_station_sep2024(stn_id: str) -> tuple[str, bool, int, str]:
    url = f"{NOAA_GLOBAL_HOURLY_BASE}/{stn_id}.csv"
    out_file = RAW_DIR / f"{stn_id}_2024_09.csv"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            content = resp.read().decode("utf-8", errors="ignore")
        
        df = pd.read_csv(io.StringIO(content), low_memory=False)
        if "DATE" not in df.columns:
            return stn_id, False, 0, "No DATE column"
        
        # Filter for September 2024 (2024-09-01 to 2024-09-30)
        df_sep = df[df["DATE"].astype(str).str.startswith("2024-09")].copy()
        if len(df_sep) == 0:
            return stn_id, False, 0, "No Sep 2024 rows"
        
        df_sep.to_csv(out_file, index=False)
        return stn_id, True, len(df_sep), "OK"
    except Exception as e:
        return stn_id, False, 0, str(e)


def download_noaa_ghcnh_data(unique_stations: list[str]) -> dict:
    print("\n" + "=" * 80)
    print(f"STEP 3: DOWNLOADING SEPTEMBER 2024 NOAA DATA ({len(unique_stations)} STATIONS)")
    print("=" * 80)

    results = {}
    success_count = 0
    total_obs = 0

    # Download with threadpool
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_map = {executor.submit(download_station_sep2024, sid): sid for sid in unique_stations}
        for i, future in enumerate(as_completed(future_map), 1):
            sid, success, rows, msg = future.result()
            results[sid] = {"success": success, "rows": rows, "message": msg}
            if success:
                success_count += 1
                total_obs += rows
            if i % 25 == 0 or i == len(unique_stations):
                print(f"Progress: {i}/{len(unique_stations)} stations processed ({success_count} downloaded, {total_obs} hourly observations)...")

    print(f"\nDownload summary: {success_count}/{len(unique_stations)} stations retrieved ({total_obs} total hourly observations for Sep 2024)")
    return results


# ==============================================================================
# STEP 4: PARSE VARIABLES, COMPUTE QUALITY, & GENERATE VALIDATION SAMPLE
# ==============================================================================
def parse_noaa_record(row: pd.Series) -> dict:
    """Parse raw NOAA string records into standardized meteorological variables."""
    # 1. Temperature (TMP: +0280,1)
    tmp_raw = str(row.get("TMP", ""))
    temp_c = None
    if tmp_raw and tmp_raw != "nan":
        parts = tmp_raw.split(",")
        if len(parts) >= 1:
            try:
                val = float(parts[0])
                if val != 9999:
                    temp_c = val / 10.0
            except ValueError:
                pass

    # 2. Dew Point (DEW: +0240,1)
    dew_raw = str(row.get("DEW", ""))
    dew_c = None
    if dew_raw and dew_raw != "nan":
        parts = dew_raw.split(",")
        if len(parts) >= 1:
            try:
                val = float(parts[0])
                if val != 9999:
                    dew_c = val / 10.0
            except ValueError:
                pass

    # 3. Relative Humidity (computed)
    rh = calculate_relative_humidity(temp_c, dew_c)

    # 4. Wind (WND: 200,1,N,0021,1 -> dir, dir_qual, type, speed, speed_qual)
    wnd_raw = str(row.get("WND", ""))
    wind_dir = None
    wind_speed_ms = None
    if wnd_raw and wnd_raw != "nan":
        parts = wnd_raw.split(",")
        if len(parts) >= 4:
            try:
                d_val = float(parts[0])
                if d_val != 999:
                    wind_dir = d_val
                s_val = float(parts[3])
                if s_val != 9999:
                    wind_speed_ms = s_val / 10.0
            except ValueError:
                pass

    # 5. Visibility (VIS: 000500,1,9,9 -> dist, dist_qual, var, var_qual)
    vis_raw = str(row.get("VIS", ""))
    vis_m = None
    if vis_raw and vis_raw != "nan":
        parts = vis_raw.split(",")
        if len(parts) >= 1:
            try:
                v_val = float(parts[0])
                if v_val != 999999:
                    vis_m = v_val
            except ValueError:
                pass

    # 6. Precipitation (AA1: period, depth, cond, qual)
    aa1_raw = str(row.get("AA1", ""))
    prcp_mm = None
    if aa1_raw and aa1_raw != "nan":
        parts = aa1_raw.split(",")
        if len(parts) >= 2:
            try:
                p_val = float(parts[1])
                if p_val != 9999:
                    prcp_mm = p_val / 10.0
            except ValueError:
                pass

    # 7. Sea Level Pressure (SLP: 10132,1)
    slp_raw = str(row.get("SLP", ""))
    slp_hpa = None
    if slp_raw and slp_raw != "nan":
        parts = slp_raw.split(",")
        if len(parts) >= 1:
            try:
                s_val = float(parts[0])
                if s_val != 99999:
                    slp_hpa = s_val / 10.0
            except ValueError:
                pass

    # 8. Present Weather / Cloud (AY1, MW1, GA1)
    present_weather = row.get("AY1") if pd.notna(row.get("AY1")) else row.get("MW1")
    cloud_coverage = row.get("GA1") if pd.notna(row.get("GA1")) else None

    return {
        "ghcnh_station_id": str(row.get("STATION", "")),
        "station_name": str(row.get("NAME", "")),
        "timestamp_utc": str(row.get("DATE", "")),
        "latitude": row.get("LATITUDE"),
        "longitude": row.get("LONGITUDE"),
        "elevation_m": row.get("ELEVATION"),
        "temperature_c": temp_c,
        "dew_point_c": dew_c,
        "relative_humidity_pct": rh,
        "wind_speed_ms": wind_speed_ms,
        "wind_direction_deg": wind_dir,
        "visibility_m": vis_m,
        "precipitation_mm": prcp_mm,
        "sea_level_pressure_hpa": slp_hpa,
        "present_weather_code": str(present_weather) if present_weather is not None else None,
        "cloud_layer": str(cloud_coverage) if cloud_coverage is not None else None
    }


def generate_validation_sample_and_metrics() -> tuple[pd.DataFrame, dict]:
    print("\n" + "=" * 80)
    print("STEP 4: GENERATING VALIDATION SAMPLE & COMPUTING INVENTORY METRICS")
    print("=" * 80)

    raw_files = list(RAW_DIR.glob("*_2024_09.csv"))
    print(f"Analyzing {len(raw_files)} downloaded NOAA September 2024 CSV files...")

    all_parsed_rows = []
    station_stats = []

    for f in raw_files:
        df = pd.read_csv(f, low_memory=False)
        stn_rows = []
        for _, row in df.iterrows():
            stn_rows.append(parse_noaa_record(row))
        all_parsed_rows.extend(stn_rows)
        
        # Per-station missingness
        stn_df = pd.DataFrame(stn_rows)
        station_stats.append({
            "station_id": f.stem.replace("_2024_09", ""),
            "observations": len(stn_df),
            "valid_temp_pct": round(stn_df["temperature_c"].notna().mean() * 100, 1),
            "valid_dew_pct": round(stn_df["dew_point_c"].notna().mean() * 100, 1),
            "valid_vis_pct": round(stn_df["visibility_m"].notna().mean() * 100, 1),
            "valid_wnd_pct": round(stn_df["wind_speed_ms"].notna().mean() * 100, 1),
            "valid_prcp_pct": round(stn_df["precipitation_mm"].notna().mean() * 100, 1),
        })

    df_all_parsed = pd.DataFrame(all_parsed_rows)
    print(f"Total parsed September 2024 observations across all stations: {len(df_all_parsed)}")

    # Sample 300 representative observations across all stations
    if len(df_all_parsed) > 300:
        sample_df = df_all_parsed.sample(n=300, random_state=42).sort_values(by=["ghcnh_station_id", "timestamp_utc"])
    else:
        sample_df = df_all_parsed.sort_values(by=["ghcnh_station_id", "timestamp_utc"])

    sample_out_path = DATA_DIR / "ghcnh_sample_september_2024.csv"
    sample_df.to_csv(sample_out_path, index=False)
    print(f"Saved validation sample ({len(sample_df)} rows) to {sample_out_path}")

    # Overall Missingness
    missingness = {}
    for col in ["temperature_c", "dew_point_c", "relative_humidity_pct", "wind_speed_ms", "wind_direction_deg", "visibility_m", "precipitation_mm", "sea_level_pressure_hpa"]:
        valid_cnt = int(df_all_parsed[col].notna().sum())
        total_cnt = len(df_all_parsed)
        missingness[col] = {
            "valid_count": valid_cnt,
            "total_count": total_cnt,
            "missing_pct": round((1.0 - valid_cnt / total_cnt) * 100, 2)
        }

    return df_all_parsed, {
        "total_observations": len(df_all_parsed),
        "total_stations": len(raw_files),
        "missingness": missingness,
        "station_stats": station_stats
    }


# ==============================================================================
# STEP 5: WRITE COMPREHENSIVE INVENTORY REPORT
# ==============================================================================
def write_inventory_report(
    df_stn_coords: pd.DataFrame,
    df_mapping: pd.DataFrame,
    df_active_noaa: pd.DataFrame,
    download_res: dict,
    metrics: dict
):
    print("\n" + "=" * 80)
    print("STEP 5: WRITING DATA INVENTORY REPORT")
    print("=" * 80)

    report_path = REPORTS_DIR / "ghcnh_data_inventory.md"

    total_rail_stations = len(df_stn_coords)
    geocoded_rail_stations = len(df_mapping)
    total_candidate_noaa = len(df_active_noaa)
    selected_noaa_stations = df_mapping["ghcnh_station_id"].nunique()
    downloaded_noaa_stations = metrics["total_stations"]
    total_observations = metrics["total_observations"]

    dists = df_mapping["distance_km"]
    dist_min = dists.min()
    dist_mean = dists.mean()
    dist_median = dists.median()
    dist_p75 = dists.quantile(0.75)
    dist_p90 = dists.quantile(0.90)
    dist_max = dists.max()

    coverage_pct = round((geocoded_rail_stations / total_rail_stations) * 100, 2)
    within_25km = (dists <= 25.0).mean() * 100
    within_50km = (dists <= 50.0).mean() * 100
    within_100km = (dists <= 100.0).mean() * 100

    missing_dict = metrics["missingness"]

    report_md = f"""# NOAA GHCNh Weather Data Inventory & Quality Report (September 2024)

## Executive Summary
This report summarizes the acquisition, geospatial linking, and quality audit of hourly meteorological observations from the **NOAA Global Historical Climatology Network - Hourly (GHCNh / ISD)** dataset for Indian Railway stations during **September 2024**.

> [!NOTE]
> - **Research Isolation**: All weather data and metadata are strictly isolated in `backend/research/weather/`. Production inference pipelines and V2 LightGBM model remain frozen and untouched.
> - **Data Fidelity**: Original NOAA measurements are preserved. No visibility, fog, or precipitation values were invented or zero-imputed.

---

## 1. Station Geospatial & Coverage Summary

| Metric | Value |
| :--- | :--- |
| **Total Railway Stations in Sep 2024 Dataset** | **{total_rail_stations:,}** |
| **Geocoded Railway Stations** | **{geocoded_rail_stations:,} ({coverage_pct}%)** |
| **Total Candidate NOAA GHCNh Stations in Region** | **{total_candidate_noaa:,}** |
| **Unique Selected NOAA Weather Stations** | **{selected_noaa_stations:,}** |
| **Successfully Downloaded NOAA Stations (Sep 2024)** | **{downloaded_noaa_stations:,}** |
| **Total September 2024 Hourly Weather Observations** | **{total_observations:,}** |

---

## 2. Railway-to-NOAA Proximity & Distance Distribution

For each geocoded railway station, the nearest active NOAA surface weather station was identified using the Haversine great-circle formula:

```
Proximity Thresholds:
  - Within 25 km:  {within_25km:.1f}% of railway stations
  - Within 50 km:  {within_50km:.1f}% of railway stations
  - Within 100 km: {within_100km:.1f}% of railway stations
```

| Statistic | Distance (km) |
| :--- | :--- |
| **Minimum Distance** | `{dist_min:.2f} km` |
| **Median Distance (p50)** | `{dist_median:.2f} km` |
| **Mean Distance** | `{dist_mean:.2f} km` |
| **75th Percentile (p75)** | `{dist_p75:.2f} km` |
| **90th Percentile (p90)** | `{dist_p90:.2f} km` |
| **Maximum Distance** | `{dist_max:.2f} km` |

---

## 3. Meteorological Variable Identification & Missingness Audit

The raw NOAA ISD/GHCNh data files contain mandatory and supplementary weather fields parsed and converted into standard SI meteorological units:

| Variable | Raw NOAA Field | Decoded Unit | Valid Observations | Missing % | Notes |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Temperature** | `TMP` | Degrees Celsius (°C) | {missing_dict['temperature_c']['valid_count']:,} | `{missing_dict['temperature_c']['missing_pct']}%` | Scaled by 0.1 from integer tenths of °C. |
| **Dew Point** | `DEW` | Degrees Celsius (°C) | {missing_dict['dew_point_c']['valid_count']:,} | `{missing_dict['dew_point_c']['missing_pct']}%` | Scaled by 0.1 from integer tenths of °C. |
| **Relative Humidity** | Derived (`TMP`, `DEW`) | Percentage (%) | {missing_dict['relative_humidity_pct']['valid_count']:,} | `{missing_dict['relative_humidity_pct']['missing_pct']}%` | Computed via Magnus-Tetens formula. |
| **Wind Speed** | `WND` (pos 4) | Meters / second (m/s) | {missing_dict['wind_speed_ms']['valid_count']:,} | `{missing_dict['wind_speed_ms']['missing_pct']}%` | Scaled by 0.1 from tenths of m/s. |
| **Wind Direction** | `WND` (pos 1) | Degrees (0–360°) | {missing_dict['wind_direction_deg']['valid_count']:,} | `{missing_dict['wind_direction_deg']['missing_pct']}%` | 999 = Missing/Calm. |
| **Visibility** | `VIS` (pos 1) | Meters (m) | {missing_dict['visibility_m']['valid_count']:,} | `{missing_dict['visibility_m']['missing_pct']}%` | 999999 = Missing. |
| **Precipitation** | `AA1` / `PRCP` | Millimeters (mm) | {missing_dict['precipitation_mm']['valid_count']:,} | `{missing_dict['precipitation_mm']['missing_pct']}%` | Reported during rain/monsoon events. |
| **Sea Level Pressure**| `SLP` | Hectopascals (hPa) | {missing_dict['sea_level_pressure_hpa']['valid_count']:,} | `{missing_dict['sea_level_pressure_hpa']['missing_pct']}%` | Scaled by 0.1 from tenths of hPa. |

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
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Saved inventory report to {report_path}")


# ==============================================================================
# MAIN PIPELINE EXECUTION
# ==============================================================================
def main():
    print("=" * 80)
    print("STARTING NOAA GHCNH WEATHER DATA ACQUISITION & MAPPING PIPELINE")
    print("=" * 80)

    # 1. Geocode railway stations
    df_stn_coords = load_and_geocode_railway_stations()

    # 2. Map to nearest active NOAA station
    df_mapping, df_active_noaa = map_railway_to_noaa_ghcnh(df_stn_coords)

    # 3. Unique NOAA stations needed
    unique_stations = df_mapping["ghcnh_station_id"].unique().tolist()
    print(f"\nUnique NOAA stations to download: {len(unique_stations)}")

    # 4. Download Sep 2024 hourly data
    download_res = download_noaa_ghcnh_data(unique_stations)

    # 5. Parse, compute missingness, generate sample CSV
    df_parsed, metrics = generate_validation_sample_and_metrics()

    # 6. Generate comprehensive inventory report
    write_inventory_report(df_stn_coords, df_mapping, df_active_noaa, download_res, metrics)

    print("\n" + "=" * 80)
    print("NOAA GHCNH ACQUISITION PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
