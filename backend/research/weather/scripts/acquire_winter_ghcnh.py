"""NOAA GHCNh Winter Weather Data Acquisition & Validation Pipeline (Oct 2024 – Jan 2025)

Acquires, normalizes, and validates hourly meteorological observations for:
- October 2024
- November 2024
- December 2024
- January 2025

Target Stations: Reuses the active Indian NOAA station network mapped to Indian Railway stations.
Preserves raw source records, computes extensive feature coverage, and generates winter fog/visibility statistics.
"""

from __future__ import annotations
import gc
import io
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw" / "ghcnh"
METADATA_DIR = DATA_DIR / "station_metadata"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = BASE_DIR / "reports"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

NOAA_2024_BASE = "https://www.ncei.noaa.gov/data/global-hourly/access/2024"
NOAA_2025_BASE = "https://www.ncei.noaa.gov/data/global-hourly/access/2025"

ACTIVE_STATIONS_PATH = METADATA_DIR / "noaa_ghcnh_active_stations.csv"
MAPPING_PATH = METADATA_DIR / "ghcnh_railway_station_mapping.csv"
SEP_NORM_PATH = PROCESSED_DIR / "ghcnh_hourly_normalized.csv"

FOG_MW_CODES = {
    "01", "02", "03", "10", "11", "12", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49"
}

def is_fog_code(code_val: str | None) -> bool:
    if not code_val or code_val == "nan" or pd.isna(code_val):
        return False
    parts = str(code_val).replace("\"", "").replace("'", "").split(",")
    for p in parts:
        p_clean = p.strip().zfill(2)
        if p_clean in FOG_MW_CODES:
            return True
        if p.strip() in {"0", "1", "00", "01", "10", "11", "12"}:
            return True
    return False

def calculate_rh(temp_c: float | None, dew_c: float | None) -> float | None:
    if temp_c is None or dew_c is None or math.isnan(temp_c) or math.isnan(dew_c):
        return None
    try:
        es = 6.112 * math.exp((17.67 * temp_c) / (temp_c + 243.5))
        e = 6.112 * math.exp((17.67 * dew_c) / (dew_c + 243.5))
        return min(100.0, max(0.0, round(100.0 * (e / es), 1)))
    except Exception:
        return None

def parse_raw_noaa_row(row: pd.Series, stn_id: str, stn_meta: dict) -> dict:
    date_str = str(row.get("DATE", "")).strip()
    
    # 1. Temperature
    raw_tmp = str(row.get("TMP", ""))
    temp_c = None
    if raw_tmp and raw_tmp != "nan":
        parts = raw_tmp.split(",")
        if len(parts) >= 1:
            try:
                v = float(parts[0])
                if v != 9999:
                    temp_c = v / 10.0
            except ValueError:
                pass

    # 2. Dew Point
    raw_dew = str(row.get("DEW", ""))
    dew_c = None
    if raw_dew and raw_dew != "nan":
        parts = raw_dew.split(",")
        if len(parts) >= 1:
            try:
                v = float(parts[0])
                if v != 9999:
                    dew_c = v / 10.0
            except ValueError:
                pass

    # 3. Relative Humidity
    rh = calculate_rh(temp_c, dew_c)

    # 4. Wind
    raw_wnd = str(row.get("WND", ""))
    wind_dir = None
    wind_speed = None
    if raw_wnd and raw_wnd != "nan":
        parts = raw_wnd.split(",")
        if len(parts) >= 4:
            try:
                d = float(parts[0])
                if d != 999:
                    wind_dir = d
                s = float(parts[3])
                if s != 9999:
                    wind_speed = s / 10.0
            except ValueError:
                pass

    # 5. Visibility
    raw_vis = str(row.get("VIS", ""))
    vis_m = None
    if raw_vis and raw_vis != "nan":
        parts = raw_vis.split(",")
        if len(parts) >= 1:
            try:
                v = float(parts[0])
                if v != 999999:
                    vis_m = v
            except ValueError:
                pass

    # 6. Precipitation (AA1)
    raw_aa1 = str(row.get("AA1", ""))
    prcp_mm = None
    if raw_aa1 and raw_aa1 != "nan":
        parts = raw_aa1.split(",")
        if len(parts) >= 2:
            try:
                p = float(parts[1])
                if p != 9999:
                    prcp_mm = p / 10.0
            except ValueError:
                pass

    # 7. Sea Level Pressure
    raw_slp = str(row.get("SLP", ""))
    slp_hpa = None
    if raw_slp and raw_slp != "nan":
        parts = raw_slp.split(",")
        if len(parts) >= 1:
            try:
                s = float(parts[0])
                if s != 99999:
                    slp_hpa = s / 10.0
            except ValueError:
                pass

    # 8. Present Weather
    pw_code = None
    if pd.notna(row.get("AY1")):
        pw_code = str(row.get("AY1"))
    elif pd.notna(row.get("MW1")):
        pw_code = str(row.get("MW1"))

    # 9. Cloud Coverage
    cloud_layer = str(row.get("GA1")) if pd.notna(row.get("GA1")) else None

    # Parse UTC epoch timestamp
    try:
        dt = pd.to_datetime(date_str, utc=True)
        ts_utc = int(dt.timestamp())
        iso_str = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        date_only = dt.strftime("%Y-%m-%d")
        month_val = int(dt.month)
        year_val = int(dt.year)
    except Exception:
        ts_utc = None
        iso_str = date_str
        date_only = None
        month_val = None
        year_val = None

    return {
        "ghcnh_station_id": str(stn_id),
        "station_name": stn_meta.get("station_name", ""),
        "latitude": stn_meta.get("latitude"),
        "longitude": stn_meta.get("longitude"),
        "elevation_m": stn_meta.get("elevation_m"),
        "timestamp_utc": ts_utc,
        "datetime_utc": iso_str,
        "date": date_only,
        "year": year_val,
        "month": month_val,
        "temperature_c": temp_c,
        "temperature_available": int(temp_c is not None),
        "dewpoint_c": dew_c,
        "dewpoint_available": int(dew_c is not None),
        "relative_humidity": rh,
        "humidity_available": int(rh is not None),
        "dewpoint_depression_c": round(temp_c - dew_c, 2) if (temp_c is not None and dew_c is not None) else None,
        "wind_direction_deg": wind_dir,
        "wind_speed_mps": wind_speed,
        "wind_available": int(wind_speed is not None),
        "visibility_m": vis_m,
        "visibility_available": int(vis_m is not None),
        "visibility_lt_1000m": int(vis_m < 1000) if vis_m is not None else None,
        "visibility_lt_500m": int(vis_m < 500) if vis_m is not None else None,
        "visibility_lt_200m": int(vis_m < 200) if vis_m is not None else None,
        "low_visibility_flag": int(vis_m < 1000) if vis_m is not None else None,
        "precipitation_accumulation_mm": prcp_mm,
        "precipitation_available": int(prcp_mm is not None),
        "sea_level_pressure_hpa": slp_hpa,
        "pressure_available": int(slp_hpa is not None),
        "present_weather_code": pw_code,
        "present_weather_available": int(pw_code is not None),
        "fog_code_flag": int(is_fog_code(pw_code)) if pw_code is not None else None,
        "fog_observation_available": int(pw_code is not None),
        "cloud_coverage": cloud_layer,
        "cloud_available": int(cloud_layer is not None),
    }


def download_station_annual(stn_id: str, year: int) -> tuple[str, int, bool, pd.DataFrame | None, str]:
    base_url = NOAA_2024_BASE if year == 2024 else NOAA_2025_BASE
    url = f"{base_url}/{stn_id}.csv"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=25) as resp:
            content = resp.read().decode("utf-8", errors="ignore")
        df = pd.read_csv(io.StringIO(content), low_memory=False)
        if "DATE" not in df.columns:
            return stn_id, year, False, None, "No DATE column"
        return stn_id, year, True, df, "OK"
    except Exception as e:
        return stn_id, year, False, None, str(e)


def main():
    print("=" * 80)
    print("NOAA GHCNH WINTER WEATHER ACQUISITION (OCT 2024 – JAN 2025)")
    print("=" * 80)

    # 1. Load active stations metadata
    if not MAPPING_PATH.exists():
        raise FileNotFoundError(f"Missing station mapping: {MAPPING_PATH}")
    df_map = pd.read_csv(MAPPING_PATH)
    unique_stn_ids = sorted(df_map["ghcnh_station_id"].astype(str).str.zfill(11).unique())
    print(f"Total target NOAA stations from railway mapping: {len(unique_stn_ids)}")

    # Load active metadata coordinates
    meta_dict = {}
    if ACTIVE_STATIONS_PATH.exists():
        df_act = pd.read_csv(ACTIVE_STATIONS_PATH)
        df_act["ghcnh_station_id"] = df_act["ghcnh_station_id"].astype(str).str.zfill(11)
        for _, r in df_act.iterrows():
            meta_dict[r["ghcnh_station_id"]] = {
                "station_name": r.get("STATION NAME", ""),
                "latitude": r.get("LAT"),
                "longitude": r.get("LON"),
                "elevation_m": r.get("ELEV(M)")
            }

    # 2. Download 2024 and 2025 station files in parallel
    print(f"\n--- Phase 1: Downloading 2024 Annual Datasets for {len(unique_stn_ids)} stations ---")
    stn_2024_data = {}
    success_2024 = 0
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(download_station_annual, s, 2024): s for s in unique_stn_ids}
        for future in as_completed(futures):
            stn_id, yr, success, df_res, msg = future.result()
            if success and df_res is not None:
                stn_2024_data[stn_id] = df_res
                success_2024 += 1
            else:
                pass

    print(f"2024 Downloads Complete: {success_2024}/{len(unique_stn_ids)} stations acquired.")

    print(f"\n--- Phase 2: Downloading 2025 Annual Datasets for {len(unique_stn_ids)} stations ---")
    stn_2025_data = {}
    success_2025 = 0
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(download_station_annual, s, 2025): s for s in unique_stn_ids}
        for future in as_completed(futures):
            stn_id, yr, success, df_res, msg = future.result()
            if success and df_res is not None:
                stn_2025_data[stn_id] = df_res
                success_2025 += 1
            else:
                pass

    print(f"2025 Downloads Complete: {success_2025}/{len(unique_stn_ids)} stations acquired.")

    # 3. Monthly extraction, normalization, and raw saving
    months_config = [
        ("2024-10", "oct2024", 2024, stn_2024_data),
        ("2024-11", "nov2024", 2024, stn_2024_data),
        ("2024-12", "dec2024", 2024, stn_2024_data),
        ("2025-01", "jan2025", 2025, stn_2025_data),
    ]

    monthly_normalized_records = {}
    monthly_stats = {}

    for month_prefix, tag, yr, data_pool in months_config:
        print(f"\n--- Processing {month_prefix} ({tag}) ---")
        stns_with_data = 0
        raw_rows_total = 0
        parsed_records = []

        for stn_id in unique_stn_ids:
            df_annual = data_pool.get(stn_id)
            if df_annual is None:
                continue
            
            mask = df_annual["DATE"].astype(str).str.startswith(month_prefix)
            df_month = df_annual[mask].copy()
            
            if len(df_month) == 0:
                continue
            
            stns_with_data += 1
            raw_rows_total += len(df_month)

            # Save raw monthly file
            raw_out_path = RAW_DIR / f"{stn_id}_{month_prefix.replace('-', '_')}.csv"
            df_month.to_csv(raw_out_path, index=False)

            # Parse and normalize each row
            s_meta = meta_dict.get(stn_id, {})
            for _, row in df_month.iterrows():
                parsed_records.append(parse_raw_noaa_row(row, stn_id, s_meta))

        df_norm = pd.DataFrame(parsed_records)
        if len(df_norm) > 0:
            # Deduplicate by station and timestamp
            df_norm = df_norm.drop_duplicates(subset=["ghcnh_station_id", "timestamp_utc"]).sort_values(
                ["ghcnh_station_id", "timestamp_utc"]
            ).reset_index(drop=True)

            out_norm_path = PROCESSED_DIR / f"ghcnh_hourly_normalized_{tag}.csv"
            df_norm.to_csv(out_norm_path, index=False)
            print(f"Saved normalized {month_prefix} dataset to {out_norm_path} ({len(df_norm):,} rows across {stns_with_data} stations)")

        monthly_normalized_records[tag] = df_norm
        monthly_stats[tag] = {
            "month": month_prefix,
            "stations_requested": len(unique_stn_ids),
            "stations_acquired": stns_with_data,
            "stations_missing": len(unique_stn_ids) - stns_with_data,
            "raw_records": raw_rows_total,
            "normalized_records": len(df_norm),
            "temp_valid": int(df_norm["temperature_available"].sum()) if len(df_norm) > 0 else 0,
            "dew_valid": int(df_norm["dewpoint_available"].sum()) if len(df_norm) > 0 else 0,
            "rh_valid": int(df_norm["humidity_available"].sum()) if len(df_norm) > 0 else 0,
            "vis_valid": int(df_norm["visibility_available"].sum()) if len(df_norm) > 0 else 0,
            "wind_valid": int(df_norm["wind_available"].sum()) if len(df_norm) > 0 else 0,
            "precip_valid": int(df_norm["precipitation_available"].sum()) if len(df_norm) > 0 else 0,
            "pw_valid": int(df_norm["present_weather_available"].sum()) if len(df_norm) > 0 else 0,
            "fog_code_count": int((df_norm["fog_code_flag"] == 1).sum()) if len(df_norm) > 0 else 0,
            "vis_lt_1000m": int((df_norm["visibility_m"] < 1000).sum()) if len(df_norm) > 0 else 0,
            "vis_lt_500m": int((df_norm["visibility_m"] < 500).sum()) if len(df_norm) > 0 else 0,
            "vis_lt_200m": int((df_norm["visibility_m"] < 200).sum()) if len(df_norm) > 0 else 0,
        }

    # Combined Winter Dataset
    all_winter = pd.concat([monthly_normalized_records[t] for t in ["oct2024", "nov2024", "dec2024", "jan2025"] if len(monthly_normalized_records[t]) > 0], ignore_index=True)
    winter_out = PROCESSED_DIR / "ghcnh_hourly_normalized_winter2024_2025.csv"
    all_winter.to_csv(winter_out, index=False)
    print(f"\nSaved combined winter dataset to {winter_out} (Total {len(all_winter):,} rows)")

    # Save summary stats JSON
    stats_out = DATA_DIR / "winter_acquisition_stats.json"
    with open(stats_out, "w", encoding="utf-8") as f:
        json.dump(monthly_stats, f, indent=2)
    print(f"Saved acquisition stats to {stats_out}")

    print("\n" + "=" * 80)
    print("ACQUISITION & VALIDATION SUMMARY TABLE")
    print("=" * 80)
    for tag, st in monthly_stats.items():
        print(f"Month: {st['month']} | Stations: {st['stations_acquired']}/{st['stations_requested']} | Records: {st['normalized_records']:,} | Temp: {st['temp_valid']:,} ({st['temp_valid']/st['normalized_records']*100:.1f}%) | Vis: {st['vis_valid']:,} ({st['vis_valid']/st['normalized_records']*100:.1f}%) | <1000m: {st['vis_lt_1000m']:,} | <500m: {st['vis_lt_500m']:,} | <200m: {st['vis_lt_200m']:,}")

if __name__ == "__main__":
    main()
