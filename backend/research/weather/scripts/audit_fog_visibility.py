"""Audit fog and visibility data quality, coverage, observation age, delay relationships, and geographic distribution."""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

RESEARCH_WEATHER_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = RESEARCH_WEATHER_DIR / "data" / "processed" / "v3_weather_features.csv"

def audit():
    print("Loading V3 dataset...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    n_tot = len(df)
    print(f"Loaded {n_tot:,} rows.")

    # 1. Confusion Table: fog_flag vs fog_observation_available
    print("\n" + "="*80)
    print("1. FOG FLAG & OBSERVATION AVAILABILITY AUDIT")
    print("="*80)
    ct = pd.crosstab(df["fog_flag"].fillna(-1), df["fog_observation_available"], margins=True)
    print("Cross-tabulation (rows: fog_flag [-1=NaN, 0.0=No Fog, 1.0=Fog], cols: fog_observation_available [0, 1]):")
    print(ct)

    # 2. Visibility Data Quality
    print("\n" + "="*80)
    print("2. VISIBILITY DATA QUALITY AUDIT")
    print("="*80)
    vis = df[df["visibility_available"] == 1]["visibility_m"]
    print(f"Visibility Available count: {len(vis):,} / {n_tot:,} ({len(vis)/n_tot*100:.2f}%)")
    print(f"Min: {vis.min():.1f} m")
    print(f"Max: {vis.max():.1f} m")
    print(f"Mean: {vis.mean():.2f} m")
    print(f"Median: {vis.median():.2f} m")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  p{p:02d}: {np.percentile(vis, p):.2f} m")

    invalid_vis = (vis < 0) | (vis > 100000)
    print(f"Invalid/out-of-range (<0 or >100km) count: {invalid_vis.sum()}")

    # 3. Fog / Visibility Coverage
    print("\n" + "="*80)
    print("3. FOG / VISIBILITY COVERAGE METRICS")
    print("="*80)
    n_w_avail = int(df["weather_available"].sum())
    n_v_avail = int(df["visibility_available"].sum())
    n_f_avail = int(df["fog_observation_available"].sum())
    n_f1 = int((df["fog_flag"] == 1).sum())
    n_v_lt_1000 = int((df["visibility_m"] < 1000).sum())
    n_v_lt_500 = int((df["visibility_m"] < 500).sum())
    n_v_lt_200 = int((df["visibility_m"] < 200).sum())
    n_v_ge_1000 = int((df["visibility_m"] >= 1000).sum())
    n_v_ge_500 = int((df["visibility_m"] >= 500).sum())

    print(f"A. weather_available:         {n_w_avail:,} ({n_w_avail/n_tot*100:.2f}%)")
    print(f"B. visibility_available:      {n_v_avail:,} ({n_v_avail/n_tot*100:.2f}%)")
    print(f"C. fog_observation_available: {n_f_avail:,} ({n_f_avail/n_tot*100:.2f}%)")
    print(f"D. fog_flag == 1:             {n_f1:,} ({n_f1/n_tot*100:.2f}%)")
    print(f"E. visibility < 1000m:        {n_v_lt_1000:,} ({n_v_lt_1000/n_tot*100:.4f}% of all, {n_v_lt_1000/n_v_avail*100:.2f}% of vis)")
    print(f"F. visibility < 500m:         {n_v_lt_500:,} ({n_v_lt_500/n_tot*100:.4f}% of all, {n_v_lt_500/n_v_avail*100:.2f}% of vis)")
    print(f"G. visibility < 200m:         {n_v_lt_200:,} ({n_v_lt_200/n_tot*100:.4f}% of all, {n_v_lt_200/n_v_avail*100:.2f}% of vis)")
    print(f"H. visibility >= 1000m:       {n_v_ge_1000:,} ({n_v_ge_1000/n_v_avail*100:.2f}% of vis)")
    print(f"I. visibility >= 500m:        {n_v_ge_500:,} ({n_v_ge_500/n_v_avail*100:.2f}% of vis)")

    # 4. Temporal Freshness Breakdown
    print("\n" + "="*80)
    print("4. TEMPORAL AGE BREAKDOWN (FOG & LOW VISIBILITY BY AGE)")
    print("="*80)
    w_joined = df[df["weather_available"] == 1]
    age_brackets = [
        ("0 to 30 min", (w_joined["weather_observation_age_minutes"] >= 0) & (w_joined["weather_observation_age_minutes"] <= 30)),
        ("31 to 60 min", (w_joined["weather_observation_age_minutes"] > 30) & (w_joined["weather_observation_age_minutes"] <= 60)),
        ("61 to 120 min", (w_joined["weather_observation_age_minutes"] > 60) & (w_joined["weather_observation_age_minutes"] <= 120)),
        ("121 to 180 min", (w_joined["weather_observation_age_minutes"] > 120) & (w_joined["weather_observation_age_minutes"] <= 180)),
    ]
    for label, mask in age_brackets:
        sub = w_joined[mask]
        v_sub = sub[sub["visibility_available"] == 1]
        v_lt_1k_cnt = int((v_sub["visibility_m"] < 1000).sum())
        fog_cnt = int((sub["fog_flag"] == 1).sum())
        print(f"{label:15s}: {len(sub):,} rows | Vis Avail: {len(v_sub)/len(sub)*100:.2f}% | Fog Avail: {sub['fog_observation_available'].mean()*100:.2f}% | Fog=1: {fog_cnt:,} ({fog_cnt/len(sub)*100:.2f}%) | Vis<1k: {v_lt_1k_cnt:,} ({v_lt_1k_cnt/len(v_sub)*100:.2f}%)")

    # 5. Visibility -> Delay Descriptive Analysis
    print("\n" + "="*80)
    print("5. VISIBILITY BINS -> DELAY ANALYSIS")
    print("="*80)
    vis_bins = [
        ("< 200m", (df["visibility_m"] < 200)),
        ("200 to 500m", (df["visibility_m"] >= 200) & (df["visibility_m"] < 500)),
        ("500 to 1000m", (df["visibility_m"] >= 500) & (df["visibility_m"] < 1000)),
        ("1000 to 2000m", (df["visibility_m"] >= 1000) & (df["visibility_m"] < 2000)),
        ("2000 to 5000m", (df["visibility_m"] >= 2000) & (df["visibility_m"] < 5000)),
        (">= 5000m", (df["visibility_m"] >= 5000)),
    ]
    vis_table = []
    for label, mask in vis_bins:
        sub = df[mask]
        n_sub = len(sub)
        if n_sub > 0:
            arr_delays = sub["target_delay"].values
            curr_delays = sub["current_arr_delay"].values
            v2_mae = float(np.mean(np.abs(arr_delays - curr_delays)))
            p_gt_15 = float(np.mean(arr_delays > 15.0) * 100.0)
            p_gt_30 = float(np.mean(arr_delays > 30.0) * 100.0)
            p_gt_60 = float(np.mean(arr_delays > 60.0) * 100.0)
            vis_table.append({
                "bin": label,
                "count": n_sub,
                "pct_of_vis": float(n_sub / n_v_avail * 100.0),
                "mean_delay": float(np.mean(arr_delays)),
                "median_delay": float(np.median(arr_delays)),
                "p75": float(np.percentile(arr_delays, 75)),
                "p90": float(np.percentile(arr_delays, 90)),
                "p95": float(np.percentile(arr_delays, 95)),
                "v2_mae": v2_mae,
                "pct_gt_15": p_gt_15,
                "pct_gt_30": p_gt_30,
                "pct_gt_60": p_gt_60,
            })
            print(f"{label:15s}: N={n_sub:,} ({n_sub/n_v_avail*100:5.2f}%) | Mean Delay={np.mean(arr_delays):5.2f}m | Median={np.median(arr_delays):4.1f}m | p90={np.percentile(arr_delays, 90):5.1f}m | p95={np.percentile(arr_delays, 95):5.1f}m | V2 Persist MAE={v2_mae:5.2f}m | >15m: {p_gt_15:4.1f}% | >30m: {p_gt_30:4.1f}% | >60m: {p_gt_60:4.1f}%")

    # 6. Fog -> Delay Analysis
    print("\n" + "="*80)
    print("6. FOG -> DELAY ANALYSIS (WHERE fog_observation_available == 1)")
    print("="*80)
    fog_obs_df = df[df["fog_observation_available"] == 1]
    fog_cases = [
        ("Confirmed Fog (fog_flag == 1)", fog_obs_df[fog_obs_df["fog_flag"] == 1]),
        ("Clear / No Fog (fog_flag == 0)", fog_obs_df[fog_obs_df["fog_flag"] == 0]),
    ]
    for label, sub in fog_cases:
        arr_delays = sub["target_delay"].values
        curr_delays = sub["current_arr_delay"].values
        v2_mae = float(np.mean(np.abs(arr_delays - curr_delays)))
        p_gt_15 = float(np.mean(arr_delays > 15.0) * 100.0)
        p_gt_30 = float(np.mean(arr_delays > 30.0) * 100.0)
        p_gt_60 = float(np.mean(arr_delays > 60.0) * 100.0)
        print(f"{label:32s}: N={len(sub):,} ({len(sub)/len(fog_obs_df)*100:5.2f}%) | Mean={np.mean(arr_delays):5.2f}m | Med={np.median(arr_delays):4.1f}m | p90={np.percentile(arr_delays, 90):5.1f}m | p95={np.percentile(arr_delays, 95):5.1f}m | V2 MAE={v2_mae:5.2f}m | >15m: {p_gt_15:4.1f}% | >30m: {p_gt_30:4.1f}% | >60m: {p_gt_60:4.1f}%")

    # 7. Geographic Analysis (Station-Level)
    print("\n" + "="*80)
    print("7. GEOGRAPHIC / STATION-LEVEL FOG & LOW-VISIBILITY CONCENTRATION")
    print("="*80)
    # Group by railway station
    stn_grp = df[df["visibility_available"] == 1].groupby("station").agg(
        total_calls=("visibility_m", "count"),
        low_vis_calls=("visibility_m", lambda s: (s < 1000).sum()),
        severe_fog_calls=("visibility_m", lambda s: (s < 200).sum()),
        mean_vis=("visibility_m", "mean"),
    ).reset_index()
    stn_grp["low_vis_rate"] = stn_grp["low_vis_calls"] / stn_grp["total_calls"] * 100.0

    print("Top 10 Stations with Most Low-Visibility Calls (< 1000m):")
    top_low_vis = stn_grp.sort_values(by="low_vis_calls", ascending=False).head(10)
    print(top_low_vis.to_string(index=False))

    print("\nTop 10 Stations with Most Fog Calls (fog_flag == 1):")
    fog_stn_grp = df[df["fog_observation_available"] == 1].groupby("station").agg(
        total_calls=("fog_flag", "count"),
        fog_calls=("fog_flag", lambda s: (s == 1).sum()),
    ).reset_index()
    fog_stn_grp["fog_rate"] = fog_stn_grp["fog_calls"] / fog_stn_grp["total_calls"] * 100.0
    print(fog_stn_grp.sort_values(by="fog_calls", ascending=False).head(10).to_string(index=False))

if __name__ == "__main__":
    audit()
