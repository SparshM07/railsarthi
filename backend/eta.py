"""Estimated Time of Arrival (ETA) calculation and downstream delay cascading engine."""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
import math
from typing import Any

from backend.geo import normalize_station_code, safe_float
from backend.stats import now_ist, parse_datetime

logger = logging.getLogger("railway_delay_api.eta")


def get_eta_confidence(
    historical_stats: dict[str, Any],
    progress_reliable: bool,
    position_source: str,
) -> str:
    """Deterministic first-ETA confidence.

    HIGH needs an exact segment with >=200 observations plus a reliable
    position/progress signal. MEDIUM allows >=50 exact observations with
    reliable position, or >=200 exact observations without progress. Every
    current-station/global fallback is LOW because it is not the requested
    physical segment.
    """
    scope = historical_stats.get("lookup_scope", "GLOBAL")
    count = safe_float(historical_stats.get("count"), 0)
    has_live_position = position_source in {
        "RailRadar live coordinates",
        "RailRadar currentLocation coordinates",
    }
    if scope == "EXACT" and count >= 200 and (progress_reliable or has_live_position):
        return "HIGH"
    if scope == "EXACT" and count >= 50 and (progress_reliable or has_live_position):
        return "MEDIUM"
    if scope == "EXACT" and count >= 200:
        return "MEDIUM"
    return "LOW"


def build_upcoming_eta(
    live_data: dict[str, Any],
    current_station: str,
    next_station: str,
    current_delay: float,
    predicted_delay: float,
    segment_progress: float,
    scheduled_segment_minutes: float,
    historical_stats: dict[str, Any],
    segment_progress_reliable: bool = False,
    position_source: str = "",
    current_time: datetime | None = None,
    previous_delay: float = 0.0,
    train_number: str | int | None = None,
    model_container: Any = None,
    segment_stats_index: Any = None,
    max_horizon: int = 5,
) -> list[dict[str, Any]]:
    """Build structured upcoming station arrival estimates with cascading delays."""
    route = live_data.get("route", [])
    if not route:
        return []

    curr_code = normalize_station_code(current_station)
    next_code = normalize_station_code(next_station)

    current_index = None
    for i, stop in enumerate(route):
        code = normalize_station_code(
            stop.get("stationCode") or stop.get("code") or stop.get("station_code")
        )
        if code == curr_code:
            current_index = i
            break

    if current_index is None:
        logger.warning("ETA Warning: Current station %s not found in route", current_station)
        return []

    if current_index >= len(route) - 1:
        # At terminal station; no upcoming stops
        return []

    # Invariant: next_station MUST be the immediate next scheduled route station after current_station
    next_index = current_index + 1
    immediate_next_code = normalize_station_code(
        route[next_index].get("stationCode")
        or route[next_index].get("code")
        or route[next_index].get("station_code")
    )
    if next_code and immediate_next_code and next_code != immediate_next_code:
        logger.warning(
            "ETA Warning: next_station %s is not the immediate route stop (%s) after %s; enforcing adjacent route stop",
            next_station,
            immediate_next_code,
            current_station,
        )

    upcoming = route[next_index:]
    if not upcoming:
        return []

    now = current_time or now_ist()
    progress = max(0.0, min(1.0, safe_float(segment_progress, 0.0)))
    cur_del = safe_float(current_delay, 0.0)
    pred_del = safe_float(predicted_delay, cur_del)

    raw_sched_seg_min = safe_float(scheduled_segment_minutes, float("nan"))
    if math.isnan(raw_sched_seg_min) or raw_sched_seg_min <= 0:
        fallback_physical_seg_min = 30.0
    else:
        fallback_physical_seg_min = raw_sched_seg_min

    remaining_segment_minutes = fallback_physical_seg_min * (
        1.0 - progress if segment_progress_reliable else 1.0
    )
    remaining_segment_minutes = max(0.0, remaining_segment_minutes)

    # -------------------------------------------------------------
    # PHASE 1: Immediate next-station ETA (Hop 1)
    # -------------------------------------------------------------
    first_stop = upcoming[0]
    first_code = normalize_station_code(
        first_stop.get("stationCode") or first_stop.get("code") or first_stop.get("station_code")
    )
    first_delay = pred_del
    additional_predicted_delay = max(0.0, pred_del - cur_del)

    first_scheduled_arrival = parse_datetime(first_stop.get("scheduledArrival"))
    first_scheduled_departure = parse_datetime(first_stop.get("scheduledDeparture"))

    current_stop_obj = route[current_index] if (current_index is not None and current_index < len(route)) else {}
    current_departure = parse_datetime(current_stop_obj.get("scheduledDeparture")) or parse_datetime(
        current_stop_obj.get("scheduledArrival")
    )

    if current_departure and first_scheduled_arrival:
        while first_scheduled_arrival < current_departure:
            first_scheduled_arrival += timedelta(days=1)

    if first_scheduled_arrival and first_scheduled_departure:
        while first_scheduled_departure < first_scheduled_arrival:
            first_scheduled_departure += timedelta(days=1)

    live_physical_eta = now + timedelta(minutes=remaining_segment_minutes)
    schedule_model_eta = None
    if first_scheduled_arrival:
        schedule_model_eta = first_scheduled_arrival + timedelta(minutes=pred_del)

    if schedule_model_eta and schedule_model_eta >= now:
        first_eta = schedule_model_eta
        eta_method = "scheduled_arrival_plus_absolute_predicted_delay"
    elif schedule_model_eta:
        first_eta = max(now, live_physical_eta)
        eta_method = "live_remaining_travel_after_stale_schedule_model_eta"
    else:
        first_eta = max(now, live_physical_eta)
        eta_method = "live_remaining_travel_no_scheduled_arrival"

    if first_eta < now:
        first_eta = now

    first_eta_minutes = max(0.0, (first_eta - now).total_seconds() / 60.0)
    eta_confidence = get_eta_confidence(
        historical_stats, segment_progress_reliable, position_source
    )

    results = [
        {
            "station_code": first_code,
            "station_name": first_stop.get("stationName"),
            "sequence": first_stop.get("sequence"),
            "distance_km": first_stop.get("distance"),
            "scheduled_arrival": (
                first_scheduled_arrival.isoformat() if first_scheduled_arrival else None
            ),
            "scheduled_departure": (
                first_scheduled_departure.isoformat() if first_scheduled_departure else None
            ),
            "delay_minutes": first_stop.get("delayMinutes"),
            "platform": first_stop.get("platform"),
            "is_halt": first_stop.get("isHalt", False),
            "predicted_delay_minutes": round(first_delay, 2),
            "additional_predicted_delay_minutes": round(additional_predicted_delay, 2),
            "is_independent_ml_prediction": True,
            "delay_propagation_source": "immediate_next_station_absolute_delay",
            "predicted_arrival": first_eta.isoformat(),
            "eta_diagnostics": {
                "current_time": now.isoformat(),
                "scheduled_arrival": (
                    first_scheduled_arrival.isoformat() if first_scheduled_arrival else None
                ),
                "current_delay_minutes": round(cur_del, 2),
                "predicted_absolute_delay_minutes": round(pred_del, 2),
                "remaining_scheduled_minutes": round(remaining_segment_minutes, 2),
                "segment_progress": round(progress, 4),
                "method": eta_method,
            },
            "eta_minutes_from_now": round(first_eta_minutes, 2),
            "confidence": eta_confidence,
            "eta_method": eta_method,
            "cascade_hop": 1,
        }
    ]

    # -------------------------------------------------------------
    # PHASE 2: Bounded Autoregressive Downstream Cascade (Hops 2+)
    # -------------------------------------------------------------
    prior_delay_state = safe_float(previous_delay, 0.0)
    current_delay_state = first_delay
    previous_eta = first_eta
    previous_stop = first_stop
    previous_scheduled_ref = first_scheduled_departure or first_scheduled_arrival

    for hop_num, stop in enumerate(upcoming[1:], start=2):
        code = normalize_station_code(
            stop.get("stationCode") or stop.get("code") or stop.get("station_code")
        )
        prev_code = normalize_station_code(
            previous_stop.get("stationCode")
            or previous_stop.get("code")
            or previous_stop.get("station_code")
        )

        prev_scheduled_arrival = parse_datetime(previous_stop.get("scheduledArrival"))
        prev_scheduled_departure = parse_datetime(previous_stop.get("scheduledDeparture"))
        current_scheduled_arrival = parse_datetime(stop.get("scheduledArrival"))
        current_scheduled_departure = parse_datetime(stop.get("scheduledDeparture"))

        if prev_scheduled_arrival and previous_scheduled_ref:
            while prev_scheduled_arrival < previous_scheduled_ref:
                prev_scheduled_arrival += timedelta(days=1)

        prev_dep_ref = prev_scheduled_departure or prev_scheduled_arrival or previous_scheduled_ref
        if prev_scheduled_departure and prev_scheduled_arrival:
            while prev_scheduled_departure < prev_scheduled_arrival:
                prev_scheduled_departure += timedelta(days=1)
        elif prev_scheduled_departure and previous_scheduled_ref:
            while prev_scheduled_departure < previous_scheduled_ref:
                prev_scheduled_departure += timedelta(days=1)

        departure_checkpoint = prev_scheduled_departure or prev_scheduled_arrival or prev_dep_ref

        if current_scheduled_arrival and departure_checkpoint:
            while current_scheduled_arrival < departure_checkpoint:
                current_scheduled_arrival += timedelta(days=1)

        if current_scheduled_departure and current_scheduled_arrival:
            while current_scheduled_departure < current_scheduled_arrival:
                current_scheduled_departure += timedelta(days=1)
        elif current_scheduled_departure and departure_checkpoint:
            while current_scheduled_departure < departure_checkpoint:
                current_scheduled_departure += timedelta(days=1)

        if departure_checkpoint and current_scheduled_arrival:
            diff_mins = (current_scheduled_arrival - departure_checkpoint).total_seconds() / 60.0
            hop_scheduled_minutes = max(0.0, diff_mins) if diff_mins >= 0 else float("nan")
        else:
            hop_scheduled_minutes = float("nan")

        if hop_num <= max_horizon and model_container is not None:
            if segment_stats_index is not None and prev_code and code:
                hop_stats, _ = segment_stats_index.get_segment_statistics(prev_code, code)
            else:
                hop_stats = {"mean": 0.0, "median": 0.0, "std": 0.0, "count": 0}

            ref_dt = (
                current_scheduled_arrival
                or departure_checkpoint
                or previous_scheduled_ref
                or now
            )
            day_of_week = ref_dt.weekday()
            month = ref_dt.month
            is_weekend = 1 if day_of_week >= 5 else 0

            hop_input = {
                "train": str(train_number) if train_number is not None else "",
                "station": prev_code,
                "next_station": code,
                "current_arr_delay": current_delay_state,
                "scheduled_segment_minutes": hop_scheduled_minutes,
                "past_segment_mean": hop_stats.get("mean", 0.0),
                "past_segment_median": hop_stats.get("median", 0.0),
                "past_segment_std": hop_stats.get("std", 0.0),
                "past_segment_count": hop_stats.get("count", 0),
                "day_of_week": day_of_week,
                "month": month,
                "is_weekend": is_weekend,
                "previous_train_delay": prior_delay_state,
            }

            try:
                hop_df = model_container.prepare_dataframe(hop_input)
                pred_hop = model_container.predict(hop_df)
                hop_delay = max(0.0, float(pred_hop))
                is_independent = True
                delay_propagation_source = "ml_autoregressive_cascade"
            except Exception as exc:
                logger.warning(
                    "Autoregressive cascade prediction failed for hop %d (%s -> %s): %s",
                    hop_num,
                    prev_code,
                    code,
                    exc,
                )
                hop_delay = current_delay_state
                is_independent = False
                delay_propagation_source = "cascade_fallback"

            prior_delay_state = current_delay_state
            current_delay_state = hop_delay
        else:
            hop_delay = current_delay_state
            is_independent = False
            delay_propagation_source = "flat_after_cascade_horizon"

        if current_scheduled_arrival:
            pred_arrival = current_scheduled_arrival + timedelta(minutes=hop_delay)
            eta_method = (
                "scheduled_arrival_plus_autoregressive_delay"
                if (hop_num <= max_horizon and is_independent)
                else "scheduled_arrival_plus_flat_delay"
            )
        else:
            travel_minutes = 5.0
            if not math.isnan(hop_scheduled_minutes) and hop_scheduled_minutes > 0:
                travel_minutes = hop_scheduled_minutes
            elif prev_scheduled_arrival and current_scheduled_arrival:
                diff_arr = (current_scheduled_arrival - prev_scheduled_arrival).total_seconds() / 60.0
                if diff_arr > 0:
                    travel_minutes = diff_arr
            pred_arrival = previous_eta + timedelta(minutes=travel_minutes)
            eta_method = "first_station_eta_plus_scheduled_arrival_interval"

        if pred_arrival < previous_eta:
            pred_arrival = previous_eta
        if pred_arrival < now:
            pred_arrival = now

        eta_minutes_from_now = max(0.0, (pred_arrival - now).total_seconds() / 60.0)

        results.append(
            {
                "station_code": code,
                "station_name": stop.get("stationName"),
                "sequence": stop.get("sequence"),
                "distance_km": stop.get("distance"),
                "scheduled_arrival": (
                    current_scheduled_arrival.isoformat() if current_scheduled_arrival else None
                ),
                "scheduled_departure": (
                    current_scheduled_departure.isoformat() if current_scheduled_departure else None
                ),
                "delay_minutes": stop.get("delayMinutes"),
                "platform": stop.get("platform"),
                "is_halt": stop.get("isHalt", False),
                "predicted_delay_minutes": round(hop_delay, 2),
                "is_independent_ml_prediction": is_independent,
                "delay_propagation_source": delay_propagation_source,
                "predicted_arrival": pred_arrival.isoformat(),
                "eta_minutes_from_now": round(eta_minutes_from_now, 2),
                "confidence": "LOW",
                "eta_method": eta_method,
                "cascade_hop": hop_num,
            }
        )

        previous_eta = pred_arrival
        previous_stop = stop
        if current_scheduled_departure or current_scheduled_arrival:
            previous_scheduled_ref = current_scheduled_departure or current_scheduled_arrival

    return results
