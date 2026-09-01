"""Estimated Time of Arrival (ETA) calculation and downstream delay cascading engine."""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
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

    next_index = None
    for i in range(current_index + 1, len(route)):
        stop = route[i]
        code = normalize_station_code(
            stop.get("stationCode") or stop.get("code") or stop.get("station_code")
        )
        if code == next_code:
            next_index = i
            break

    if next_index is None:
        logger.warning("ETA Warning: Next station %s not found in route", next_station)
        return []

    upcoming = route[next_index:]
    if not upcoming:
        return []

    now = current_time or now_ist()
    progress = max(0.0, min(1.0, safe_float(segment_progress, 0.0)))
    cur_del = safe_float(current_delay, 0.0)
    pred_del = safe_float(predicted_delay, cur_del)
    sched_seg_min = max(0.0, safe_float(scheduled_segment_minutes, 30.0))

    remaining_segment_minutes = sched_seg_min * (
        1.0 - progress if segment_progress_reliable else 1.0
    )
    remaining_segment_minutes = max(0.0, remaining_segment_minutes)

    first_stop = upcoming[0]
    first_code = normalize_station_code(
        first_stop.get("stationCode") or first_stop.get("code") or first_stop.get("station_code")
    )
    first_delay = pred_del
    additional_predicted_delay = max(0.0, pred_del - cur_del)

    first_scheduled_arrival = parse_datetime(first_stop.get("scheduledArrival"))
    first_scheduled_departure = parse_datetime(first_stop.get("scheduledDeparture"))

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
        first_eta = live_physical_eta
        eta_method = "live_remaining_travel_after_stale_schedule_model_eta"
    else:
        first_eta = live_physical_eta
        eta_method = "live_remaining_travel_no_scheduled_arrival"

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
        }
    ]

    previous_eta = first_eta
    previous_stop = first_stop

    for stop in upcoming[1:]:
        code = normalize_station_code(
            stop.get("stationCode") or stop.get("code") or stop.get("station_code")
        )
        previous_scheduled = parse_datetime(previous_stop.get("scheduledArrival"))
        previous_scheduled_departure = parse_datetime(previous_stop.get("scheduledDeparture"))
        current_scheduled = parse_datetime(stop.get("scheduledArrival"))

        travel_minutes = 5.0
        if previous_scheduled and current_scheduled:
            if previous_scheduled_departure:
                while previous_scheduled_departure < previous_scheduled:
                    previous_scheduled_departure += timedelta(days=1)
            chronology_ref = previous_scheduled_departure or previous_scheduled
            while current_scheduled < chronology_ref:
                current_scheduled += timedelta(days=1)
            difference = (current_scheduled - previous_scheduled).total_seconds() / 60.0
            if difference > 0:
                travel_minutes = difference

        previous_eta = previous_eta + timedelta(minutes=travel_minutes)
        explicit_delay = stop.get("delayMinutes")
        scheduled_departure = parse_datetime(stop.get("scheduledDeparture"))

        if current_scheduled and scheduled_departure:
            while scheduled_departure < current_scheduled:
                scheduled_departure += timedelta(days=1)

        results.append(
            {
                "station_code": code,
                "station_name": stop.get("stationName"),
                "sequence": stop.get("sequence"),
                "distance_km": stop.get("distance"),
                "scheduled_arrival": (
                    current_scheduled.isoformat() if current_scheduled else None
                ),
                "scheduled_departure": (
                    scheduled_departure.isoformat() if scheduled_departure else None
                ),
                "delay_minutes": explicit_delay,
                "platform": stop.get("platform"),
                "is_halt": stop.get("isHalt", False),
                "predicted_delay_minutes": round(first_delay, 2),
                "is_independent_ml_prediction": False,
                "delay_propagation_source": "immediate_next_station_absolute_delay",
                "predicted_arrival": previous_eta.isoformat(),
                "eta_minutes_from_now": round((previous_eta - now).total_seconds() / 60.0, 2),
                "confidence": "LOW",
                "eta_method": "first_station_eta_plus_scheduled_arrival_interval",
            }
        )
        previous_stop = stop

    return results
