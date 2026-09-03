"""Historical segment statistics lookup and timetable schedule parsing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
import math
from pathlib import Path
from typing import Any
import pandas as pd

from backend.geo import (
    find_route_stop,
    get_stop_code,
    normalize_station_code,
    safe_float,
)

logger = logging.getLogger("railway_delay_api.stats")

IST = timezone(timedelta(hours=5, minutes=30))
MIN_EXACT_SEGMENT_SAMPLES = 20
MIN_FALLBACK_SEGMENT_SAMPLES = 50


def now_ist() -> datetime:
    """Return current datetime localized to Indian Standard Time (IST)."""
    return datetime.now(IST)


def parse_datetime(value: Any) -> datetime | None:
    """Parse ISO formatted string or datetime instance into IST datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=IST)
        return value.astimezone(IST)
    try:
        text = str(value).strip()
        if not text:
            return None
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=IST)
        return parsed.astimezone(IST)
    except (TypeError, ValueError):
        return None


def aggregate_segment_statistics(
    candidates: list[tuple[str, dict[str, Any]]], lookup_scope: str
) -> dict[str, Any] | None:
    """Aggregate actual segment-stat rows using their observation counts.

    The source CSV has per-segment mean, median, standard deviation and
    count, rather than raw observations. Mean and standard deviation can be
    pooled from those values; the returned median is the weighted median of
    the per-segment medians.
    """
    valid = [
        stats for _, stats in candidates if safe_float(stats.get("count"), 0) > 0
    ]
    total_count = sum(int(stats["count"]) for stats in valid)
    if total_count < MIN_FALLBACK_SEGMENT_SAMPLES:
        return None

    mean = sum(stats["mean"] * stats["count"] for stats in valid) / total_count

    if total_count > 1:
        variance_numerator = sum(
            max(0, stats["count"] - 1) * (stats["std"] ** 2)
            + stats["count"] * ((stats["mean"] - mean) ** 2)
            for stats in valid
        )
        std = math.sqrt(variance_numerator / (total_count - 1))
    else:
        std = 0.0

    weighted_medians = sorted((stats["median"], stats["count"]) for stats in valid)
    halfway = total_count / 2
    cumulative_count = 0
    median = weighted_medians[-1][0]
    for candidate_median, count in weighted_medians:
        cumulative_count += count
        if cumulative_count >= halfway:
            median = candidate_median
            break

    return {
        "mean": mean,
        "median": median,
        "std": std,
        "count": total_count,
        "lookup_scope": lookup_scope,
    }


class SegmentStatsIndex:
    """Indexed historical segment stats supporting exact and hierarchical lookups."""

    def __init__(self, stats_path: Path | str):
        self.stats_path = Path(stats_path)
        self.segment_lookup: dict[str, dict[str, Any]] = {}
        self.outgoing_segment_stats: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        self.incoming_segment_stats: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        self._load()

    def _load(self) -> None:
        if not self.stats_path.exists():
            raise FileNotFoundError(f"Segment stats CSV not found: {self.stats_path}")
        df = pd.read_csv(self.stats_path)
        for _, row in df.iterrows():
            segment = str(row["segment"]).strip().upper()
            try:
                mean = float(row["mean"])
                median = float(row["median"])
                std = float(row["std"]) if pd.notna(row["std"]) else 0.0
                count = int(row["count"])
                stats = {"mean": mean, "median": median, "std": std, "count": count}
                self.segment_lookup[segment] = stats
                if "->" in segment:
                    start, end = segment.split("->", 1)
                    self.outgoing_segment_stats.setdefault(start, []).append((segment, stats))
                    self.incoming_segment_stats.setdefault(end, []).append((segment, stats))
            except (TypeError, ValueError):
                continue
        logger.info("Loaded %d historical segment statistics", len(self.segment_lookup))

    def get_segment_statistics(
        self, current_station: str | None, next_station: str | None
    ) -> tuple[dict[str, Any], str]:
        """Return segment statistics with graceful hierarchical fallback."""
        curr_code = normalize_station_code(current_station)
        next_code = normalize_station_code(next_station)

        if not curr_code or not next_code:
            return (
                {"mean": 0.0, "median": 0.0, "std": 0.0, "count": 0, "lookup_scope": "TERMINAL"},
                "TERMINAL",
            )

        requested_segment = f"{curr_code}->{next_code}"
        exact = self.segment_lookup.get(requested_segment)
        if exact and exact["count"] >= MIN_EXACT_SEGMENT_SAMPLES:
            stats = {**exact, "lookup_scope": "EXACT"}
            return stats, requested_segment

        current_candidates = self.outgoing_segment_stats.get(curr_code, [])
        current_stats = aggregate_segment_statistics(current_candidates, "CURRENT_STATION")
        if current_stats:
            return current_stats, f"CURRENT_STATION:{curr_code}->*"

        next_candidates = self.incoming_segment_stats.get(next_code, [])
        next_stats = aggregate_segment_statistics(next_candidates, "NEXT_STATION")
        if next_stats:
            return next_stats, f"NEXT_STATION:*->{next_code}"

        all_candidates = list(self.segment_lookup.items())
        if not all_candidates:
            raise ValueError("Historical segment statistics are empty.")

        global_stats = aggregate_segment_statistics(all_candidates, "GLOBAL")
        if global_stats is None:
            raise ValueError("Historical segment statistics have insufficient observations.")

        return global_stats, "GLOBAL"


def get_scheduled_segment_minutes(
    live_data: dict[str, Any], current_station: str, next_station: str
) -> float:
    """Calculate timetable difference in minutes between current and next stop."""
    route = live_data.get("route", [])
    current_stop = find_route_stop(route, current_station)
    next_stop = find_route_stop(route, next_station)

    current_departure = None
    next_arrival = None

    if current_stop:
        current_departure = parse_datetime(current_stop.get("scheduledDeparture"))
        if current_departure is None:
            current_departure = parse_datetime(current_stop.get("scheduledArrival"))

    if next_stop:
        next_arrival = parse_datetime(next_stop.get("scheduledArrival"))

    if current_departure and next_arrival:
        while next_arrival < current_departure:
            next_arrival += timedelta(days=1)
        minutes = (next_arrival - current_departure).total_seconds() / 60.0
        if minutes >= 0:
            return max(0.0, minutes)

    return float("nan")


def get_previous_station_delay(live_data: dict[str, Any], current_station: str) -> float:
    """Extract arrival delay from the previous station or previous halt."""
    previous_halt = live_data.get("previousHalt") or {}
    previous_code = previous_halt.get("stationCode") or previous_halt.get("code")
    route = live_data.get("route", [])

    if previous_code:
        previous_stop = find_route_stop(route, previous_code)
        if previous_stop:
            val = previous_stop.get("delayArrival")
            if val is None:
                val = previous_stop.get("delayMinutes")
            return safe_float(val, 0.0)

    normalized_current = normalize_station_code(current_station)
    current_index = None
    for i, stop in enumerate(route):
        if get_stop_code(stop) == normalized_current:
            current_index = i
            break

    if current_index is not None and current_index > 0:
        previous_stop = route[current_index - 1]
        val = previous_stop.get("delayArrival")
        if val is None:
            val = previous_stop.get("delayMinutes")
        return safe_float(val, 0.0)

    return safe_float(live_data.get("delayMinutes"), 0.0)


def get_upcoming_stations(
    live_data: dict[str, Any], current_station: str
) -> list[dict[str, Any]]:
    """Return all downstream route stops following current_station."""
    route = live_data.get("route", [])
    normalized_current = normalize_station_code(current_station)
    current_index = None

    for i, stop in enumerate(route):
        if not isinstance(stop, dict):
            continue
        if get_stop_code(stop) == normalized_current:
            current_index = i
            break

    if current_index is None:
        return []
    return route[current_index + 1 :]
