"""Geospatial and route-geometry utilities for railway position tracking."""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger("railway_delay_api.geo")


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert an arbitrary value to finite float or return default."""
    try:
        if value is None:
            return default
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return default
        result = float(value)
        if not math.isfinite(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def normalize_station_code(value: Any) -> str | None:
    """Normalize a station code to uppercase trimmed string or None."""
    if value is None:
        return None
    value = str(value).strip().upper()
    return value if value else None


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Compute the great-circle distance (in km) between two coordinates."""
    r = 6371.0
    lon1_rad = math.radians(float(lon1))
    lat1_rad = math.radians(float(lat1))
    lon2_rad = math.radians(float(lon2))
    lat2_rad = math.radians(float(lat2))

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def nearest_route_index(
    coordinates: list[list[float]], target: list[float] | tuple[float, float]
) -> tuple[int | None, float]:
    """Find the index of the closest point in coordinates to target [lon, lat]."""
    target_lon, target_lat = target
    best_index = None
    best_distance = float("inf")

    for i, point in enumerate(coordinates):
        if not point or len(point) < 2:
            continue
        try:
            lon = float(point[0])
            lat = float(point[1])
            distance = haversine(target_lon, target_lat, lon, lat)
            if distance < best_distance:
                best_distance = distance
                best_index = i
        except (TypeError, ValueError):
            continue

    return best_index, best_distance


def get_station_coordinates(route_data: dict[str, Any]) -> dict[str, list[float]]:
    """Extract a mapping of stationCode -> [lon, lat] from route data stops."""
    result: dict[str, list[float]] = {}
    for stop in route_data.get("stops", []):
        if not isinstance(stop, dict):
            continue
        code = stop.get("code") or stop.get("stationCode") or stop.get("station_code")
        if not code:
            continue
        normalized_code = normalize_station_code(code)
        if not normalized_code:
            continue

        lat = stop.get("lat") if stop.get("lat") is not None else stop.get("latitude")
        lng = (
            stop.get("lng")
            if stop.get("lng") is not None
            else (stop.get("lon") if stop.get("lon") is not None else stop.get("longitude"))
        )

        if lat is None or lng is None:
            coords = stop.get("coordinates") or {}
            if isinstance(coords, dict):
                if lat is None:
                    lat = coords.get("lat") or coords.get("latitude")
                if lng is None:
                    lng = coords.get("lng") or coords.get("lon") or coords.get("longitude")

        if lat is None or lng is None:
            continue

        try:
            lat = float(lat)
            lng = float(lng)
            if not (math.isfinite(lat) and math.isfinite(lng)):
                continue
            result[normalized_code] = [lng, lat]
        except (TypeError, ValueError):
            continue

    return result


def estimate_train_position(
    route_coordinates: list[list[float]],
    station_coordinates: dict[str, list[float]],
    current_station: str,
    next_station: str,
    segment_progress: float,
) -> dict[str, float]:
    """Interpolate train coordinates along route geometry between stations."""
    current_code = normalize_station_code(current_station)
    next_code = normalize_station_code(next_station)

    if not current_code or current_code not in station_coordinates:
        raise ValueError(f"No route coordinates available for current station {current_station}")
    if not next_code or next_code not in station_coordinates:
        raise ValueError(f"No route coordinates available for next station {next_station}")

    current_point = station_coordinates[current_code]
    next_point = station_coordinates[next_code]

    current_index, current_error = nearest_route_index(route_coordinates, current_point)
    next_index, next_error = nearest_route_index(route_coordinates, next_point)

    if current_index is None or next_index is None:
        raise ValueError("Unable to locate stations on route geometry.")

    if next_index <= current_index:
        raise ValueError(
            f"Invalid route ordering: {current_station} index={current_index}, {next_station} index={next_index}"
        )

    segment = route_coordinates[current_index : next_index + 1]
    if len(segment) < 2:
        return {"latitude": current_point[1], "longitude": current_point[0]}

    cumulative = [0.0]
    for i in range(1, len(segment)):
        lon1, lat1 = segment[i - 1][0], segment[i - 1][1]
        lon2, lat2 = segment[i][0], segment[i][1]
        cumulative.append(cumulative[-1] + haversine(lon1, lat1, lon2, lat2))

    total_distance = cumulative[-1]
    if total_distance <= 0:
        return {"latitude": segment[0][1], "longitude": segment[0][0]}

    progress = max(0.0, min(1.0, safe_float(segment_progress, 0.0)))
    target_distance = total_distance * progress

    estimated_lon = segment[-1][0]
    estimated_lat = segment[-1][1]

    for i in range(1, len(cumulative)):
        if cumulative[i] < target_distance:
            continue
        previous_distance = cumulative[i - 1]
        current_distance = cumulative[i]
        if current_distance == previous_distance:
            local_progress = 0.0
        else:
            local_progress = (target_distance - previous_distance) / (
                current_distance - previous_distance
            )
        lon1, lat1 = segment[i - 1][0], segment[i - 1][1]
        lon2, lat2 = segment[i][0], segment[i][1]
        estimated_lon = lon1 + local_progress * (lon2 - lon1)
        estimated_lat = lat1 + local_progress * (lat2 - lat1)
        break

    return {"latitude": estimated_lat, "longitude": estimated_lon}


def derive_segment_progress_from_position(
    route_data: dict[str, Any],
    current_station: str,
    next_station: str,
    latitude: float,
    longitude: float,
) -> float | None:
    """Safely derive progress only from a live coordinate snapped to the
    current station-to-next station geometry. Returns None when the geometry
    is missing, out of order, or the live point is more than 2 km from route.
    """
    geometry = (route_data.get("geojson") or {}).get("geometry") or {}
    coordinates = geometry.get("coordinates") or []
    station_coordinates = get_station_coordinates(route_data)
    current_code = normalize_station_code(current_station)
    next_code = normalize_station_code(next_station)
    if not current_code or not next_code:
        return None
    current_point = station_coordinates.get(current_code)
    next_point = station_coordinates.get(next_code)
    if not coordinates or not current_point or not next_point:
        return None
    current_index, _ = nearest_route_index(coordinates, current_point)
    next_index, _ = nearest_route_index(coordinates, next_point)
    live_index, live_distance = nearest_route_index(coordinates, [longitude, latitude])
    if (
        current_index is None
        or next_index is None
        or live_index is None
        or next_index <= current_index
        or not current_index <= live_index <= next_index
        or live_distance > 2.0
    ):
        return None
    segment = coordinates[current_index : next_index + 1]
    cumulative = [0.0]
    for index in range(1, len(segment)):
        cumulative.append(
            cumulative[-1]
            + haversine(
                segment[index - 1][0],
                segment[index - 1][1],
                segment[index][0],
                segment[index][1],
            )
        )
    total_distance = cumulative[-1]
    if total_distance <= 0:
        return None
    return max(0.0, min(1.0, cumulative[live_index - current_index] / total_distance))


def find_route_stop(route: list[dict[str, Any]], station_code: str | None) -> dict[str, Any] | None:
    """Find a route stop object matching station_code."""
    target_code = normalize_station_code(station_code)
    if not target_code:
        return None
    for stop in route:
        if not isinstance(stop, dict):
            continue
        code = normalize_station_code(
            stop.get("stationCode") or stop.get("code") or stop.get("station_code")
        )
        if code == target_code:
            return stop
    return None


def get_stop_code(stop: Any) -> str | None:
    """Return a normalized code only for a route stop object."""
    if not isinstance(stop, dict):
        return None
    return normalize_station_code(
        stop.get("stationCode") or stop.get("code") or stop.get("station_code")
    )


def get_stop_name(stop: Any) -> str:
    """Return a safe display name for a route stop object."""
    if not isinstance(stop, dict):
        return ""
    return str(stop.get("stationName") or stop.get("name") or "").strip()


def find_route_index(
    route: list[dict[str, Any]], station_code: str | None, start_index: int = 0
) -> int | None:
    """Find a real route-stop index; geometry points are never considered."""
    target_code = normalize_station_code(station_code)
    if not target_code:
        return None
    for index, stop in enumerate(route[start_index:], start=start_index):
        if get_stop_code(stop) == target_code:
            return index
    return None


def get_next_station_from_route(
    route: list[dict[str, Any]], current_station: str
) -> tuple[str | None, dict[str, Any] | None]:
    """Return the next real stop after current_station, if one exists."""
    current_index = find_route_index(route, current_station)
    if current_index is None:
        return None, None
    for stop in route[current_index + 1 :]:
        code = get_stop_code(stop)
        if code:
            return code, stop
    return None, None


def get_live_position(
    live_data: dict[str, Any],
    route_data: dict[str, Any],
    current_station: str,
    next_station: str | None,
    segment_progress: float,
) -> dict[str, Any]:
    """Determine best available train coordinates from live data or route interpolation."""
    current = live_data.get("currentLocation") or {}
    coordinates = current.get("coordinates") or {}

    if isinstance(coordinates, dict):
        lat = coordinates.get("lat") or coordinates.get("latitude")
        lng = coordinates.get("lng") or coordinates.get("lon") or coordinates.get("longitude")
        if lat is not None and lng is not None:
            try:
                return {
                    "latitude": float(lat),
                    "longitude": float(lng),
                    "source": "RailRadar live coordinates",
                }
            except (TypeError, ValueError):
                pass

    lat = current.get("lat") or current.get("latitude")
    lng = current.get("lng") or current.get("lon") or current.get("longitude")
    if lat is not None and lng is not None:
        try:
            return {
                "latitude": float(lat),
                "longitude": float(lng),
                "source": "RailRadar currentLocation coordinates",
            }
        except (TypeError, ValueError):
            pass

    # Terminal station fallback
    if not next_station:
        station_coordinates = get_station_coordinates(route_data)
        current_point = station_coordinates.get(normalize_station_code(current_station) or "")
        if current_point:
            return {
                "latitude": current_point[1],
                "longitude": current_point[0],
                "source": "RailRadar terminal station coordinates",
            }

    geojson = route_data.get("geojson") or {}
    geometry = geojson.get("geometry") or {}
    route_coordinates = geometry.get("coordinates") or []
    station_coordinates = get_station_coordinates(route_data)

    if route_coordinates and station_coordinates and next_station:
        try:
            pos = estimate_train_position(
                route_coordinates,
                station_coordinates,
                current_station,
                next_station,
                segment_progress,
            )
            pos["source"] = "RailRadar route geometry + segment progress"
            return pos
        except ValueError as e:
            logger.debug("Position estimation fallback triggered: %s", e)

    normalized_current = normalize_station_code(current_station)
    for stop in route_data.get("stops", []):
        if not isinstance(stop, dict):
            continue
        code = get_stop_code(stop)
        if code != normalized_current:
            continue
        lat = stop.get("lat") or stop.get("latitude")
        lng = stop.get("lng") or stop.get("lon") or stop.get("longitude")
        if lat is not None and lng is not None:
            try:
                return {
                    "latitude": float(lat),
                    "longitude": float(lng),
                    "source": "RailRadar station stop coordinates",
                }
            except (TypeError, ValueError):
                pass

    raise ValueError("Unable to obtain train coordinates from RailRadar.")
