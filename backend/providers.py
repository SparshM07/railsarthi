"""Client adapters and caching for external RailRadar and Open-Meteo services."""

from __future__ import annotations

import json
import logging
from typing import Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from backend.runtime import RuntimeMetrics, TTLCache

logger = logging.getLogger("railway_delay_api.providers")


def build_http_session() -> requests.Session:
    """Construct a requests session with bounded retry for idempotent GET requests."""
    session = requests.Session()
    session.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(
                total=2,
                connect=2,
                read=2,
                backoff_factor=0.25,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=frozenset({"GET"}),
                raise_on_status=False,
            )
        ),
    )
    return session


def railradar_headers(api_key: str) -> dict[str, str]:
    """Return standard auth and accept headers for RailRadar API requests."""
    return {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }


def get_cached_provider_json(
    name: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    ttl_seconds: float,
    cache: TTLCache,
    metrics: RuntimeMetrics,
    session: requests.Session,
) -> dict[str, Any]:
    """Fetch an external JSON endpoint with bounded TTL caching and metrics."""
    cache_key = json.dumps(
        {"name": name, "url": url, "params": params or {}},
        sort_keys=True,
        separators=(",", ":"),
    )

    def fetch() -> dict[str, Any]:
        metrics.increment(f"provider.{name}.network_request")
        response = session.get(url, headers=headers, params=params, timeout=20)
        response.raise_for_status()
        return response.json()

    before = cache.snapshot()["hits"]
    result = cache.get_or_set(cache_key, ttl_seconds, fetch)
    after = cache.snapshot()["hits"]
    metrics.increment(
        f"provider.{name}.cache_hit" if after > before else f"provider.{name}.cache_miss"
    )
    return result


def fetch_railradar_live(
    train_number: int,
    api_key: str,
    session: requests.Session,
    cache: TTLCache,
    metrics: RuntimeMetrics,
) -> dict[str, Any]:
    """Retrieve live GPS and halt data from RailRadar."""
    url = f"https://api.railradar.in/v1/trains/{train_number}/live"
    headers = railradar_headers(api_key)
    result = get_cached_provider_json(
        "railradar_live",
        url,
        headers=headers,
        ttl_seconds=15,
        cache=cache,
        metrics=metrics,
        session=session,
    )
    if not result.get("success"):
        raise ValueError(f"RailRadar live API error: {result}")
    return result["data"]


def fetch_railradar_route(
    train_number: int,
    api_key: str,
    session: requests.Session,
    cache: TTLCache,
    metrics: RuntimeMetrics,
) -> dict[str, Any]:
    """Retrieve static GeoJSON route geometry and stop timetable from RailRadar."""
    url = f"https://api.railradar.in/v1/trains/{train_number}/route"
    headers = railradar_headers(api_key)
    params = {"format": "geojson", "stops": "true"}
    result = get_cached_provider_json(
        "railradar_route",
        url,
        headers=headers,
        params=params,
        ttl_seconds=3600,
        cache=cache,
        metrics=metrics,
        session=session,
    )
    if not result.get("success"):
        raise ValueError(f"RailRadar route API error: {result}")
    return result["data"]


def fetch_open_meteo_weather(
    latitude: float,
    longitude: float,
    session: requests.Session,
    cache: TTLCache,
    metrics: RuntimeMetrics,
) -> dict[str, Any]:
    """Fetch current weather observation for coordinate from Open-Meteo."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "rain,"
            "weather_code,"
            "wind_speed_10m"
        ),
    }
    result = get_cached_provider_json(
        "open_meteo",
        url,
        params=params,
        ttl_seconds=600,
        cache=cache,
        metrics=metrics,
        session=session,
    )
    return result.get("current", {})
