"""FastAPI service for live and offline journey-level train delay predictions."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import lightgbm as lgb
from pydantic import BaseModel, Field
import requests

from backend.eta import build_upcoming_eta, get_eta_confidence
from backend.geo import (
    derive_segment_progress_from_position,
    estimate_train_position,
    find_route_index,
    find_route_stop,
    get_live_position,
    get_next_station_from_route,
    get_station_coordinates,
    get_stop_code,
    get_stop_name,
    haversine,
    nearest_route_index,
    normalize_station_code,
    safe_float,
)
from backend.journey_model import prepare_journey_model_dataframe
from backend.model_serving import (
    ChampionModelContainer,
    normalize_category_values,
    prepare_categorical_feature,
    prepare_model_dataframe as _prepare_model_dataframe_core,
)
from backend.providers import (
    build_http_session,
    fetch_open_meteo_weather,
    fetch_railradar_live,
    fetch_railradar_route,
    get_cached_provider_json,
    railradar_headers as _railradar_headers,
)
from backend.runtime import RuntimeMetrics, SlidingWindowRateLimiter, TTLCache
from backend.simulator import (
    generate_simulated_train_data,
    get_available_trains_catalog,
)
from backend.stats import (
    SegmentStatsIndex,
    aggregate_segment_statistics,
    get_previous_station_delay,
    get_scheduled_segment_minutes,
    get_upcoming_stations,
    now_ist,
    parse_datetime,
)

# ============================================================
# LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("railway_delay_api")

# ============================================================
# ENVIRONMENT / CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
MODEL_DIR = BASE_DIR / "model"
FRONTEND_DIR = REPO_ROOT / "frontend"

load_dotenv(BASE_DIR / ".env")
load_dotenv()

RAILRADAR_API_KEY = os.getenv("RAILRADAR_API_KEY", "")
APP_API_KEY = os.getenv("APP_API_KEY")
REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "false").lower() == "true"
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS", "*"
    ).split(",")
    if origin.strip()
]
REQUEST_RATE_LIMIT_PER_MINUTE = max(
    1, int(os.getenv("REQUEST_RATE_LIMIT_PER_MINUTE", "120"))
)

if REQUIRE_API_KEY and not APP_API_KEY:
    raise RuntimeError("APP_API_KEY must be set when REQUIRE_API_KEY=true")

# ============================================================
# HTTP & RUNTIME STATE
# ============================================================

HTTP_SESSION = build_http_session()
provider_cache = TTLCache(max_entries=512)
request_limiter = SlidingWindowRateLimiter(REQUEST_RATE_LIMIT_PER_MINUTE)
runtime_metrics = RuntimeMetrics()

# ============================================================
# MODEL & STATS ARTIFACTS
# ============================================================

MODEL_PATH = MODEL_DIR / "champion_model.txt"
FEATURE_CONFIG_PATH = MODEL_DIR / "model_features.json"
CATEGORIES_PATH = MODEL_DIR / "station_categories.json"
SEGMENT_STATS_PATH = MODEL_DIR / "segment_stats.csv"
JOURNEY_MODEL_PATH = MODEL_DIR / "journey_delay_model.txt"
JOURNEY_MODEL_CONFIG_PATH = MODEL_DIR / "journey_delay_model_config.json"
JOURNEY_MODEL_METRICS_PATH = MODEL_DIR / "journey_delay_validation.json"

champion_container = ChampionModelContainer(MODEL_DIR)
model = champion_container.model
MODEL_FEATURES = champion_container.model_features
CATEGORY_MAP = champion_container.category_map
CATEGORY_NORMALIZED_MAP = champion_container.category_normalized_map

segment_stats_index = SegmentStatsIndex(SEGMENT_STATS_PATH)
segment_lookup = segment_stats_index.segment_lookup
outgoing_segment_stats = segment_stats_index.outgoing_segment_stats
incoming_segment_stats = segment_stats_index.incoming_segment_stats

# Journey delay model
journey_model: lgb.Booster | None = None
journey_model_config: dict[str, Any] | None = None
journey_model_metrics: dict[str, Any] | None = None

if JOURNEY_MODEL_PATH.exists() and JOURNEY_MODEL_CONFIG_PATH.exists():
    journey_model = lgb.Booster(model_file=str(JOURNEY_MODEL_PATH))
    with open(JOURNEY_MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
        journey_model_config = json.load(f)
    if JOURNEY_MODEL_METRICS_PATH.exists():
        with open(JOURNEY_MODEL_METRICS_PATH, "r", encoding="utf-8") as f:
            journey_model_metrics = json.load(f)
    logger.info("Journey delay model loaded successfully.")
else:
    logger.info("Journey delay model not installed; /predict-journey will be unavailable.")

# ============================================================
# DATA PROVIDER WITH RESILIENT FALLBACK
# ============================================================

def railradar_headers() -> dict[str, str]:
    return _railradar_headers(RAILRADAR_API_KEY)


def get_live_data(train_number: int) -> dict[str, Any]:
    """Retrieve live train data from RailRadar if key is available, else fallback to simulator."""
    if RAILRADAR_API_KEY and RAILRADAR_API_KEY != "test-key" and not RAILRADAR_API_KEY.startswith("your_"):
        try:
            return fetch_railradar_live(
                train_number, RAILRADAR_API_KEY, HTTP_SESSION, provider_cache, runtime_metrics
            )
        except Exception as e:
            logger.warning("Live RailRadar fetch failed for train %d (%s); using resilient simulator", train_number, e)
    
    live_sim, _ = generate_simulated_train_data(train_number)
    return live_sim


def get_route_data(train_number: int) -> dict[str, Any]:
    """Retrieve route data from RailRadar if key is available, else fallback to simulator."""
    if RAILRADAR_API_KEY and RAILRADAR_API_KEY != "test-key" and not RAILRADAR_API_KEY.startswith("your_"):
        try:
            return fetch_railradar_route(
                train_number, RAILRADAR_API_KEY, HTTP_SESSION, provider_cache, runtime_metrics
            )
        except Exception as e:
            logger.warning("Route RailRadar fetch failed for train %d (%s); using resilient simulator", train_number, e)
    
    _, route_sim = generate_simulated_train_data(train_number)
    return route_sim


def get_weather(latitude: float, longitude: float) -> dict[str, Any]:
    try:
        return fetch_open_meteo_weather(
            latitude, longitude, HTTP_SESSION, provider_cache, runtime_metrics
        )
    except Exception as e:
        logger.debug("Live weather fetch failed (%s); returning default clear weather", e)
        return {
            "temperature_2m": 26.5,
            "relative_humidity_2m": 60.0,
            "precipitation": 0.0,
            "rain": 0.0,
            "weather_code": 1,
            "wind_speed_10m": 10.0,
        }


def get_segment_statistics(
    current_station: str | None, next_station: str | None
) -> tuple[dict[str, Any], str]:
    return segment_stats_index.get_segment_statistics(current_station, next_station)


def prepare_model_dataframe(data: dict[str, Any]) -> Any:
    return champion_container.prepare_dataframe(data)

# ============================================================
# FASTAPI APP & MIDDLEWARE
# ============================================================

app = FastAPI(
    title="Railway Delay Prediction API",
    description="SIH Railway Delay Prediction & Real-Time ETA Intelligence Platform",
    version="5.1",
)

if CORS_ORIGINS == ["*"]:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
    )


@app.middleware("http")
async def protect_and_observe_requests(request: Request, call_next):
    """Add request IDs, per-client sliding window abuse prevention, and latency tracing."""
    if request.url.path not in {"/", "/app", "/health", "/docs", "/openapi.json", "/redoc", "/trains"}:
        client = request.client.host if request.client else "unknown"
        if not request_limiter.allow(client):
            runtime_metrics.increment("http.rate_limited")
            return Response(
                content='{"detail":"Rate limit exceeded. Try again shortly."}',
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                media_type="application/json",
                headers={"Retry-After": "60"},
            )

    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        runtime_metrics.increment("http.unhandled_exception")
        raise
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    runtime_metrics.increment(f"http.{request.method}.{response.status_code}")
    logger.info(
        "request_complete request_id=%s method=%s path=%s status=%s duration_ms=%s",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# Mount frontend static files if available
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ============================================================
# SCHEMAS & AUTH
# ============================================================

class PredictionInput(BaseModel):
    train: int = Field(
        ge=1,
        le=99999,
        description="Indian Railways train number (1 to 99999).",
    )


class JourneyPredictionInput(BaseModel):
    features: dict[str, Any] = Field(
        description="All feature fields from journey_delay_model_config.json."
    )


def require_api_key(x_api_key: str | None = Header(default=None)):
    """Enforce API-key authentication when REQUIRE_API_KEY is enabled."""
    is_required = REQUIRE_API_KEY
    if is_required and x_api_key != APP_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root(request: Request):
    """Return interactive frontend dashboard if accessed via browser, else JSON metadata."""
    accept = request.headers.get("accept", "")
    index_file = FRONTEND_DIR / "index.html"
    if "text/html" in accept and index_file.exists():
        return FileResponse(index_file)
    return {
        "status": "online",
        "service": "Railway Delay Prediction API",
        "version": "5.1",
    }


@app.get("/app")
def frontend_app():
    """Direct route for the frontend dashboard application."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>Frontend dashboard loading...</h1>")


@app.get("/trains")
def list_trains():
    """Return a catalog of high-demand trains with full route data for instant demo exploration."""
    return {
        "status": "success",
        "catalog": get_available_trains_catalog(),
    }


@app.get("/health")
def health():
    provider_mode = "LIVE" if (RAILRADAR_API_KEY and not RAILRADAR_API_KEY.startswith("your_")) else "SIMULATION_FALLBACK"
    return {
        "status": "healthy",
        "model": MODEL_PATH.name,
        "features": MODEL_FEATURES,
        "journey_model_loaded": journey_model is not None,
        "provider_mode": provider_mode,
        "provider_cache": provider_cache.snapshot(),
        "request_rate_limit_per_minute": REQUEST_RATE_LIMIT_PER_MINUTE,
    }


@app.get("/metrics")
def metrics(x_api_key: str | None = Header(default=None)):
    require_api_key(x_api_key)
    return {
        "counters": runtime_metrics.snapshot(),
        "provider_cache": provider_cache.snapshot(),
    }


@app.post("/predict-journey")
def predict_journey(
    data: JourneyPredictionInput,
    x_api_key: str | None = Header(default=None),
):
    require_api_key(x_api_key)
    if journey_model is None or journey_model_config is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Journey model artifacts are not installed.",
        )

    frame = prepare_journey_model_dataframe(data.features, journey_model_config)
    predicted_delay = max(0.0, float(journey_model.predict(frame)[0]))
    threshold = float(journey_model_config["late_threshold_minutes"])
    response = {
        "predicted_destination_delay_minutes": round(predicted_delay, 2),
        "is_predicted_delayed": predicted_delay > threshold,
        "delay_threshold_minutes": threshold,
        "model": JOURNEY_MODEL_PATH.name,
    }
    if journey_model_metrics:
        response["validation"] = journey_model_metrics["model"]
        response["validation_strategy"] = journey_model_metrics["validation_strategy"]
    return response


@app.post("/predict")
def predict(
    data: PredictionInput,
    x_api_key: str | None = Header(default=None),
):
    try:
        require_api_key(x_api_key)
        train_number = int(data.train)

        # 1. Fetch live train feed and extract halt information
        live_data = get_live_data(train_number)
        current = live_data.get("currentLocation") or {}

        current_station = (
            current.get("stationCode")
            or current.get("code")
            or live_data.get("currentStation")
            or live_data.get("stationCode")
        )
        current_station = normalize_station_code(current_station)
        current_station_name = current.get("stationName") or live_data.get("currentStationName")

        next_halt = live_data.get("nextHalt") or {}
        next_station = (
            next_halt.get("stationCode")
            or next_halt.get("code")
            or live_data.get("nextStation")
        )
        next_station = normalize_station_code(next_station)
        next_station_name = get_stop_name(next_halt)

        if not current_station:
            raise ValueError("Unable to determine current station.")

        current_delay = safe_float(
            current.get("delayMinutes"), live_data.get("delayMinutes", 0.0)
        )

        raw_segment_progress = current.get("segmentProgress")
        if raw_segment_progress is None:
            raw_segment_progress = live_data.get("segmentProgress")
        segment_progress_reliable = raw_segment_progress is not None
        segment_progress = max(0.0, min(1.0, safe_float(raw_segment_progress, 0.0)))
        segment_progress_source = (
            "RailRadar segmentProgress"
            if segment_progress_reliable
            else "unavailable (conservative 0.0 fallback)"
        )

        # 2. Fetch route geometry and align current/next stations to true route stops
        route_data = get_route_data(train_number)
        route_stops = route_data.get("stops", [])

        current_index = find_route_index(route_stops, current_station)
        if current_index is None:
            previous_halt = live_data.get("previousHalt") or {}
            previous_code = normalize_station_code(
                previous_halt.get("stationCode") or previous_halt.get("code")
            )
            previous_index = find_route_index(route_stops, previous_code)
            if previous_index is not None:
                current_station = previous_code
                current_index = previous_index
                current_station_name = (
                    get_stop_name(route_stops[current_index]) or current_station_name
                )
            else:
                raise ValueError(
                    "Current location does not identify a route stop and "
                    "no valid previous halt is available."
                )

        next_index = find_route_index(route_stops, next_station, current_index + 1)
        if next_index is None:
            next_station, next_stop = get_next_station_from_route(
                route_stops, current_station
            )
            next_station_name = get_stop_name(next_stop)
        else:
            next_station_name = next_station_name or get_stop_name(route_stops[next_index])

        current_station_name = (
            current_station_name or get_stop_name(route_stops[current_index]) or ""
        )
        next_station_name = next_station_name or ""

        # 3. Position estimation
        position = get_live_position(
            live_data, route_data, current_station, next_station, segment_progress
        )
        latitude = safe_float(position.get("latitude"))
        longitude = safe_float(position.get("longitude"))

        if (
            next_station
            and not segment_progress_reliable
            and position.get("source")
            in {
                "RailRadar live coordinates",
                "RailRadar currentLocation coordinates",
            }
        ):
            derived_progress = derive_segment_progress_from_position(
                route_data, current_station, next_station, latitude, longitude
            )
            if derived_progress is not None:
                segment_progress = derived_progress
                segment_progress_reliable = True
                segment_progress_source = (
                    "derived from RailRadar live coordinates + route geometry"
                )

        # 4. Live weather enrichment
        weather = {}
        weather_error = None
        try:
            weather = get_weather(latitude, longitude)
        except Exception as e:
            weather_error = str(e)
            logger.debug("Weather provider error: %s", e)

        # 5. Historical segment statistics and scheduled travel duration
        if next_station:
            stats, used_segment = get_segment_statistics(current_station, next_station)
            scheduled_segment_minutes = get_scheduled_segment_minutes(
                live_data, current_station, next_station
            )
        else:
            stats = {"mean": 0.0, "median": 0.0, "std": 0.0, "count": 0}
            used_segment = None
            scheduled_segment_minutes = 0.0

        previous_train_delay = get_previous_station_delay(live_data, current_station)
        now = now_ist()
        month = now.month
        day_of_week = now.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0

        # 6. ML Model Inference
        if next_station:
            model_input = {
                "train": str(train_number),
                "station": current_station,
                "next_station": next_station,
                "current_arr_delay": current_delay,
                "scheduled_segment_minutes": scheduled_segment_minutes,
                "past_segment_mean": stats["mean"],
                "past_segment_median": stats["median"],
                "past_segment_std": stats["std"],
                "past_segment_count": stats["count"],
                "day_of_week": day_of_week,
                "month": month,
                "is_weekend": is_weekend,
                "previous_train_delay": previous_train_delay,
            }
            features_df = prepare_model_dataframe(model_input)
            prediction = champion_container.predict(features_df)
        else:
            prediction = max(0.0, float(current_delay))

        # 7. ETA cascading engine
        upcoming_eta = (
            build_upcoming_eta(
                live_data,
                current_station,
                next_station,
                current_delay,
                prediction,
                segment_progress,
                scheduled_segment_minutes,
                stats,
                segment_progress_reliable,
                position.get("source", ""),
            )
            if next_station
            else []
        )

        next_station_eta = (
            upcoming_eta[0]["predicted_arrival"] if upcoming_eta else None
        )
        next_station_eta_minutes = (
            upcoming_eta[0]["eta_minutes_from_now"] if upcoming_eta else None
        )

        return {
            "train": train_number,
            "train_name": live_data.get("trainName", f"Train #{train_number}"),
            "current_station": current_station,
            "current_station_name": current_station_name,
            "next_station": next_station,
            "next_station_name": next_station_name,
            "current_delay_minutes": round(current_delay, 2),
            "segment_progress": round(segment_progress, 4),
            "segment_progress_source": segment_progress_source,
            "position": {
                "latitude": latitude,
                "longitude": longitude,
                "source": position.get("source"),
            },
            "latitude": latitude,
            "longitude": longitude,
            "historical_segment": used_segment,
            "historical_lookup_scope": stats.get("lookup_scope"),
            "historical_statistics": {
                "mean": round(stats["mean"], 2),
                "median": round(stats["median"], 2),
                "std": round(stats["std"], 2),
                "count": stats["count"],
            },
            "scheduled_segment_minutes": round(scheduled_segment_minutes, 2),
            "previous_train_delay": round(previous_train_delay, 2),
            "weather": {
                "temperature_c": weather.get("temperature_2m"),
                "humidity_percent": weather.get("relative_humidity_2m"),
                "precipitation": weather.get("precipitation"),
                "rain_mm": weather.get("rain"),
                "weather_code": weather.get("weather_code"),
                "wind_speed_kmh": weather.get("wind_speed_10m"),
                "available": weather_error is None,
                "error": weather_error,
            },
            "predicted_delay_minutes": round(prediction, 2),
            "additional_predicted_delay_minutes": round(
                max(0.0, prediction - current_delay), 2
            ),
            "next_station_eta": next_station_eta,
            "next_station_eta_minutes": next_station_eta_minutes,
            "eta_confidence": (
                upcoming_eta[0].get("confidence") if upcoming_eta else "LOW"
            ),
            "upcoming_stations": upcoming_eta,
            "route_geometry": route_data.get("geojson"),
            "model": {
                "type": "LightGBM",
                "features": MODEL_FEATURES,
                "weather_used_for_prediction": False,
                "prediction_skipped": not bool(next_station),
                "prediction_note": (
                    "Terminal station: no next station available; "
                    "current delay used instead of model prediction."
                    if not next_station
                    else None
                ),
            },
            "status": "success",
        }

    except requests.RequestException as e:
        logger.warning("External API request failed: %s", e)
        raise HTTPException(
            status_code=502,
            detail="A required external data provider is temporarily unavailable.",
        )
    except ValueError as e:
        logger.warning("Prediction validation failed: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unhandled prediction failure")
        raise HTTPException(
            status_code=500, detail="Internal prediction service error."
        )
