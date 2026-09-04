"""FastAPI service for live and offline journey-level train delay predictions."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import os
from pathlib import Path
import secrets
import time
from typing import Any
from uuid import uuid4

# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, Header, HTTPException, Request, Response, status
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from fastapi.responses import FileResponse, HTMLResponse
# pyrefly: ignore [missing-import]
from fastapi.staticfiles import StaticFiles
# pyrefly: ignore [missing-import]
import lightgbm as lgb
# pyrefly: ignore [missing-import]
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
    resolve_active_route_segment,
    safe_float,
)
from backend.journey_model import build_journey_model_metadata, prepare_journey_model_dataframe
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

MODEL_PATH = MODEL_DIR / "champion_model_scheduled_segment_v2.txt"
FEATURE_CONFIG_PATH = MODEL_DIR / "model_features_scheduled_segment_v2.json"
CATEGORIES_PATH = MODEL_DIR / "station_categories_scheduled_segment_v2.json"
SEGMENT_STATS_PATH = MODEL_DIR / "segment_stats.csv"
JOURNEY_MODEL_PATH = MODEL_DIR / "journey_delay_model.txt"
JOURNEY_MODEL_METRICS_PATH = MODEL_DIR / "journey_delay_validation.json"
JOURNEY_MODEL_URL = os.getenv("JOURNEY_MODEL_URL", "").strip()


def install_journey_model_if_configured() -> str:
    """Install the optional journey model from a configured HTTPS release URL."""
    if JOURNEY_MODEL_PATH.exists():
        return "installed"
    if not JOURNEY_MODEL_URL:
        return "not_configured"
    if not JOURNEY_MODEL_URL.startswith("https://"):
        logger.warning("Journey model download rejected: JOURNEY_MODEL_URL must use HTTPS.")
        return "invalid_url"
    try:
        response = requests.get(JOURNEY_MODEL_URL, timeout=(5, 90), stream=True)
        response.raise_for_status()
        temporary_path = JOURNEY_MODEL_PATH.with_suffix(".download")
        with open(temporary_path, "wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
        temporary_path.replace(JOURNEY_MODEL_PATH)
        logger.info("Journey model downloaded from configured release asset.")
        return "downloaded"
    except (OSError, requests.RequestException) as exc:
        logger.warning("Journey model download failed: %s", exc)
        return "download_failed"


JOURNEY_MODEL_ARTIFACT_STATUS = install_journey_model_if_configured()

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

if JOURNEY_MODEL_PATH.exists():
    if not JOURNEY_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Journey model path is not a regular file: {JOURNEY_MODEL_PATH.resolve()}"
        )
    if JOURNEY_MODEL_PATH.stat().st_size == 0:
        raise RuntimeError(
            f"Journey model artifact is empty (0 bytes): {JOURNEY_MODEL_PATH.resolve()}"
        )
    try:
        with open(JOURNEY_MODEL_PATH, "rb") as f:
            header = f.read(4096)
        if b"\r\n" in header:
            logger.warning(
                "Journey model %s contains CRLF line endings; normalizing to LF for LightGBM parser",
                JOURNEY_MODEL_PATH,
            )
            content = JOURNEY_MODEL_PATH.read_bytes()
            JOURNEY_MODEL_PATH.write_bytes(content.replace(b"\r\n", b"\n"))
    except OSError as e:
        logger.warning("Could not auto-normalize journey model line endings on disk: %s", e)

    journey_model = lgb.Booster(model_file=str(JOURNEY_MODEL_PATH))
    if JOURNEY_MODEL_METRICS_PATH.exists():
        with open(JOURNEY_MODEL_METRICS_PATH, "r", encoding="utf-8") as f:
            journey_model_metrics = json.load(f)
    journey_model_config = build_journey_model_metadata(
        journey_model, journey_model_metrics
    )
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
            result = fetch_railradar_live(
                train_number, RAILRADAR_API_KEY, HTTP_SESSION, provider_cache, runtime_metrics
            )
            result["_provider_mode"] = "LIVE"
            return result
        except Exception as e:
            logger.warning("Live RailRadar fetch failed for train %d (%s); using resilient simulator", train_number, e)
    
    live_sim, _ = generate_simulated_train_data(train_number)
    live_sim["_provider_mode"] = "SIMULATION_FALLBACK"
    return live_sim


def get_route_data(train_number: int) -> dict[str, Any]:
    """Retrieve route data from RailRadar if key is available, else fallback to simulator."""
    if RAILRADAR_API_KEY and RAILRADAR_API_KEY != "test-key" and not RAILRADAR_API_KEY.startswith("your_"):
        try:
            result = fetch_railradar_route(
                train_number, RAILRADAR_API_KEY, HTTP_SESSION, provider_cache, runtime_metrics
            )
            result["_provider_mode"] = "LIVE"
            return result
        except Exception as e:
            logger.warning("Route RailRadar fetch failed for train %d (%s); using resilient simulator", train_number, e)
    
    _, route_sim = generate_simulated_train_data(train_number)
    route_sim["_provider_mode"] = "SIMULATION_FALLBACK"
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
    title="RailsArthi Railway Intelligence API",
    description="RailsArthi - Indian Railways Delay Prediction & Real-Time ETA Intelligence Platform",
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

SESSION_COOKIE_NAME = "railsarthi_session"
SESSION_TTL_SECONDS = 86400  # 24 hours


def _get_session_secret() -> bytes:
    key_material = APP_API_KEY or "railsarthi_secure_session_secret_key"
    return hashlib.sha256(f"railsarthi_session_secret_{key_material}".encode("utf-8")).digest()


def create_session_token() -> str:
    """Generate an HMAC-SHA256 signed session token for same-origin browser clients."""
    timestamp = int(time.time())
    nonce = secrets.token_hex(8)
    payload = f"{timestamp}:{nonce}"
    signature = hmac.new(_get_session_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload}:{signature}"


def validate_session_token(token: str | None) -> bool:
    """Validate timestamp and cryptographic signature of a session token."""
    if not token or not isinstance(token, str):
        return False
    parts = token.split(":")
    if len(parts) != 3:
        return False
    timestamp_str, nonce, signature = parts
    try:
        timestamp = int(timestamp_str)
    except ValueError:
        return False
    current_time = int(time.time())
    # Reject expired or future-skewed tokens
    if current_time - timestamp > SESSION_TTL_SECONDS or timestamp > current_time + 300:
        return False
    payload = f"{timestamp_str}:{nonce}"
    expected_sig = hmac.new(_get_session_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, expected_sig)


class PredictionInput(BaseModel):
    train: int = Field(
        ge=1,
        le=99999,
        description="Indian Railways train number (1 to 99999).",
    )


class JourneyPredictionInput(BaseModel):
    features: dict[str, Any] = Field(
        description="All feature fields required by the journey delay model."
    )


def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None),
):
    """Enforce API-key or server-signed session authentication when REQUIRE_API_KEY is enabled."""
    if not REQUIRE_API_KEY:
        return True

    # 1. Direct API key validation for external clients
    if x_api_key and APP_API_KEY and hmac.compare_digest(x_api_key, APP_API_KEY):
        return True

    # 2. Server-signed same-origin session cookie for browser dashboard
    session_token = request.cookies.get(SESSION_COOKIE_NAME)
    if session_token and validate_session_token(session_token):
        return True

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key or session.",
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
        response = FileResponse(index_file)
        session_token = request.cookies.get(SESSION_COOKIE_NAME)
        if not session_token or not validate_session_token(session_token):
            response.set_cookie(
                key=SESSION_COOKIE_NAME,
                value=create_session_token(),
                max_age=SESSION_TTL_SECONDS,
                httponly=True,
                samesite="lax",
                path="/",
            )
        return response

    response = Response(
        content=json.dumps({
            "status": "online",
            "service": "RailsArthi Railway Intelligence API",
            "version": "5.1",
        }),
        media_type="application/json",
    )
    session_token = request.cookies.get(SESSION_COOKIE_NAME)
    if not session_token or not validate_session_token(session_token):
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=create_session_token(),
            max_age=SESSION_TTL_SECONDS,
            httponly=True,
            samesite="lax",
            path="/",
        )
    return response


@app.get("/app")
def frontend_app(request: Request):
    """Direct route for the frontend dashboard application."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        response = FileResponse(index_file)
        session_token = request.cookies.get(SESSION_COOKIE_NAME)
        if not session_token or not validate_session_token(session_token):
            response.set_cookie(
                key=SESSION_COOKIE_NAME,
                value=create_session_token(),
                max_age=SESSION_TTL_SECONDS,
                httponly=True,
                samesite="lax",
                path="/",
            )
        return response
    return HTMLResponse("<h1>RailsArthi dashboard loading...</h1>")


@app.get("/trains")
def list_trains():
    """Return a catalog of high-demand trains with full route data for instant demo exploration."""
    return {
        "status": "success",
        "catalog": get_available_trains_catalog(),
    }


@app.get("/health")
def health():
    provider_mode = "LIVE_READY" if (RAILRADAR_API_KEY and not RAILRADAR_API_KEY.startswith("your_")) else "SIMULATION_FALLBACK"
    return {
        "status": "healthy",
        "service": "RailsArthi",
        "version": "5.1",
        "provider_mode": provider_mode,
        "journey_model_loaded": journey_model is not None,
        "provider_cache": provider_cache.snapshot(),
        "request_rate_limit_per_minute": REQUEST_RATE_LIMIT_PER_MINUTE,
    }


@app.get("/metrics")
def metrics(
    request: Request,
    x_api_key: str | None = Header(default=None),
):
    require_api_key(request, x_api_key)
    return {
        "counters": runtime_metrics.snapshot(),
        "provider_cache": provider_cache.snapshot(),
    }


@app.post("/predict-journey")
def predict_journey(
    data: JourneyPredictionInput,
    request: Request,
    x_api_key: str | None = Header(default=None),
):
    require_api_key(request, x_api_key)
    if journey_model is None or journey_model_config is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Journey model artifacts are not installed.",
        )

    frame = prepare_journey_model_dataframe(data.features, journey_model_config)
    predicted_delay = max(0.0, float(journey_model.predict(frame)[0]))  # type: ignore
    threshold = float(journey_model_config.get("late_threshold_minutes", 15.0))
    result: dict[str, Any] = {
        "predicted_destination_delay_minutes": round(predicted_delay, 2),
        "is_predicted_delayed": predicted_delay > threshold,
        "delay_threshold_minutes": threshold,
    }
    if journey_model_metrics:
        result["validation"] = journey_model_metrics
        result["validation_metrics"] = journey_model_metrics
    return result


@app.post("/predict")
def predict(
    data: PredictionInput,
    request: Request,
    x_api_key: str | None = Header(default=None),
):
    try:
        require_api_key(request, x_api_key)
        train_number = int(data.train)

        # 1. Fetch live train feed and extract halt information
        live_data = get_live_data(train_number)
        data_provider_mode = live_data.pop("_provider_mode", "SIMULATION_FALLBACK")
        current = live_data.get("currentLocation") or {}

        raw_current_station = (
            current.get("stationCode")
            or current.get("code")
            or live_data.get("currentStation")
            or live_data.get("stationCode")
        )
        raw_current_station = normalize_station_code(raw_current_station)

        previous_halt = live_data.get("previousHalt") or {}
        previous_code = normalize_station_code(
            previous_halt.get("stationCode") or previous_halt.get("code")
        )

        if not raw_current_station and not previous_code:
            raise ValueError("Unable to determine current station.")

        current_delay = safe_float(
            current.get("delayMinutes"), live_data.get("delayMinutes", 0.0)
        )

        # 2. Fetch authoritative route geometry and stops
        route_data = get_route_data(train_number)
        route_provider_mode = route_data.pop("_provider_mode", "SIMULATION_FALLBACK")
        provider_mode = "LIVE" if data_provider_mode == route_provider_mode == "LIVE" else "SIMULATION_FALLBACK"

        route_stops = route_data.get("stops") or live_data.get("route") or []
        if len(live_data.get("route", [])) > len(route_stops):
            route_stops = live_data.get("route", [])

        if not route_stops:
            raise ValueError("No route stops available for train.")

        # Ensure live_data["route"] contains route_stops if missing
        if not live_data.get("route"):
            live_data["route"] = route_stops

        # Resolve the active adjacent segment with strict invariant enforcement:
        # next_station MUST be the immediate next scheduled route station after current_station.
        (
            current_index,
            current_station,
            current_station_name,
            next_station,
            next_station_name,
        ) = resolve_active_route_segment(
            route_stops,
            raw_current_station,
            live_route=live_data.get("route", []),
            previous_halt_code=previous_code,
        )

        current_station_name = (
            current_station_name
            or current.get("stationName")
            or live_data.get("currentStationName")
            or ""
        )
        next_station_name = next_station_name or ""

        # 3. Segment progress resolution for the active adjacent segment
        if not next_station:
            segment_progress = 1.0
            segment_progress_reliable = True
            segment_progress_source = "terminal station reached"
        else:
            raw_segment_progress = None
            cur_telemetry_code = normalize_station_code(
                current.get("stationCode")
                or current.get("code")
                or live_data.get("currentStation")
            )
            if cur_telemetry_code == current_station:
                raw_segment_progress = current.get("segmentProgress")
                if raw_segment_progress is None:
                    raw_segment_progress = live_data.get("segmentProgress")

            if raw_segment_progress is not None:
                segment_progress = max(0.0, min(1.0, safe_float(raw_segment_progress, 0.0)))
                segment_progress_reliable = True
                segment_progress_source = "RailRadar segmentProgress"
            else:
                # Check distance traveled along active segment
                cur_stop = route_stops[current_index]
                nxt_stop = route_stops[current_index + 1]
                cur_dist = safe_float(cur_stop.get("distance"), -1.0)
                nxt_dist = safe_float(nxt_stop.get("distance"), -1.0)
                dist_from_last = safe_float(current.get("distanceFromLastStationKm"), -1.0)
                if dist_from_last >= 0 and cur_dist >= 0 and nxt_dist > cur_dist:
                    seg_km = nxt_dist - cur_dist
                    segment_progress = max(0.0, min(1.0, dist_from_last / seg_km))
                    segment_progress_reliable = True
                    segment_progress_source = "derived from distance traveled along active segment"
                else:
                    segment_progress = 0.0
                    segment_progress_reliable = False
                    segment_progress_source = "unavailable (conservative 0.0 fallback)"

        # 4. Position estimation & derivation along adjacent segment
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
                segment_progress = max(0.0, min(1.0, derived_progress))
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
                previous_delay=previous_train_delay,
                train_number=train_number,
                model_container=champion_container,
                segment_stats_index=segment_stats_index,
                max_horizon=5,
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
            "scheduled_segment_minutes": (
                round(scheduled_segment_minutes, 2)
                if (scheduled_segment_minutes is not None and not math.isnan(scheduled_segment_minutes))
                else None
            ),
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
            "data_freshness": {
                "provider_mode": provider_mode,
                "generated_at": now.isoformat(),
                "refresh_recommended_seconds": 15,
            },
            "prediction_explanation": {
                "summary": (
                    f"Prediction starts from the current {round(current_delay, 1)} minute delay and "
                    f"uses {stats['count']} historical observations for the selected segment."
                ),
                "factors": [
                    {"name": "Current delay", "value": round(current_delay, 1), "unit": "minutes"},
                    {"name": "Historical segment median", "value": round(stats["median"], 1), "unit": "minutes"},
                    {"name": "Previous station delay", "value": round(previous_train_delay, 1), "unit": "minutes"},
                    {
                        "name": "Scheduled segment",
                        "value": (
                            round(scheduled_segment_minutes, 1)
                            if (scheduled_segment_minutes is not None and not math.isnan(scheduled_segment_minutes))
                            else None
                        ),
                        "unit": "minutes",
                    },
                    {"name": "Historical sample size", "value": stats["count"], "unit": "observations"},
                ],
                "weather_note": "Weather is displayed for context and is not currently a live-model feature.",
            },
            "upcoming_stations": upcoming_eta,
            "route_geometry": route_data.get("geojson"),
            "prediction": {
                "available": bool(next_station),
                "note": (
                    "Terminal station: no next station available; "
                    "current delay maintained."
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
