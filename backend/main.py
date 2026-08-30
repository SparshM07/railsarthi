from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import lightgbm as lgb
import pandas as pd
import json
import os
import math
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from pathlib import Path


# ============================================================
# ENVIRONMENT / PATHS
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

RAILRADAR_API_KEY = os.getenv("RAILRADAR_API_KEY")

if not RAILRADAR_API_KEY:
    raise RuntimeError("RAILRADAR_API_KEY not found in .env")


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Railway Delay Prediction API",
    description="SIH Railway Delay Prediction Backend",
    version="5.0"
)


# ============================================================
# MODEL FILES
# ============================================================

MODEL_PATH = MODEL_DIR / "champion_model.txt"
FEATURE_CONFIG_PATH = MODEL_DIR / "model_features.json"
CATEGORIES_PATH = MODEL_DIR / "station_categories.json"
SEGMENT_STATS_PATH = MODEL_DIR / "segment_stats.csv"


# ============================================================
# CHECK FILES
# ============================================================

print("\n======================================")
print("CHECKING MODEL FILES")
print("======================================")

required_files = [
    MODEL_PATH,
    FEATURE_CONFIG_PATH,
    CATEGORIES_PATH,
    SEGMENT_STATS_PATH
]

for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required model file not found: {file_path}"
        )

    print("FOUND:", file_path)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n======================================")
print("LOADING MODEL")
print("======================================")

model = lgb.Booster(
    model_file=str(MODEL_PATH)
)

MODEL_FEATURES = model.feature_name()

print("Model loaded successfully.")
print("\nMODEL FEATURES:")

for i, feature in enumerate(MODEL_FEATURES):

    print(i, feature)


# ============================================================
# LOAD FEATURE CONFIG
# ============================================================

print("\n======================================")
print("LOADING FEATURE CONFIGURATION")
print("======================================")

with open(
    FEATURE_CONFIG_PATH,
    "r",
    encoding="utf-8"
) as f:

    feature_config = json.load(f)


if isinstance(feature_config, list):

    EXPORTED_FEATURES = feature_config

elif isinstance(feature_config, dict):

    EXPORTED_FEATURES = (
        feature_config.get("features")
        or feature_config.get("feature_order")
        or feature_config.get("model_features")
        or []
    )

else:

    EXPORTED_FEATURES = []


print("Feature configuration loaded.")
print("EXPORTED FEATURE ORDER:")
print(EXPORTED_FEATURES)


if EXPORTED_FEATURES:

    if EXPORTED_FEATURES != MODEL_FEATURES:

        print("\nWARNING:")
        print(
            "model_features.json order differs from "
            "LightGBM model feature order."
        )

        print(
            "JSON FEATURES:",
            EXPORTED_FEATURES
        )

        print(
            "MODEL FEATURES:",
            MODEL_FEATURES
        )

    else:

        print(
            "Feature configuration matches model feature order."
        )


# ============================================================
# LOAD CATEGORIES
# ============================================================

print("\n======================================")
print("LOADING CATEGORIES")
print("======================================")

with open(
    CATEGORIES_PATH,
    "r",
    encoding="utf-8"
) as f:

    categories = json.load(f)


print("Categories loaded.")

for feature, values in categories.items():

    print(
        feature,
        "->",
        len(values),
        "categories"
    )


# ============================================================
# NORMALIZE CATEGORY LISTS
# ============================================================

def normalize_category_values(values):

    result = []

    for value in values:

        value = str(value).strip()

        if value:

            result.append(value)

    return result


CATEGORY_MAP = {}

for feature in [
    "train",
    "station",
    "next_station"
]:

    raw_values = categories.get(
        feature,
        []
    )

    CATEGORY_MAP[feature] = (
        normalize_category_values(
            raw_values
        )
    )


# ============================================================
# HISTORICAL SEGMENT STATISTICS
# ============================================================

print("\n======================================")
print("LOADING HISTORICAL SEGMENT STATISTICS")
print("======================================")

segment_stats_df = pd.read_csv(
    SEGMENT_STATS_PATH
)

print(
    "Historical segment rows:",
    len(segment_stats_df)
)

print(
    "Columns:",
    segment_stats_df.columns.tolist()
)


segment_lookup = {}

for _, row in segment_stats_df.iterrows():

    segment = str(
        row["segment"]
    ).strip().upper()

    try:

        segment_lookup[segment] = {

            "mean":
                float(row["mean"]),

            "median":
                float(row["median"]),

            "std":
                (
                    float(row["std"])
                    if pd.notna(row["std"])
                    else 0.0
                ),

            "count":
                int(row["count"])
        }

    except (
        TypeError,
        ValueError
    ):

        continue


print(
    "Historical segments loaded:",
    len(segment_lookup)
)


# ============================================================
# RELIABILITY
# ============================================================

MIN_EXACT_SEGMENT_SAMPLES = 20
MIN_FALLBACK_SEGMENT_SAMPLES = 50


# ============================================================
# INPUT
# ============================================================

class PredictionInput(BaseModel):

    train: int


# ============================================================
# TIMEZONE
# ============================================================

IST = timezone(
    timedelta(
        hours=5,
        minutes=30
    )
)


def now_ist():

    return datetime.now(IST)


# ============================================================
# RAILRADAR HEADERS
# ============================================================

def railradar_headers():

    return {

        "Authorization":
            f"Bearer {RAILRADAR_API_KEY}",

        "Accept":
            "application/json"
    }


# ============================================================
# SAFE FLOAT
# ============================================================

def safe_float(
    value,
    default=0.0
):

    try:

        if value is None:

            return default

        if isinstance(
            value,
            str
        ):

            value = value.strip()

            if not value:

                return default

        result = float(value)

        if not math.isfinite(result):

            return default

        return result

    except (
        TypeError,
        ValueError
    ):

        return default


# ============================================================
# NORMALIZE STATION
# ============================================================

def normalize_station_code(value):

    if value is None:

        return None

    value = str(
        value
    ).strip().upper()

    if not value:

        return None

    return value


# ============================================================
# PARSE DATETIME
# ============================================================

def parse_datetime(value):

    if value is None:

        return None

    if isinstance(
        value,
        datetime
    ):

        if value.tzinfo is None:

            return value.replace(
                tzinfo=IST
            )

        return value

    try:

        text = str(
            value
        ).strip()

        if not text:

            return None

        parsed = datetime.fromisoformat(
            text.replace(
                "Z",
                "+00:00"
            )
        )

        if parsed.tzinfo is None:

            parsed = parsed.replace(
                tzinfo=IST
            )

        return parsed

    except (
        TypeError,
        ValueError
    ):

        return None


# ============================================================
# LIVE TRAIN DATA
# ============================================================

def get_live_data(
    train_number
):

    url = (
        "https://api.railradar.in/v1/"
        f"trains/{train_number}/live"
    )

    response = requests.get(
        url,
        headers=railradar_headers(),
        timeout=20
    )

    response.raise_for_status()

    result = response.json()

    if not result.get("success"):

        raise ValueError(
            f"RailRadar live API error: {result}"
        )

    return result["data"]


# ============================================================
# ROUTE DATA
# ============================================================

def get_route_data(
    train_number
):

    url = (
        "https://api.railradar.in/v1/"
        f"trains/{train_number}/route"
    )

    params = {

        "format":
            "geojson",

        "stops":
            "true"
    }

    response = requests.get(
        url,
        headers=railradar_headers(),
        params=params,
        timeout=30
    )

    response.raise_for_status()

    result = response.json()

    if not result.get("success"):

        raise ValueError(
            f"RailRadar route API error: {result}"
        )

    return result["data"]


# ============================================================
# HAVERSINE
# ============================================================

def haversine(
    lon1,
    lat1,
    lon2,
    lat2
):

    R = 6371.0

    lon1 = math.radians(
        float(lon1)
    )

    lat1 = math.radians(
        float(lat1)
    )

    lon2 = math.radians(
        float(lon2)
    )

    lat2 = math.radians(
        float(lat2)
    )

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        math.sin(dlat / 2) ** 2
        +
        math.cos(lat1)
        *
        math.cos(lat2)
        *
        math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return R * c


# ============================================================
# NEAREST ROUTE INDEX
# ============================================================

def nearest_route_index(
    coordinates,
    target
):

    target_lon, target_lat = target

    best_index = None
    best_distance = float("inf")

    for i, point in enumerate(
        coordinates
    ):

        if (
            not point
            or
            len(point) < 2
        ):

            continue

        try:

            lon = float(point[0])
            lat = float(point[1])

            distance = haversine(
                target_lon,
                target_lat,
                lon,
                lat
            )

            if distance < best_distance:

                best_distance = distance
                best_index = i

        except (
            TypeError,
            ValueError
        ):

            continue

    return (
        best_index,
        best_distance
    )


# ============================================================
# STATION COORDINATES
# ============================================================

def get_station_coordinates(
    route_data
):

    result = {}

    for stop in route_data.get(
        "stops",
        []
    ):

        if not isinstance(
            stop,
            dict
        ):

            continue

        code = (

            stop.get("code")

            or stop.get("stationCode")

            or stop.get("station_code")
        )

        if not code:

            continue

        normalized_code = (
            normalize_station_code(
                code
            )
        )

        if not normalized_code:

            continue

        lat = stop.get("lat")

        if lat is None:

            lat = stop.get(
                "latitude"
            )

        lng = stop.get("lng")

        if lng is None:

            lng = stop.get(
                "lon"
            )

        if lng is None:

            lng = stop.get(
                "longitude"
            )

        coordinates = (
            stop.get(
                "coordinates"
            )
            or {}
        )

        if (
            lat is None
            or
            lng is None
        ):

            if isinstance(
                coordinates,
                dict
            ):

                if lat is None:

                    lat = (
                        coordinates.get("lat")
                        or
                        coordinates.get("latitude")
                    )

                if lng is None:

                    lng = (
                        coordinates.get("lng")
                        or
                        coordinates.get("lon")
                        or
                        coordinates.get("longitude")
                    )

        if (
            lat is None
            or
            lng is None
        ):

            continue

        try:

            lat = float(lat)
            lng = float(lng)

            if not (
                math.isfinite(lat)
                and
                math.isfinite(lng)
            ):

                continue

            result[
                normalized_code
            ] = [
                lng,
                lat
            ]

        except (
            TypeError,
            ValueError
        ):

            continue

    return result


# ============================================================
# ESTIMATE TRAIN POSITION
# ============================================================

def estimate_train_position(
    route_coordinates,
    station_coordinates,
    current_station,
    next_station,
    segment_progress
):

    current_station = (
        normalize_station_code(
            current_station
        )
    )

    next_station = (
        normalize_station_code(
            next_station
        )
    )

    if (
        current_station
        not in station_coordinates
    ):

        raise ValueError(
            "No route coordinates available "
            f"for current station {current_station}"
        )

    if (
        next_station
        not in station_coordinates
    ):

        raise ValueError(
            "No route coordinates available "
            f"for next station {next_station}"
        )

    current_point = (
        station_coordinates[
            current_station
        ]
    )

    next_point = (
        station_coordinates[
            next_station
        ]
    )

    current_index, current_error = (
        nearest_route_index(
            route_coordinates,
            current_point
        )
    )

    next_index, next_error = (
        nearest_route_index(
            route_coordinates,
            next_point
        )
    )

    if (
        current_index is None
        or
        next_index is None
    ):

        raise ValueError(
            "Unable to locate stations "
            "on route geometry."
        )

    print("\n======================================")
    print("POSITION ESTIMATION")
    print("======================================")

    print(
        "CURRENT STATION:",
        current_station
    )

    print(
        "NEXT STATION:",
        next_station
    )

    print(
        "CURRENT ROUTE INDEX:",
        current_index
    )

    print(
        "NEXT ROUTE INDEX:",
        next_index
    )

    print(
        "CURRENT STATION DISTANCE:",
        round(current_error, 3),
        "km"
    )

    print(
        "NEXT STATION DISTANCE:",
        round(next_error, 3),
        "km"
    )

    if next_index <= current_index:

        raise ValueError(
            "Invalid route ordering: "
            f"{current_station} index={current_index}, "
            f"{next_station} index={next_index}"
        )

    segment = route_coordinates[
        current_index:
        next_index + 1
    ]

    if len(segment) < 2:

        return {

            "latitude":
                current_point[1],

            "longitude":
                current_point[0]
        }

    cumulative = [0.0]

    for i in range(
        1,
        len(segment)
    ):

        lon1 = segment[i - 1][0]
        lat1 = segment[i - 1][1]

        lon2 = segment[i][0]
        lat2 = segment[i][1]

        cumulative.append(

            cumulative[-1]
            +
            haversine(
                lon1,
                lat1,
                lon2,
                lat2
            )
        )

    total_distance = cumulative[-1]

    if total_distance <= 0:

        return {

            "latitude":
                segment[0][1],

            "longitude":
                segment[0][0]
        }

    progress = safe_float(
        segment_progress,
        0.0
    )

    progress = max(
        0.0,
        min(
            1.0,
            progress
        )
    )

    print(
        "SEGMENT PROGRESS:",
        progress
    )

    target_distance = (
        total_distance
        *
        progress
    )

    estimated_lon = segment[-1][0]
    estimated_lat = segment[-1][1]

    for i in range(
        1,
        len(cumulative)
    ):

        if (
            cumulative[i]
            <
            target_distance
        ):

            continue

        previous_distance = (
            cumulative[i - 1]
        )

        current_distance = (
            cumulative[i]
        )

        if (
            current_distance
            ==
            previous_distance
        ):

            local_progress = 0.0

        else:

            local_progress = (

                target_distance
                -
                previous_distance

            ) / (

                current_distance
                -
                previous_distance
            )

        lon1 = segment[i - 1][0]
        lat1 = segment[i - 1][1]

        lon2 = segment[i][0]
        lat2 = segment[i][1]

        estimated_lon = (

            lon1
            +
            local_progress
            *
            (
                lon2 - lon1
            )
        )

        estimated_lat = (

            lat1
            +
            local_progress
            *
            (
                lat2 - lat1
            )
        )

        break

    print(
        "ESTIMATED LONGITUDE:",
        estimated_lon
    )

    print(
        "ESTIMATED LATITUDE:",
        estimated_lat
    )

    return {

        "latitude":
            estimated_lat,

        "longitude":
            estimated_lon
    }


def derive_segment_progress_from_position(
    route_data, current_station, next_station, latitude, longitude
):
    """Safely derive progress only from a live coordinate snapped to the
    current station-to-next station geometry. Returns None when the geometry
    is missing, out of order, or the live point is more than 2 km from route.
    """
    geometry = (route_data.get("geojson") or {}).get("geometry") or {}
    coordinates = geometry.get("coordinates") or []
    station_coordinates = get_station_coordinates(route_data)
    current_point = station_coordinates.get(normalize_station_code(current_station))
    next_point = station_coordinates.get(normalize_station_code(next_station))
    if not coordinates or not current_point or not next_point:
        return None
    current_index, _ = nearest_route_index(coordinates, current_point)
    next_index, _ = nearest_route_index(coordinates, next_point)
    live_index, live_distance = nearest_route_index(
        coordinates, [longitude, latitude]
    )
    if (
        current_index is None or next_index is None or live_index is None
        or next_index <= current_index
        or not current_index <= live_index <= next_index
        or live_distance > 2.0
    ):
        return None
    segment = coordinates[current_index:next_index + 1]
    cumulative = [0.0]
    for index in range(1, len(segment)):
        cumulative.append(cumulative[-1] + haversine(
            segment[index - 1][0], segment[index - 1][1],
            segment[index][0], segment[index][1]
        ))
    total_distance = cumulative[-1]
    if total_distance <= 0:
        return None
    return max(0.0, min(1.0, cumulative[live_index - current_index] / total_distance))


# ============================================================
# FIND ROUTE STOP
# ============================================================

def find_route_stop(
    route,
    station_code
):

    station_code = (
        normalize_station_code(
            station_code
        )
    )

    for stop in route:

        if not isinstance(
            stop,
            dict
        ):

            continue

        code = (

            stop.get("stationCode")

            or
            stop.get("code")

            or
            stop.get("station_code")
        )

        code = normalize_station_code(
            code
        )

        if code == station_code:

            return stop

    return None


def get_stop_code(stop):
    """Return a normalized code only for a route stop object."""
    if not isinstance(stop, dict):
        return None
    return normalize_station_code(
        stop.get("stationCode")
        or stop.get("code")
        or stop.get("station_code")
    )


def get_stop_name(stop):
    """Return a safe display name for a route stop object."""
    if not isinstance(stop, dict):
        return ""
    return str(stop.get("stationName") or stop.get("name") or "").strip()


def find_route_index(route, station_code, start_index=0):
    """Find a real route-stop index; geometry points are never considered."""
    station_code = normalize_station_code(station_code)
    if not station_code:
        return None
    for index, stop in enumerate(route[start_index:], start=start_index):
        if get_stop_code(stop) == station_code:
            return index
    return None


def get_next_station_from_route(route, current_station):
    """Return the next real stop after current_station, if one exists."""
    current_index = find_route_index(route, current_station)
    if current_index is None:
        return None, None
    for stop in route[current_index + 1:]:
        code = get_stop_code(stop)
        if code:
            return code, stop
    return None, None


# ============================================================
# WEATHER
# ============================================================

def get_weather(
    latitude,
    longitude
):

    url = (
        "https://api.open-meteo.com/v1/forecast"
    )

    params = {

        "latitude":
            latitude,

        "longitude":
            longitude,

        "current":
            (
                "temperature_2m,"
                "relative_humidity_2m,"
                "precipitation,"
                "rain,"
                "weather_code,"
                "wind_speed_10m"
            )
    }

    response = requests.get(
        url,
        params=params,
        timeout=20
    )

    response.raise_for_status()

    result = response.json()

    return result.get(
        "current",
        {}
    )


# ============================================================
# LIVE POSITION
# ============================================================

def get_live_position(
    live_data,
    route_data,
    current_station,
    next_station,
    segment_progress
):

    current = (
        live_data.get(
            "currentLocation"
        )
        or {}
    )

    coordinates = (
        current.get(
            "coordinates"
        )
        or {}
    )

    if isinstance(
        coordinates,
        dict
    ):

        lat = (
            coordinates.get("lat")
            or
            coordinates.get("latitude")
        )

        lng = (
            coordinates.get("lng")
            or
            coordinates.get("lon")
            or
            coordinates.get("longitude")
        )

        if (
            lat is not None
            and
            lng is not None
        ):

            try:

                return {

                    "latitude":
                        float(lat),

                    "longitude":
                        float(lng),

                    "source":
                        "RailRadar live coordinates"
                }

            except (
                TypeError,
                ValueError
            ):

                pass

    lat = (
        current.get("lat")
        or
        current.get("latitude")
    )

    lng = (
        current.get("lng")
        or
        current.get("lon")
        or
        current.get("longitude")
    )

    if (
        lat is not None
        and
        lng is not None
    ):

        try:

            return {

                "latitude":
                    float(lat),

                "longitude":
                    float(lng),

                "source":
                    "RailRadar currentLocation coordinates"
            }

        except (
            TypeError,
            ValueError
        ):

            pass

    # --------------------------------------------------------
    # Terminal station fallback
    # --------------------------------------------------------
    # RailRadar can return no next station at the final stop.
    # There is then no segment to interpolate, so use the
    # current station coordinate directly.

    if not next_station:

        station_coordinates = get_station_coordinates(
            route_data
        )

        current_point = station_coordinates.get(
            normalize_station_code(current_station)
        )

        if current_point:

            return {
                "latitude": current_point[1],
                "longitude": current_point[0],
                "source": "RailRadar terminal station coordinates"
            }

    geojson = (
        route_data.get(
            "geojson",
            {}
        )
        or {}
    )

    geometry = (
        geojson.get(
            "geometry",
            {}
        )
        or {}
    )

    route_coordinates = (
        geometry.get(
            "coordinates",
            []
        )
        or []
    )

    station_coordinates = (
        get_station_coordinates(
            route_data
        )
    )

    print(
        "STATION COORDINATES:",
        len(station_coordinates)
    )

    if (
        route_coordinates
        and
        station_coordinates
    ):

        try:

            position = (
                estimate_train_position(
                    route_coordinates,
                    station_coordinates,
                    current_station,
                    next_station,
                    segment_progress
                )
            )

            position["source"] = (
                "RailRadar route geometry "
                "+ segment progress"
            )

            return position

        except ValueError as e:

            print(
                "POSITION ESTIMATION WARNING:",
                str(e)
            )

    normalized_current = (
        normalize_station_code(
            current_station
        )
    )

    for stop in route_data.get(
        "stops",
        []
    ):

        if not isinstance(
            stop,
            dict
        ):

            continue

        code = (

            stop.get("code")

            or
            stop.get("stationCode")

            or
            stop.get("station_code")
        )

        code = normalize_station_code(
            code
        )

        if code != normalized_current:

            continue

        lat = (
            stop.get("lat")
            or
            stop.get("latitude")
        )

        lng = (
            stop.get("lng")
            or
            stop.get("lon")
            or
            stop.get("longitude")
        )

        if (
            lat is not None
            and
            lng is not None
        ):

            try:

                return {

                    "latitude":
                        float(lat),

                    "longitude":
                        float(lng),

                    "source":
                        "RailRadar station stop coordinates"
                }

            except (
                TypeError,
                ValueError
            ):

                pass

    raise ValueError(
        "Unable to obtain train coordinates "
        "from RailRadar."
    )


# ============================================================
# HISTORICAL SEGMENT
# ============================================================

def get_segment_statistics(
    current_station,
    next_station
):

    current_station = (
        normalize_station_code(
            current_station
        )
    )

    next_station = (
        normalize_station_code(
            next_station
        )
    )

    if not current_station or not next_station:

        return (
            {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "count": 0,
                "lookup_scope": "TERMINAL"
            },
            "TERMINAL"
        )

    requested_segment = (
        f"{current_station}->{next_station}"
    )

    print("\n======================================")
    print("HISTORICAL SEGMENT LOOKUP")
    print("======================================")

    print(
        "REQUESTED SEGMENT:",
        requested_segment
    )

    exact = segment_lookup.get(
        requested_segment
    )

    if (
        exact
        and
        exact["count"]
        >=
        MIN_EXACT_SEGMENT_SAMPLES
    ):

        print(
            "EXACT SEGMENT FOUND"
        )

        print(
            "MEAN:",
            exact["mean"]
        )

        print(
            "MEDIAN:",
            exact["median"]
        )

        print(
            "STD:",
            exact["std"]
        )

        print(
            "COUNT:",
            exact["count"]
        )

        print(
            "SEGMENT STATUS: RELIABLE EXACT"
        )

        return ({**exact, "lookup_scope": "EXACT"}, requested_segment)

    if exact:

        print(
            "EXACT SEGMENT FOUND BUT "
            "SAMPLE COUNT IS LOW:",
            exact["count"]
        )

    else:

        print(
            "EXACT SEGMENT NOT FOUND"
        )

    # A segment ending at the same next station is not a substitute for
    # CURRENT->NEXT.  It describes a different physical segment and was the
    # cause of JRO->LAR being incorrectly reported as BINA->LAR.
    prefix = f"{current_station}->"
    candidates = []

    for segment, stats in (
        segment_lookup.items()
    ):

        if segment == requested_segment or not segment.startswith(prefix):

            continue

        if (
            stats["count"]
            <
            MIN_FALLBACK_SEGMENT_SAMPLES
        ):

            continue

        candidates.append(
            (
                segment,
                stats
            )
        )

    if candidates:

        fallback_segment, fallback_stats = max(
            candidates,
            key=lambda item:
                item[1]["count"]
        )

        print("CURRENT-STATION FALLBACK SEGMENT:", fallback_segment)

        print(
            "FALLBACK REASON: another outgoing segment from the "
            "same current station; it is not the requested segment"
        )

        print(
            "MEAN:",
            fallback_stats["mean"]
        )

        print(
            "MEDIAN:",
            fallback_stats["median"]
        )

        print(
            "STD:",
            fallback_stats["std"]
        )

        print(
            "COUNT:",
            fallback_stats["count"]
        )

        print(
            "SEGMENT STATUS: RELIABLE FALLBACK"
        )

        return (
            {**fallback_stats, "lookup_scope": "CURRENT_STATION_OUTGOING"},
            f"CURRENT_STATION_OUTGOING:{fallback_segment}"
        )

    print(
        "NO RELIABLE SEGMENT FALLBACK AVAILABLE"
    )

    print(
        "USING GLOBAL HISTORICAL STATISTICS"
    )

    all_stats = list(
        segment_lookup.values()
    )

    if not all_stats:

        raise ValueError(
            "Historical segment statistics "
            "are empty."
        )

    total_count = sum(
        x["count"]
        for x in all_stats
    )

    if total_count <= 0:

        total_count = len(
            all_stats
        )

    global_mean = (

        sum(

            x["mean"]
            *
            max(
                1,
                x["count"]
            )

            for x in all_stats

        )

        /
        total_count
    )

    global_median = (

        sum(

            x["median"]
            *
            max(
                1,
                x["count"]
            )

            for x in all_stats

        )

        /
        total_count
    )

    global_std = (

        sum(

            x["std"]
            *
            max(
                1,
                x["count"]
            )

            for x in all_stats

        )

        /
        total_count
    )

    stats = {

        "mean":
            global_mean,

        "median":
            global_median,

        "std":
            global_std,

        "count":
            total_count,

        "lookup_scope": "GLOBAL"
    }

    print(
        "MEAN:",
        stats["mean"]
    )

    print(
        "MEDIAN:",
        stats["median"]
    )

    print(
        "STD:",
        stats["std"]
    )

    print(
        "COUNT:",
        stats["count"]
    )

    print(
        "SEGMENT STATUS: GLOBAL"
    )

    return (
        stats,
        "GLOBAL"
    )


# ============================================================
# SCHEDULED SEGMENT MINUTES
# ============================================================

def get_scheduled_segment_minutes(
    live_data,
    current_station,
    next_station
):

    route = live_data.get(
        "route",
        []
    )

    current_stop = (
        find_route_stop(
            route,
            current_station
        )
    )

    next_stop = (
        find_route_stop(
            route,
            next_station
        )
    )

    current_departure = None
    next_arrival = None

    if current_stop:

        current_departure = (
            parse_datetime(
                current_stop.get(
                    "scheduledDeparture"
                )
            )
        )

        if current_departure is None:

            current_departure = (
                parse_datetime(
                    current_stop.get(
                        "scheduledArrival"
                    )
                )
            )

    if next_stop:

        next_arrival = (
            parse_datetime(
                next_stop.get(
                    "scheduledArrival"
                )
            )
        )

    if (
        current_departure
        and
        next_arrival
    ):

        if next_arrival < current_departure:

            next_arrival += timedelta(
                days=1
            )

        minutes = (

            next_arrival
            -
            current_departure
        ).total_seconds() / 60.0

        if minutes >= 0:

            print(
                "SCHEDULED SEGMENT:",
                f"{current_station}->{next_station}"
            )

            print(
                "SCHEDULED DEPARTURE:",
                current_departure.isoformat()
            )

            print(
                "SCHEDULED ARRIVAL:",
                next_arrival.isoformat()
            )

            print(
                "SCHEDULED SEGMENT MINUTES:",
                minutes
            )

            return max(
                0.0,
                minutes
            )

    if (
        current_stop
        and
        next_stop
    ):

        try:

            d1 = safe_float(
                current_stop.get(
                    "distance"
                ),
                0.0
            )

            d2 = safe_float(
                next_stop.get(
                    "distance"
                ),
                0.0
            )

            distance_km = max(
                0.0,
                d2 - d1
            )

            train_info = (
                live_data.get(
                    "train",
                    {}
                )
                or {}
            )

            avg_speed = safe_float(
                train_info.get(
                    "avgSpeed"
                ),
                55.0
            )

            if (
                avg_speed > 0
                and
                distance_km > 0
            ):

                distance_minutes = (

                    distance_km
                    /
                    avg_speed
                ) * 60.0

                print(
                    "SCHEDULED SEGMENT "
                    "FALLBACK MINUTES:",
                    distance_minutes
                )

                return distance_minutes

        except Exception:

            pass

    print(
        "SCHEDULED SEGMENT FALLBACK: "
        "30 minutes"
    )

    return 30.0


# ============================================================
# PREVIOUS STATION DELAY
# ============================================================

def get_previous_station_delay(
    live_data,
    current_station
):

    previous_halt = (
        live_data.get(
            "previousHalt"
        )
        or {}
    )

    previous_code = (

        previous_halt.get(
            "stationCode"
        )

        or

        previous_halt.get(
            "code"
        )
    )

    route = live_data.get(
        "route",
        []
    )

    if previous_code:

        previous_stop = (
            find_route_stop(
                route,
                previous_code
            )
        )

        if previous_stop:

            value = (
                previous_stop.get(
                    "delayArrival"
                )
            )

            if value is None:

                value = (
                    previous_stop.get(
                        "delayMinutes"
                    )
                )

            return safe_float(
                value,
                0.0
            )

    current_index = None

    normalized_current = (
        normalize_station_code(
            current_station
        )
    )

    for i, stop in enumerate(
        route
    ):

        code = (

            stop.get("stationCode")
            or
            stop.get("code")
            or
            stop.get("station_code")
        )

        code = normalize_station_code(
            code
        )

        if code == normalized_current:

            current_index = i

            break

    if (
        current_index is not None
        and
        current_index > 0
    ):

        previous_stop = route[
            current_index - 1
        ]

        value = (
            previous_stop.get(
                "delayArrival"
            )
        )

        if value is None:

            value = (
                previous_stop.get(
                    "delayMinutes"
                )
            )

        return safe_float(
            value,
            0.0
        )

    return safe_float(
        live_data.get(
            "delayMinutes"
        ),
        0.0
    )


# ============================================================
# UPCOMING STATIONS
# ============================================================

def get_upcoming_stations(
    live_data,
    current_station
):

    route = live_data.get(
        "route",
        []
    )

    normalized_current = (
        normalize_station_code(
            current_station
        )
    )

    current_index = None

    for i, stop in enumerate(
        route
    ):

        if not isinstance(
            stop,
            dict
        ):

            continue

        code = (

            stop.get("stationCode")
            or
            stop.get("code")
            or
            stop.get("station_code")
        )

        code = normalize_station_code(
            code
        )

        if code == normalized_current:

            current_index = i

            break

    if current_index is None:

        return []

    return route[
        current_index + 1:
    ]


# ============================================================
# CATEGORICAL PREPARATION
# ============================================================

def prepare_categorical_feature(
    dataframe,
    feature
):

    if feature not in dataframe.columns:

        return

    allowed_categories = (
        CATEGORY_MAP.get(
            feature,
            []
        )
    )

    if not allowed_categories:

        print(
            f"WARNING: No category list for {feature}"
        )

        return

    # --------------------------------------------------------
    # IMPORTANT FIX
    #
    # LightGBM requires the categorical dtype in the prediction
    # dataframe to match the categorical feature used during
    # training.
    #
    # Unknown values are converted to NaN instead of causing
    # an exception.
    # --------------------------------------------------------

    values = (
        dataframe[feature]
        .astype(str)
        .str.strip()
    )

    dataframe[feature] = pd.Categorical(
        values,
        categories=allowed_categories
    )

    unknown_mask = (
        dataframe[feature].isna()
    )

    if unknown_mask.any():

        unknown_values = (
            values[
                unknown_mask
            ].tolist()
        )

        print(
            f"WARNING: Unknown categorical "
            f"value(s) for '{feature}':",
            unknown_values
        )

        print(
            f"These values will be treated "
            f"as missing by LightGBM."
        )


# ============================================================
# MODEL DATAFRAME
# ============================================================

def prepare_model_dataframe(
    data
):

    dataframe = pd.DataFrame(
        [data]
    )

    # --------------------------------------------------------
    # Categorical columns
    # --------------------------------------------------------

    for feature in [
        "train",
        "station",
        "next_station"
    ]:

        prepare_categorical_feature(
            dataframe,
            feature
        )

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    numeric_features = [

        "current_arr_delay",

        "scheduled_segment_minutes",

        "past_segment_mean",

        "past_segment_median",

        "past_segment_std",

        "past_segment_count",

        "day_of_week",

        "month",

        "is_weekend",

        "previous_train_delay"
    ]

    for feature in numeric_features:

        if feature in dataframe.columns:

            dataframe[feature] = pd.to_numeric(
                dataframe[feature],
                errors="coerce"
            )

    # --------------------------------------------------------
    # Correct feature order
    # --------------------------------------------------------

    missing = [

        feature

        for feature
        in MODEL_FEATURES

        if feature
        not in dataframe.columns
    ]

    if missing:

        raise ValueError(
            "Missing model features: "
            +
            str(missing)
        )

    dataframe = dataframe[
        MODEL_FEATURES
    ]

    return dataframe


# ============================================================
# UPCOMING ETA ENGINE
# ============================================================

def get_eta_confidence(historical_stats, progress_reliable, position_source):
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
        "RailRadar currentLocation coordinates"
    }
    if scope == "EXACT" and count >= 200 and (progress_reliable or has_live_position):
        return "HIGH"
    if scope == "EXACT" and count >= 50 and (progress_reliable or has_live_position):
        return "MEDIUM"
    if scope == "EXACT" and count >= 200:
        return "MEDIUM"
    return "LOW"

def build_upcoming_eta(
    live_data,
    current_station,
    next_station,
    current_delay,
    predicted_delay,
    segment_progress,
    scheduled_segment_minutes,
    historical_stats,
    segment_progress_reliable=False,
    position_source=""
):

    route = live_data.get(
        "route",
        []
    )

    if not route:
        return []

    current_station = normalize_station_code(
        current_station
    )

    next_station = normalize_station_code(
        next_station
    )

    # ========================================================
    # FIND CURRENT STATION INDEX
    # ========================================================

    current_index = None

    for i, stop in enumerate(route):

        code = (
            stop.get("stationCode")
            or stop.get("code")
            or stop.get("station_code")
        )

        code = normalize_station_code(code)

        if code == current_station:

            current_index = i
            break

    if current_index is None:

        print(
            "ETA WARNING: Current station not found:",
            current_station
        )

        return []

    # ========================================================
    # FIND NEXT STATION INDEX
    # ========================================================

    next_index = None

    for i in range(
        current_index + 1,
        len(route)
    ):

        stop = route[i]

        code = (
            stop.get("stationCode")
            or stop.get("code")
            or stop.get("station_code")
        )

        code = normalize_station_code(code)

        if code == next_station:

            next_index = i
            break

    # ========================================================
    # IF NEXT STATION WAS NOT FOUND
    # ========================================================

    if next_index is None:

        print(
            "ETA WARNING: Next station not found:",
            next_station
        )

        return []

    # ========================================================
    # UPCOMING STATIONS
    #
    # IMPORTANT:
    # Start exactly from next_station.
    # ========================================================

    upcoming = route[
        next_index:
    ]

    if not upcoming:

        return []

    # ========================================================
    # CURRENT TIME
    # ========================================================

    now = now_ist()

    # ========================================================
    # NORMALIZE VALUES
    # ========================================================

    progress = max(
        0.0,
        min(
            1.0,
            safe_float(
                segment_progress,
                0.0
            )
        )
    )

    current_delay = safe_float(
        current_delay,
        0.0
    )

    predicted_delay = safe_float(
        predicted_delay,
        current_delay
    )

    scheduled_segment_minutes = max(
        0.0,
        safe_float(
            scheduled_segment_minutes,
            30.0
        )
    )

    # The model target is target_next_arr_delay: the absolute arrival delay
    # at the immediate next station. current_delay is the live delay already
    # incurred, while additional_predicted_delay is only the non-negative
    # change between the model's absolute next-station estimate and it.
    # It is descriptive and is never added separately to ETA.
    #
    # ETA = max(now + remaining physical travel time,
    #           scheduled next arrival + predicted absolute delay).
    # This reconciles a historical/schedule estimate with the train's live
    # state and prevents adding an already-incurred delay twice. Progress is
    # used only when RailRadar supplied it or it was safely derived.
    remaining_segment_minutes = scheduled_segment_minutes * (
        1.0 - progress if segment_progress_reliable else 1.0
    )

    remaining_segment_minutes = max(
        0.0,
        remaining_segment_minutes
    )

    # ========================================================
    # FIRST STOP = ACTUAL NEXT STATION
    # ========================================================

    first_stop = upcoming[0]

    first_code = (
        first_stop.get("stationCode")
        or first_stop.get("code")
        or first_stop.get("station_code")
    )

    first_code = normalize_station_code(
        first_code
    )

    # ========================================================
    # ABSOLUTE NEXT-STATION DELAY / ETA
    # ========================================================

    first_delay = predicted_delay

    additional_predicted_delay = max(
        0.0,
        predicted_delay - current_delay
    )

    first_scheduled_arrival = parse_datetime(
        first_stop.get(
            "scheduledArrival"
        )
    )

    first_scheduled_departure = parse_datetime(
        first_stop.get(
            "scheduledDeparture"
        )
    )

    # A route can express an after-midnight departure with an earlier clock
    # time than its arrival.  Keep the schedule chronology intact without
    # changing the immediate-station ETA calculation below.
    if (
        first_scheduled_arrival
        and
        first_scheduled_departure
    ):

        while (
            first_scheduled_departure
            <
            first_scheduled_arrival
        ):

            first_scheduled_departure += timedelta(days=1)

    live_physical_eta = now + timedelta(minutes=remaining_segment_minutes)
    schedule_model_eta = None
    if first_scheduled_arrival:
        schedule_model_eta = first_scheduled_arrival + timedelta(
            minutes=predicted_delay
        )
    first_eta = max(
        live_physical_eta,
        schedule_model_eta or live_physical_eta
    )
    first_eta_minutes = max(
        0.0,
        (first_eta - now).total_seconds() / 60.0
    )
    eta_confidence = get_eta_confidence(
        historical_stats,
        segment_progress_reliable,
        position_source
    )

    results = [

        {

            "station_code":
                first_code,

            "station_name":
                first_stop.get(
                    "stationName"
                ),

            "sequence":
                first_stop.get(
                    "sequence"
                ),

            "distance_km":
                first_stop.get(
                    "distance"
                ),

            "scheduled_arrival":
                (
                    first_scheduled_arrival.isoformat()
                    if first_scheduled_arrival
                    else None
                ),

            "scheduled_departure":
                (
                    first_scheduled_departure.isoformat()
                    if first_scheduled_departure
                    else None
                ),

            "delay_minutes":
                first_stop.get(
                    "delayMinutes"
                ),

            "platform":
                first_stop.get(
                    "platform"
                ),

            "is_halt":
                first_stop.get(
                    "isHalt",
                    False
                ),

            "predicted_delay_minutes":
                round(
                    first_delay,
                    2
                ),

            "additional_predicted_delay_minutes":
                round(
                    additional_predicted_delay,
                    2
                ),

            "is_independent_ml_prediction": True,

            "predicted_arrival":
                first_eta.isoformat(),

            "eta_minutes_from_now":
                round(
                    first_eta_minutes,
                    2
                ),

            "confidence": eta_confidence,

            "eta_method":
                "max(live_remaining_travel,schedule_plus_absolute_delay)"
        }

    ]

    # ========================================================
    # REMAINING STATIONS
    # ========================================================

    previous_eta = first_eta
    previous_stop = first_stop

    for stop in upcoming[1:]:

        code = (
            stop.get("stationCode")
            or stop.get("code")
            or stop.get("station_code")
        )

        code = normalize_station_code(
            code
        )

        # ----------------------------------------------------
        # Scheduled arrival times
        # ----------------------------------------------------

        previous_scheduled = parse_datetime(
            previous_stop.get(
                "scheduledArrival"
            )
        )

        previous_scheduled_departure = parse_datetime(
            previous_stop.get(
                "scheduledDeparture"
            )
        )

        current_scheduled = parse_datetime(
            stop.get(
                "scheduledArrival"
            )
        )

        # ----------------------------------------------------
        # Travel time between stations
        # ----------------------------------------------------

        travel_minutes = 5.0

        if previous_scheduled and current_scheduled:

            # Arrival-to-arrival schedule time is the correct propagation
            # interval because it contains both the prior station's dwell
            # and the following running time.  Normalize each timestamp in
            # chronology order so routes that cross midnight remain valid.
            if previous_scheduled_departure:

                while (
                    previous_scheduled_departure
                    <
                    previous_scheduled
                ):

                    previous_scheduled_departure += timedelta(days=1)

            chronology_reference = (
                previous_scheduled_departure
                or previous_scheduled
            )

            while current_scheduled < chronology_reference:

                current_scheduled += timedelta(days=1)

            difference = (
                current_scheduled
                -
                previous_scheduled
            ).total_seconds() / 60.0

            if difference > 0:

                travel_minutes = (
                    difference
                )

        # ----------------------------------------------------
        # Propagate ETA
        # ----------------------------------------------------

        previous_eta = (
            previous_eta
            +
            timedelta(
                minutes=travel_minutes
            )
        )

        # ----------------------------------------------------
        # Delay propagation
        # ----------------------------------------------------

        explicit_delay = (
            stop.get(
                "delayMinutes"
            )
        )

        # ----------------------------------------------------
        # Scheduled departure
        # ----------------------------------------------------

        scheduled_departure = parse_datetime(
            stop.get(
                "scheduledDeparture"
            )
        )

        if current_scheduled and scheduled_departure:

            while scheduled_departure < current_scheduled:

                scheduled_departure += timedelta(days=1)

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        results.append({

            "station_code":
                code,

            "station_name":
                stop.get(
                    "stationName"
                ),

            "sequence":
                stop.get(
                    "sequence"
                ),

            "distance_km":
                stop.get(
                    "distance"
                ),

            "scheduled_arrival":
                (
                    current_scheduled.isoformat()
                    if current_scheduled
                    else None
                ),

            "scheduled_departure":
                (
                    scheduled_departure.isoformat()
                    if scheduled_departure
                    else None
                ),

            "delay_minutes":
                explicit_delay,

            "platform":
                stop.get(
                    "platform"
                ),

            "is_halt":
                stop.get(
                    "isHalt",
                    False
                ),

            "predicted_delay_minutes":
                round(
                    first_delay,
                    2
                ),

            "is_independent_ml_prediction": False,

            "delay_propagation_source":
                "immediate_next_station_absolute_delay",

            "predicted_arrival":
                previous_eta.isoformat(),

            "eta_minutes_from_now":
                round(
                    (
                        previous_eta
                        -
                        now
                    ).total_seconds()
                    / 60.0,
                    2
                ),

            "confidence":
                "LOW",

            "eta_method":
                "first_station_eta_plus_scheduled_arrival_interval"
        })

        previous_stop = stop

    return results

# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {

        "status":
            "online",

        "service":
            "Railway Delay Prediction API",

        "version":
            "5.0"
    }


@app.get("/health")
def health():

    return {

        "status":
            "healthy",

        "model":
            MODEL_PATH.name,

        "features":
            MODEL_FEATURES
    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
def predict(
    data: PredictionInput
):
    try:

        # ====================================================
        # 1. TRAIN NUMBER
        # ====================================================

        train_number = int(
            data.train
        )

        print("\n======================================")
        print("LIVE TRAIN DATA")
        print("======================================")

        # ====================================================
        # 2. LIVE DATA
        # ====================================================
        live_data = get_live_data(
            train_number
        )   
    
        current = (
            live_data.get(
                "currentLocation"
            )
            or {}
        )

        current_station = (

            current.get(
                "stationCode"
            )

            or

            current.get(
                "code"
            )

            or

            live_data.get(
                "currentStation"
            )

            or

            live_data.get(
                "stationCode"
            )
        )

        current_station = normalize_station_code(
            current_station
        )

        current_station_name = (
            current.get(
                "stationName"
            )
        )

        if not current_station_name:

            current_station_name = (
                live_data.get(
                    "currentStationName"
                )
            )

        next_halt = (
            live_data.get(
                "nextHalt"
            )
            or
            {}
        )

        next_station = (

            next_halt.get(
                "stationCode"
            )

            or

            next_halt.get(
                "code"
            )

            or

            live_data.get(
                "nextStation"
            )
        )

        next_station = normalize_station_code(
            next_station
        )

        # Always initialise this before any fallback path or response use.
        next_station_name = get_stop_name(next_halt)

        if not current_station:

            raise ValueError(
                "Unable to determine current station."
            )

        if not next_station:

            print(
                "NEXT STATION: None (terminal station)"
            )

        current_delay = safe_float(

            current.get(
                "delayMinutes"
            ),

            live_data.get(
                "delayMinutes",
                0.0
            )
        )

        raw_segment_progress = current.get("segmentProgress")
        if raw_segment_progress is None:
            raw_segment_progress = live_data.get("segmentProgress")
        segment_progress_reliable = raw_segment_progress is not None
        segment_progress = safe_float(raw_segment_progress, 0.0)
        segment_progress = max(0.0, min(1.0, segment_progress))
        segment_progress_source = (
            "RailRadar segmentProgress"
            if segment_progress_reliable
            else "unavailable (conservative 0.0 fallback)"
        )

        print(
            "TRAIN:",
            train_number
        )

        print(
            "CURRENT:",
            current_station
        )

        print(
            "CURRENT NAME:",
            current_station_name
        )

        print(
            "NEXT:",
            next_station
        )

        print(
            "NEXT NAME:",
            next_halt.get(
                "stationName"
            )
        )

        print(
            "CURRENT DELAY:",
            current_delay
        )

        print(
            "SEGMENT PROGRESS:",
            segment_progress
        )

        # ====================================================
        # 3. ROUTE
        # ====================================================


        route_data = get_route_data(
            train_number
        )


        geojson = (
            route_data.get(
                "geojson",
                {}
            )
            or {}
        )

        geometry = (
            geojson.get(
                "geometry",
                {}
            )
            or {}
        )

        route_coordinates = (
            geometry.get(
                "coordinates",
                []
            )
            or []
        )

        route_stops = (
            route_data.get(
                "stops",
                []
            )
            or []
        )

        # currentLocation can describe a point between stations. Only retain
        # a code that is a real route stop; otherwise previousHalt is the
        # safest station anchor for the current segment.
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
                    get_stop_name(route_stops[current_index])
                    or current_station_name
                )
            else:
                raise ValueError(
                    "Current location does not identify a route stop and "
                    "no valid previous halt is available."
                )

        # nextHalt wins only when it is a downstream route stop. Otherwise
        # derive the next real stop; no GeoJSON route point can become a
        # station code.
        next_index = find_route_index(route_stops, next_station, current_index + 1)
        if next_index is None:
            next_station, next_stop = get_next_station_from_route(
                route_stops, current_station
            )
            next_station_name = get_stop_name(next_stop)
        else:
            next_station_name = (
                next_station_name or get_stop_name(route_stops[next_index])
            )
        current_station_name = (
            current_station_name or get_stop_name(route_stops[current_index]) or ""
        )
        next_station_name = next_station_name or ""

        print("\n======================================")
        print("ROUTE DATA")
        print("======================================")

        print(
            "ROUTE GEOMETRY TYPE:",
            geometry.get(
                "type"
            )
        )

        print(
            "ROUTE COORDINATES:",
            len(route_coordinates)
        )

        print(
            "ROUTE STOPS:",
            len(route_stops)
        )

        if route_coordinates:

            print(
                "ROUTE FIRST POINT:",
                route_coordinates[0]
            )

            print(
                "ROUTE LAST POINT:",
                route_coordinates[-1]
            )

        # ====================================================
        # 4. POSITION
        # ====================================================

        position = get_live_position(

            live_data,
            route_data,
            current_station,
            next_station,
            segment_progress
        )

        latitude = safe_float(
            position.get(
                "latitude"
            )
        )

        longitude = safe_float(
            position.get(
                "longitude"
            )
        )

        # A missing RailRadar progress value is not evidence that the train
        # is at the current station. Derive it only when a live coordinate
        # snaps inside this exact route segment; otherwise retain a
        # conservative 0.0 and do not shorten scheduled travel time.
        if (
            next_station
            and not segment_progress_reliable
            and position.get("source") in {
                "RailRadar live coordinates",
                "RailRadar currentLocation coordinates"
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

        print("\n======================================")
        print("TRAIN POSITION")
        print("======================================")

        print(
            "LATITUDE:",
            latitude
        )

        print(
            "LONGITUDE:",
            longitude
        )

        print(
            "POSITION SOURCE:",
            position.get(
                "source"
            )
        )

        # ====================================================
        # 5. WEATHER
        # ====================================================

        weather = {}
        weather_error = None

        try:


            weather = get_weather(
                latitude,
                longitude
            )


        except Exception as e:

            weather_error = str(e)

        print("\n======================================")
        print("WEATHER")
        print("======================================")

        print(
            "TEMPERATURE:",
            weather.get(
                "temperature_2m"
            )
        )

        print(
            "HUMIDITY:",
            weather.get(
                "relative_humidity_2m"
            )
        )

        print(
            "PRECIPITATION:",
            weather.get(
                "precipitation"
            )
        )

        print(
            "RAIN:",
            weather.get(
                "rain"
            )
        )

        print(
            "WEATHER CODE:",
            weather.get(
                "weather_code"
            )
        )

        print(
            "WIND SPEED:",
            weather.get(
                "wind_speed_10m"
            )
        )

        # ====================================================
        # 6. HISTORICAL SEGMENT / SCHEDULED SEGMENT
        # ====================================================

        if next_station:


            stats, used_segment = (
                get_segment_statistics(
                    current_station,
                    next_station
                )
            )


            scheduled_segment_minutes = (
                get_scheduled_segment_minutes(
                    live_data,
                    current_station,
                    next_station
                )
            )

        else:

            # Terminal station: there is no next segment.
            stats = {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "count": 0
            }
            used_segment = None
            scheduled_segment_minutes = 0.0

            print(
                "TERMINAL STATION: no historical/scheduled segment"
            )

        # ====================================================
        # 8. PREVIOUS DELAY
        # ====================================================

        previous_train_delay = (
            get_previous_station_delay(
                live_data,
                current_station
            )
        )

        print(
            "PREVIOUS STATION DELAY:",
            previous_train_delay
        )

        # ====================================================
        # 9. DATE FEATURES
        # ====================================================

        now = now_ist()

        month = now.month

        day_of_week = now.weekday()

        is_weekend = (

            1
            if day_of_week >= 5
            else 0
        )

        # ====================================================
        # 10. RAW MODEL DATA
        # ====================================================

        model_input = {

            "train":
                str(train_number),

            "station":
                current_station,

            "next_station":
                next_station,

            "current_arr_delay":
                current_delay,

            "scheduled_segment_minutes":
                scheduled_segment_minutes,

            "past_segment_mean":
                stats["mean"],

            "past_segment_median":
                stats["median"],

            "past_segment_std":
                stats["std"],

            "past_segment_count":
                stats["count"],

            "day_of_week":
                day_of_week,

            "month":
                month,

            "is_weekend":
                is_weekend,

            "previous_train_delay":
                previous_train_delay
        }

        # ====================================================
        # 11. MODEL DATAFRAME / PREDICTION
        # ====================================================

        features = None

        if next_station:

            features = prepare_model_dataframe(
                model_input
            )

            print("\n======================================")
            print("MODEL FEATURES")
            print("======================================")

            print(MODEL_FEATURES)

            print("\nMODEL INPUT:")
            print(
                features.to_dict(
                    orient="records"
                )[0]
            )

            print("\nMODEL DTYPES:")
            print(features.dtypes)


            prediction = model.predict(
                features
            )[0]


            prediction = max(
                0.0,
                float(prediction)
            )

            print(
                "RAW MODEL PREDICTION:",
                prediction
            )

        else:

            # The trained model requires next_station. At the
            # terminal station that feature does not exist, so
            # do not fabricate a category. Use the live delay as
            # the terminal delay value instead.
            prediction = max(
                0.0,
                float(current_delay)
            )

            print(
                "TERMINAL STATION: model skipped; "
                "using current delay:",
                prediction
            )

        # ====================================================
        # 13. ETA ENGINE
        # ====================================================
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
                position.get("source", "")
            )
            if next_station
            else []
        )

        next_station_eta = (

            upcoming_eta[0][
                "predicted_arrival"
            ]

            if upcoming_eta

            else None
        )

        next_station_eta_minutes = (

            upcoming_eta[0][
                "eta_minutes_from_now"
            ]

            if upcoming_eta

            else None
        )

        print("\n======================================")
        print("ETA ENGINE")
        print("======================================")

        print(
            "CURRENT SEGMENT PROGRESS:",
            segment_progress
        )

        print(
            "REMAINING SCHEDULED MINUTES:",
            round(

                scheduled_segment_minutes
                *
                (
                    1.0
                    -
                    segment_progress
                ),

                2
            )
        )

        print(
            "PREDICTED DELAY:",
            prediction
        )

        print(
            "NEXT STATION:",
            next_station
        )

        print(
            "NEXT STATION ETA:",
            next_station_eta
        )

        print(
            "NEXT STATION ETA MINUTES:",
            next_station_eta_minutes
        )

        print(
            "UPCOMING STATIONS:",
            len(upcoming_eta)
        )

        print(
            "======================================"
        )

        # ====================================================
        # 15. RESPONSE
        # ====================================================

        return {

            "train":
                train_number,

            "current_station":
                current_station,

            "current_station_name":
                current_station_name,

            "next_station":
                next_station,

            "next_station_name":
                next_station_name,

            "current_delay_minutes":
                round(
                    current_delay,
                    2
                ),

            "segment_progress":
                round(
                    segment_progress,
                    4
                ),

            "segment_progress_source": segment_progress_source,

            "position": {

                "latitude":
                    latitude,

                "longitude":
                    longitude,

                "source":
                    position.get(
                        "source"
                    )
            },

            "latitude":
                latitude,

            "longitude":
                longitude,

            "historical_segment":
                used_segment,

            "historical_lookup_scope": stats.get("lookup_scope"),

            "historical_statistics": {

                "mean":
                    round(
                        stats["mean"],
                        2
                    ),

                "median":
                    round(
                        stats["median"],
                        2
                    ),

                "std":
                    round(
                        stats["std"],
                        2
                    ),

                "count":
                    stats["count"]
            },

            "scheduled_segment_minutes":
                round(
                    scheduled_segment_minutes,
                    2
                ),

            "previous_train_delay":
                round(
                    previous_train_delay,
                    2
                ),

            "weather": {

                "temperature_c":
                    weather.get(
                        "temperature_2m"
                    ),

                "humidity_percent":
                    weather.get(
                        "relative_humidity_2m"
                    ),

                "precipitation_mm":
                    weather.get(
                        "precipitation"
                    ),

                "rain_mm":
                    weather.get(
                        "rain"
                    ),

                "weather_code":
                    weather.get(
                        "weather_code"
                    ),

                "wind_speed_kmh":
                    weather.get(
                        "wind_speed_10m"
                    ),

                "available":
                    weather_error is None,

                "error":
                    weather_error
            },

            "predicted_delay_minutes":
                round(
                    prediction,
                    2
                ),

            "additional_predicted_delay_minutes":
                round(
                    max(
                        0.0,
                        prediction - current_delay
                    ),
                    2
                ),

            "next_station_eta":
                next_station_eta,

            "next_station_eta_minutes":
                next_station_eta_minutes,

            "eta_confidence":
                (
                    upcoming_eta[0].get("confidence")
                    if upcoming_eta
                    else "LOW"
                ),

            "upcoming_stations":
                upcoming_eta,

            "model": {

                "type":
                    "LightGBM",

                "features":
                    MODEL_FEATURES,

                "weather_used_for_prediction":
                    False,

                "prediction_skipped":
                    not bool(next_station),

                "prediction_note":
                    (
                        "Terminal station: no next station available; "
                        "current delay used instead of model prediction."
                        if not next_station
                        else None
                    )
            },

            "status":
                "success"
        }

    except requests.HTTPError as e:

        print(
            "\nEXTERNAL API ERROR:",
            str(e)
        )

        raise HTTPException(
            status_code=502,
            detail=
                f"External API error: {str(e)}"
        )

    # ========================================================
    # HTTP EXCEPTION
    # ========================================================

    except HTTPException:

        raise

    # ========================================================
    # GENERAL ERROR
    # ========================================================

    except Exception as e:

        print(
            "\nERROR:",
            str(e)
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
