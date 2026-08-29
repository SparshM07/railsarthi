from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import lightgbm as lgb
import pandas as pd

import json
import os
import requests

from dotenv import load_dotenv
from datetime import datetime, timedelta


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

RAILRADAR_API_KEY = os.getenv("RAILRADAR_API_KEY")

if not RAILRADAR_API_KEY:
    raise RuntimeError(
        "RAILRADAR_API_KEY not found in .env"
    )


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Railway Delay Prediction API",
    description=(
        "SIH Railway Real-Time ETA and Delay "
        "Prediction Backend"
    ),
    version="3.0"
)


# ============================================================
# MODEL FILES
# ============================================================

MODEL_PATH = "model/champion_model.txt"
CATEGORIES_PATH = "model/station_categories.json"
SEGMENT_STATS_PATH = "model/segment_stats.csv"


# ============================================================
# MODEL SETTINGS
# ============================================================

MIN_RELIABLE_SEGMENT_SAMPLES = 30

DEFAULT_SCHEDULED_SEGMENT_MINUTES = 75.0


# ============================================================
# LOAD MODEL
# ============================================================

print("\n======================================")
print("LOADING MODEL")
print("======================================")

model = lgb.Booster(
    model_file=MODEL_PATH
)

print("Model loaded successfully.")

MODEL_FEATURES = model.feature_name()

print("MODEL FEATURES:")
print(MODEL_FEATURES)


# ============================================================
# LOAD CATEGORIES
# ============================================================

with open(
    CATEGORIES_PATH,
    "r"
) as f:

    categories = json.load(f)


print("Station/train categories loaded.")


# ============================================================
# LOAD HISTORICAL SEGMENT STATISTICS
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


# ============================================================
# CREATE FAST SEGMENT LOOKUP
# ============================================================

segment_lookup = {}


for _, row in segment_stats_df.iterrows():

    segment = str(
        row["segment"]
    )

    segment_lookup[segment] = {

        "mean":
            float(row["mean"]),

        "median":
            float(row["median"]),

        "std":
            float(row["std"]),

        "count":
            int(row["count"])
    }


print(
    "Historical segments loaded:",
    len(segment_lookup)
)


# ============================================================
# INPUT MODEL
# ============================================================

class PredictionInput(BaseModel):

    train: int


# ============================================================
# RAILRADAR HEADERS
# ============================================================

def railradar_headers():

    return {

        "Authorization":
            f"Bearer {RAILRADAR_API_KEY}"
    }


# ============================================================
# GENERIC REQUEST HELPER
# ============================================================

def safe_get(
    url,
    params=None,
    timeout=20
):

    response = requests.get(

        url,

        headers=railradar_headers()
        if "railradar.in" in url
        else None,

        params=params,

        timeout=timeout
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# GET LIVE TRAIN DATA
# ============================================================

def get_live_data(train_number):

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
            "RailRadar live API error: "
            f"{result}"
        )

    return result["data"]


# ============================================================
# GET REAL LIVE TRAIN POSITION
# ============================================================

def get_live_map_data(train_number):

    url = (
        "https://api.railradar.in/v1/"
        "legacy/trains/live-map"
    )

    response = requests.get(

        url,

        headers=railradar_headers(),

        timeout=30
    )

    response.raise_for_status()

    result = response.json()

    if not result.get("success"):

        raise ValueError(
            "RailRadar live-map API error: "
            f"{result}"
        )

    train_number = str(train_number)

    for train in result.get(
        "data",
        []
    ):

        if str(
            train.get("train_number")
        ) == train_number:

            return train

    raise ValueError(
        f"Train {train_number} not found "
        "in RailRadar live map"
    )


# ============================================================
# GET WEATHER
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

        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "rain,"
            "weather_code,"
            "wind_speed_10m"
        )
    }

    try:

        response = requests.get(

            url,

            params=params,

            timeout=15
        )

        response.raise_for_status()

        result = response.json()

        return result.get(
            "current",
            {}
        )

    except requests.RequestException as e:

        # Weather is an auxiliary feature.
        # Prediction should NOT fail merely
        # because Open-Meteo is temporarily unavailable.

        print(
            "WARNING: Weather API unavailable:",
            str(e)
        )

        return {

            "temperature_2m":
                None,

            "relative_humidity_2m":
                None,

            "precipitation":
                None,

            "rain":
                None,

            "weather_code":
                None,

            "wind_speed_10m":
                None
        }


# ============================================================
# SAFE ISO DATETIME PARSER
# ============================================================

def parse_datetime(
    value
):

    if not isinstance(
        value,
        str
    ):

        return None

    value = value.strip()

    if not value:

        return None

    try:

        return datetime.fromisoformat(
            value
        )

    except ValueError:

        return None


# ============================================================
# GET SCHEDULED SEGMENT MINUTES
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

    current_stop = None
    next_stop = None

    current_index = None
    next_index = None


    # --------------------------------------------------------
    # Find stations
    # --------------------------------------------------------

    for i, stop in enumerate(route):

        code = stop.get(
            "stationCode"
        )

        if code == current_station:

            current_stop = stop
            current_index = i

        if code == next_station:

            next_stop = stop
            next_index = i

        if (
            current_stop is not None
            and next_stop is not None
        ):

            break


    # --------------------------------------------------------
    # Safety
    # --------------------------------------------------------

    if (
        current_stop is None
        or next_stop is None
    ):

        print(
            "WARNING: Scheduled segment "
            "stations not found:",
            f"{current_station}->{next_station}"
        )

        return DEFAULT_SCHEDULED_SEGMENT_MINUTES


    # --------------------------------------------------------
    # Make sure next station really comes after
    # current station.
    # --------------------------------------------------------

    if (
        current_index is not None
        and next_index is not None
        and next_index <= current_index
    ):

        print(
            "WARNING: Invalid station order:",
            f"{current_station}->{next_station}"
        )

        return DEFAULT_SCHEDULED_SEGMENT_MINUTES


    # --------------------------------------------------------
    # Get scheduled times
    # --------------------------------------------------------

    departure = current_stop.get(
        "scheduledDeparture"
    )

    arrival = next_stop.get(
        "scheduledArrival"
    )


    departure_time = parse_datetime(
        departure
    )

    arrival_time = parse_datetime(
        arrival
    )


    # --------------------------------------------------------
    # Handle invalid/missing times
    # --------------------------------------------------------

    if (
        departure_time is None
        or arrival_time is None
    ):

        print(
            "WARNING: Invalid scheduled time for:",
            f"{current_station}->{next_station}"
        )

        return DEFAULT_SCHEDULED_SEGMENT_MINUTES


    # --------------------------------------------------------
    # Journey-day difference
    # --------------------------------------------------------

    try:

        departure_day = int(
            current_stop.get(
                "departureDay",
                1
            )
        )

    except (
        TypeError,
        ValueError
    ):

        departure_day = 1


    try:

        arrival_day = int(
            next_stop.get(
                "arrivalDay",
                departure_day
            )
        )

    except (
        TypeError,
        ValueError
    ):

        arrival_day = departure_day


    day_difference = (
        arrival_day - departure_day
    )


    if day_difference > 0:

        arrival_time += timedelta(
            days=day_difference
        )


    # --------------------------------------------------------
    # Calculate duration
    # --------------------------------------------------------

    duration = (

        arrival_time
        - departure_time

    ).total_seconds() / 60.0


    # --------------------------------------------------------
    # Safety
    # --------------------------------------------------------

    if duration <= 0:

        print(
            "WARNING: Invalid scheduled duration:",
            duration
        )

        return DEFAULT_SCHEDULED_SEGMENT_MINUTES


    # --------------------------------------------------------
    # Debug
    # --------------------------------------------------------

    print(
        "SCHEDULED SEGMENT:",
        f"{current_station}->{next_station}"
    )

    print(
        "SCHEDULED DEPARTURE:",
        departure
    )

    print(
        "SCHEDULED ARRIVAL:",
        arrival
    )

    print(
        "SCHEDULED SEGMENT MINUTES:",
        round(duration, 2)
    )


    return duration


# ============================================================
# GET PREVIOUS STATION DELAY
# ============================================================

def get_previous_station_delay(
    live_data,
    current_station
):

    route = live_data.get(
        "route",
        []
    )

    current_index = None


    # --------------------------------------------------------
    # Find current station
    # --------------------------------------------------------

    for i, stop in enumerate(route):

        if stop.get(
            "stationCode"
        ) == current_station:

            current_index = i

            break


    if current_index is None:

        print(
            "WARNING: Current station not found "
            "for previous delay."
        )

        return 0.0


    # --------------------------------------------------------
    # First station
    # --------------------------------------------------------

    if current_index == 0:

        print(
            "PREVIOUS STATION DELAY: 0.0"
        )

        return 0.0


    previous_stop = route[
        current_index - 1
    ]


    previous_delay = previous_stop.get(
        "delayArrival"
    )


    if previous_delay is None:

        previous_delay = previous_stop.get(
            "delayDeparture"
        )


    if previous_delay is None:

        previous_delay = 0.0


    try:

        previous_delay = float(
            previous_delay
        )

    except (
        TypeError,
        ValueError
    ):

        previous_delay = 0.0


    print(
        "PREVIOUS STATION:",
        previous_stop.get(
            "stationCode"
        )
    )

    print(
        "PREVIOUS STATION NAME:",
        previous_stop.get(
            "stationName"
        )
    )

    print(
        "PREVIOUS TRAIN DELAY:",
        previous_delay
    )


    return previous_delay


# ============================================================
# GET HISTORICAL SEGMENT STATISTICS
# ============================================================

def get_segment_statistics(
    current_station,
    next_station
):

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


    # ========================================================
    # 1. EXACT MATCH
    # ========================================================

    if requested_segment in segment_lookup:

        stats = segment_lookup[
            requested_segment
        ]


        if (
            stats["count"]
            >= MIN_RELIABLE_SEGMENT_SAMPLES
        ):

            print(
                "EXACT SEGMENT FOUND:",
                requested_segment
            )

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
                "SEGMENT STATUS:",
                "RELIABLE EXACT"
            )

            return (
                stats,
                requested_segment
            )


        print(
            "WARNING: Exact segment has only",
            stats["count"],
            "historical samples."
        )

        print(
            "MINIMUM REQUIRED:",
            MIN_RELIABLE_SEGMENT_SAMPLES
        )

        print(
            "Trying reliable fallback..."
        )


    else:

        print(
            "EXACT SEGMENT NOT FOUND"
        )


    # ========================================================
    # 2. FALLBACK BY NEXT STATION
    # ========================================================

    suffix = (
        f"->{next_station}"
    )

    candidates = []


    for segment, stats in segment_lookup.items():

        if segment.endswith(
            suffix
        ):

            if (
                stats["count"]
                >= MIN_RELIABLE_SEGMENT_SAMPLES
            ):

                candidates.append(
                    (
                        segment,
                        stats
                    )
                )


    if candidates:

        fallback_segment, stats = max(

            candidates,

            key=lambda x:
                x[1]["count"]
        )


        print(
            "FALLBACK SEGMENT:",
            fallback_segment
        )

        print(
            "FALLBACK REASON:",
            "Exact segment unavailable or unreliable"
        )

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
            "SEGMENT STATUS:",
            "RELIABLE FALLBACK"
        )


        return (
            stats,
            fallback_segment
        )


    # ========================================================
    # 3. GLOBAL FALLBACK
    # ========================================================

    print(
        "NO RELIABLE SEGMENT ENDING AT NEXT STATION"
    )

    print(
        "USING GLOBAL HISTORICAL STATISTICS"
    )


    all_stats = [

        x for x in segment_lookup.values()

        if x["count"]
        >= MIN_RELIABLE_SEGMENT_SAMPLES
    ]


    if not all_stats:

        all_stats = list(
            segment_lookup.values()
        )


    if not all_stats:

        raise ValueError(
            "Historical segment statistics are empty."
        )


    global_mean = (

        sum(
            x["mean"]
            for x in all_stats
        )
        /
        len(all_stats)
    )


    global_median = (

        sum(
            x["median"]
            for x in all_stats
        )
        /
        len(all_stats)
    )


    global_std = (

        sum(
            x["std"]
            for x in all_stats
        )
        /
        len(all_stats)
    )


    global_count = sum(

        x["count"]
        for x in all_stats
    )


    stats = {

        "mean":
            global_mean,

        "median":
            global_median,

        "std":
            global_std,

        "count":
            global_count
    }


    print(
        "GLOBAL MEAN:",
        global_mean
    )

    print(
        "GLOBAL MEDIAN:",
        global_median
    )

    print(
        "GLOBAL STD:",
        global_std
    )

    print(
        "GLOBAL COUNT:",
        global_count
    )


    return (
        stats,
        "GLOBAL"
    )


# ============================================================
# GET STATION LIVE BOARD
# ============================================================

def get_station_live_board(
    station_code,
    hours=4,
    include_intermediate=False
):

    allowed_hours = {
        2,
        4,
        6,
        8
    }


    if hours not in allowed_hours:

        hours = 4


    url = (
        "https://api.railradar.in/v1/"
        f"stations/{station_code}/live"
    )


    params = {

        "hours":
            hours,

        "includeIntermediate":
            str(
                include_intermediate
            ).lower()
    }


    response = requests.get(

        url,

        headers=railradar_headers(),

        params=params,

        timeout=20
    )


    response.raise_for_status()

    result = response.json()


    if not result.get(
        "success"
    ):

        raise ValueError(
            "RailRadar station live API error: "
            f"{result}"
        )


    return result["data"]


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


    current_index = None


    # --------------------------------------------------------
    # Find current station
    # --------------------------------------------------------

    for i, stop in enumerate(route):

        if stop.get(
            "stationCode"
        ) == current_station:

            current_index = i

            break


    if current_index is None:

        return []


    upcoming = []


    # --------------------------------------------------------
    # Collect future stations
    # --------------------------------------------------------

    for stop in route[
        current_index + 1:
    ]:

        station_code = stop.get(
            "stationCode"
        )

        station_name = stop.get(
            "stationName"
        )


        if not station_code:

            continue


        # Only stations after current location
        upcoming.append({

            "station_code":
                station_code,

            "station_name":
                station_name,

            "sequence":
                stop.get(
                    "sequence"
                ),

            "distance_km":
                stop.get(
                    "distance"
                ),

            "scheduled_arrival":
                stop.get(
                    "scheduledArrival"
                ),

            "scheduled_departure":
                stop.get(
                    "scheduledDeparture"
                ),

            "delay_minutes":
                stop.get(
                    "delayArrival",
                    stop.get(
                        "delayMinutes"
                    )
                ),

            "platform":
                stop.get(
                    "platform"
                )
        })


    return upcoming


# ============================================================
# BUILD UPCOMING ETA
# ============================================================

def build_upcoming_eta(
    live_data,
    current_station,
    predicted_delay
):

    upcoming = get_upcoming_stations(
        live_data,
        current_station
    )


    result = []


    for stop in upcoming:

        scheduled_arrival = stop.get(
            "scheduled_arrival"
        )


        arrival_time = parse_datetime(
            scheduled_arrival
        )


        predicted_arrival = None


        if arrival_time is not None:

            predicted_arrival = (
                arrival_time
                + timedelta(
                    minutes=predicted_delay
                )
            ).isoformat()


        result.append({

            **stop,

            "predicted_delay_minutes":
                round(
                    predicted_delay,
                    2
                ),

            "predicted_arrival":
                predicted_arrival
        })


    return result


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {

        "message":
            "Railway Delay Prediction API is running!",

        "version":
            "3.0",

        "endpoints": [

            "/predict",

            "/station/{station_code}/live"
        ]
    }


# ============================================================
# STATION LIVE ENDPOINT
# ============================================================

@app.get(
    "/station/{station_code}/live"
)
def station_live(
    station_code: str,
    hours: int = 4,
    includeIntermediate: bool = False
):

    try:

        station_code = (
            station_code
            .strip()
            .upper()
        )


        if not station_code:

            raise HTTPException(

                status_code=400,

                detail="Station code is required."
            )


        data = get_station_live_board(

            station_code,

            hours,

            includeIntermediate
        )


        return {

            "station":
                data.get(
                    "station"
                ),

            "window":
                data.get(
                    "window"
                ),

            "count":
                data.get(
                    "count",
                    0
                ),

            "trains":
                data.get(
                    "trains",
                    []
                )
        }


    except requests.HTTPError as e:

        print(
            "STATION API ERROR:",
            str(e)
        )


        raise HTTPException(

            status_code=502,

            detail=(
                "RailRadar station API error: "
                f"{str(e)}"
            )
        )


    except HTTPException:

        raise


    except Exception as e:

        print(
            "STATION ERROR:",
            str(e)
        )


        raise HTTPException(

            status_code=500,

            detail=str(e)
        )


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
def predict(
    data: PredictionInput
):

    try:

        # ====================================================
        # 1. LIVE TRAIN DATA
        # ====================================================

        live_data = get_live_data(
            data.train
        )


        current = live_data.get(
            "currentLocation"
        )

        next_halt = live_data.get(
            "nextHalt"
        )


        if not current:

            raise ValueError(
                "RailRadar did not return currentLocation."
            )


        if not next_halt:

            raise ValueError(
                "RailRadar did not return nextHalt."
            )


        current_station = (
            current.get(
                "stationCode"
            )
        )


        next_station = (
            next_halt.get(
                "stationCode"
            )
        )


        if not current_station:

            raise ValueError(
                "Current station code missing."
            )


        if not next_station:

            raise ValueError(
                "Next station code missing."
            )


        current_delay = current.get(
            "delayMinutes",
            live_data.get(
                "delayMinutes",
                0
            )
        )


        try:

            current_delay = float(
                current_delay
            )

        except (
            TypeError,
            ValueError
        ):

            current_delay = 0.0


        segment_progress = current.get(
            "segmentProgress",
            0.0
        )


        try:

            segment_progress = float(
                segment_progress
            )

        except (
            TypeError,
            ValueError
        ):

            segment_progress = 0.0


        print("\n\n")

        print("======================================")
        print("LIVE TRAIN DATA")
        print("======================================")

        print(
            "TRAIN:",
            data.train
        )

        print(
            "CURRENT:",
            current_station
        )

        print(
            "NEXT:",
            next_station
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
        # 2. REAL LIVE GPS
        # ====================================================

        live_map = get_live_map_data(
            data.train
        )


        try:

            latitude = float(
                live_map.get(
                    "current_lat"
                )
            )

            longitude = float(
                live_map.get(
                    "current_lng"
                )
            )

        except (
            TypeError,
            ValueError
        ):

            raise ValueError(
                "Invalid live GPS coordinates."
            )


        print(
            "LIVE LATITUDE:",
            latitude
        )

        print(
            "LIVE LONGITUDE:",
            longitude
        )

        print(
            "CURRENT DISTANCE:",
            live_map.get(
                "curr_distance"
            )
        )

        print(
            "NEXT DISTANCE:",
            live_map.get(
                "next_distance"
            )
        )

        print("======================================")


        # ====================================================
        # 3. WEATHER
        # ====================================================

        weather = get_weather(

            latitude,

            longitude
        )


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

        print("======================================")


        # ====================================================
        # 4. HISTORICAL SEGMENT
        # ====================================================

        stats, used_segment = (
            get_segment_statistics(

                current_station,

                next_station
            )
        )


        # ====================================================
        # 5. DATE FEATURES
        # ====================================================

        now = datetime.now()

        month = now.month

        day_of_week = now.weekday()

        is_weekend = (

            1

            if day_of_week >= 5

            else 0
        )


        # ====================================================
        # 6. SCHEDULED SEGMENT
        # ====================================================

        scheduled_segment_minutes = (

            get_scheduled_segment_minutes(

                live_data,

                current_station,

                next_station
            )
        )


        # ====================================================
        # 7. PREVIOUS STATION DELAY
        # ====================================================

        previous_train_delay = (

            get_previous_station_delay(

                live_data,

                current_station
            )
        )


        # ====================================================
        # 8. MODEL FEATURES
        # ====================================================

        features = pd.DataFrame([{

            "train":
                data.train,

            "current_station":
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
        }])


        # ====================================================
        # 9. CATEGORICAL FEATURES
        # ====================================================

        categorical_features = [

            "train",

            "current_station",

            "next_station"
        ]


        for feature in categorical_features:

            if feature not in categories:

                continue


            allowed_categories = categories[
                feature
            ]


            value = features.at[
                0,
                feature
            ]


            # ------------------------------------------------
            # IMPORTANT:
            #
            # Never allow an unknown station to become NaN.
            # ------------------------------------------------

            if value not in allowed_categories:

                print(
                    f"WARNING: {feature} "
                    f"not present in training categories:",
                    value
                )


                # Try UNKNOWN
                if "UNKNOWN" in allowed_categories:

                    value = "UNKNOWN"


                # Otherwise use first valid category.
                # This keeps LightGBM from receiving NaN
                # because of an unseen categorical value.
                else:

                    if not allowed_categories:

                        raise ValueError(
                            f"No categories available "
                            f"for {feature}"
                        )


                    value = (
                        allowed_categories[0]
                    )


                print(
                    f"Using compatibility category "
                    f"for {feature}:",
                    value
                )


            features.at[
                0,
                feature
            ] = value


            features[feature] = pd.Categorical(

                features[feature],

                categories=allowed_categories
            )


        # ====================================================
        # 10. EXACT MODEL FEATURE CHECK
        # ====================================================

        print("\n======================================")
        print("MODEL FEATURES")
        print("======================================")


        print(
            MODEL_FEATURES
        )


        missing_features = [

            feature

            for feature in MODEL_FEATURES

            if feature not in features.columns
        ]


        if missing_features:

            raise ValueError(

                "Missing model features: "
                +
                str(missing_features)
            )


        # ====================================================
        # KEEP EXACT MODEL ORDER
        # ====================================================

        features = features[
            MODEL_FEATURES
        ]


        print("\nMODEL INPUT:")


        print(
            features.to_dict(
                orient="records"
            )[0]
        )


        # ====================================================
        # 11. MODEL PREDICTION
        # ====================================================

        prediction = model.predict(
            features
        )[0]


        prediction = max(

            0.0,

            float(prediction)
        )


        # ====================================================
        # 12. UPCOMING STATIONS + ETA
        # ====================================================

        upcoming_eta = build_upcoming_eta(

            live_data,

            current_station,

            prediction
        )


        print("\n======================================")
        print("PREDICTION")
        print("======================================")


        print(
            "PREDICTED DELAY:",
            prediction
        )


        print(
            "HISTORICAL SEGMENT USED:",
            used_segment
        )


        print(
            "UPCOMING STATIONS:",
            len(upcoming_eta)
        )


        print("======================================")


        # ====================================================
        # 13. RESPONSE
        # ====================================================

        return {

            "train":
                int(data.train),


            "current_station":
                current_station,


            "current_station_name":
                current.get(
                    "stationName"
                ),


            "next_station":
                next_station,


            "next_station_name":
                next_halt.get(
                    "stationName"
                ),


            "current_delay_minutes":
                current_delay,


            "segment_progress":
                segment_progress,


            "latitude":
                latitude,


            "longitude":
                longitude,


            "current_distance_km":
                live_map.get(
                    "curr_distance"
                ),


            "next_distance_km":
                live_map.get(
                    "next_distance"
                ),


            "historical_segment":
                used_segment,


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
                    )
            },


            "predicted_delay_minutes":
                round(
                    prediction,
                    2
                ),


            "upcoming_stations":
                upcoming_eta
        }


    # ========================================================
    # EXTERNAL API ERROR
    # ========================================================

    except requests.HTTPError as e:

        print(
            "\nEXTERNAL API ERROR:",
            str(e)
        )


        raise HTTPException(

            status_code=502,

            detail=(
                "External API error: "
                f"{str(e)}"
            )
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