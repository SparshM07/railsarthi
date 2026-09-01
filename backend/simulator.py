"""High-fidelity local Indian Railways timetable and route simulator.

Provides realistic real-time GPS coordinates, delay simulations, and route
geometry for popular express routes and arbitrary train numbers, ensuring
zero-dependency deployments and uninterrupted evaluation.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import random
from typing import Any

IST = timezone(timedelta(hours=5, minutes=30))


def now_ist() -> datetime:
    return datetime.now(IST)


# Catalog of famous trains with realistic stations, coordinates, and timetable schedules
POPULAR_TRAINS: dict[int, dict[str, Any]] = {
    12919: {
        "trainNumber": "12919",
        "trainName": "Malwa Superfast Express",
        "source": "DADN (Dr. Ambedkar Nagar)",
        "destination": "SVDK (Shri Mata Vaishno Devi Katra)",
        "type": "Superfast Express",
        "avgSpeed": 62.0,
        "stops": [
            {"code": "DADN", "name": "Dr. Ambedkar Nagar", "coordinates": [75.7667, 22.5500], "sch_arr": "11:50", "sch_dep": "12:15", "distance": 0, "platform": "1"},
            {"code": "INDB", "name": "Indore Junction", "coordinates": [75.8648, 22.7196], "sch_arr": "12:45", "sch_dep": "12:55", "distance": 21, "platform": "4"},
            {"code": "UJN", "name": "Ujjain Junction", "coordinates": [75.7772, 23.1765], "sch_arr": "13:55", "sch_dep": "14:05", "distance": 79, "platform": "1"},
            {"code": "BPL", "name": "Bhopal Junction", "coordinates": [77.4126, 23.2599], "sch_arr": "17:25", "sch_dep": "17:35", "distance": 262, "platform": "2"},
            {"code": "VGLJ", "name": "VGL Jhansi Junction", "coordinates": [78.5685, 25.4484], "sch_arr": "21:30", "sch_dep": "21:38", "distance": 554, "platform": "4"},
            {"code": "GWL", "name": "Gwalior Junction", "coordinates": [78.1828, 26.2183], "sch_arr": "22:48", "sch_dep": "22:50", "distance": 651, "platform": "2"},
            {"code": "AGC", "name": "Agra Cantt", "coordinates": [78.0081, 27.1767], "sch_arr": "00:45", "sch_dep": "00:50", "distance": 769, "platform": "3"},
            {"code": "NDLS", "name": "New Delhi", "coordinates": [77.2197, 28.6448], "sch_arr": "04:15", "sch_dep": "04:30", "distance": 964, "platform": "5"},
            {"code": "LDH", "name": "Ludhiana Junction", "coordinates": [75.8573, 30.9010], "sch_arr": "08:10", "sch_dep": "08:20", "distance": 1277, "platform": "3"},
            {"code": "JAT", "name": "Jammu Tawi", "coordinates": [74.8723, 32.7060], "sch_arr": "14:15", "sch_dep": "14:25", "distance": 1541, "platform": "1"},
            {"code": "SVDK", "name": "Shri Mata Vaishno Devi Katra", "coordinates": [74.9525, 32.9915], "sch_arr": "16:30", "sch_dep": "16:30", "distance": 1619, "platform": "2"},
        ],
    },
    12002: {
        "trainNumber": "12002",
        "trainName": "New Delhi - Bhopal Shatabdi Express",
        "source": "NDLS (New Delhi)",
        "destination": "RKMP (Rani Kamalapati)",
        "type": "Shatabdi Express",
        "avgSpeed": 84.0,
        "stops": [
            {"code": "NDLS", "name": "New Delhi", "coordinates": [77.2197, 28.6448], "sch_arr": "05:50", "sch_dep": "06:00", "distance": 0, "platform": "1"},
            {"code": "MTJ", "name": "Mathura Junction", "coordinates": [77.6737, 27.4924], "sch_arr": "07:19", "sch_dep": "07:20", "distance": 141, "platform": "1"},
            {"code": "AGC", "name": "Agra Cantt", "coordinates": [78.0081, 27.1767], "sch_arr": "07:50", "sch_dep": "07:55", "distance": 195, "platform": "1"},
            {"code": "GWL", "name": "Gwalior Junction", "coordinates": [78.1828, 26.2183], "sch_arr": "09:23", "sch_dep": "09:28", "distance": 313, "platform": "1"},
            {"code": "VGLJ", "name": "VGL Jhansi Junction", "coordinates": [78.5685, 25.4484], "sch_arr": "10:45", "sch_dep": "10:50", "distance": 410, "platform": "1"},
            {"code": "BPL", "name": "Bhopal Junction", "coordinates": [77.4126, 23.2599], "sch_arr": "14:05", "sch_dep": "14:10", "distance": 702, "platform": "1"},
            {"code": "RKMP", "name": "Rani Kamalapati", "coordinates": [77.4526, 23.2299], "sch_arr": "14:40", "sch_dep": "14:40", "distance": 708, "platform": "1"},
        ],
    },
    22436: {
        "trainNumber": "22436",
        "trainName": "Vande Bharat Express (NDLS - BSB)",
        "source": "NDLS (New Delhi)",
        "destination": "BSB (Varanasi Junction)",
        "type": "Vande Bharat Express",
        "avgSpeed": 95.0,
        "stops": [
            {"code": "NDLS", "name": "New Delhi", "coordinates": [77.2197, 28.6448], "sch_arr": "05:50", "sch_dep": "06:00", "distance": 0, "platform": "16"},
            {"code": "CNB", "name": "Kanpur Central", "coordinates": [80.3507, 26.4537], "sch_arr": "10:08", "sch_dep": "10:10", "distance": 440, "platform": "5"},
            {"code": "PRYJ", "name": "Prayagraj Junction", "coordinates": [81.8340, 25.4497], "sch_arr": "12:08", "sch_dep": "12:10", "distance": 634, "platform": "6"},
            {"code": "BSB", "name": "Varanasi Junction", "coordinates": [82.9904, 25.3283], "sch_arr": "14:00", "sch_dep": "14:00", "distance": 759, "platform": "1"},
        ],
    },
    12424: {
        "trainNumber": "12424",
        "trainName": "Dibrugarh Rajdhani Express",
        "source": "NDLS (New Delhi)",
        "destination": "DBRG (Dibrugarh)",
        "type": "Rajdhani Express",
        "avgSpeed": 76.0,
        "stops": [
            {"code": "NDLS", "name": "New Delhi", "coordinates": [77.2197, 28.6448], "sch_arr": "16:00", "sch_dep": "16:20", "distance": 0, "platform": "16"},
            {"code": "CNB", "name": "Kanpur Central", "coordinates": [80.3507, 26.4537], "sch_arr": "21:02", "sch_dep": "21:07", "distance": 440, "platform": "4"},
            {"code": "DDU", "name": "Pt. DD Upadhyaya Junction", "coordinates": [83.1160, 25.2818], "sch_arr": "01:23", "sch_dep": "01:33", "distance": 787, "platform": "2"},
            {"code": "PNBE", "name": "Patna Junction", "coordinates": [85.1376, 25.6022], "sch_arr": "04:10", "sch_dep": "04:20", "distance": 998, "platform": "1"},
            {"code": "KIR", "name": "Katihar Junction", "coordinates": [87.5750, 25.5450], "sch_arr": "09:40", "sch_dep": "09:50", "distance": 1285, "platform": "1"},
            {"code": "NJP", "name": "New Jalpaiguri", "coordinates": [88.4419, 26.6852], "sch_arr": "13:05", "sch_dep": "13:15", "distance": 1471, "platform": "1A"},
            {"code": "GHY", "name": "Guwahati", "coordinates": [91.7539, 26.1862], "sch_arr": "19:20", "sch_dep": "19:35", "distance": 1878, "platform": "1"},
            {"code": "DBRG", "name": "Dibrugarh", "coordinates": [94.9120, 27.4728], "sch_arr": "07:00", "sch_dep": "07:00", "distance": 2434, "platform": "1"},
        ],
    },
    12952: {
        "trainNumber": "12952",
        "trainName": "Mumbai Rajdhani Express",
        "source": "NDLS (New Delhi)",
        "destination": "MMCT (Mumbai Central)",
        "type": "Rajdhani Express",
        "avgSpeed": 88.0,
        "stops": [
            {"code": "NDLS", "name": "New Delhi", "coordinates": [77.2197, 28.6448], "sch_arr": "16:40", "sch_dep": "16:55", "distance": 0, "platform": "3"},
            {"code": "KOTA", "name": "Kota Junction", "coordinates": [75.8648, 25.2138], "sch_arr": "21:30", "sch_dep": "21:40", "distance": 465, "platform": "2"},
            {"code": "RTM", "name": "Ratlam Junction", "coordinates": [75.0445, 23.3341], "sch_arr": "00:57", "sch_dep": "01:00", "distance": 732, "platform": "4"},
            {"code": "BRC", "name": "Vadodara Junction", "coordinates": [73.1812, 22.3107], "sch_arr": "04:15", "sch_dep": "04:23", "distance": 993, "platform": "1"},
            {"code": "ST", "name": "Surat", "coordinates": [72.8407, 21.2049], "sch_arr": "05:53", "sch_dep": "05:58", "distance": 1122, "platform": "2"},
            {"code": "BVI", "name": "Borivali", "coordinates": [72.8569, 19.2291], "sch_arr": "07:58", "sch_dep": "08:00", "distance": 1356, "platform": "7"},
            {"code": "MMCT", "name": "Mumbai Central", "coordinates": [72.8194, 18.9696], "sch_arr": "08:35", "sch_dep": "08:35", "distance": 1386, "platform": "1"},
        ],
    },
}


def get_available_trains_catalog() -> list[dict[str, Any]]:
    """Return a list of predefined popular trains for quick discovery in frontend/API."""
    catalog = []
    for train_no, data in POPULAR_TRAINS.items():
        catalog.append({
            "train_number": train_no,
            "train_name": data["trainName"],
            "source": data["source"],
            "destination": data["destination"],
            "train_type": data["type"],
            "total_stops": len(data["stops"]),
            "distance_km": data["stops"][-1]["distance"],
        })
    return catalog


def _build_timestamps(stops: list[dict[str, Any]], ref_date: datetime) -> list[dict[str, Any]]:
    """Generate ISO scheduled arrival and departure timestamps anchored around ref_date."""
    enriched_stops = []
    base_date = ref_date.replace(hour=0, minute=0, second=0, microsecond=0)
    current_day = 0
    prev_minutes = -1

    for idx, stop in enumerate(stops):
        arr_str = stop["sch_arr"]
        dep_str = stop["sch_dep"]

        arr_h, arr_m = map(int, arr_str.split(":"))
        dep_h, dep_m = map(int, dep_str.split(":"))

        arr_total = arr_h * 60 + arr_m
        dep_total = dep_h * 60 + dep_m

        if prev_minutes >= 0 and arr_total < prev_minutes:
            current_day += 1

        arr_dt = base_date + timedelta(days=current_day, hours=arr_h, minutes=arr_m)
        if dep_total < arr_total:
            dep_dt = base_date + timedelta(days=current_day + 1, hours=dep_h, minutes=dep_m)
        else:
            dep_dt = base_date + timedelta(days=current_day, hours=dep_h, minutes=dep_m)

        prev_minutes = dep_total

        enriched = dict(stop)
        enriched["scheduledArrival"] = arr_dt.isoformat()
        enriched["scheduledDeparture"] = dep_dt.isoformat()
        enriched["sequence"] = idx + 1
        enriched["stationCode"] = stop["code"]
        enriched["stationName"] = stop["name"]
        enriched_stops.append(enriched)

    return enriched_stops


def _generate_dense_route_coordinates(stops: list[dict[str, Any]]) -> list[list[float]]:
    """Interpolate intermediate coordinate points between station coordinates."""
    coordinates: list[list[float]] = []
    for i in range(len(stops) - 1):
        p1 = stops[i]["coordinates"]
        p2 = stops[i + 1]["coordinates"]
        steps = 8
        for s in range(steps):
            t = s / steps
            lon = p1[0] + t * (p2[0] - p1[0])
            lat = p1[1] + t * (p2[1] - p1[1])
            coordinates.append([round(lon, 4), round(lat, 4)])
    coordinates.append(stops[-1]["coordinates"])
    return coordinates


def generate_simulated_train_data(train_number: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate realistic live data and route data payloads for a train number."""
    now = now_ist()
    train_def = POPULAR_TRAINS.get(train_number)

    if not train_def:
        # Fallback dynamic generator for unseen train numbers
        train_def = {
            "trainNumber": str(train_number),
            "trainName": f"Express Special #{train_number}",
            "source": "NDLS (New Delhi)",
            "destination": "BPL (Bhopal Junction)",
            "type": "Express",
            "avgSpeed": 65.0,
            "stops": [
                {"code": "NDLS", "name": "New Delhi", "coordinates": [77.2197, 28.6448], "sch_arr": "06:00", "sch_dep": "06:15", "distance": 0, "platform": "3"},
                {"code": "AGC", "name": "Agra Cantt", "coordinates": [78.0081, 27.1767], "sch_arr": "08:45", "sch_dep": "08:50", "distance": 195, "platform": "1"},
                {"code": "GWL", "name": "Gwalior Junction", "coordinates": [78.1828, 26.2183], "sch_arr": "10:30", "sch_dep": "10:35", "distance": 313, "platform": "2"},
                {"code": "VGLJ", "name": "VGL Jhansi Junction", "coordinates": [78.5685, 25.4484], "sch_arr": "12:15", "sch_dep": "12:25", "distance": 410, "platform": "4"},
                {"code": "BPL", "name": "Bhopal Junction", "coordinates": [77.4126, 23.2599], "sch_arr": "16:45", "sch_dep": "16:45", "distance": 702, "platform": "1"},
            ],
        }

    stops = _build_timestamps(train_def["stops"], now)
    route_coords = _generate_dense_route_coordinates(stops)

    # Determine simulated current position based on time of day
    # Seed deterministic delay variance from train_number + current hour
    random.seed(train_number + now.hour)
    base_delay = round(random.uniform(2.0, 18.0), 1)

    # Find the active segment based on current clock time
    active_idx = 0
    progress = 0.35

    for i in range(len(stops) - 1):
        dep_time = datetime.fromisoformat(stops[i]["scheduledDeparture"])
        next_arr_time = datetime.fromisoformat(stops[i + 1]["scheduledArrival"])
        if now < dep_time:
            active_idx = i
            progress = 0.0
            break
        elif dep_time <= now <= next_arr_time:
            active_idx = i
            total_duration = max(1.0, (next_arr_time - dep_time).total_seconds())
            progress = min(0.95, max(0.05, (now - dep_time).total_seconds() / total_duration))
            break
        else:
            active_idx = i + 1

    if active_idx >= len(stops) - 1:
        # Train near/at final terminal
        current_stop = stops[-1]
        next_stop = None
        current_code = current_stop["code"]
        next_code = None
        curr_coords = current_stop["coordinates"]
        lat, lon = curr_coords[1], curr_coords[0]
        progress = 1.0
    else:
        current_stop = stops[active_idx]
        next_stop = stops[active_idx + 1]
        current_code = current_stop["code"]
        next_code = next_stop["code"]
        c1 = current_stop["coordinates"]
        c2 = next_stop["coordinates"]
        lon = c1[0] + progress * (c2[0] - c1[0])
        lat = c1[1] + progress * (c2[1] - c1[1])

    # Construct RailRadar-compatible live data structure
    live_data: dict[str, Any] = {
        "success": True,
        "trainNumber": str(train_number),
        "trainName": train_def["trainName"],
        "train": {"avgSpeed": train_def["avgSpeed"], "type": train_def["type"]},
        "currentStation": current_code,
        "currentStationName": current_stop["name"],
        "nextStation": next_code,
        "delayMinutes": base_delay,
        "segmentProgress": round(progress, 4),
        "currentLocation": {
            "stationCode": current_code,
            "stationName": current_stop["name"],
            "delayMinutes": base_delay,
            "latitude": round(lat, 5),
            "longitude": round(lon, 5),
            "segmentProgress": round(progress, 4),
        },
        "nextHalt": {
            "stationCode": next_code,
            "stationName": next_stop["name"] if next_stop else None,
            "scheduledArrival": next_stop["scheduledArrival"] if next_stop else None,
            "scheduledDeparture": next_stop["scheduledDeparture"] if next_stop else None,
            "distance": next_stop["distance"] if next_stop else None,
            "platform": next_stop.get("platform", "1") if next_stop else None,
        } if next_stop else {},
        "previousHalt": {
            "stationCode": stops[active_idx - 1]["code"],
            "stationName": stops[active_idx - 1]["name"],
            "delayArrival": max(0.0, base_delay - 2.0),
            "delayMinutes": max(0.0, base_delay - 2.0),
        } if active_idx > 0 else {},
        "route": stops,
    }

    # Construct RailRadar-compatible route structure
    route_data: dict[str, Any] = {
        "success": True,
        "trainNumber": str(train_number),
        "stops": [
            {
                "code": s["code"],
                "name": s["name"],
                "coordinates": s["coordinates"],
                "distance": s["distance"],
                "sequence": s["sequence"],
                "platform": s.get("platform", "1"),
            }
            for s in stops
        ],
        "geojson": {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": route_coords,
            },
        },
    }

    return live_data, route_data
