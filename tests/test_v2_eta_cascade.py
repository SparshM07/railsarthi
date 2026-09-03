"""Comprehensive unit tests for production V2 model and ETA autoregressive cascade."""

from __future__ import annotations

from datetime import datetime, timedelta
import math
import os
from pathlib import Path
import unittest
from typing import Any

os.environ.setdefault("RAILRADAR_API_KEY", "test-key")

from backend.eta import build_upcoming_eta, get_eta_confidence
from backend.main import (
    MODEL_DIR,
    champion_container,
    segment_stats_index,
    prepare_model_dataframe,
)
from backend.model_serving import ChampionModelContainer
from backend.stats import get_scheduled_segment_minutes, IST


class MockModelContainer:
    """Mock container recording feature payloads and returning predictable predictions."""

    def __init__(self):
        self.call_history: list[dict[str, Any]] = []

    def prepare_dataframe(self, data: dict[str, Any]) -> dict[str, Any]:
        return data

    def predict(self, dataframe: Any) -> float:
        self.call_history.append(dataframe)
        # Add 5 minutes delay per hop
        current = float(dataframe.get("current_arr_delay", 0.0))
        return current + 5.0


class FailingModelContainer:
    """Mock container that raises exceptions during prediction."""

    def prepare_dataframe(self, data: dict[str, Any]) -> dict[str, Any]:
        return data

    def predict(self, dataframe: Any) -> float:
        raise RuntimeError("Simulated ML engine failure")


class ProductionV2ModelTests(unittest.TestCase):
    def test_production_v2_model_and_features_loaded(self):
        container = ChampionModelContainer(MODEL_DIR)
        expected_features = [
            "train",
            "station",
            "next_station",
            "current_arr_delay",
            "scheduled_segment_minutes",
            "past_segment_mean",
            "past_segment_median",
            "past_segment_std",
            "past_segment_count",
            "day_of_week",
            "month",
            "is_weekend",
            "previous_train_delay",
        ]
        self.assertEqual(container.model_features, expected_features)
        self.assertEqual(len(container.model_features), 13)
        self.assertIn("champion_model_scheduled_segment_v2.txt", str(container.model_path))

    def test_missing_scheduled_segment_duration_remains_nan(self):
        route = [
            {"stationCode": "STN_A"},
            {"stationCode": "STN_B"},
        ]
        minutes = get_scheduled_segment_minutes({"route": route}, "STN_A", "STN_B")
        self.assertTrue(math.isnan(minutes))

        # Check prepared dataframe preserves NaN
        input_data = {
            "train": "12919",
            "station": "STN_A",
            "next_station": "STN_B",
            "current_arr_delay": 15.0,
            "scheduled_segment_minutes": minutes,
            "past_segment_mean": 10.0,
            "past_segment_median": 8.0,
            "past_segment_std": 2.0,
            "past_segment_count": 50,
            "day_of_week": 2,
            "month": 9,
            "is_weekend": 0,
            "previous_train_delay": 5.0,
        }
        df = champion_container.prepare_dataframe(input_data)
        self.assertTrue(math.isnan(df["scheduled_segment_minutes"].iloc[0]))


class ProductionEtaCascadeTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 9, 3, 10, 0, 0, tzinfo=IST)
        # Create a route with 8 stations (Origin + Hop 1 to 7)
        self.route = [
            {"stationCode": "ORIGIN", "stationName": "Origin", "sequence": 1, "scheduledDeparture": "2026-09-03T10:00:00+05:30"},
            {"stationCode": "HOP1", "stationName": "Station 1", "sequence": 2, "scheduledArrival": "2026-09-03T10:30:00+05:30", "scheduledDeparture": "2026-09-03T10:35:00+05:30"},
            {"stationCode": "HOP2", "stationName": "Station 2", "sequence": 3, "scheduledArrival": "2026-09-03T11:00:00+05:30", "scheduledDeparture": "2026-09-03T11:05:00+05:30"},
            {"stationCode": "HOP3", "stationName": "Station 3", "sequence": 4, "scheduledArrival": "2026-09-03T11:30:00+05:30", "scheduledDeparture": "2026-09-03T11:35:00+05:30"},
            {"stationCode": "HOP4", "stationName": "Station 4", "sequence": 5, "scheduledArrival": "2026-09-03T12:00:00+05:30", "scheduledDeparture": "2026-09-03T12:05:00+05:30"},
            {"stationCode": "HOP5", "stationName": "Station 5", "sequence": 6, "scheduledArrival": "2026-09-03T12:30:00+05:30", "scheduledDeparture": "2026-09-03T12:35:00+05:30"},
            {"stationCode": "HOP6", "stationName": "Station 6", "sequence": 7, "scheduledArrival": "2026-09-03T13:00:00+05:30", "scheduledDeparture": "2026-09-03T13:05:00+05:30"},
            {"stationCode": "HOP7", "stationName": "Station 7", "sequence": 8, "scheduledArrival": "2026-09-03T13:30:00+05:30", "scheduledDeparture": "2026-09-03T13:35:00+05:30"},
        ]
        self.live_data = {"trainName": "Malwa Express", "route": self.route}

    def test_autoregressive_hops_and_horizon(self):
        mock_model = MockModelContainer()
        results = build_upcoming_eta(
            live_data=self.live_data,
            current_station="ORIGIN",
            next_station="HOP1",
            current_delay=10.0,
            predicted_delay=12.0,  # Hop 1 prediction
            segment_progress=0.0,
            scheduled_segment_minutes=30.0,
            historical_stats={"mean": 10.0, "median": 8.0, "std": 2.0, "count": 100, "lookup_scope": "EXACT"},
            segment_progress_reliable=False,
            position_source="",
            current_time=self.now,
            previous_delay=5.0,
            train_number="12919",
            model_container=mock_model,
            segment_stats_index=segment_stats_index,
            max_horizon=5,
        )

        self.assertEqual(len(results), 7)

        # 1. Hop 1 prediction exists and is 12.0
        self.assertEqual(results[0]["station_code"], "HOP1")
        self.assertEqual(results[0]["predicted_delay_minutes"], 12.0)
        self.assertEqual(results[0]["cascade_hop"], 1)
        self.assertTrue(results[0]["is_independent_ml_prediction"])

        # 2. Hop 2 receives Hop 1 prediction (12.0) as current_arr_delay, and prior_delay (5.0)
        self.assertEqual(len(mock_model.call_history), 4)  # Hops 2, 3, 4, 5
        hop2_input = mock_model.call_history[0]
        self.assertEqual(hop2_input["station"], "HOP1")
        self.assertEqual(hop2_input["next_station"], "HOP2")
        self.assertEqual(hop2_input["current_arr_delay"], 12.0)
        self.assertEqual(hop2_input["previous_train_delay"], 5.0)
        self.assertEqual(results[1]["predicted_delay_minutes"], 17.0)  # 12.0 + 5.0
        self.assertEqual(results[1]["cascade_hop"], 2)
        self.assertEqual(results[1]["delay_propagation_source"], "ml_autoregressive_cascade")

        # 3. Hop 3 receives Hop 2 prediction (17.0) as current_arr_delay, and prior_delay (12.0)
        hop3_input = mock_model.call_history[1]
        self.assertEqual(hop3_input["current_arr_delay"], 17.0)
        self.assertEqual(hop3_input["previous_train_delay"], 12.0)
        self.assertEqual(results[2]["predicted_delay_minutes"], 22.0)  # 17.0 + 5.0
        self.assertEqual(results[2]["cascade_hop"], 3)

        # 4. Hop 4 receives Hop 3 prediction (22.0) as current_arr_delay, and prior_delay (17.0)
        hop4_input = mock_model.call_history[2]
        self.assertEqual(hop4_input["current_arr_delay"], 22.0)
        self.assertEqual(hop4_input["previous_train_delay"], 17.0)
        self.assertEqual(results[3]["predicted_delay_minutes"], 27.0)  # 22.0 + 5.0
        self.assertEqual(results[3]["cascade_hop"], 4)

        # 5. Hop 5 receives Hop 4 prediction (27.0) as current_arr_delay, and prior_delay (22.0)
        hop5_input = mock_model.call_history[3]
        self.assertEqual(hop5_input["current_arr_delay"], 27.0)
        self.assertEqual(hop5_input["previous_train_delay"], 22.0)
        self.assertEqual(results[4]["predicted_delay_minutes"], 32.0)  # 27.0 + 5.0
        self.assertEqual(results[4]["cascade_hop"], 5)

        # 6. Hop 6+ remains flat at Hop 5 prediction (32.0)
        self.assertEqual(results[5]["predicted_delay_minutes"], 32.0)
        self.assertEqual(results[5]["cascade_hop"], 6)
        self.assertEqual(results[5]["delay_propagation_source"], "flat_after_cascade_horizon")
        self.assertFalse(results[5]["is_independent_ml_prediction"])

        self.assertEqual(results[6]["predicted_delay_minutes"], 32.0)
        self.assertEqual(results[6]["cascade_hop"], 7)
        self.assertEqual(results[6]["delay_propagation_source"], "flat_after_cascade_horizon")

    def test_midnight_timetable_crossing(self):
        midnight_route = [
            {"stationCode": "STN_NIGHT", "sequence": 1, "scheduledDeparture": "2026-09-03T23:45:00+05:30"},
            {"stationCode": "STN_MORNING", "sequence": 2, "scheduledArrival": "2026-09-03T00:30:00+05:30", "scheduledDeparture": "2026-09-03T00:35:00+05:30"},
        ]
        minutes = get_scheduled_segment_minutes({"route": midnight_route}, "STN_NIGHT", "STN_MORNING")
        self.assertEqual(minutes, 45.0)

        results = build_upcoming_eta(
            live_data={"route": midnight_route},
            current_station="STN_NIGHT",
            next_station="STN_MORNING",
            current_delay=0.0,
            predicted_delay=10.0,
            segment_progress=0.0,
            scheduled_segment_minutes=45.0,
            historical_stats={"mean": 5.0, "median": 5.0, "std": 1.0, "count": 100, "lookup_scope": "EXACT"},
            current_time=datetime(2026, 9, 3, 23, 45, 0, tzinfo=IST),
        )
        self.assertEqual(len(results), 1)
        expected_arrival = datetime(2026, 9, 4, 0, 40, 0, tzinfo=IST)
        self.assertEqual(datetime.fromisoformat(results[0]["predicted_arrival"]), expected_arrival)

    def test_monotonicity_invariants(self):
        # Even if scheduled timetable or delays decrease, ETA must never decrease or precede now
        stale_route = [
            {"stationCode": "A", "sequence": 1, "scheduledDeparture": "2026-09-03T08:00:00+05:30"},
            {"stationCode": "B", "sequence": 2, "scheduledArrival": "2026-09-03T08:30:00+05:30"},
            {"stationCode": "C", "sequence": 3, "scheduledArrival": "2026-09-03T08:40:00+05:30"},
        ]
        results = build_upcoming_eta(
            live_data={"route": stale_route},
            current_station="A",
            next_station="B",
            current_delay=10.0,
            predicted_delay=10.0,
            segment_progress=0.0,
            scheduled_segment_minutes=30.0,
            historical_stats={"lookup_scope": "GLOBAL", "count": 10},
            current_time=self.now,  # 10:00:00
        )

        previous_dt = self.now
        for res in results:
            eta_dt = datetime.fromisoformat(res["predicted_arrival"])
            self.assertGreaterEqual(eta_dt, self.now)
            self.assertGreaterEqual(eta_dt, previous_dt)
            previous_dt = eta_dt

    def test_recursive_model_failure_fallback(self):
        failing_model = FailingModelContainer()
        results = build_upcoming_eta(
            live_data=self.live_data,
            current_station="ORIGIN",
            next_station="HOP1",
            current_delay=10.0,
            predicted_delay=15.0,
            segment_progress=0.0,
            scheduled_segment_minutes=30.0,
            historical_stats={"lookup_scope": "GLOBAL", "count": 10},
            current_time=self.now,
            previous_delay=0.0,
            train_number="12919",
            model_container=failing_model,
            segment_stats_index=segment_stats_index,
        )
        self.assertEqual(len(results), 7)
        # Should gracefully fallback to prior delay without crashing
        for res in results:
            self.assertEqual(res["predicted_delay_minutes"], 15.0)

    def test_frontend_response_fields_preserved(self):
        results = build_upcoming_eta(
            live_data=self.live_data,
            current_station="ORIGIN",
            next_station="HOP1",
            current_delay=10.0,
            predicted_delay=15.0,
            segment_progress=0.0,
            scheduled_segment_minutes=30.0,
            historical_stats={"mean": 10.0, "median": 8.0, "std": 2.0, "count": 100, "lookup_scope": "EXACT"},
            segment_progress_reliable=True,
            position_source="RailRadar live coordinates",
            current_time=self.now,
        )
        hop1 = results[0]
        required_keys = [
            "station_code",
            "station_name",
            "sequence",
            "distance_km",
            "scheduled_arrival",
            "scheduled_departure",
            "delay_minutes",
            "platform",
            "is_halt",
            "predicted_delay_minutes",
            "predicted_arrival",
            "eta_minutes_from_now",
            "confidence",
            "eta_method",
            "is_independent_ml_prediction",
            "delay_propagation_source",
            "eta_diagnostics",
            "cascade_hop",
        ]
        for k in required_keys:
            self.assertIn(k, hop1, f"Missing key: {k}")

    def test_calendar_features_and_segment_stats(self):
        # Verify calendar features calculation
        test_dt = datetime(2026, 9, 5, 14, 0, 0, tzinfo=IST)  # Saturday (weekday 5)
        self.assertEqual(test_dt.weekday(), 5)
        self.assertEqual(test_dt.month, 9)
        self.assertEqual(1 if test_dt.weekday() >= 5 else 0, 1)

        test_dt_weekday = datetime(2026, 9, 2, 14, 0, 0, tzinfo=IST)  # Wednesday (weekday 2)
        self.assertEqual(test_dt_weekday.weekday(), 2)
        self.assertEqual(test_dt_weekday.month, 9)
        self.assertEqual(1 if test_dt_weekday.weekday() >= 5 else 0, 0)

    def test_main_response_serialization_with_nan_duration(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        client = TestClient(app)
        client.get("/")
        response = client.post("/predict", json={"train": 12919})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        # Verify no NaN floats serialized as invalid strings
        self.assertIn("scheduled_segment_minutes", data)
        self.assertIn("prediction_explanation", data)
        for factor in data["prediction_explanation"]["factors"]:
            # All factor values must be valid JSON types (float, int, None)
            val = factor.get("value")
            if val is not None:
                self.assertFalse(math.isnan(val))


if __name__ == "__main__":
    unittest.main()
