"""Integration tests for all FastAPI endpoints and middlewares."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch
import requests

from fastapi.testclient import TestClient

from backend.main import app, request_limiter


BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "backend" / "model" / "journey_delay_model_config.json"
DATA_PATH = BASE_DIR / "backend" / "dataset" / "ir_train.csv"


SAMPLE_LIVE_DATA = {
    "trainNumber": "12919",
    "currentLocation": {
        "stationCode": "DR",
        "stationName": "Dadar",
        "delayMinutes": 10,
        "latitude": 19.0178,
        "longitude": 72.8478,
    },
    "nextHalt": {
        "stationCode": "CSMT",
        "stationName": "Mumbai CSMT",
        "scheduledArrival": "2026-09-01T21:30:00+05:30",
    },
    "route": [
        {
            "stationCode": "DR",
            "stationName": "Dadar",
            "scheduledArrival": "2026-09-01T20:30:00+05:30",
            "distance": 0,
            "day": 1,
        },
        {
            "stationCode": "CSMT",
            "stationName": "Mumbai CSMT",
            "scheduledArrival": "2026-09-01T21:30:00+05:30",
            "distance": 9,
            "day": 1,
        },
    ],
}

SAMPLE_ROUTE_DATA = {
    "trainNumber": "12919",
    "stops": [
        {"code": "DR", "name": "Dadar", "coordinates": [72.8478, 19.0178]},
        {"code": "CSMT", "name": "Mumbai CSMT", "coordinates": [72.8358, 18.9400]},
    ],
    "geojson": {
        "geometry": {
            "coordinates": [[72.8478, 19.0178], [72.8358, 18.9400]],
        }
    },
}

SAMPLE_WEATHER_DATA = {
    "temperature_2m": 28.0,
    "relative_humidity_2m": 75.0,
    "precipitation": 0.0,
    "rain": 0.0,
    "weather_code": 1,
    "wind_speed_10m": 12.0,
}


class ApiEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        request_limiter._requests.clear()

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("version", data)
        self.assertEqual(data["service"], "Railway Delay Prediction API")
        self.assertIn("X-Request-ID", response.headers)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["model"], "champion_model.txt")
        self.assertTrue(data["journey_model_loaded"])
        self.assertIn("provider_cache", data)

    def test_metrics_endpoint(self):
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("counters", data)
        self.assertIn("provider_cache", data)

    def test_predict_journey_success(self):
        import pandas as pd
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        row = pd.read_csv(DATA_PATH, nrows=1).iloc[0]
        features = {
            name: row[name].item() if hasattr(row[name], "item") else row[name]
            for name in config["features"]
        }
        response = self.client.post("/predict-journey", json={"features": features})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("predicted_destination_delay_minutes", data)
        self.assertIn("is_predicted_delayed", data)
        self.assertIn("delay_threshold_minutes", data)
        self.assertGreaterEqual(data["predicted_destination_delay_minutes"], 0)

    def test_predict_journey_invalid_input(self):
        response = self.client.post("/predict-journey", json={"features": {"bad_feature": 123}})
        self.assertEqual(response.status_code, 422)

    def test_predict_success_with_mock_providers(self):
        with patch("backend.main.get_live_data", return_value=SAMPLE_LIVE_DATA), \
             patch("backend.main.get_route_data", return_value=SAMPLE_ROUTE_DATA), \
             patch("backend.main.get_weather", return_value=SAMPLE_WEATHER_DATA):
            response = self.client.post("/predict", json={"train": 12919})
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "success")
            self.assertEqual(data["train"], 12919)
            self.assertEqual(data["current_station"], "DR")
            self.assertEqual(data["next_station"], "CSMT")
            self.assertIn("predicted_delay_minutes", data)
            self.assertIn("upcoming_stations", data)
            self.assertIn("weather", data)
            self.assertIn("eta_confidence", data)
            self.assertIn("prediction_explanation", data)
            self.assertEqual(len(data["prediction_explanation"]["factors"]), 5)
            self.assertEqual(data["data_freshness"]["provider_mode"], "SIMULATION_FALLBACK")

    def test_predict_invalid_train_number(self):
        response = self.client.post("/predict", json={"train": 0})
        self.assertEqual(response.status_code, 422)

        response = self.client.post("/predict", json={"train": 100000})
        self.assertEqual(response.status_code, 422)

        response = self.client.post("/predict", json={"train": "invalid"})
        self.assertEqual(response.status_code, 422)

    def test_predict_external_provider_network_error(self):
        with patch("backend.main.get_live_data", side_effect=requests.ConnectionError("Provider down")):
            response = self.client.post("/predict", json={"train": 12919})
            self.assertEqual(response.status_code, 502)
            self.assertEqual(
                response.json()["detail"],
                "A required external data provider is temporarily unavailable.",
            )

    def test_predict_external_provider_business_error(self):
        with patch("backend.main.get_live_data", side_effect=ValueError("Train not running today")):
            response = self.client.post("/predict", json={"train": 12919})
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json()["detail"], "Train not running today")

    def test_rate_limiting(self):
        with patch.object(request_limiter, "allow", return_value=False):
            response = self.client.post("/predict-journey", json={"features": {}})
            self.assertEqual(response.status_code, 429)
            self.assertEqual(response.headers.get("Retry-After"), "60")

    def test_authentication_enforcement(self):
        with patch("backend.main.REQUIRE_API_KEY", True), \
             patch("backend.main.APP_API_KEY", "secret-key-123"):
            res_missing = self.client.get("/metrics")
            self.assertEqual(res_missing.status_code, 401)

            res_wrong = self.client.get("/metrics", headers={"X-API-Key": "wrong-key"})
            self.assertEqual(res_wrong.status_code, 401)

            res_valid = self.client.get("/metrics", headers={"X-API-Key": "secret-key-123"})
            self.assertEqual(res_valid.status_code, 200)


if __name__ == "__main__":
    unittest.main()
