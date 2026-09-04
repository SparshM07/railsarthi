"""Offline regression checks for business logic that must not call providers."""

import os
from pathlib import Path
import unittest
from datetime import datetime, timezone

os.environ.setdefault("RAILRADAR_API_KEY", "test-key")

from backend.main import (  # noqa: E402
    PredictionInput,
    build_upcoming_eta,
    get_segment_statistics,
    prepare_model_dataframe,
)


class PredictionInputTests(unittest.TestCase):
    def test_accepts_valid_train_number(self):
        self.assertEqual(PredictionInput(train=12919).train, 12919)

    def test_rejects_out_of_range_train_numbers(self):
        for train in (0, -1, 100000):
            with self.assertRaises(Exception):
                PredictionInput(train=train)

    def test_champion_model_uses_lf_line_endings(self):
        model_path = (
            Path(__file__).resolve().parents[1]
            / "backend"
            / "model"
            / "champion_model_scheduled_segment_v2.txt"
        )
        self.assertNotIn(b"\r\n", model_path.read_bytes())


class HistoricalStatisticsTests(unittest.TestCase):
    def test_known_segment_uses_exact_statistics(self):
        stats, segment = get_segment_statistics("DR", "CSMT")
        self.assertEqual(segment, "DR->CSMT")
        self.assertEqual(stats["lookup_scope"], "EXACT")
        self.assertGreaterEqual(stats["count"], 20)

    def test_unknown_segment_has_a_labelled_fallback(self):
        stats, _ = get_segment_statistics("NOT_A_STATION", "ALSO_UNKNOWN")
        self.assertIn(
            stats["lookup_scope"],
            {"CURRENT_STATION", "NEXT_STATION", "GLOBAL"},
        )
        self.assertGreater(stats["count"], 0)


class FeaturePreparationTests(unittest.TestCase):
    def test_features_are_model_ordered_and_unknown_categories_are_safe(self):
        features = prepare_model_dataframe({
            "train": "99999",
            "station": "UNKNOWN",
            "next_station": "UNKNOWN",
            "current_arr_delay": 10,
            "scheduled_segment_minutes": 30,
            "past_segment_mean": 10,
            "past_segment_median": 8,
            "past_segment_std": 2,
            "past_segment_count": 100,
            "day_of_week": 1,
            "month": 1,
            "is_weekend": 0,
            "previous_train_delay": 5,
        })
        self.assertEqual(list(features.columns), [
            "train", "station", "next_station", "current_arr_delay",
            "scheduled_segment_minutes", "past_segment_mean",
            "past_segment_median", "past_segment_std", "past_segment_count",
            "day_of_week", "month", "is_weekend", "previous_train_delay",
        ])
        self.assertTrue(features["station"].isna().iloc[0])


class EtaTests(unittest.TestCase):
    def test_past_schedule_model_eta_uses_live_remaining_travel(self):
        route = [
            {"stationCode": "AAA", "scheduledArrival": "2026-09-01T20:00:00+05:30"},
            {"stationCode": "BBB", "stationName": "B", "scheduledArrival": "2026-09-01T20:10:00+05:30"},
        ]
        now = datetime.fromisoformat("2026-09-01T21:00:00+05:30")
        result = build_upcoming_eta(
            {"route": route}, "AAA", "BBB", 10, 10, 0.5, 20,
            {"lookup_scope": "EXACT", "count": 200}, True,
            "RailRadar live coordinates", current_time=now,
        )
        self.assertEqual(
            result[0]["eta_method"],
            "live_remaining_travel_after_stale_schedule_model_eta",
        )
        self.assertEqual(result[0]["eta_minutes_from_now"], 10.0)
        self.assertGreaterEqual(
            datetime.fromisoformat(result[0]["predicted_arrival"]), now,
        )


if __name__ == "__main__":
    unittest.main()
