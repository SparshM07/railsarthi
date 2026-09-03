"""Focused regression tests for adjacent route segment resolution and stop ordering."""

from datetime import datetime
import unittest
from fastapi.testclient import TestClient

from backend.geo import resolve_active_route_segment, find_route_index, get_stop_code
from backend.eta import build_upcoming_eta
from backend.main import app, SESSION_COOKIE_NAME, champion_container, segment_stats_index


class AdjacentSegmentResolutionTests(unittest.TestCase):
    """Test the invariant that next_station is always the immediate scheduled stop after current_station."""

    def setUp(self):
        self.route_5_stops = [
            {"code": "A", "name": "Station A", "distance": 0.0, "scheduledArrival": "2026-09-04T10:00:00+05:30", "scheduledDeparture": "2026-09-04T10:05:00+05:30"},
            {"code": "B", "name": "Station B", "distance": 20.0, "scheduledArrival": "2026-09-04T10:25:00+05:30", "scheduledDeparture": "2026-09-04T10:30:00+05:30"},
            {"code": "C", "name": "Station C", "distance": 45.0, "scheduledArrival": "2026-09-04T10:55:00+05:30", "scheduledDeparture": "2026-09-04T11:00:00+05:30"},
            {"code": "D", "name": "Station D", "distance": 70.0, "scheduledArrival": "2026-09-04T11:30:00+05:30", "scheduledDeparture": "2026-09-04T11:35:00+05:30"},
            {"code": "E", "name": "Station E", "distance": 100.0, "scheduledArrival": "2026-09-04T12:10:00+05:30", "scheduledDeparture": "2026-09-04T12:10:00+05:30"},
        ]

    def test_five_stop_route_sequential_adjacency(self):
        """Verify for A -> B -> C -> D -> E:
        current=A -> next=B
        current=B -> next=C
        current=C -> next=D
        current=D -> next=E
        current=E -> next=None
        """
        expected_pairs = [
            ("A", "B", "Station B", 0),
            ("B", "C", "Station C", 1),
            ("C", "D", "Station D", 2),
            ("D", "E", "Station E", 3),
            ("E", None, None, 4),
        ]

        for cur_code, expected_next, expected_next_name, expected_idx in expected_pairs:
            idx, cur, cur_name, nxt, nxt_name = resolve_active_route_segment(
                self.route_5_stops,
                cur_code,
            )
            self.assertEqual(idx, expected_idx, f"Index mismatch for current={cur_code}")
            self.assertEqual(cur, cur_code, f"Current station mismatch")
            self.assertEqual(nxt, expected_next, f"Next station mismatch for current={cur_code}")
            self.assertEqual(nxt_name, expected_next_name, f"Next station name mismatch for current={cur_code}")

    def test_distant_station_is_never_selected_when_intermediate_stops_exist(self):
        """Verify that even if live telemetry has a distant nextHalt (e.g. E when at A),
        the active segment is strictly A -> B, never A -> E.
        """
        idx, cur, cur_name, nxt, nxt_name = resolve_active_route_segment(
            self.route_5_stops,
            "A",
        )
        self.assertEqual(cur, "A")
        self.assertEqual(nxt, "B")
        self.assertNotEqual(nxt, "E", "Distant station E must NEVER be selected when intermediate stop B exists")

    def test_advances_to_next_segment_when_station_passed(self):
        """If telemetry route log confirms the train departed a downstream stop,
        current_index advances to the departed station and next becomes the adjacent station.
        """
        live_route = [
            {"stationCode": "A", "status": "departed"},
            {"stationCode": "B", "status": "departed"},
            {"stationCode": "C", "status": "upcoming"},
            {"stationCode": "D", "status": "upcoming"},
            {"stationCode": "E", "status": "upcoming"},
        ]
        # Telemetry reports A, but live_route confirms train departed B
        idx, cur, cur_name, nxt, nxt_name = resolve_active_route_segment(
            self.route_5_stops,
            "A",
            live_route=live_route,
        )
        self.assertEqual(idx, 1)
        self.assertEqual(cur, "B")
        self.assertEqual(nxt, "C")

    def test_terminal_station_resolution(self):
        """At the terminal station, next_station must be None."""
        idx, cur, cur_name, nxt, nxt_name = resolve_active_route_segment(
            self.route_5_stops,
            "E",
        )
        self.assertEqual(idx, 4)
        self.assertEqual(cur, "E")
        self.assertIsNone(nxt)
        self.assertIsNone(nxt_name)

    def test_build_upcoming_eta_preserves_all_intermediate_stops(self):
        """Verify build_upcoming_eta starts at immediate next station (Hop 1)
        and cascades across all downstream stops in strict order without skipping.
        """
        now = datetime.fromisoformat("2026-09-04T10:00:00+05:30")
        etas = build_upcoming_eta(
            live_data={"route": self.route_5_stops},
            current_station="A",
            next_station="B",
            current_delay=5.0,
            predicted_delay=7.0,
            segment_progress=0.25,
            scheduled_segment_minutes=20.0,
            historical_stats={"mean": 5.0, "median": 4.0, "std": 1.0, "count": 50},
            segment_progress_reliable=True,
            current_time=now,
            model_container=champion_container,
            segment_stats_index=segment_stats_index,
        )

        station_codes = [e["station_code"] for e in etas]
        self.assertEqual(station_codes, ["B", "C", "D", "E"], "Upcoming ETA must not skip any route stops")
        self.assertEqual(etas[0]["cascade_hop"], 1)
        self.assertEqual(etas[1]["cascade_hop"], 2)
        self.assertEqual(etas[2]["cascade_hop"], 3)
        self.assertEqual(etas[3]["cascade_hop"], 4)

    def test_build_upcoming_eta_guards_against_distant_next_station(self):
        """Even if caller accidentally passes a distant next_station='E' when at 'A',
        build_upcoming_eta must enforce the adjacent route stop 'B' as Hop 1.
        """
        now = datetime.fromisoformat("2026-09-04T10:00:00+05:30")
        etas = build_upcoming_eta(
            live_data={"route": self.route_5_stops},
            current_station="A",
            next_station="E",  # Distant station
            current_delay=5.0,
            predicted_delay=7.0,
            segment_progress=0.25,
            scheduled_segment_minutes=20.0,
            historical_stats={"mean": 5.0, "median": 4.0, "std": 1.0, "count": 50},
            segment_progress_reliable=True,
            current_time=now,
            model_container=champion_container,
            segment_stats_index=segment_stats_index,
        )

        station_codes = [e["station_code"] for e in etas]
        self.assertEqual(station_codes, ["B", "C", "D", "E"], "Must enforce adjacent stop B as Hop 1")


class ApiPredictionAdjacencyTests(unittest.TestCase):
    """Test /predict API for real and simulated trains enforcing adjacent segment invariant."""

    def setUp(self):
        self.client = TestClient(app)
        res = self.client.get("/", headers={"accept": "text/html"})
        self.cookie = res.cookies.get(SESSION_COOKIE_NAME)

    def _verify_adjacent_segment_invariant(self, train_number: int):
        res = self.client.post(
            "/predict",
            json={"train": train_number},
            cookies={SESSION_COOKIE_NAME: self.cookie},
        )
        self.assertEqual(res.status_code, 200, f"Train {train_number} prediction failed: {res.text}")
        data = res.json()

        cur = data.get("current_station")
        nxt = data.get("next_station")
        progress = data.get("segment_progress")

        # Progress must be strictly in [0.0, 1.0]
        self.assertIsNotNone(progress)
        self.assertGreaterEqual(progress, 0.0)
        self.assertLessEqual(progress, 1.0)

        upcoming = data.get("upcoming_stations", [])

        if nxt is not None:
            # First upcoming station must be next_station
            self.assertGreater(len(upcoming), 0)
            self.assertEqual(upcoming[0]["station_code"], nxt, "Hop 1 must match next_station")

        return data

    def test_train_12429_adjacent_segment_resolution(self):
        """Live or fallback test for train 12429 ensuring next_station is immediate next route stop,
        NOT a distant commercial halt like SPN.
        """
        data = self._verify_adjacent_segment_invariant(12429)
        cur = data["current_station"]
        nxt = data["next_station"]
        if cur in {"UTA", "SAN"}:
            self.assertNotEqual(nxt, "SPN", f"Train 12429 at {cur} must not skip intermediate stops to SPN")
        if cur == "DLQ":
            self.assertEqual(nxt, "BLM", "Immediate stop after DLQ must be BLM")

    def test_train_12919_adjacent_segment_resolution(self):
        self._verify_adjacent_segment_invariant(12919)

    def test_train_12229_adjacent_segment_resolution(self):
        self._verify_adjacent_segment_invariant(12229)

    def test_train_12230_adjacent_segment_resolution(self):
        self._verify_adjacent_segment_invariant(12230)

    def test_train_12002_adjacent_segment_resolution(self):
        self._verify_adjacent_segment_invariant(12002)

    def test_train_22416_adjacent_segment_resolution(self):
        self._verify_adjacent_segment_invariant(22416)


if __name__ == "__main__":
    unittest.main()
