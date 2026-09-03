"""Regression checks for the dataset-backed journey model serving contract."""

import json
import unittest
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from fastapi import HTTPException

from backend.journey_model import (
    build_journey_model_metadata,
    prepare_journey_model_dataframe,
)


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "backend" / "model" / "journey_delay_model.txt"
VALIDATION_PATH = BASE_DIR / "backend" / "model" / "journey_delay_validation.json"
DATA_PATH = BASE_DIR / "backend" / "dataset" / "ir_train.csv"


class JourneyModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.booster = lgb.Booster(model_file=str(MODEL_PATH))
        cls.validation = (
            json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))
            if VALIDATION_PATH.exists()
            else {}
        )
        cls.config = build_journey_model_metadata(cls.booster, cls.validation)
        row = pd.read_csv(DATA_PATH, nrows=1).iloc[0]
        cls.features = {
            name: row[name].item() if hasattr(row[name], "item") else row[name]
            for name in cls.config["features"]
        }

    def test_features_prepare_in_trained_order_and_predict(self):
        frame = prepare_journey_model_dataframe(self.features, self.config)
        self.assertEqual(list(frame.columns), self.config["features"])
        prediction = self.booster.predict(frame)[0]
        self.assertGreaterEqual(prediction, 0)

    def test_missing_feature_is_rejected(self):
        features = dict(self.features)
        features.pop("distance_km")
        with self.assertRaises(HTTPException) as context:
            prepare_journey_model_dataframe(features, self.config)
        self.assertEqual(context.exception.status_code, 422)

    def test_non_finite_numeric_feature_is_rejected(self):
        features = dict(self.features)
        features["distance_km"] = float("inf")
        with self.assertRaises(HTTPException) as context:
            prepare_journey_model_dataframe(features, self.config)
        self.assertEqual(context.exception.status_code, 422)

    def test_invalid_categorical_feature_is_rejected(self):
        features = dict(self.features)
        features["train_type"] = "NonExistentHyperloop"
        with self.assertRaises(HTTPException) as context:
            prepare_journey_model_dataframe(features, self.config)
        self.assertEqual(context.exception.status_code, 422)
        self.assertIn("invalid_category", context.exception.detail)


if __name__ == "__main__":
    unittest.main()
