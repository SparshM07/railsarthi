"""Model loading, category validation, and feature preparation for live delay inference."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
import lightgbm as lgb
import pandas as pd

logger = logging.getLogger("railway_delay_api.model_serving")


def normalize_category_values(values: list[Any]) -> list[str]:
    """Strip whitespace and drop empty strings from categorical vocabulary."""
    result = []
    for value in values:
        val_str = str(value).strip()
        if val_str:
            result.append(val_str)
    return result


def prepare_categorical_feature(
    dataframe: pd.DataFrame,
    feature: str,
    category_map: dict[str, list[str]],
    category_normalized_map: dict[str, dict[str, str]],
) -> None:
    """Safely map categorical feature column to pandas Categorical matching training vocabulary."""
    if feature not in dataframe.columns:
        return
    allowed_categories = category_map.get(feature, [])
    if not allowed_categories:
        logger.warning("No category list found for feature: %s", feature)
        return

    raw_values = dataframe[feature].astype(str).str.strip()
    normalized_categories = category_normalized_map.get(feature, {})
    values = raw_values.map(lambda val: normalized_categories.get(val.upper()))
    dataframe[feature] = pd.Categorical(values, categories=allowed_categories)

    unknown_mask = dataframe[feature].isna()
    if unknown_mask.any():
        unseen_values = raw_values[unknown_mask].tolist()
        logger.info(
            "Controlled model category fallback: unseen %s value(s): %s",
            feature,
            unseen_values,
        )


def prepare_model_dataframe(
    data: dict[str, Any],
    model_features: list[str],
    category_map: dict[str, list[str]],
    category_normalized_map: dict[str, dict[str, str]],
) -> pd.DataFrame:
    """Convert input feature mapping into a typed, ordered DataFrame for LightGBM."""
    dataframe = pd.DataFrame([data])

    for feature in ["train", "station", "next_station"]:
        prepare_categorical_feature(
            dataframe, feature, category_map, category_normalized_map
        )

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
        "previous_train_delay",
    ]

    for feature in numeric_features:
        if feature in dataframe.columns:
            dataframe[feature] = pd.to_numeric(dataframe[feature], errors="coerce")

    missing = [f for f in model_features if f not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")

    return dataframe[model_features]


class ChampionModelContainer:
    """Encapsulates the live LightGBM booster model, feature configuration, and category maps."""

    def __init__(self, model_dir: Path | str):
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / "champion_model_scheduled_segment_v2.txt"
        self.feature_config_path = self.model_dir / "model_features_scheduled_segment_v2.json"
        self.categories_path = self.model_dir / "station_categories_scheduled_segment_v2.json"
        self._check_files()
        self.model = lgb.Booster(model_file=str(self.model_path))
        self.model_features = self.model.feature_name()
        self.category_map: dict[str, list[str]] = {}
        self.category_normalized_map: dict[str, dict[str, str]] = {}
        self._load_categories()

    def _check_files(self) -> None:
        for p in [self.model_path, self.feature_config_path, self.categories_path]:
            if not p.exists():
                raise FileNotFoundError(f"Required model artifact not found: {p}")

    def _load_categories(self) -> None:
        with open(self.categories_path, "r", encoding="utf-8") as f:
            categories = json.load(f)
        for feature in ["train", "station", "next_station"]:
            raw = categories.get(feature, [])
            norm_list = normalize_category_values(raw)
            self.category_map[feature] = norm_list
            self.category_normalized_map[feature] = {
                str(v).strip().upper(): v for v in norm_list
            }
        logger.info(
            "Loaded model categories: train (%d), station (%d), next_station (%d)",
            len(self.category_map.get("train", [])),
            len(self.category_map.get("station", [])),
            len(self.category_map.get("next_station", [])),
        )

    def prepare_dataframe(self, data: dict[str, Any]) -> pd.DataFrame:
        return prepare_model_dataframe(
            data,
            self.model_features,
            self.category_map,
            self.category_normalized_map,
        )

    def predict(self, dataframe: pd.DataFrame) -> float:
        pred = self.model.predict(dataframe)[0]
        return max(0.0, float(pred))
