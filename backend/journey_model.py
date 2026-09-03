"""Serving helpers for the separately trained journey-level delay model."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
from fastapi import HTTPException, status


def build_journey_model_metadata(
    booster: Any,
    validation_data: dict[str, Any] | None = None,
    late_threshold_minutes: float = 15.0,
) -> dict[str, Any]:
    """Extract feature schema, category definitions, and thresholds from Booster and validation metadata."""
    features = booster.feature_name()
    validation_data = validation_data or {}
    categorical_features = validation_data.get("categorical_features", [])

    booster_cats = getattr(booster, "pandas_categorical", None) or []
    categories: dict[str, list[str]] = {}
    if booster_cats and len(booster_cats) == len(categorical_features):
        categories = dict(zip(categorical_features, booster_cats))

    return {
        "features": features,
        "categorical_features": categorical_features,
        "categories": categories,
        "late_threshold_minutes": late_threshold_minutes,
    }


def prepare_journey_model_dataframe(
    features: dict[str, Any], config: dict[str, Any]
) -> pd.DataFrame:
    """Validate and type a request exactly as the training model expects."""
    expected = config["features"]
    supplied = set(features)
    missing = [name for name in expected if name not in supplied]
    unexpected = sorted(supplied - set(expected))
    if missing or unexpected:
        detail: dict[str, list[str]] = {}
        if missing:
            detail["missing_features"] = missing
        if unexpected:
            detail["unexpected_features"] = unexpected
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=detail)

    frame = pd.DataFrame([[features[name] for name in expected]], columns=expected)
    categorical_features = config.get("categorical_features", [])
    categories = config.get("categories", {})
    for column in categorical_features:
        value = str(frame.at[0, column]).strip()
        if categories and column in categories:
            allowed = categories[column]
            if value not in allowed:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail={"invalid_category": {"feature": column, "value": value, "allowed": allowed}},
                )
            frame[column] = pd.Categorical([value], categories=allowed)
        else:
            frame[column] = pd.Categorical([value])

    numeric_features = set(expected) - set(categorical_features)
    for column in numeric_features:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={"invalid_numeric_feature": column},
            ) from None
        if not math.isfinite(float(frame.at[0, column])):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={"invalid_numeric_feature": column},
            )
    return frame
