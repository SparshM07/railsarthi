"""Serving helpers for the separately trained journey-level delay model."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
from fastapi import HTTPException, status


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
    for column in config["categorical_features"]:
        value = str(frame.at[0, column]).strip()
        allowed = config["categories"][column]
        if value not in allowed:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={"invalid_category": {"feature": column, "value": value, "allowed": allowed}},
            )
        frame[column] = pd.Categorical([value], categories=allowed)

    numeric_features = set(expected) - set(config["categorical_features"])
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
