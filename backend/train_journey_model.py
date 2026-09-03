"""Train and validate the dataset-backed end-to-end journey delay model.

The existing API model predicts a *next-station* delay from live RailRadar
data.  This script deliberately trains a separate model: the supplied data
contains journey-level route and operating characteristics and a final
destination delay.  It must not replace the live model.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "ir_train.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "journey_delay_model.txt"
METRICS_PATH = MODEL_DIR / "journey_delay_validation.json"

TARGET = "delay_minutes"
HOLDOUT_YEAR = 2024
EXCLUDED_COLUMNS = {
    "journey_id",          # identifier, not a stable predictive feature
    "departure_date",      # represented by date-derived feature columns
    "primary_delay_cause", # known only after the journey: target leakage
    "delay_minutes",
    "is_delayed",
}


def metric_summary(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Return regression plus business-relevant >15 minute late metrics."""
    actual_late = actual.gt(15)
    predicted_late = predicted > 15
    return {
        "mae_minutes": round(float(mean_absolute_error(actual, predicted)), 3),
        "rmse_minutes": round(float(mean_squared_error(actual, predicted) ** 0.5), 3),
        "late_arrival_accuracy": round(float(accuracy_score(actual_late, predicted_late)), 4),
        "late_arrival_precision": round(float(precision_score(actual_late, predicted_late, zero_division=0)), 4),
        "late_arrival_recall": round(float(recall_score(actual_late, predicted_late, zero_division=0)), 4),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    features = [column for column in data.columns if column not in EXCLUDED_COLUMNS]
    categorical_features = data[features].select_dtypes(include=["object", "string"]).columns.tolist()

    # Fix the category vocabulary before splitting so categorical codes are
    # consistent between the training, validation, and serving paths.
    categories: dict[str, list[str]] = {}
    for column in categorical_features:
        values = sorted(data[column].dropna().astype(str).unique().tolist())
        categories[column] = values
        data[column] = pd.Categorical(data[column].astype(str), categories=values)

    train = data.loc[data["year"] < HOLDOUT_YEAR].copy()
    validation = data.loc[data["year"] == HOLDOUT_YEAR].copy()
    if train.empty or validation.empty:
        raise ValueError("Time split produced an empty train or validation partition.")

    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        train[features],
        train[TARGET],
        categorical_feature=categorical_features,
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
        eval_set=[(validation[features], validation[TARGET])],
        eval_metric="l1",
    )

    prediction = pd.Series(model.predict(validation[features]), index=validation.index).clip(lower=0)
    baseline_prediction = pd.Series(float(train[TARGET].median()), index=validation.index)
    metrics = {
        "validation_strategy": "Train on 2018-2023; validate on unseen 2024 journeys.",
        "excluded_for_leakage": sorted(EXCLUDED_COLUMNS),
        "feature_count": len(features),
        "categorical_features": categorical_features,
        "training_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
        "model": metric_summary(validation[TARGET], prediction),
        "median_delay_baseline": metric_summary(validation[TARGET], baseline_prediction),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(MODEL_PATH), num_iteration=model.best_iteration_)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
