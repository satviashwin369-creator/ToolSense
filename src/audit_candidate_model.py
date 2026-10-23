
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "toolsense_model_candidate.joblib"
METRICS_PATH = ROOT / "reports" / "candidate_model_metrics.json"
PREDICTIONS_PATH = ROOT / "reports" / "candidate_model_predictions.csv"
IMPORTANCE_PATH = ROOT / "reports" / "candidate_feature_importance.csv"


def main():
    print("=" * 65)
    print("TOOLSENSE CANDIDATE MODEL AUDIT")
    print("=" * 65)

    # 1. Check model artifact
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    model_features = artifact["features"]

    print("\n1. MODEL ARTIFACT")
    print(f"Model type: {artifact['model_type']}")
    print(f"Declared feature count: {artifact['feature_count']}")
    print(f"Actual model input count: {model.n_features_in_}")
    print(f"Feature list length: {len(model_features)}")

    assert len(model_features) == 50
    assert model.n_features_in_ == 50
    assert len(set(model_features)) == 50
    print("PASS: Model has 50 unique features.")

    # 2. Check validation metrics
    with METRICS_PATH.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    overall = metrics["overall"]

    print("\n2. VALIDATION METRICS")
    for key in (
        "MAE_mm",
        "RMSE_mm",
        "R2",
        "high_wear_MAE_mm",
        "high_wear_signed_error_mm",
    ):
        print(f"{key}: {overall[key]}")

    assert metrics["feature_count"] == 50
    assert len(metrics["features"]) == 50
    assert metrics["model_type"] == artifact["model_type"]
    print("PASS: Metrics identify the same model and feature count.")

    # 3. Check prediction records and error calculations
    predictions = pd.read_csv(PREDICTIONS_PATH)

    required_columns = {
        "case",
        "run",
        "VB",
        "predicted_VB",
        "signed_error_mm",
        "absolute_error_mm",
    }

    print("\n3. PREDICTION REPORT")
    print(f"Prediction rows: {len(predictions)}")
    print(f"Columns: {list(predictions.columns)}")

    assert required_columns.issubset(predictions.columns)
    assert len(predictions) == 146
    assert predictions[list(required_columns)].notna().all().all()

    expected_signed_error = (
        predictions["VB"] - predictions["predicted_VB"]
    )
    expected_absolute_error = expected_signed_error.abs()

    assert np.allclose(
        predictions["signed_error_mm"], expected_signed_error
    )
    assert np.allclose(
        predictions["absolute_error_mm"], expected_absolute_error
    )

    print("PASS: 146 records and error calculations are consistent.")
    print("Signed error convention: actual VB - predicted VB.")

    # 4. Check feature importance report
    importance = pd.read_csv(IMPORTANCE_PATH)

    print("\n4. FEATURE IMPORTANCE")
    print(f"Importance rows: {len(importance)}")
    print(f"Columns: {list(importance.columns)}")

    assert len(importance) == 50

    feature_column = next(
        (
            col for col in importance.columns
            if col.lower() in {"feature", "feature_name"}
        ),
        None,
    )
    value_column = next(
        (
            col for col in importance.columns
            if "importance" in col.lower()
        ),
        None,
    )

    assert feature_column is not None, (
        "Could not identify the feature-name column."
    )
    assert value_column is not None, (
        "Could not identify the importance-value column."
    )

    assert set(importance[feature_column]) == set(model_features)

    values = importance[value_column].to_numpy(dtype=float)
    assert np.isfinite(values).all()
    assert (values >= 0).all()
    assert np.isclose(values.sum(), 1.0, atol=1e-5)

    print("PASS: Importance report matches all 50 model features.")
    print(f"Importance total: {values.sum():.6f}")

    print("\n" + "=" * 65)
    print("AUDIT COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
