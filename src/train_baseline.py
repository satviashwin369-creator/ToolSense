
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "feature_dataset.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "baseline_predictions.csv"

def main():
    df = pd.read_csv(DATA_PATH)

    # Missing VB means an unavailable measurement, not zero wear.
    df = df.dropna(subset=["VB"]).copy()

    # Use sensor features and machining conditions.
    # Keep case/run/time out of the initial feature set.
    sensor_features = [
        col for col in df.columns
        if any(
            col.startswith(sensor + "_")
            for sensor in [
                "smcAC", "smcDC", "vib_table",
                "vib_spindle", "AE_table", "AE_spindle"
            ]
        )
    ]

    condition_features = ["DOC", "feed", "material"]
    feature_columns = sensor_features + condition_features

    X = df[feature_columns]
    y = df["VB"]
    groups = df["case"]

    # Every case stays entirely within one fold.
    cv = GroupKFold(n_splits=5)

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    predictions = cross_val_predict(
        model,
        X,
        y,
        groups=groups,
        cv=cv
    )

    mae = mean_absolute_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r2 = r2_score(y, predictions)

    print("=" * 55)
    print("TOOLSENSE - RANDOM FOREST BASELINE")
    print("=" * 55)
    print("Measured-label records:", len(df))
    print("Sensor features:", len(sensor_features))
    print("Condition features:", len(condition_features))
    print("Total model features:", len(feature_columns))
    print("Cross-validation: 5-fold GroupKFold by case")
    print("\nEvaluation results:")
    print(f"MAE:  {mae:.4f} mm")
    print(f"RMSE: {rmse:.4f} mm")
    print(f"R²:   {r2:.4f}")

    # Save out-of-fold predictions for later analysis and dashboard use.
    results = df[["case", "run", "VB"]].copy()
    results["predicted_VB"] = predictions
    results["absolute_error"] = np.abs(y.to_numpy() - predictions)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORT_PATH, index=False)

    print("\nPredictions saved to:", REPORT_PATH)
    print("\nSample predictions:")
    print(results.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
