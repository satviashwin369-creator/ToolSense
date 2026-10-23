
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "feature_dataset.csv"
REPORTS_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"

SENSORS = [
    "smcAC", "smcDC", "vib_table",
    "vib_spindle", "AE_table", "AE_spindle"
]

def get_sensor_features(df):
    return [
        col for col in df.columns
        if any(col.startswith(sensor + "_") for sensor in SENSORS)
    ]

def evaluate_model(X, y, groups, cv):
    predictions = np.full(len(y), np.nan)

    for train_idx, test_idx in cv.split(X, y, groups):
        model = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        predictions[test_idx] = model.predict(X.iloc[test_idx])

    metrics = {
        "MAE_mm": float(mean_absolute_error(y, predictions)),
        "RMSE_mm": float(np.sqrt(mean_squared_error(y, predictions))),
        "R2": float(r2_score(y, predictions)),
    }

    return predictions, metrics

def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["VB"]).reset_index(drop=True)

    sensor_features = get_sensor_features(df)

    feature_sets = {
        "sensors_plus_DOC_feed": sensor_features + ["DOC", "feed"],
        "sensors_plus_all_conditions": (
            sensor_features + ["DOC", "feed", "material"]
        ),
    }

    y = df["VB"]
    groups = df["case"]
    cv = GroupKFold(n_splits=5)

    all_metrics = {}
    prediction_output = df[["case", "run", "VB"]].copy()

    print("=" * 60)
    print("TOOLSENSE - FINALIST MODEL COMPARISON")
    print("=" * 60)

    for name, columns in feature_sets.items():
        predictions, metrics = evaluate_model(
            df[columns], y, groups, cv
        )

        prediction_output[f"predicted_{name}"] = predictions
        all_metrics[name] = {
            "feature_count": len(columns),
            **metrics
        }

        print(f"\n{name}")
        print("Features:", len(columns))
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    prediction_output.to_csv(
        REPORTS_DIR / "finalist_predictions.csv", index=False
    )

    with open(REPORTS_DIR / "model_metrics.json", "w") as file:
        json.dump(all_metrics, file, indent=4)

    # Use the 38-feature candidate as the provisional final model.
    final_features = feature_sets["sensors_plus_DOC_feed"]

    final_model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(df[final_features], y)

    joblib.dump(
        {
            "model": final_model,
            "features": final_features,
            "target": "VB",
            "target_unit": "mm",
            "model_type": "RandomForestRegressor",
        },
        MODEL_DIR / "toolsense_model.joblib"
    )

    print("\nSaved:")
    print(" - reports/finalist_predictions.csv")
    print(" - reports/model_metrics.json")
    print(" - models/toolsense_model.joblib")
    print("\nThe saved model is provisional; evaluation metrics")
    print("come from grouped cross-validation, not training fit.")

if __name__ == "__main__":
    main()
