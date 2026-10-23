
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "feature_dataset.csv"

SENSORS = [
    "smcAC", "smcDC", "vib_table",
    "vib_spindle", "AE_table", "AE_spindle"
]

def main():
    df = pd.read_csv(DATA_PATH).dropna(subset=["VB"]).copy()

    groups = df["case"]
    y = df["VB"]

    configurations = {
        "All six sensors": SENSORS,
        "smcDC only": ["smcDC"],
        "smcDC + AE_spindle": ["smcDC", "AE_spindle"],
        "smcDC + AE_spindle + vib_table": [
            "smcDC", "AE_spindle", "vib_table"
        ],
    }

    cv = GroupKFold(n_splits=5)

    print("=" * 60)
    print("TOOLSENSE - SENSOR GROUP COMPARISON")
    print("=" * 60)

    for config_name, selected_sensors in configurations.items():
        columns = [
            col for col in df.columns
            if any(col.startswith(sensor + "_")
                   for sensor in selected_sensors)
        ]

        X = df[columns]

        model = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        predictions = cross_val_predict(
            model, X, y, groups=groups, cv=cv
        )

        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)

        print(f"\n{config_name}")
        print("Sensors:", ", ".join(selected_sensors))
        print("Feature count:", len(columns))
        print(f"MAE:  {mae:.4f} mm")
        print(f"RMSE: {rmse:.4f} mm")
        print(f"R2:   {r2:.4f}")

if __name__ == "__main__":
    main()
