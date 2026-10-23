
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "feature_dataset.csv"

SENSOR_NAMES = [
    "smcAC", "smcDC", "vib_table",
    "vib_spindle", "AE_table", "AE_spindle"
]

def main():
    df = pd.read_csv(DATA_PATH).dropna(subset=["VB"]).copy()

    sensor_features = [
        col for col in df.columns
        if any(col.startswith(name + "_") for name in SENSOR_NAMES)
    ]

    feature_sets = {
        "Sensors only": sensor_features,
        "Sensors + DOC": sensor_features + ["DOC"],
        "Sensors + DOC + feed": sensor_features + ["DOC", "feed"],
        "Sensors + all conditions": sensor_features + [
            "DOC", "feed", "material"
        ],
    }

    y = df["VB"]
    groups = df["case"]
    cv = GroupKFold(n_splits=5)

    print("=" * 60)
    print("TOOLSENSE - MACHINING CONDITION COMPARISON")
    print("=" * 60)

    for name, columns in feature_sets.items():
        model = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        predictions = cross_val_predict(
            model, df[columns], y, groups=groups, cv=cv
        )

        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)

        print(f"\n{name}")
        print("Feature count:", len(columns))
        print(f"MAE:  {mae:.4f} mm")
        print(f"RMSE: {rmse:.4f} mm")
        print(f"R2:   {r2:.4f}")

if __name__ == "__main__":
    main()
