
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "feature_dataset.csv"

def main():
    df = pd.read_csv(DATA_PATH).dropna(subset=["VB"]).copy()

    sensor_names = [
        "smcAC", "smcDC", "vib_table",
        "vib_spindle", "AE_table", "AE_spindle"
    ]

    sensor_features = [
        col for col in df.columns
        if any(col.startswith(name + "_") for name in sensor_names)
    ]

    condition_features = ["DOC", "feed", "material"]
    feature_columns = sensor_features + condition_features

    X = df[feature_columns]
    y = df["VB"]
    groups = df["case"]

    cv = GroupKFold(n_splits=5)
    fold_scores = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups), 1
    ):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        result = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="neg_mean_absolute_error",
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )

        fold_scores.append(result.importances_mean)

        print(f"Completed importance analysis for fold {fold}")

    importance = pd.DataFrame(
        fold_scores, columns=feature_columns
    ).mean(axis=0)

    feature_results = (
        importance.rename("importance")
        .sort_values(ascending=False)
        .rename_axis("feature")
        .reset_index()
    )

    def get_sensor(feature):
        for name in sensor_names:
            if feature.startswith(name + "_"):
                return name
        return "machining_conditions"

    feature_results["sensor"] = feature_results["feature"].map(get_sensor)

    sensor_results = (
        feature_results.groupby("sensor")["importance"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\n" + "=" * 60)
    print("TOOLSENSE - PERMUTATION IMPORTANCE")
    print("=" * 60)
    print("\nSensor-level importance (summed feature scores):")
    print(sensor_results.round(5).to_string())

    print("\nTop 15 individual features:")
    print(feature_results.head(15).round(5).to_string(index=False))

    output_path = PROJECT_ROOT / "reports" / "sensor_importance.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_results.to_csv(output_path, index=False)

    print("\nSaved feature importance to:", output_path)

if __name__ == "__main__":
    main()
