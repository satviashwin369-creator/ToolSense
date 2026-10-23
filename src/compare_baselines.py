
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "feature_dataset.csv"

def evaluate(y_true, predictions):
    return {
        "MAE": mean_absolute_error(y_true, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_true, predictions)),
        "R2": r2_score(y_true, predictions),
    }

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

    feature_columns = sensor_features + ["DOC", "feed", "material"]

    X = df[feature_columns]
    y = df["VB"]
    groups = df["case"]

    cv = GroupKFold(n_splits=5)

    rf_predictions = np.full(len(df), np.nan)
    mean_predictions = np.full(len(df), np.nan)

    print("=" * 60)
    print("TOOLSENSE - BASELINE COMPARISON")
    print("=" * 60)

    print("\n1. HELD-OUT CASES AND WEAR RANGES")

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        train_cases = sorted(groups.iloc[train_idx].unique())
        test_cases = sorted(groups.iloc[test_idx].unique())

        rf = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        rf_predictions[test_idx] = rf.predict(X_test)

        dummy = DummyRegressor(strategy="mean")
        dummy.fit(X_train, y_train)
        mean_predictions[test_idx] = dummy.predict(X_test)

        print(f"\nFold {fold}")
        print("Test cases:", test_cases)
        print(
            f"Training VB range: {y_train.min():.2f} to "
            f"{y_train.max():.2f} mm"
        )
        print(
            f"Testing VB range:  {y_test.min():.2f} to "
            f"{y_test.max():.2f} mm"
        )

    print("\n2. OVERALL OUT-OF-FOLD COMPARISON")

    for name, predictions in [
        ("Mean baseline", mean_predictions),
        ("Random Forest", rf_predictions),
    ]:
        scores = evaluate(y, predictions)
        print(f"\n{name}")
        print(f"MAE:  {scores['MAE']:.4f} mm")
        print(f"RMSE: {scores['RMSE']:.4f} mm")
        print(f"R2:   {scores['R2']:.4f}")

    print("\n3. RANDOM FOREST IMPROVEMENT OVER MEAN BASELINE")
    dummy_mae = mean_absolute_error(y, mean_predictions)
    rf_mae = mean_absolute_error(y, rf_predictions)

    improvement = (dummy_mae - rf_mae) / dummy_mae * 100
    print(f"Mean baseline MAE: {dummy_mae:.4f} mm")
    print(f"Random Forest MAE: {rf_mae:.4f} mm")
    print(f"MAE improvement:   {improvement:.2f}%")

if __name__ == "__main__":
    main()
