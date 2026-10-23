
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "feature_dataset.csv"
REPORTS_DIR = ROOT / "reports"

SENSORS = [
    "smcAC",
    "smcDC",
    "vib_table",
    "vib_spindle",
    "AE_table",
    "AE_spindle",
]

RANDOM_STATE = 42
N_SPLITS = 5
HIGH_WEAR_THRESHOLD = 0.5


def get_sensor_features(df):
    return [
        column
        for column in df.columns
        if any(column.startswith(sensor + "_") for sensor in SENSORS)
    ]


def build_models():
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            loss="huber",
            random_state=RANDOM_STATE,
        ),
        "Histogram Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.05,
            max_leaf_nodes=7,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
    }


def calculate_metrics(y_true, predictions, high_wear_mask):
    return {
        "MAE_mm": float(mean_absolute_error(y_true, predictions)),
        "RMSE_mm": float(
            np.sqrt(mean_squared_error(y_true, predictions))
        ),
        "R2": float(r2_score(y_true, predictions)),
        "High_wear_count": int(high_wear_mask.sum()),
        "High_wear_MAE_mm": float(
            mean_absolute_error(
                y_true[high_wear_mask],
                predictions[high_wear_mask],
            )
        ) if high_wear_mask.any() else None,
    }


def main():
    print("=" * 72)
    print("TOOLSENSE - FAIR ALGORITHM COMPARISON")
    print("=" * 72)

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["VB"]).reset_index(drop=True)

    sensor_features = get_sensor_features(df)
    feature_columns = sensor_features + ["DOC", "feed"]

    missing_columns = [
        column for column in feature_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required feature columns are missing: {missing_columns}"
        )

    if df[feature_columns].isna().any().any():
        raise ValueError(
            "Missing feature values found. Investigate them before training."
        )

    X = df[feature_columns].astype(float)
    y = df["VB"].astype(float)
    groups = df["case"]

    if groups.nunique() < N_SPLITS:
        raise ValueError(
            f"Need at least {N_SPLITS} distinct cases for GroupKFold."
        )

    high_wear_mask = y.to_numpy() > HIGH_WEAR_THRESHOLD
    cv = GroupKFold(n_splits=N_SPLITS)

    print(f"Measured-label records: {len(df)}")
    print(f"Distinct cases: {groups.nunique()}")
    print(f"Sensor features: {len(sensor_features)}")
    print(f"Total features: {len(feature_columns)}")
    print(f"High-wear threshold: VB > {HIGH_WEAR_THRESHOLD:.2f} mm")
    print(f"High-wear records: {int(high_wear_mask.sum())}")
    print(f"Validation: {N_SPLITS}-fold GroupKFold by case")
    print("\nAll models use the same features and fold assignments.")

    prediction_report = df[["case", "run", "VB"]].copy()
    results = []

    for model_name, model_template in build_models().items():
        print(f"\nEvaluating: {model_name}")

        predictions = np.full(len(df), np.nan)

        for fold, (train_idx, test_idx) in enumerate(
            cv.split(X, y, groups), start=1
        ):
            train_cases = set(groups.iloc[train_idx])
            test_cases = set(groups.iloc[test_idx])

            if train_cases.intersection(test_cases):
                raise RuntimeError(
                    "Data leakage detected: a case appears in both folds."
                )

            # Each fold receives a fresh, unfitted model.
            from sklearn.base import clone
            model = clone(model_template)

            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            predictions[test_idx] = model.predict(X.iloc[test_idx])

            print(
                f"  Fold {fold}/{N_SPLITS} complete "
                f"(held-out cases: {len(test_cases)})"
            )

        if not np.isfinite(predictions).all():
            raise RuntimeError(
                f"Incomplete out-of-fold predictions for {model_name}."
            )

        scores = calculate_metrics(
            y.to_numpy(),
            predictions,
            high_wear_mask,
        )

        prediction_column = (
            "predicted_" + model_name.lower().replace(" ", "_")
        )
        prediction_report[prediction_column] = predictions

        results.append({
            "Model": model_name,
            "Feature_count": len(feature_columns),
            **scores,
        })

        print(f"  MAE:            {scores['MAE_mm']:.4f} mm")
        print(f"  RMSE:           {scores['RMSE_mm']:.4f} mm")
        print(f"  R2:             {scores['R2']:.4f}")
        print(
            "  High-wear MAE:  "
            f"{scores['High_wear_MAE_mm']:.4f} mm"
        )

    results_df = pd.DataFrame(results).sort_values(
        by="MAE_mm",
        ascending=True,
    ).reset_index(drop=True)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results_path = REPORTS_DIR / "algorithm_comparison.csv"
    predictions_path = REPORTS_DIR / "algorithm_predictions.csv"

    results_df.to_csv(results_path, index=False)
    prediction_report.to_csv(predictions_path, index=False)

    print("\n" + "=" * 72)
    print("FINAL COMPARISON — LOWER MAE/RMSE IS BETTER")
    print("=" * 72)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nSaved reports:")
    print(f"  {results_path.relative_to(ROOT)}")
    print(f"  {predictions_path.relative_to(ROOT)}")

    print("\nIMPORTANT:")
    print("- These are grouped out-of-fold evaluation results.")
    print("- The current deployed model has not been changed.")
    print("- The lowest overall MAE alone does not guarantee the best model.")
    print("- Consider high-wear error and fold-to-fold stability before selection.")


if __name__ == "__main__":
    main()
