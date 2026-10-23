
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import skew, kurtosis
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "feature_dataset.csv"
MAT_PATH = ROOT / "data" / "raw" / "mill.mat"
MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

CANDIDATE_PATH = MODEL_DIR / "toolsense_model_candidate.joblib"
METRICS_PATH = REPORTS_DIR / "candidate_model_metrics.json"
PREDICTIONS_PATH = REPORTS_DIR / "candidate_model_predictions.csv"

SENSORS = [
    "smcAC", "smcDC", "vib_table",
    "vib_spindle", "AE_table", "AE_spindle",
]
BASE_STATS = [
    "mean", "std", "rms",
    "minimum", "maximum", "peak_to_peak",
]
SHAPE_STATS = ["skewness", "kurtosis"]

N_SPLITS = 5
HIGH_WEAR_THRESHOLD = 0.50


def extract_shape_features():
    """Calculate shape features from the original sensor signals."""
    records = loadmat(MAT_PATH)["mill"].ravel()
    rows = []

    for record in records:
        case_id = int(np.asarray(record["case"]).squeeze())
        run_id = int(np.asarray(record["run"]).squeeze())
        row = {"case": case_id, "run": run_id}

        for sensor in SENSORS:
            signal = np.asarray(record[sensor], dtype=float).ravel()

            if signal.size == 0 or not np.isfinite(signal).all():
                raise ValueError(
                    f"Invalid {sensor} signal in case {case_id}, run {run_id}"
                )

            if np.std(signal) == 0:
                skewness_value = 0.0
                kurtosis_value = 0.0
            else:
                skewness_value = (
                    float(skew(signal, bias=False))
                    if signal.size >= 3 else 0.0
                )
                kurtosis_value = (
                    float(kurtosis(signal, fisher=True, bias=False))
                    if signal.size >= 4 else 0.0
                )

            row[f"{sensor}_skewness"] = skewness_value
            row[f"{sensor}_kurtosis"] = kurtosis_value

        rows.append(row)

    return pd.DataFrame(rows)


def calculate_metrics(y, predictions):
    high = y > HIGH_WEAR_THRESHOLD

    return {
        "MAE_mm": float(mean_absolute_error(y, predictions)),
        "RMSE_mm": float(np.sqrt(mean_squared_error(y, predictions))),
        "R2": float(r2_score(y, predictions)),
        "high_wear_threshold_mm": HIGH_WEAR_THRESHOLD,
        "high_wear_count": int(high.sum()),
        "high_wear_MAE_mm": (
            float(mean_absolute_error(y[high], predictions[high]))
            if high.any() else None
        ),
        "high_wear_signed_error_mm": (
            float(np.mean(y[high] - predictions[high]))
            if high.any() else None
        ),
    }


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["VB"]).copy()

    shape_df = extract_shape_features()

    df = df.merge(
        shape_df,
        on=["case", "run"],
        how="left",
        validate="one_to_one",
    )

    sensor_stats = [
        f"{sensor}_{stat}"
        for sensor in SENSORS
        for stat in BASE_STATS
    ]
    shape_stats = [
        f"{sensor}_{stat}"
        for sensor in SENSORS
        for stat in SHAPE_STATS
    ]

    # Exact 50-feature configuration evaluated during tuning.
    features = sensor_stats + shape_stats + ["DOC", "feed"]

    if len(features) != 50:
        raise RuntimeError(f"Expected 50 features, got {len(features)}")

    if df[features].isna().any().any():
        raise ValueError("Missing values found in candidate model features.")

    X = df[features].to_numpy(dtype=float)
    y = df["VB"].to_numpy(dtype=float)
    groups = df["case"].to_numpy()

    if pd.Series(groups).nunique() < N_SPLITS:
        raise ValueError("Not enough distinct cases for grouped validation.")

    # Selected candidate from the controlled tuning experiment.
    model_settings = {
        "n_estimators": 200,
        "min_samples_leaf": 1,
        "max_features": 1.0,
        "max_depth": 12,
        "random_state": 42,
        "n_jobs": -1,
    }

    cv = GroupKFold(n_splits=N_SPLITS)
    oof_predictions = np.full(len(df), np.nan)
    fold_results = []

    print("=" * 72)
    print("TOOLSENSE - CANDIDATE MODEL TRAINING")
    print("=" * 72)
    print(f"Measured records: {len(df)}")
    print(f"Distinct cases: {pd.Series(groups).nunique()}")
    print(f"Features: {len(features)}")
    print(f"Model: ExtraTreesRegressor")
    print(f"Settings: {model_settings}")
    print("Validation: 5-fold GroupKFold by case")

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups), start=1
    ):
        if set(groups[train_idx]).intersection(set(groups[test_idx])):
            raise RuntimeError("Case leakage detected.")

        fold_model = ExtraTreesRegressor(**model_settings)
        fold_model.fit(X[train_idx], y[train_idx])
        predictions = fold_model.predict(X[test_idx])

        oof_predictions[test_idx] = predictions

        fold_metrics = calculate_metrics(y[test_idx], predictions)
        fold_results.append({
            "fold": fold,
            "test_records": len(test_idx),
            **fold_metrics,
        })

        print(
            f"Fold {fold}/{N_SPLITS}: "
            f"MAE={fold_metrics['MAE_mm']:.5f} mm | "
            f"RMSE={fold_metrics['RMSE_mm']:.5f} mm"
        )

    if np.isnan(oof_predictions).any():
        raise RuntimeError("Some records lack out-of-fold predictions.")

    overall_metrics = calculate_metrics(y, oof_predictions)

    # Save out-of-fold results for honest evaluation.
    prediction_output = df[["case", "run", "VB"]].copy()
    prediction_output["predicted_VB"] = oof_predictions
    prediction_output["signed_error_mm"] = y - oof_predictions
    prediction_output["absolute_error_mm"] = np.abs(y - oof_predictions)
    prediction_output.to_csv(PREDICTIONS_PATH, index=False)

    metrics_output = {
        "model_name": "Extra Trees",
        "model_type": "ExtraTreesRegressor",
        "feature_count": len(features),
        "features": features,
        "validation": "5-fold GroupKFold by case",
        "evaluation_type": "out-of-fold cross-validation",
        "settings": model_settings,
        "overall": overall_metrics,
        "fold_results": fold_results,
        "warning": (
            "Research prototype. Metrics are exploratory cross-validation "
            "estimates on a small dataset, not an independent test result."
        ),
    }

    import json

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics_output, file, indent=2)

    # Fit the candidate on all measured records only after CV evaluation.
    final_model = ExtraTreesRegressor(**model_settings)
    final_model.fit(X, y)

    artifact = {
        "model": final_model,
        "features": features,
        "target": "VB",
        "target_unit": "mm",
        "model_type": "ExtraTreesRegressor",
        "model_settings": model_settings,
        "feature_count": len(features),
        "feature_definition": (
            "36 sensor statistics + 12 signal shape statistics + DOC + feed"
        ),
        "validation": "5-fold GroupKFold by case",
        "cross_validation_metrics": overall_metrics,
    }

    # Save to a separate candidate filename; preserve the current model.
    joblib.dump(artifact, CANDIDATE_PATH)

    print("\nOVERALL OUT-OF-FOLD METRICS")
    for key, value in overall_metrics.items():
        print(f"{key}: {value}")

    print("\nSaved candidate only:")
    print(CANDIDATE_PATH)
    print(METRICS_PATH)
    print(PREDICTIONS_PATH)
    print("\nThe existing toolsense_model.joblib was not changed.")


if __name__ == "__main__":
    main()
