
from pathlib import Path
from itertools import product

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
REPORT_PATH = ROOT / "reports" / "extra_trees_tuning.csv"
PREDICTIONS_PATH = ROOT / "reports" / "extra_trees_tuning_predictions.csv"

SENSORS = [
    "smcAC", "smcDC", "vib_table",
    "vib_spindle", "AE_table", "AE_spindle",
]

BASE_STATS = [
    "mean", "std", "rms",
    "minimum", "maximum", "peak_to_peak",
]

HIGH_WEAR_THRESHOLD = 0.50
N_SPLITS = 5
N_ESTIMATORS = 200
RANDOM_STATE = 42


def extract_shape_features():
    """Extract skewness and Fisher kurtosis from raw sensor signals."""
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
                signal_skew = 0.0
                signal_kurtosis = 0.0
            else:
                signal_skew = (
                    float(skew(signal, bias=False))
                    if signal.size >= 3 else 0.0
                )
                signal_kurtosis = (
                    float(kurtosis(signal, fisher=True, bias=False))
                    if signal.size >= 4 else 0.0
                )

            row[f"{sensor}_skewness"] = signal_skew
            row[f"{sensor}_kurtosis"] = signal_kurtosis

        rows.append(row)

    return pd.DataFrame(rows)


def metrics(y, pred):
    high = y > HIGH_WEAR_THRESHOLD

    return {
        "MAE_mm": mean_absolute_error(y, pred),
        "RMSE_mm": np.sqrt(mean_squared_error(y, pred)),
        "R2": r2_score(y, pred),
        "High_wear_count": int(high.sum()),
        "High_wear_MAE_mm": (
            mean_absolute_error(y[high], pred[high])
            if high.any() else np.nan
        ),
        "High_wear_signed_error_mm": (
            np.mean(y[high] - pred[high])
            if high.any() else np.nan
        ),
    }


def main():
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
        for stat in ["skewness", "kurtosis"]
    ]

    # Keep the same 50-feature set throughout this experiment.
    features = sensor_stats + shape_stats + ["DOC", "feed"]

    if len(features) != 50:
        raise RuntimeError(f"Expected 50 features, got {len(features)}")

    if df[features].isna().any().any():
        raise ValueError("Missing feature values found.")

    X = df[features].to_numpy(dtype=float)
    y = df["VB"].to_numpy(dtype=float)
    groups = df["case"].to_numpy()

    if pd.Series(groups).nunique() < N_SPLITS:
        raise ValueError("Not enough distinct cases for GroupKFold.")

    # A small, controlled hyperparameter search.
    configurations = list(product(
        [1, 2, 4],       # min_samples_leaf
        [0.7, 1.0],      # max_features
        [None, 12],      # max_depth
    ))

    splitter = GroupKFold(n_splits=N_SPLITS)
    summary_rows = []
    fold_rows = []
    prediction_rows = []

    print("=" * 76)
    print("TOOLSENSE - EXTRA TREES CONTROLLED TUNING")
    print("=" * 76)
    print(f"Measured records: {len(df)}")
    print(f"Distinct cases: {pd.Series(groups).nunique()}")
    print(f"Features: {len(features)}")
    print(f"Configurations: {len(configurations)}")
    print(f"Validation: {N_SPLITS}-fold GroupKFold by case")
    print(f"Trees per fit: {N_ESTIMATORS}")
    print("\nThis is an exploratory tuning comparison, not a final")
    print("independent estimate of the selected model's performance.")

    for config_no, (leaf, max_features, depth) in enumerate(
        configurations, start=1
    ):
        config_name = (
            f"leaf={leaf},max_features={max_features},depth={depth}"
        )
        oof_pred = np.full(len(df), np.nan)

        for fold, (train_idx, test_idx) in enumerate(
            splitter.split(X, y, groups), start=1
        ):
            train_cases = set(groups[train_idx])
            test_cases = set(groups[test_idx])

            if train_cases.intersection(test_cases):
                raise RuntimeError("Case leakage detected.")

            model = ExtraTreesRegressor(
                n_estimators=N_ESTIMATORS,
                min_samples_leaf=leaf,
                max_features=max_features,
                max_depth=depth,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )

            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            oof_pred[test_idx] = pred

            fold_rows.append({
                "Configuration": config_name,
                "Fold": fold,
                **metrics(y[test_idx], pred),
            })

        if np.isnan(oof_pred).any():
            raise RuntimeError("Some records have no out-of-fold prediction.")

        overall = metrics(y, oof_pred)
        fold_subset = [
            row["MAE_mm"]
            for row in fold_rows
            if row["Configuration"] == config_name
        ]

        summary_rows.append({
            "Configuration": config_name,
            "min_samples_leaf": leaf,
            "max_features": max_features,
            "max_depth": "None" if depth is None else depth,
            **overall,
            "Fold_MAE_mean": float(np.mean(fold_subset)),
            "Fold_MAE_std": float(np.std(fold_subset)),
        })

        for index, pred in enumerate(oof_pred):
            prediction_rows.append({
                "Configuration": config_name,
                "case": df.iloc[index]["case"],
                "run": df.iloc[index]["run"],
                "VB": y[index],
                "prediction": pred,
            })

        print(
            f"[{config_no:02d}/{len(configurations)}] {config_name} | "
            f"MAE={overall['MAE_mm']:.5f} | "
            f"High-wear MAE={overall['High_wear_MAE_mm']:.5f}"
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["MAE_mm", "High_wear_MAE_mm"]
    )
    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)

    summary.to_csv(REPORT_PATH, index=False)
    folds.to_csv(ROOT / "reports" / "extra_trees_tuning_folds.csv", index=False)
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    print("\n" + "=" * 76)
    print("TOP CONFIGURATIONS BY OVERALL MAE")
    print("=" * 76)
    print(
        summary.head(5)[
            [
                "Configuration", "MAE_mm", "RMSE_mm", "R2",
                "High_wear_MAE_mm", "Fold_MAE_mean", "Fold_MAE_std",
            ]
        ].round(5).to_string(index=False)
    )

    print("\nSaved reports:")
    print(REPORT_PATH)
    print(ROOT / "reports" / "extra_trees_tuning_folds.csv")
    print(PREDICTIONS_PATH)


if __name__ == "__main__":
    main()
