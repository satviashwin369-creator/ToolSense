
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "feature_dataset.csv"
RESULT_PATH = ROOT / "reports" / "shape_feature_comparison.csv"
PREDICTION_PATH = ROOT / "reports" / "shape_feature_predictions.csv"

SENSORS = [
    "smcAC",
    "smcDC",
    "vib_table",
    "vib_spindle",
    "AE_table",
    "AE_spindle",
]

BASE_STATS = [
    "mean",
    "std",
    "rms",
    "minimum",
    "maximum",
    "peak_to_peak",
]

HIGH_WEAR_THRESHOLD = 0.50
N_SPLITS = 5
RANDOM_STATE = 42


def make_model():
    return ExtraTreesRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def calculate_metrics(y, pred):
    high = y > HIGH_WEAR_THRESHOLD

    return {
        "MAE_mm": mean_absolute_error(y, pred),
        "RMSE_mm": np.sqrt(mean_squared_error(y, pred)),
        "R2": r2_score(y, pred),
        "High_wear_count": int(high.sum()),
        "High_wear_MAE_mm": (
            mean_absolute_error(y[high], pred[high])
            if high.any()
            else np.nan
        ),
        "High_wear_signed_error_mm": (
            np.mean(y[high] - pred[high])
            if high.any()
            else np.nan
        ),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["VB"]).copy().reset_index(drop=True)

    # The baseline must contain the same 38 features as the
    # earlier Extra Trees experiment.
    sensor_base = [
        f"{sensor}_{stat}"
        for sensor in SENSORS
        for stat in BASE_STATS
    ]

    baseline_features = sensor_base + ["DOC", "feed"]
    shape_features = [
        f"{sensor}_{stat}"
        for sensor in SENSORS
        for stat in ["skewness", "kurtosis"]
    ]

    required = set(
        baseline_features + ["material", "case", "run", "VB"]
    )
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Required columns missing from feature dataset: {sorted(missing)}"
        )

    if len(baseline_features) != 38:
        raise RuntimeError("Baseline feature count is not 38.")

    # Use the available stored sensor statistics to assess shape features.
    # Raw sensor arrays are not stored in feature_dataset.csv, so this
    # script computes skewness/kurtosis only if their source signals
    # are available in a separate step. Do not approximate them from
    # existing summary statistics.
    #
    # We must therefore load mill.mat and extract the raw signals here.
    from scipy.io import loadmat

    mat_path = ROOT / "data" / "raw" / "mill.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {mat_path}")

    records = loadmat(mat_path)["mill"].ravel()
    signal_rows = []

    for record in records:
        case_id = int(np.asarray(record["case"]).squeeze())
        run_id = int(np.asarray(record["run"]).squeeze())

        row = {"case": case_id, "run": run_id}

        for sensor in SENSORS:
            signal = np.asarray(record[sensor], dtype=float).ravel()

            if signal.size == 0 or not np.isfinite(signal).all():
                raise ValueError(
                    f"Invalid signal in case {case_id}, run {run_id}, "
                    f"sensor {sensor}"
                )

            # Bias correction is not needed for feature comparison;
            # Fisher kurtosis is used (normal distribution ~= 0).
            row[f"{sensor}_skewness"] = (
                float(skew(signal, bias=False))
                if signal.size >= 3 and np.std(signal) > 0
                else 0.0
            )
            row[f"{sensor}_kurtosis"] = (
                float(kurtosis(signal, fisher=True, bias=False))
                if signal.size >= 4 and np.std(signal) > 0
                else 0.0
            )

        signal_rows.append(row)

    shape_df = pd.DataFrame(signal_rows)

    # Merge using record identifiers; never rely on row order.
    df = df.merge(
        shape_df,
        on=["case", "run"],
        how="left",
        validate="one_to_one",
    )

    if df[shape_features].isna().any().any():
        raise ValueError("Missing shape features after merging raw signals.")

    feature_sets = {
        "Baseline_38": baseline_features,
        "Baseline_plus_shape_50": baseline_features + shape_features,
        "Shape_plus_all_conditions_51": (
            sensor_base + shape_features + ["DOC", "feed", "material"]
        ),
    }

    y = df["VB"].to_numpy(dtype=float)
    groups = df["case"].to_numpy()
    splitter = GroupKFold(n_splits=N_SPLITS)

    predictions = {
        name: np.full(len(df), np.nan)
        for name in feature_sets
    }
    fold_rows = []

    print("=" * 76)
    print("TOOLSENSE - SENSOR SHAPE FEATURE EXPERIMENT")
    print("=" * 76)
    print(f"Measured records: {len(df)}")
    print(f"Distinct cases: {df['case'].nunique()}")
    print(f"Shape features: {len(shape_features)}")
    print("Validation: 5-fold GroupKFold by case")

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(df, y, groups), start=1
    ):
        if set(groups[train_idx]).intersection(set(groups[test_idx])):
            raise RuntimeError("Case leakage detected.")

        for name, features in feature_sets.items():
            X_train = df.iloc[train_idx][features].to_numpy(dtype=float)
            X_test = df.iloc[test_idx][features].to_numpy(dtype=float)

            model = make_model()
            model.fit(X_train, y[train_idx])
            pred = model.predict(X_test)
            predictions[name][test_idx] = pred

            fold_rows.append({
                "Model": name,
                "Fold": fold,
                "Test_records": len(test_idx),
                **calculate_metrics(y[test_idx], pred),
            })

        print(f"Fold {fold}/{N_SPLITS} complete")

    results = []
    for name, pred in predictions.items():
        if np.isnan(pred).any():
            raise RuntimeError(f"Missing out-of-fold predictions for {name}")

        results.append({
            "Model": name,
            "Feature_count": len(feature_sets[name]),
            **calculate_metrics(y, pred),
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULT_PATH, index=False)
    pd.DataFrame(fold_rows).to_csv(
        ROOT / "reports" / "shape_feature_fold_results.csv",
        index=False,
    )

    pred_df = df[["case", "run", "VB"]].copy()
    for name, pred in predictions.items():
        column = name.lower()
        pred_df[f"predicted_{column}"] = pred
        pred_df[f"error_{column}"] = pred_df["VB"] - pred

    pred_df.to_csv(PREDICTION_PATH, index=False)

    print("\nOVERALL OUT-OF-FOLD RESULTS")
    print(results_df.round(5).to_string(index=False))

    print("\nFOLD-BY-FOLD MAE")
    fold_df = pd.DataFrame(fold_rows)
    print(
        fold_df.pivot(
            index="Fold", columns="Model", values="MAE_mm"
        ).round(5).to_string()
    )

    print("\nSaved:")
    print(RESULT_PATH)
    print(ROOT / "reports" / "shape_feature_fold_results.csv")
    print(PREDICTION_PATH)


if __name__ == "__main__":
    main()
