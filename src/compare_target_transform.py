
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "feature_dataset.csv"
COMPARISON_PATH = ROOT / "reports" / "target_transform_comparison.csv"
PREDICTIONS_PATH = ROOT / "reports" / "target_transform_predictions.csv"

N_SPLITS = 5
HIGH_WEAR_THRESHOLD = 0.50
RANDOM_STATE = 42


def make_model():
    return ExtraTreesRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def calculate_metrics(y_true, y_pred):
    high_mask = y_true > HIGH_WEAR_THRESHOLD

    return {
        "MAE_mm": mean_absolute_error(y_true, y_pred),
        "RMSE_mm": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "High_wear_count": int(high_mask.sum()),
        "High_wear_MAE_mm": (
            mean_absolute_error(y_true[high_mask], y_pred[high_mask])
            if high_mask.any()
            else np.nan
        ),
        # Positive signed error means underprediction.
        "High_wear_signed_error_mm": (
            np.mean(y_true[high_mask] - y_pred[high_mask])
            if high_mask.any()
            else np.nan
        ),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["VB"]).copy()

    # Use the same 38 features as the existing Extra Trees experiment:
    # 36 sensor features + DOC + feed.
    excluded = {"case", "run", "VB", "time", "material"}
    features = [column for column in df.columns if column not in excluded]

    if len(features) != 38:
        raise ValueError(
            f"Expected 38 features (36 sensor + DOC + feed), "
            f"but found {len(features)}: {features}"
        )

    if df[features].isna().any().any():
        bad_columns = df[features].columns[
            df[features].isna().any()
        ].tolist()
        raise ValueError(
            f"Missing feature values found in: {bad_columns}. "
            "Resolve them before evaluating."
        )

    X = df[features].to_numpy(dtype=float)
    y = df["VB"].to_numpy(dtype=float)
    groups = df["case"].to_numpy()

    if np.any(y < 0):
        raise ValueError(
            "Log1p target transformation requires non-negative wear values."
        )

    if pd.Series(groups).nunique() < N_SPLITS:
        raise ValueError("Not enough distinct cases for GroupKFold.")

    splitter = GroupKFold(n_splits=N_SPLITS)

    prediction_columns = {
        "Extra Trees - Original": "predicted_original",
        "Extra Trees - Log1p target": "predicted_log1p",
    }

    predictions = {
        name: np.full(len(df), np.nan)
        for name in prediction_columns
    }

    fold_rows = []

    print("=" * 76)
    print("TOOLSENSE - EXTRA TREES TARGET TRANSFORMATION EXPERIMENT")
    print("=" * 76)
    print(f"Measured records: {len(df)}")
    print(f"Distinct cases: {pd.Series(groups).nunique()}")
    print(f"Features: {len(features)}")
    print(f"Validation: {N_SPLITS}-fold GroupKFold by case")
    print(f"High-wear threshold: VB > {HIGH_WEAR_THRESHOLD:.2f} mm")
    print("\nNo existing model or report will be overwritten.")

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X, y, groups), start=1
    ):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        train_cases = set(groups[train_idx])
        test_cases = set(groups[test_idx])

        if train_cases.intersection(test_cases):
            raise RuntimeError("Case leakage detected between train and test.")

        # Experiment A: train on the original wear scale.
        original_model = make_model()
        original_model.fit(X_train, y_train)
        pred_original = original_model.predict(X_test)

        # Experiment B: transform training targets only.
        # Predictions are converted back to millimetres before scoring.
        transformed_model = make_model()
        transformed_model.fit(X_train, np.log1p(y_train))
        pred_log = np.expm1(transformed_model.predict(X_test))

        # Wear cannot be negative; clip only numerical/model outputs.
        pred_log = np.maximum(pred_log, 0.0)

        predictions["Extra Trees - Original"][test_idx] = pred_original
        predictions["Extra Trees - Log1p target"][test_idx] = pred_log

        for name, pred in [
            ("Extra Trees - Original", pred_original),
            ("Extra Trees - Log1p target", pred_log),
        ]:
            metrics = calculate_metrics(y_test, pred)
            fold_rows.append({
                "Model": name,
                "Fold": fold,
                "Train_cases": len(train_cases),
                "Test_cases": len(test_cases),
                "Test_records": len(test_idx),
                **metrics,
            })

        print(
            f"Fold {fold}/{N_SPLITS} complete | "
            f"held-out cases: {len(test_cases)}"
        )

    # Ensure every record received exactly one out-of-fold prediction.
    for name, pred in predictions.items():
        if np.isnan(pred).any():
            raise RuntimeError(f"Missing out-of-fold predictions for {name}.")

    comparison_rows = []

    for name, pred in predictions.items():
        metrics = calculate_metrics(y, pred)
        comparison_rows.append({
            "Model": name,
            "Feature_count": len(features),
            **metrics,
        })

    comparison = pd.DataFrame(comparison_rows)
    fold_results = pd.DataFrame(fold_rows)

    # Save overall comparison and fold-by-fold results together.
    comparison["Result_type"] = "Overall OOF"
    fold_results["Result_type"] = "Fold"
    fold_results["Feature_count"] = len(features)

    common_columns = [
        "Result_type", "Model", "Feature_count", "Fold",
        "Train_cases", "Test_cases", "Test_records",
        "MAE_mm", "RMSE_mm", "R2", "High_wear_count",
        "High_wear_MAE_mm", "High_wear_signed_error_mm",
    ]

    for column in common_columns:
        if column not in comparison:
            comparison[column] = np.nan
        if column not in fold_results:
            fold_results[column] = np.nan

    combined = pd.concat(
        [comparison[common_columns], fold_results[common_columns]],
        ignore_index=True,
    )

    prediction_output = df[["case", "run", "VB"]].copy()
    prediction_output["predicted_original"] = predictions[
        "Extra Trees - Original"
    ]
    prediction_output["predicted_log1p"] = predictions[
        "Extra Trees - Log1p target"
    ]
    prediction_output["original_signed_error"] = (
        prediction_output["VB"] - prediction_output["predicted_original"]
    )
    prediction_output["log1p_signed_error"] = (
        prediction_output["VB"] - prediction_output["predicted_log1p"]
    )

    comparison.to_csv(COMPARISON_PATH, index=False)
    combined.to_csv(ROOT / "reports" / "target_transform_fold_results.csv",
                    index=False)
    prediction_output.to_csv(PREDICTIONS_PATH, index=False)

    print("\n" + "=" * 76)
    print("OVERALL OUT-OF-FOLD COMPARISON")
    print("=" * 76)
    print(comparison.round(5).to_string(index=False))

    print("\nFOLD-BY-FOLD MAE")
    print(
        fold_results.pivot(
            index="Fold", columns="Model", values="MAE_mm"
        ).round(5).to_string()
    )

    print("\nSaved reports:")
    print(f"  {COMPARISON_PATH}")
    print(f"  {ROOT / 'reports' / 'target_transform_fold_results.csv'}")
    print(f"  {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
