
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = ROOT / "reports" / "algorithm_predictions.csv"
OUTPUT_PATH = ROOT / "reports" / "algorithm_case_stability.csv"

MODEL_COLUMNS = {
    "Random Forest": "predicted_random_forest",
    "Extra Trees": "predicted_extra_trees",
    "Gradient Boosting": "predicted_gradient_boosting",
    "Histogram Gradient Boosting": "predicted_histogram_gradient_boosting",
}

HIGH_WEAR_THRESHOLD = 0.50


def main():
    df = pd.read_csv(PREDICTIONS_PATH)

    required = {"case", "run", "VB", *MODEL_COLUMNS.values()}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["VB"]).copy()

    print("=" * 76)
    print("TOOLSENSE - MODEL STABILITY BY CASE")
    print("=" * 76)
    print(f"Measured records: {len(df)}")
    print(f"Distinct cases: {df['case'].nunique()}")
    print(f"High-wear threshold: VB > {HIGH_WEAR_THRESHOLD:.2f} mm")

    results = []

    for model_name, prediction_column in MODEL_COLUMNS.items():
        df["absolute_error"] = (
            df["VB"] - df[prediction_column]
        ).abs()

        df["squared_error"] = (
            df["VB"] - df[prediction_column]
        ) ** 2

        df["high_wear"] = df["VB"] > HIGH_WEAR_THRESHOLD

        overall_mae = mean_absolute_error(
            df["VB"], df[prediction_column]
        )
        overall_rmse = np.sqrt(
            mean_squared_error(df["VB"], df[prediction_column])
        )
        overall_r2 = r2_score(df["VB"], df[prediction_column])

        print(f"\n{'-' * 76}")
        print(model_name)
        print(
            f"Overall: MAE={overall_mae:.4f} mm | "
            f"RMSE={overall_rmse:.4f} mm | R2={overall_r2:.4f}"
        )

        case_results = []

        for case_id, group in df.groupby("case"):
            case_mae = mean_absolute_error(
                group["VB"], group[prediction_column]
            )

            case_results.append({
                "Model": model_name,
                "case": case_id,
                "Records": len(group),
                "MAE_mm": case_mae,
                "High_wear_records": int(
                    (group["VB"] > HIGH_WEAR_THRESHOLD).sum()
                ),
                "High_wear_MAE_mm": (
                    mean_absolute_error(
                        group.loc[
                            group["VB"] > HIGH_WEAR_THRESHOLD, "VB"
                        ],
                        group.loc[
                            group["VB"] > HIGH_WEAR_THRESHOLD,
                            prediction_column,
                        ],
                    )
                    if (group["VB"] > HIGH_WEAR_THRESHOLD).any()
                    else np.nan
                ),
            })

        case_df = pd.DataFrame(case_results)
        results.extend(case_results)

        print(
            f"Case-level MAE: mean={case_df['MAE_mm'].mean():.4f} mm | "
            f"median={case_df['MAE_mm'].median():.4f} mm | "
            f"worst={case_df['MAE_mm'].max():.4f} mm"
        )

        print("Cases with largest errors:")
        print(
            case_df.nlargest(5, "MAE_mm")[
                ["case", "Records", "MAE_mm", "High_wear_records"]
            ].to_string(index=False)
        )

        high_wear = df[df["high_wear"]]

        if not high_wear.empty:
            high_wear_mae = mean_absolute_error(
                high_wear["VB"], high_wear[prediction_column]
            )
            mean_actual = high_wear["VB"].mean()
            mean_predicted = high_wear[prediction_column].mean()

            print(
                f"High-wear MAE: {high_wear_mae:.4f} mm | "
                f"actual mean={mean_actual:.4f} mm | "
                f"predicted mean={mean_predicted:.4f} mm"
            )

    output = pd.DataFrame(results)
    output.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved case-level results to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
