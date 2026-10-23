
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "reports" / "algorithm_predictions.csv"
OUTPUT_PATH = ROOT / "reports" / "prediction_error_analysis.csv"

PREDICTIONS = {
    "Random Forest": "predicted_random_forest",
    "Extra Trees": "predicted_extra_trees",
    "Gradient Boosting": "predicted_gradient_boosting",
    "Histogram Gradient Boosting": "predicted_histogram_gradient_boosting",
}

HIGH_WEAR_THRESHOLD = 0.50


def main():
    df = pd.read_csv(INPUT_PATH)
    df = df.dropna(subset=["VB"]).copy()

    # Calculate prediction errors: positive means underestimation.
    for name, column in PREDICTIONS.items():
        safe_name = name.lower().replace(" ", "_")
        df[f"{safe_name}_error"] = df["VB"] - df[column]
        df[f"{safe_name}_abs_error"] = df[f"{safe_name}_error"].abs()

    print("=" * 78)
    print("TOOLSENSE - INDIVIDUAL PREDICTION ERROR ANALYSIS")
    print("=" * 78)

    print("\n1. CASE 13 RECORDS")
    case13 = df[df["case"] == 13].copy()

    if case13.empty:
        print("No records found for Case 13.")
    else:
        display_columns = [
            "case", "run", "VB",
            PREDICTIONS["Extra Trees"],
            PREDICTIONS["Histogram Gradient Boosting"],
            "extra_trees_error",
            "histogram_gradient_boosting_error",
        ]
        print(
            case13[display_columns]
            .sort_values("VB", ascending=False)
            .round(4)
            .to_string(index=False)
        )

    print("\n2. HIGH-WEAR RECORDS (VB > 0.50 mm)")
    high = df[df["VB"] > HIGH_WEAR_THRESHOLD].copy()

    print(f"High-wear record count: {len(high)}")
    print(f"Actual mean wear: {high['VB'].mean():.4f} mm")

    for name, column in PREDICTIONS.items():
        safe_name = name.lower().replace(" ", "_")
        bias = (high["VB"] - high[column]).mean()
        mae = mean_absolute_error(high["VB"], high[column])

        print(
            f"{name}: MAE={mae:.4f} mm | "
            f"mean signed error={bias:+.4f} mm"
        )

    print("\n3. LARGEST EXTRA TREES ERRORS")
    largest = df.assign(
        absolute_error=(df["VB"] - df[PREDICTIONS["Extra Trees"]]).abs(),
        signed_error=df["VB"] - df[PREDICTIONS["Extra Trees"]],
    ).nlargest(10, "absolute_error")

    print(
        largest[
            ["case", "run", "VB", PREDICTIONS["Extra Trees"],
             "signed_error", "absolute_error"]
        ].round(4).to_string(index=False)
    )

    # Save a copy with signed and absolute errors for every model.
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved detailed analysis to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
