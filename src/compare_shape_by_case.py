
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "reports" / "shape_feature_predictions.csv"
OUTPUT_PATH = ROOT / "reports" / "shape_case_analysis.csv"

THRESHOLD = 0.50

MODELS = {
    "Baseline_38": "predicted_baseline_38",
    "Shape_50": "predicted_baseline_plus_shape_50",
    "Shape_conditions_51": "predicted_shape_plus_all_conditions_51",
}


def main():
    df = pd.read_csv(INPUT_PATH).dropna(subset=["VB"]).copy()

    required = {"case", "run", "VB", *MODELS.values()}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["high_wear"] = df["VB"] > THRESHOLD

    print("=" * 76)
    print("TOOLSENSE - SHAPE FEATURE CASE ANALYSIS")
    print("=" * 76)

    print("\nOVERALL AND HIGH-WEAR METRICS")
    for name, column in MODELS.items():
        errors = df["VB"] - df[column]
        high = df["high_wear"]

        high_mae = (
            mean_absolute_error(df.loc[high, "VB"], df.loc[high, column])
            if high.any()
            else np.nan
        )
        high_bias = errors[high].mean() if high.any() else np.nan

        print(
            f"{name}: "
            f"overall MAE={errors.abs().mean():.5f} mm | "
            f"high-wear MAE={high_mae:.5f} mm | "
            f"high-wear signed error={high_bias:+.5f} mm"
        )

    rows = []

    for case_id, group in df.groupby("case"):
        high = group[group["high_wear"]]
        row = {
            "case": case_id,
            "Records": len(group),
            "High_wear_records": len(high),
        }

        for name, column in MODELS.items():
            errors = group["VB"] - group[column]
            row[f"{name}_MAE_mm"] = errors.abs().mean()

            row[f"{name}_High_wear_MAE_mm"] = (
                (high["VB"] - high[column]).abs().mean()
                if len(high)
                else np.nan
            )

        # Positive means that the 51-feature model reduced case MAE
        # relative to the 50-feature model.
        row["Material_condition_gain_mm"] = (
            row["Shape_50_MAE_mm"]
            - row["Shape_conditions_51_MAE_mm"]
        )
        rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_PATH, index=False)

    print("\nCASES MOST IMPROVED BY ADDING MATERIAL")
    print(
        result.nlargest(5, "Material_condition_gain_mm")[
            [
                "case", "Records", "High_wear_records",
                "Shape_50_MAE_mm",
                "Shape_conditions_51_MAE_mm",
                "Material_condition_gain_mm",
            ]
        ].round(5).to_string(index=False)
    )

    print("\nCASES MOST WORSENED BY ADDING MATERIAL")
    print(
        result.nsmallest(5, "Material_condition_gain_mm")[
            [
                "case", "Records", "High_wear_records",
                "Shape_50_MAE_mm",
                "Shape_conditions_51_MAE_mm",
                "Material_condition_gain_mm",
            ]
        ].round(5).to_string(index=False)
    )

    print("\nHIGH-WEAR CASE COMPARISON")
    high_cases = result[result["High_wear_records"] > 0]
    print(
        high_cases[
            [
                "case", "High_wear_records",
                "Baseline_38_High_wear_MAE_mm",
                "Shape_50_High_wear_MAE_mm",
                "Shape_conditions_51_High_wear_MAE_mm",
            ]
        ].round(5).to_string(index=False)
    )

    print(f"\nSaved detailed results to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
