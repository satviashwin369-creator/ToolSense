
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "reports" / "target_transform_predictions.csv"
OUTPUT_PATH = ROOT / "reports" / "transform_case_comparison.csv"

THRESHOLD = 0.50


def main():
    df = pd.read_csv(INPUT_PATH)
    df = df.dropna(subset=["VB"]).copy()

    required = {
        "case", "run", "VB",
        "predicted_original", "predicted_log1p",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["original_abs_error"] = (
        df["VB"] - df["predicted_original"]
    ).abs()
    df["log1p_abs_error"] = (
        df["VB"] - df["predicted_log1p"]
    ).abs()

    df["original_signed_error"] = (
        df["VB"] - df["predicted_original"]
    )
    df["log1p_signed_error"] = (
        df["VB"] - df["predicted_log1p"]
    )

    df["high_wear"] = df["VB"] > THRESHOLD

    print("=" * 76)
    print("TOOLSENSE - TARGET TRANSFORMATION CASE ANALYSIS")
    print("=" * 76)

    print("\nOVERALL METRICS")
    for name, column in [
        ("Original Extra Trees", "predicted_original"),
        ("Log1p Extra Trees", "predicted_log1p"),
    ]:
        y = df["VB"].to_numpy()
        pred = df[column].to_numpy()
        high = y > THRESHOLD

        print(
            f"{name}: "
            f"MAE={mean_absolute_error(y, pred):.5f} mm | "
            f"RMSE={np.sqrt(mean_squared_error(y, pred)):.5f} mm | "
            f"R2={r2_score(y, pred):.5f} | "
            f"High-wear MAE="
            f"{mean_absolute_error(y[high], pred[high]):.5f} mm | "
            f"High-wear signed error="
            f"{np.mean(y[high] - pred[high]):+.5f} mm"
        )

    case_rows = []
    for case_id, group in df.groupby("case"):
        high = group[group["high_wear"]]

        original_mae = group["original_abs_error"].mean()
        log_mae = group["log1p_abs_error"].mean()

        case_rows.append({
            "case": case_id,
            "Records": len(group),
            "High_wear_records": len(high),
            "Original_MAE_mm": original_mae,
            "Log1p_MAE_mm": log_mae,
            # Positive means log1p reduced error.
            "MAE_improvement_mm": original_mae - log_mae,
            "Original_high_wear_MAE_mm": (
                high["original_abs_error"].mean()
                if len(high) else np.nan
            ),
            "Log1p_high_wear_MAE_mm": (
                high["log1p_abs_error"].mean()
                if len(high) else np.nan
            ),
        })

    result = pd.DataFrame(case_rows).sort_values(
        "MAE_improvement_mm", ascending=False
    )
    result.to_csv(OUTPUT_PATH, index=False)

    print("\nCASE-LEVEL COMPARISON")
    print("Positive improvement means the log transform reduced case MAE.")
    print(result.round(5).to_string(index=False))

    print("\nCASES WHERE LOG TRANSFORM WORSENED MAE")
    worsened = result[result["MAE_improvement_mm"] < 0]
    if worsened.empty:
        print("None.")
    else:
        print(worsened.round(5).to_string(index=False))

    print("\nHIGH-WEAR CASE COMPARISON")
    high_cases = result[result["High_wear_records"] > 0]
    print(
        high_cases[
            [
                "case", "High_wear_records",
                "Original_high_wear_MAE_mm",
                "Log1p_high_wear_MAE_mm",
            ]
        ].round(5).to_string(index=False)
    )

    print(f"\nSaved report:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
