
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "reports" / "baseline_predictions.csv"

df = pd.read_csv(DATA_PATH)

df["error"] = df["predicted_VB"] - df["VB"]
df["absolute_error"] = df["error"].abs()

print("=" * 55)
print("TOOLSENSE - BASELINE ERROR ANALYSIS")
print("=" * 55)

print("\n1. OVERALL ERROR SUMMARY")
print(f"Mean actual VB: {df['VB'].mean():.4f} mm")
print(f"Mean predicted VB: {df['predicted_VB'].mean():.4f} mm")
print(f"Median absolute error: {df['absolute_error'].median():.4f} mm")
print(f"90th percentile absolute error: "
      f"{df['absolute_error'].quantile(0.90):.4f} mm")
print(f"Maximum absolute error: {df['absolute_error'].max():.4f} mm")

print("\n2. ERROR BY CASE")
by_case = df.groupby("case").agg(
    records=("VB", "size"),
    actual_mean=("VB", "mean"),
    predicted_mean=("predicted_VB", "mean"),
    MAE=("absolute_error", "mean")
)
print(by_case.round(4).to_string())

print("\n3. ERROR BY ACTUAL WEAR RANGE")
bins = [-np.inf, 0.15, 0.30, 0.50, np.inf]
labels = ["VB <= 0.15", "0.15 < VB <= 0.30",
          "0.30 < VB <= 0.50", "VB > 0.50"]

df["wear_range"] = pd.cut(
    df["VB"], bins=bins, labels=labels, include_lowest=True
)

by_range = df.groupby("wear_range", observed=False).agg(
    records=("VB", "size"),
    actual_mean=("VB", "mean"),
    predicted_mean=("predicted_VB", "mean"),
    MAE=("absolute_error", "mean")
)
print(by_range.round(4).to_string())

print("\n4. LARGEST PREDICTION ERRORS")
print(
    df.sort_values("absolute_error", ascending=False)
    [["case", "run", "VB", "predicted_VB", "absolute_error"]]
    .head(10)
    .round(4)
    .to_string(index=False)
)
