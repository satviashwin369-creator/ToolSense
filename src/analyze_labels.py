
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "feature_dataset.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 55)
print("TOOLSENSE - LABEL ANALYSIS")
print("=" * 55)

# Overall label information
print("\n1. OVERALL VB LABEL COUNTS")
print("Total records:", len(df))
print("Measured VB:", df["VB"].notna().sum())
print("Missing VB:", df["VB"].isna().sum())
print("VB equal to zero:", (df["VB"] == 0).sum())
print("VB greater than zero:", (df["VB"] > 0).sum())

# Statistics using only measured labels
measured = df.dropna(subset=["VB"])

print("\n2. MEASURED VB STATISTICS")
print(measured["VB"].describe().to_string())

# Show every run and whether its wear measurement exists
print("\n3. VB MEASUREMENTS BY CASE AND RUN")
print(
    df[["case", "run", "time", "VB"]]
    .sort_values(["case", "run"])
    .to_string(index=False)
)

# Count measured and missing labels per case
print("\n4. LABEL COVERAGE BY CASE")
summary = (
    df.groupby("case")
    .agg(
        total_records=("run", "size"),
        measured_VB=("VB", "count"),
        missing_VB=("VB", lambda x: x.isna().sum()),
        zero_VB=("VB", lambda x: (x == 0).sum()),
    )
)

print(summary.to_string())
