
from pathlib import Path
from scipy.io import loadmat
import numpy as np

# Find the project directory automatically
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "mill.mat"

print("=" * 55)
print("TOOLSENSE - DATASET INSPECTION")
print("=" * 55)

# Load the original dataset
data = loadmat(DATA_PATH)
mill = data["mill"].ravel()

print(f"\nTotal records: {len(mill)}")
print(f"Fields: {mill.dtype.names}")

# Summarize metadata and wear measurements
wear_values = []

for record in mill:
    case = int(np.asarray(record["case"]).squeeze())
    run = int(np.asarray(record["run"]).squeeze())
    vb = float(np.asarray(record["VB"]).squeeze())

    wear_values.append(vb)

    print(
        f"Case: {case:2d} | "
        f"Run: {run:2d} | "
        f"VB: {vb:.4f} mm"
    )

wear_values = np.array(wear_values)
valid_wear = np.isfinite(wear_values) & (wear_values > 0)

print("\n--- Wear summary ---")
print("Records with positive VB:", valid_wear.sum())
print("Records with zero VB:", (wear_values == 0).sum())
print("Records with non-finite VB:", (~np.isfinite(wear_values)).sum())
print("Minimum positive VB:",
      wear_values[valid_wear].min() if valid_wear.any() else "None")
print("Maximum VB:",
      wear_values[valid_wear].max() if valid_wear.any() else "None")


import pandas as pd

# Create a table containing the record metadata
rows = []

for record in mill:
    rows.append({
        "case": int(np.asarray(record["case"]).squeeze()),
        "run": int(np.asarray(record["run"]).squeeze()),
        "VB": float(np.asarray(record["VB"]).squeeze()),
        "time": float(np.asarray(record["time"]).squeeze()),
        "DOC": float(np.asarray(record["DOC"]).squeeze()),
        "feed": float(np.asarray(record["feed"]).squeeze()),
        "material": int(np.asarray(record["material"]).squeeze()),
    })

df = pd.DataFrame(rows)

# Summarize labels by experimental case
summary = df.groupby("case").agg(
    total_records=("run", "count"),
    measured_VB=("VB", "count"),
    missing_VB=("VB", lambda s: s.isna().sum()),
    min_VB=("VB", "min"),
    max_VB=("VB", "max"),
)

print("\n--- Wear-label summary by case ---")
print(summary.to_string())

print("\n--- Overall label summary ---")
print("Total records:", len(df))
print("Missing VB:", df["VB"].isna().sum())
print("Measured VB including zeros:", df["VB"].notna().sum())
print("VB = 0:", (df["VB"] == 0).sum())


print("\n--- Sensor signal inspection ---")

sensor_fields = [
    "smcAC",
    "smcDC",
    "vib_table",
    "vib_spindle",
    "AE_table",
    "AE_spindle",
]

for field in sensor_fields:
    lengths = []
    non_finite_counts = []

    for record in mill:
        signal = np.asarray(record[field], dtype=float).ravel()
        lengths.append(len(signal))
        non_finite_counts.append(
            int((~np.isfinite(signal)).sum())
        )

    print(f"\n{field}")
    print("  Signal lengths:", sorted(set(lengths)))
    print("  Total samples:", sum(lengths))
    print("  Non-finite samples:", sum(non_finite_counts))

# Display one sensor signal from the first record
signal = np.asarray(mill[0]["smcAC"], dtype=float).ravel()

print("\n--- First record: smcAC ---")
print("Number of samples:", len(signal))
print("Minimum:", np.min(signal))
print("Maximum:", np.max(signal))
print("Mean:", np.mean(signal))
print("Standard deviation:", np.std(signal))
