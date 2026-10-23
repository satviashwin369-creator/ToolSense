
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.io import loadmat

# Allow imports from the src folder
sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import extract_record_features


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "mill.mat"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "feature_dataset.csv"


def get_scalar(record, field):
    """Convert a MATLAB scalar field to a Python number."""
    return float(np.asarray(record[field]).squeeze())


def build_dataset():
    data = loadmat(DATA_PATH)
    records = data["mill"].ravel()

    rows = []

    for record in records:
        row = {
            "case": int(get_scalar(record, "case")),
            "run": int(get_scalar(record, "run")),
            "VB": get_scalar(record, "VB"),
            "time": get_scalar(record, "time"),
            "DOC": get_scalar(record, "DOC"),
            "feed": get_scalar(record, "feed"),
            "material": int(get_scalar(record, "material")),
        }

        row.update(extract_record_features(record))
        rows.append(row)

    df = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("=" * 55)
    print("TOOLSENSE - FEATURE DATASET BUILT")
    print("=" * 55)
    print("Dataset shape:", df.shape)
    print("Total records:", len(df))
    print("Features extracted per record:", 36)
    print("Missing VB labels:", df["VB"].isna().sum())
    print("Saved to:", OUTPUT_PATH)

    print("\nPreview:")
    print(df.head(5).to_string(index=False))

    print("\nColumn names:")
    print(df.columns.tolist())


if __name__ == "__main__":
    build_dataset()
