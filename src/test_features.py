
from pathlib import Path
from scipy.io import loadmat

from features import extract_record_features

project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "raw" / "mill.mat"

mill = loadmat(data_path)["mill"].ravel()

features = extract_record_features(mill[0])

print("Features extracted:", len(features))
print("\nFirst six features:")

for name, value in list(features.items())[:6]:
    print(f"{name}: {value:.6f}")
