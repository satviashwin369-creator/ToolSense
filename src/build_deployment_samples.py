
"""
Build deployment_samples.json from the NASA milling dataset.

Extracts the same 50 features that the Flask API's extract_features()
uses, so the generated file is drop-in compatible with /api/sample and
/api/predict.  Records with a missing or NaN VB label are kept (with
measured_VB_mm set to null) because their sensor features may still be
valid prediction inputs.

Usage:
    python src/build_deployment_samples.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.stats import kurtosis, skew

ROOT = Path(__file__).resolve().parent.parent
MAT_PATH = ROOT / "data" / "raw" / "mill.mat"
OUTPUT_PATH = ROOT / "data" / "processed" / "deployment_samples.json"

SENSORS = [
    "smcAC",
    "smcDC",
    "vib_table",
    "vib_spindle",
    "AE_table",
    "AE_spindle",
]

BASE_STATISTICS = [
    "mean", "std", "rms", "minimum", "maximum", "peak_to_peak",
]
SHAPE_STATISTICS = ["skewness", "kurtosis"]

EXPECTED_FEATURE_COUNT = 50


def scalar_from_record(record, field):
    """Read a scalar MATLAB field as a Python float."""
    value = np.asarray(record[field]).squeeze()
    if value.size == 0:
        raise ValueError(f"Empty field '{field}'")
    return float(value.reshape(-1)[0])


def signal_from_record(record, field):
    """Read a sensor signal as a 1-D float64 NumPy array."""
    signal = np.asarray(record[field], dtype=np.float64).squeeze().reshape(-1)
    if signal.size == 0 or not np.isfinite(signal).all():
        raise ValueError(f"Signal '{field}' is empty or non-finite.")
    return signal


def calculate_signal_features(signal):
    """
    Compute the same eight statistics that app.py uses.

    This intentionally mirrors app.calculate_signal_features() so that
    the generated deployment samples match what the live API extracts.
    """
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)

    if signal.size == 0 or not np.isfinite(signal).all():
        raise ValueError("Signal must contain finite numeric values.")

    result = {
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "rms": float(np.sqrt(np.mean(np.square(signal)))),
        "minimum": float(np.min(signal)),
        "maximum": float(np.max(signal)),
        "peak_to_peak": float(np.ptp(signal)),
        "skewness": float(skew(signal)),
        "kurtosis": float(kurtosis(signal)),
    }

    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError("Calculated statistics are non-finite.")

    return result


def extract_features(record):
    """
    Extract the exact 50 model input features from one MATLAB record.

    Feature ordering matches the saved model artifact:
      - 36 base sensor statistics  (6 sensors × 6 stats)
      - 12 shape statistics        (6 sensors × 2 stats)
      - DOC
      - feed
    """
    features = {}

    for sensor in SENSORS:
        statistics = calculate_signal_features(signal_from_record(record, sensor))
        for name in BASE_STATISTICS + SHAPE_STATISTICS:
            features[f"{sensor}_{name}"] = statistics[name]

    features["DOC"] = scalar_from_record(record, "DOC")
    features["feed"] = scalar_from_record(record, "feed")

    if len(features) != EXPECTED_FEATURE_COUNT:
        raise ValueError(f"Expected {EXPECTED_FEATURE_COUNT} features; got {len(features)}.")

    return features


def read_vb(record):
    """
    Read the measured flank wear value.

    Returns None when the value is NaN (unmeasured) or missing.
    Never returns 0.0 for a genuinely NaN label.
    """
    try:
        raw = scalar_from_record(record, "VB")
    except (ValueError, KeyError):
        return None

    if not math.isfinite(raw):
        return None

    return raw


def main():
    if not MAT_PATH.exists():
        print(f"ERROR: Dataset not found at {MAT_PATH}", file=sys.stderr)
        sys.exit(1)

    data = loadmat(MAT_PATH)
    records = data["mill"].ravel()

    samples = []
    skipped = []

    for idx, record in enumerate(records):
        case_id = int(scalar_from_record(record, "case"))
        run_id = int(scalar_from_record(record, "run"))

        try:
            features = extract_features(record)
        except (ValueError, KeyError) as exc:
            skipped.append({
                "index": idx,
                "case": case_id,
                "run": run_id,
                "reason": str(exc),
            })
            continue

        # Validate every feature is numeric and finite
        invalid = [
            name for name, val in features.items()
            if not isinstance(val, (int, float)) or not math.isfinite(val)
        ]
        if invalid:
            skipped.append({
                "index": idx,
                "case": case_id,
                "run": run_id,
                "reason": f"Non-finite features: {invalid}",
            })
            continue

        vb = read_vb(record)

        samples.append({
            "case": case_id,
            "run": run_id,
            "feature_count": len(features),
            "features": features,
            "measured_VB_mm": vb,
            "notice": (
                "Features were calculated from the original sensor signals. "
                "Measured VB is reference information, not an input feature."
            ),
            "source": "data/raw/mill.mat",
        })

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(samples, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    # Summary
    with_label = sum(1 for s in samples if s["measured_VB_mm"] is not None)
    without_label = len(samples) - with_label

    print("=" * 60)
    print("TOOLSENSE — DEPLOYMENT SAMPLES BUILT")
    print("=" * 60)
    print(f"Total dataset records:  {len(records)}")
    print(f"Valid samples written:  {len(samples)}")
    print(f"  With measured VB:     {with_label}")
    print(f"  Without measured VB:  {without_label}")
    print(f"Skipped records:        {len(skipped)}")
    for entry in skipped:
        print(f"  Case {entry['case']} Run {entry['run']}: {entry['reason']}")
    print(f"Features per sample:    {EXPECTED_FEATURE_COUNT}")
    try:
        print(f"Output file:            {OUTPUT_PATH}")
    except UnicodeEncodeError:
        print(f"Output file:            {OUTPUT_PATH.relative_to(ROOT)}")

    # Sanity checks
    cases = sorted({(s["case"], s["run"]) for s in samples})
    print(f"Unique (case, run):     {len(cases)}")

    if not samples:
        print("WARNING: No valid samples were produced!", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
