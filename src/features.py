
import numpy as np

SENSOR_FIELDS = [
    "smcAC",
    "smcDC",
    "vib_table",
    "vib_spindle",
    "AE_table",
    "AE_spindle",
]


def extract_signal_features(signal):
    """Extract basic statistical features from one sensor signal."""
    signal = np.asarray(signal, dtype=float).ravel()

    if signal.size == 0:
        raise ValueError("Sensor signal cannot be empty.")

    if not np.isfinite(signal).all():
        raise ValueError("Sensor signal contains non-finite values.")

    features = {
        "mean": np.mean(signal),
        "std": np.std(signal),
        "rms": np.sqrt(np.mean(signal ** 2)),
        "minimum": np.min(signal),
        "maximum": np.max(signal),
        "peak_to_peak": np.ptp(signal),
    }

    return features


def extract_record_features(record):
    """Extract features from all six sensors in one machining record."""
    result = {}

    for sensor_name in SENSOR_FIELDS:
        signal = record[sensor_name]
        sensor_features = extract_signal_features(signal)

        for feature_name, value in sensor_features.items():
            result[f"{sensor_name}_{feature_name}"] = float(value)

    return result
