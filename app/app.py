
import json
import math
import os
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request
from scipy.io import loadmat
from scipy.stats import kurtosis, skew


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "toolsense_model_candidate.joblib"
METRICS_PATH = ROOT / "reports" / "candidate_model_metrics.json"
MAT_PATH = ROOT / "data" / "raw" / "mill.mat"
SAMPLE_PATH = ROOT / "data" / "processed" / "deployment_sample.json"
SAMPLES_PATH = ROOT / "data" / "processed" / "deployment_samples.json"

EXPECTED_FEATURE_COUNT = 50
HIGH_WEAR_THRESHOLD_MM = 0.5

SENSORS = [
    "smcAC",
    "smcDC",
    "vib_table",
    "vib_spindle",
    "AE_table",
    "AE_spindle",
]

BASE_STATISTICS = [
    "mean", "std", "rms", "minimum", "maximum", "peak_to_peak"
]
SHAPE_STATISTICS = ["skewness", "kurtosis"]

app = Flask(__name__)


# --------------------------------------------------
# CORS
# --------------------------------------------------

def allowed_origins():
    origins = {"http://localhost:5173"}
    configured_origin = os.environ.get("FRONTEND_ORIGIN", "").strip()

    if configured_origin:
        origins.add(configured_origin.rstrip("/"))

    return origins


@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin", "").rstrip("/")

    if origin and origin in allowed_origins():
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"

    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/api", defaults={"unused": ""}, methods=["OPTIONS"])
@app.route("/api/<path:unused>", methods=["OPTIONS"])
def api_preflight(unused):
    return "", 204


# --------------------------------------------------
# Model loading
# --------------------------------------------------

model = None
FEATURES = []
MODEL_ARTIFACT = None
MODEL_LOAD_ERROR = None

try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")

    MODEL_ARTIFACT = joblib.load(MODEL_PATH)
    model = MODEL_ARTIFACT["model"]
    FEATURES = list(MODEL_ARTIFACT["features"])

    if len(FEATURES) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected 50 features; artifact contains {len(FEATURES)}."
        )

    if getattr(model, "n_features_in_", None) != EXPECTED_FEATURE_COUNT:
        raise ValueError("The model does not expect 50 input features.")

except Exception as exc:
    model = None
    FEATURES = []
    MODEL_LOAD_ERROR = str(exc)


# --------------------------------------------------
# Dataset and feature utilities
# --------------------------------------------------

def scalar_from_record(record, field):
    try:
        value = np.asarray(record[field]).squeeze()

        if value.size == 0:
            raise ValueError("Empty field")

        return float(value.reshape(-1)[0])

    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"Could not read scalar field '{field}'.") from exc


def signal_from_record(record, field):
    try:
        signal = np.asarray(
            record[field], dtype=np.float64
        ).squeeze().reshape(-1)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Could not read sensor field '{field}'.") from exc

    if signal.size == 0 or not np.isfinite(signal).all():
        raise ValueError(f"Sensor signal '{field}' is empty or non-finite.")

    return signal


def calculate_signal_features(signal):
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

    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Calculated signal statistics are non-finite.")

    return result


def extract_features(record):
    features = {}

    for sensor in SENSORS:
        statistics = calculate_signal_features(
            signal_from_record(record, sensor)
        )

        for name in BASE_STATISTICS + SHAPE_STATISTICS:
            features[f"{sensor}_{name}"] = statistics[name]

    features["DOC"] = scalar_from_record(record, "DOC")
    features["feed"] = scalar_from_record(record, "feed")

    if len(features) != EXPECTED_FEATURE_COUNT:
        raise ValueError(f"Expected 50 features; got {len(features)}.")

    return features


def load_raw_records():
    if not MAT_PATH.exists():
        raise FileNotFoundError("The original mill.mat dataset is not included.")

    try:
        return loadmat(MAT_PATH)["mill"].ravel()
    except (OSError, KeyError, ValueError) as exc:
        raise RuntimeError(f"Could not read milling dataset: {exc}") from exc


def make_sample_response(record):
    features = extract_features(record)

    return {
        "case": int(scalar_from_record(record, "case")),
        "run": int(scalar_from_record(record, "run")),
        "feature_count": len(features),
        "features": features,
        "measured_VB_mm": scalar_from_record(record, "VB"),
        "notice": (
            "Features were calculated from the original sensor signals. "
            "Measured VB is reference information, not an input feature."
        ),
        "source": "data/raw/mill.mat",
    }


def load_saved_deployment_sample():
    if not SAMPLE_PATH.exists():
        raise FileNotFoundError("The saved deployment sample is unavailable.")

    try:
        sample = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read deployment sample: {exc}") from exc

    features = sample.get("features")

    if not isinstance(features, dict) or len(features) != EXPECTED_FEATURE_COUNT:
        raise ValueError("Saved sample must contain exactly 50 features.")

    if FEATURES and set(features) != set(FEATURES):
        raise ValueError("Saved sample feature names do not match the model.")

    for name, value in features.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Feature '{name}' must be a number.")

        if not math.isfinite(value):
            raise ValueError(f"Feature '{name}' must be finite.")

    sample["feature_count"] = len(features)
    return sample


def validate_deployment_sample(sample):
    """Validate a single record loaded from deployment_samples.json."""
    features = sample.get("features")

    if not isinstance(features, dict) or len(features) != EXPECTED_FEATURE_COUNT:
        raise ValueError("Sample must contain exactly 50 features.")

    if FEATURES and set(features) != set(FEATURES):
        raise ValueError("Sample feature names do not match the model.")

    for name, value in features.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Feature '{name}' must be a number.")
        if not math.isfinite(value):
            raise ValueError(f"Feature '{name}' must be finite.")

    return True


_cached_deployment_samples = None


def load_saved_deployment_samples():
    """Load and cache the multi-sample deployment file."""
    global _cached_deployment_samples

    if _cached_deployment_samples is not None:
        return _cached_deployment_samples

    if not SAMPLES_PATH.exists():
        raise FileNotFoundError(
            f"Multi-sample deployment file not found: {SAMPLES_PATH.name}"
        )

    try:
        data = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Could not read deployment samples: {exc}"
        ) from exc

    if not isinstance(data, list) or not data:
        raise ValueError("deployment_samples.json must be a non-empty list.")

    for sample in data:
        validate_deployment_sample(sample)
        sample["feature_count"] = len(sample["features"])

    _cached_deployment_samples = data
    return data


# --------------------------------------------------
# Health
# --------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    loaded = model is not None

    return jsonify({
        "status": "ok" if loaded else "error",
        "model_loaded": loaded,
        "model_name": type(model).__name__ if loaded else None,
        "expected_features": len(FEATURES),
        "feature_definition": (
            "36 sensor statistics + 12 signal shape statistics + DOC + feed"
        ),
        "target": "VB",
        "target_unit": "mm",
        "validation": "GroupKFold",
        "sample_endpoint_available": MAT_PATH.exists() or SAMPLE_PATH.exists(),
        "error": MODEL_LOAD_ERROR,
    }), (200 if loaded else 500)


# --------------------------------------------------
# Metrics
# --------------------------------------------------

@app.route("/api/metrics", methods=["GET"])
def metrics():
    if not METRICS_PATH.exists():
        return jsonify({"error": "Model metrics file is unavailable."}), 503

    try:
        data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        return jsonify(data), 200
    except (OSError, json.JSONDecodeError) as exc:
        return jsonify({"error": f"Could not load metrics: {exc}"}), 500


# --------------------------------------------------
# Sample
# --------------------------------------------------

@app.route("/api/samples", methods=["GET"])
def samples_list():
    """Return metadata for all available deployment samples."""
    try:
        all_samples = load_saved_deployment_samples()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        app.logger.exception("Could not load deployment samples")
        return jsonify({
            "error": "Sample data is unavailable.",
            "detail": str(exc),
        }), 503

    records = []
    for s in all_samples:
        records.append({
            "case": s["case"],
            "run": s["run"],
            "measured_VB_mm": s["measured_VB_mm"],
            "has_measured_wear": s["measured_VB_mm"] is not None,
        })

    return jsonify({
        "count": len(records),
        "records": records,
    }), 200


@app.route("/api/sample", methods=["GET"])
def sample():
    # Support case/run query parameters for specific record selection
    requested_case = request.args.get("case")
    requested_run = request.args.get("run")

    if requested_case is not None or requested_run is not None:
        return _get_selected_sample(requested_case, requested_run)

    # Original default behaviour: return case 1, run 1
    if MAT_PATH.exists():
        try:
            for record in load_raw_records():
                case_number = int(scalar_from_record(record, "case"))
                run_number = int(scalar_from_record(record, "run"))

                if case_number == 1 and run_number == 1:
                    return jsonify(make_sample_response(record)), 200

            return jsonify({"error": "Case 1, Run 1 was not found."}), 404

        except Exception as exc:
            app.logger.exception("Could not load sample from mill.mat")
            return jsonify({
                "error": "Could not load sample from original dataset.",
                "detail": str(exc),
            }), 500

    try:
        return jsonify(load_saved_deployment_sample()), 200
    except FileNotFoundError as exc:
        return jsonify({
            "error": "Sample data is unavailable on this deployment.",
            "detail": str(exc),
            "sample_endpoint_available": False,
        }), 503
    except (RuntimeError, ValueError, OSError) as exc:
        app.logger.exception("Could not load saved deployment sample")
        return jsonify({
            "error": "The saved deployment sample could not be loaded.",
            "detail": str(exc),
        }), 500


def _get_selected_sample(raw_case, raw_run):
    """Return a specific sample identified by case and run."""
    # Validate parameters
    if raw_case is None or raw_run is None:
        return jsonify({
            "error": "Both 'case' and 'run' query parameters are required.",
        }), 400

    try:
        target_case = int(raw_case)
        target_run = int(raw_run)
    except (ValueError, TypeError):
        return jsonify({
            "error": "'case' and 'run' must be integers.",
        }), 400

    # Try deployment samples first (works on Vercel)
    try:
        all_samples = load_saved_deployment_samples()
        for s in all_samples:
            if s["case"] == target_case and s["run"] == target_run:
                return jsonify(s), 200

        return jsonify({
            "error": f"No record found for case {target_case}, run {target_run}.",
        }), 404

    except (FileNotFoundError, RuntimeError, ValueError):
        pass  # Fall through to raw dataset

    # Fall back to raw dataset extraction
    if MAT_PATH.exists():
        try:
            for record in load_raw_records():
                case_number = int(scalar_from_record(record, "case"))
                run_number = int(scalar_from_record(record, "run"))

                if case_number == target_case and run_number == target_run:
                    return jsonify(make_sample_response(record)), 200

            return jsonify({
                "error": f"No record found for case {target_case}, run {target_run}.",
            }), 404

        except Exception as exc:
            app.logger.exception("Could not load sample from mill.mat")
            return jsonify({
                "error": "Could not load sample from original dataset.",
                "detail": str(exc),
            }), 500

    return jsonify({
        "error": "Sample data is unavailable on this deployment.",
    }), 503


# --------------------------------------------------
# Prediction
# --------------------------------------------------

@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "error": "The ML model is not available.",
            "detail": MODEL_LOAD_ERROR,
        }), 503

    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    values = payload.get("features", payload)

    if not isinstance(values, dict):
        return jsonify({"error": "'features' must be a JSON object."}), 400

    missing = [name for name in FEATURES if name not in values]

    if missing:
        return jsonify({
            "error": "Missing required model features.",
            "missing_features": missing,
            "missing_count": len(missing),
            "expected_features": len(FEATURES),
        }), 400

    validated = []

    for name in FEATURES:
        value = values[name]

        if isinstance(value, bool):
            return jsonify({
                "error": f"Feature '{name}' must be a number."
            }), 400

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return jsonify({
                "error": f"Feature '{name}' must be a number."
            }), 400

        if not math.isfinite(numeric_value):
            return jsonify({
                "error": f"Feature '{name}' must be finite."
            }), 400

        validated.append(numeric_value)

    try:
        prediction_value = float(
            model.predict(np.asarray([validated], dtype=np.float64))[0]
        )

        if not math.isfinite(prediction_value):
            raise ValueError("Model returned a non-finite prediction.")

        return jsonify({
            "predicted_VB_mm": prediction_value,
            "unit": "mm",
            "model": type(model).__name__,
            "feature_count": len(FEATURES),
            "input_feature_count": len(validated),
            "high_wear_threshold_mm": HIGH_WEAR_THRESHOLD_MM,
            "high_wear": prediction_value >= HIGH_WEAR_THRESHOLD_MM,
            "research_only": True,
            "notice": (
                "This is a research estimate, not a certified "
                "tool-change recommendation."
            ),
        }), 200

    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed.",
            "detail": str(exc),
        }), 500


# --------------------------------------------------
# Local development
# --------------------------------------------------

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
