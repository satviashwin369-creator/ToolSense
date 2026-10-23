
from pathlib import Path
import json
import math
import os

import joblib
import numpy as np
from scipy.io import loadmat
from scipy.stats import skew, kurtosis
from flask import Flask, jsonify, request


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "toolsense_model_candidate.joblib"
METRICS_PATH = ROOT / "reports" / "candidate_model_metrics.json"
MAT_PATH = ROOT / "data" / "raw" / "mill.mat"

app = Flask(__name__)

MODEL_NAME = "ExtraTreesRegressor"
TARGET_NAME = "VB"
TARGET_UNIT = "mm"
WEAR_THRESHOLD_MM = 0.5

SENSORS = [
    "smcAC",
    "smcDC",
    "vib_table",
    "vib_spindle",
    "AE_table",
    "AE_spindle",
]

BASE_STATS = [
    "mean",
    "std",
    "rms",
    "minimum",
    "maximum",
    "peak_to_peak",
]

SHAPE_STATS = ["skewness", "kurtosis"]


# ---------------------------------------------------------------------
# CORS configuration
#
# Local development is allowed automatically.
# For deployment, set FRONTEND_ORIGIN to your frontend's exact URL.
# Example: https://toolsense-frontend.vercel.app
# ---------------------------------------------------------------------
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
    response.headers["Access-Control-Allow-Methods"] = (
        "GET, POST, OPTIONS"
    )

    return response


# ---------------------------------------------------------------------
# Load and verify the candidate model
# ---------------------------------------------------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Candidate model not found at {MODEL_PATH}. "
        "Ensure models/toolsense_model_candidate.joblib is included "
        "in the deployment."
    )

saved_artifact = joblib.load(MODEL_PATH)

model = saved_artifact["model"]
FEATURES = saved_artifact["features"]

if len(FEATURES) != model.n_features_in_:
    raise ValueError(
        f"Artifact mismatch: {len(FEATURES)} feature names but "
        f"the model expects {model.n_features_in_} features."
    )

if len(FEATURES) != 50:
    raise ValueError(
        f"Expected the 50-feature candidate model, found {len(FEATURES)}."
    )

EXPECTED_FEATURES = [
    f"{sensor}_{stat}"
    for sensor in SENSORS
    for stat in BASE_STATS
] + [
    f"{sensor}_{stat}"
    for sensor in SENSORS
    for stat in SHAPE_STATS
] + ["DOC", "feed"]

if FEATURES != EXPECTED_FEATURES:
    raise ValueError(
        "The model's feature order does not match the expected "
        "50-feature definition. Check the saved training artifact."
    )

print("=" * 64)
print("ToolSense backend")
print(f"Model: {MODEL_NAME}")
print(f"Features: {len(FEATURES)}")
print(f"Raw dataset: {MAT_PATH}")
print(f"Artifact: {MODEL_PATH}")
print(f"Raw dataset available: {MAT_PATH.exists()}")
print("=" * 64)


# ---------------------------------------------------------------------
# Load raw dataset records
# ---------------------------------------------------------------------
def load_raw_records():
    """Load original milling records from mill.mat."""
    if not MAT_PATH.exists():
        raise FileNotFoundError(
            "The original mill.mat dataset is not included in this "
            "deployment. The sample endpoint is unavailable."
        )

    try:
        records = loadmat(MAT_PATH)["mill"].ravel()
    except (OSError, KeyError, ValueError) as error:
        raise RuntimeError(
            f"Could not read the milling dataset: {error}"
        ) from error

    return records


def scalar_from_record(record, field):
    """Read a scalar MATLAB struct field safely."""
    try:
        value = np.asarray(record[field]).squeeze()

        if value.size != 1:
            raise ValueError(f"{field} is not a scalar.")

        return float(value)

    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Could not read field '{field}' from the raw record."
        ) from error


def extract_record_features(record):
    """
    Extract the same 50 features and in the same order as training.

    Base statistics:
      mean, std, RMS, minimum, maximum, peak-to-peak

    Shape statistics:
      scipy skew(bias=False)
      scipy kurtosis(fisher=True, bias=False)
    """
    result = {}

    for sensor in SENSORS:
        try:
            signal = np.asarray(
                record[sensor], dtype=float
            ).ravel()
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Could not read sensor '{sensor}'."
            ) from error

        if signal.size == 0 or not np.isfinite(signal).all():
            raise ValueError(
                f"Invalid {sensor} signal in the selected record."
            )

        # These definitions match src/train_candidate_model.py.
        result[f"{sensor}_mean"] = float(np.mean(signal))
        result[f"{sensor}_std"] = float(np.std(signal))
        result[f"{sensor}_rms"] = float(
            np.sqrt(np.mean(signal ** 2))
        )
        result[f"{sensor}_minimum"] = float(np.min(signal))
        result[f"{sensor}_maximum"] = float(np.max(signal))
        result[f"{sensor}_peak_to_peak"] = float(np.ptp(signal))

        if np.std(signal) == 0:
            skewness_value = 0.0
            kurtosis_value = 0.0
        else:
            skewness_value = (
                float(skew(signal, bias=False))
                if signal.size >= 3
                else 0.0
            )

            kurtosis_value = (
                float(
                    kurtosis(
                        signal,
                        fisher=True,
                        bias=False,
                    )
                )
                if signal.size >= 4
                else 0.0
            )

        result[f"{sensor}_skewness"] = skewness_value
        result[f"{sensor}_kurtosis"] = kurtosis_value

    # Read machining conditions from the actual dataset record.
    result["DOC"] = scalar_from_record(record, "DOC")
    result["feed"] = scalar_from_record(record, "feed")

    if len(result) != 50:
        raise ValueError(
            f"Expected 50 extracted features, got {len(result)}."
        )

    missing = [
        feature for feature in FEATURES if feature not in result
    ]

    if missing:
        raise ValueError(
            f"Feature extraction omitted required fields: {missing}"
        )

    for feature in FEATURES:
        if not math.isfinite(result[feature]):
            raise ValueError(
                f"Feature '{feature}' is not finite."
            )

    return result


# ---------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------
@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "project": "ToolSense",
        "model_loaded": model is not None,
        "model_name": MODEL_NAME,
        "expected_features": len(FEATURES),
        "target": TARGET_NAME,
        "target_unit": TARGET_UNIT,
        "feature_definition": saved_artifact.get(
            "feature_definition",
            "50 sensor and machining-condition features",
        ),
        "validation": saved_artifact.get("validation"),
        "sample_endpoint_available": MAT_PATH.exists(),
        "notice": (
            "Research prototype. Predictions are not certified "
            "industrial safety decisions."
        ),
    })


# ---------------------------------------------------------------------
# Candidate-model evaluation metrics
# ---------------------------------------------------------------------
@app.get("/api/metrics")
def metrics():
    if not METRICS_PATH.exists():
        return jsonify({
            "error": "Candidate model metrics file not found."
        }), 404

    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            saved_metrics = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        app.logger.exception("Could not read candidate model metrics.")
        return jsonify({
            "error": "Could not read candidate model metrics.",
            "details": str(error),
        }), 500

    return jsonify(saved_metrics)


# ---------------------------------------------------------------------
# Real dataset sample endpoint
#
# If mill.mat is absent in a deployment, return a clear 503 response.
# Never fabricate sample features or replace missing wear with zero.
# ---------------------------------------------------------------------
@app.get("/api/sample")
def sample():
    if not MAT_PATH.exists():
        return jsonify({
            "error": "Sample data is unavailable on this deployment.",
            "detail": (
                "The original mill.mat dataset is not included. "
                "The model prediction endpoint remains available "
                "when valid features are supplied."
            ),
            "sample_endpoint_available": False,
        }), 503

    try:
        records = load_raw_records()

        # Prefer a record with a measured wear value.
        # Do not replace missing wear labels with zero.
        measured_record = None

        for record in records:
            try:
                vb = scalar_from_record(record, "VB")
            except ValueError:
                continue

            if math.isfinite(vb):
                measured_record = record
                break

        if measured_record is None:
            return jsonify({
                "error": (
                    "No record with a valid measured VB value "
                    "was found in the raw dataset."
                )
            }), 404

        features = extract_record_features(measured_record)

        case_id = int(scalar_from_record(measured_record, "case"))
        run_id = int(scalar_from_record(measured_record, "run"))
        measured_vb = scalar_from_record(measured_record, "VB")

        return jsonify({
            "features": {
                feature: features[feature]
                for feature in FEATURES
            },
            "case": case_id,
            "run": run_id,
            "measured_VB_mm": measured_vb,
            "feature_count": len(FEATURES),
            "source": "data/raw/mill.mat",
            "notice": (
                "Features were calculated from the original sensor "
                "signals. Measured VB is reference information, not "
                "an input feature."
            ),
        })

    except (OSError, RuntimeError, ValueError) as error:
        app.logger.exception("Could not load a raw dataset sample.")
        return jsonify({
            "error": str(error)
        }), 500


# ---------------------------------------------------------------------
# Wear prediction endpoint
# Expects every named feature used during model training.
# ---------------------------------------------------------------------
@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({
            "error": (
                "Request body must be a JSON object containing features."
            )
        }), 400

    missing = [
        feature for feature in FEATURES
        if feature not in payload
    ]

    if missing:
        return jsonify({
            "error": "Some required features are missing.",
            "missing_features": missing,
            "missing_count": len(missing),
            "expected_feature_count": len(FEATURES),
        }), 400

    values = {}

    for feature in FEATURES:
        value = payload[feature]

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            return jsonify({
                "error": f"Feature '{feature}' must be a number."
            }), 400

        if not math.isfinite(value):
            return jsonify({
                "error": f"Feature '{feature}' must be finite."
            }), 400

        values[feature] = float(value)

    input_data = [
        [values[feature] for feature in FEATURES]
    ]

    try:
        prediction = float(model.predict(input_data)[0])
    except Exception:
        app.logger.exception("Candidate-model prediction failed.")
        return jsonify({
            "error": (
                "The model could not process the supplied features."
            )
        }), 500

    # Do not silently clip the raw model prediction.
    return jsonify({
        "predicted_VB_mm": round(prediction, 4),
        "unit": TARGET_UNIT,
        "model": MODEL_NAME,
        "input_feature_count": len(FEATURES),
        "high_wear_threshold_mm": WEAR_THRESHOLD_MM,
        "research_only": True,
        "notice": (
            "Experimental wear estimate. It is not a certified "
            "tool-change recommendation or machine-safety decision."
        ),
    })


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )
