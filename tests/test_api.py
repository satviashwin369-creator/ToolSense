
import math

import pytest

from app.app import app, FEATURES


@pytest.fixture
def client():
    app.config["TESTING"] = True

    with app.test_client() as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/api/health")

    assert response.status_code == 200

    data = response.get_json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["model_name"] == "ExtraTreesRegressor"
    assert data["expected_features"] == 50


def test_metrics_endpoint(client):
    response = client.get("/api/metrics")

    assert response.status_code == 200

    data = response.get_json()

    assert data["feature_count"] == 50
    assert data["model_type"] == "ExtraTreesRegressor"
    assert "overall" in data
    assert "MAE_mm" in data["overall"]
    assert "RMSE_mm" in data["overall"]
    assert "R2" in data["overall"]


def test_sample_endpoint_returns_all_model_features(client):
    response = client.get("/api/sample")

    assert response.status_code == 200

    data = response.get_json()

    assert data["feature_count"] == 50
    assert isinstance(data["features"], dict)
    assert set(data["features"].keys()) == set(FEATURES)

    for feature in FEATURES:
        value = data["features"][feature]
        assert isinstance(value, (int, float))
        assert math.isfinite(value)


def test_prediction_accepts_real_sample(client):
    sample_response = client.get("/api/sample")
    assert sample_response.status_code == 200

    features = sample_response.get_json()["features"]

    response = client.post("/api/predict", json=features)

    assert response.status_code == 200

    data = response.get_json()

    assert isinstance(data["predicted_VB_mm"], (int, float))
    assert math.isfinite(data["predicted_VB_mm"])
    assert data["input_feature_count"] == 50
    assert data["research_only"] is True


def test_prediction_rejects_missing_features(client):
    response = client.post("/api/predict", json={})

    assert response.status_code == 400

    data = response.get_json()

    assert "missing_features" in data
    assert data["missing_count"] == 50


def test_prediction_rejects_non_numeric_feature(client):
    sample_response = client.get("/api/sample")
    assert sample_response.status_code == 200

    features = sample_response.get_json()["features"].copy()
    features[FEATURES[0]] = "not-a-number"

    response = client.post("/api/predict", json=features)

    assert response.status_code == 400
    assert "must be a number" in response.get_json()["error"]


def test_prediction_rejects_non_finite_feature(client):
    sample_response = client.get("/api/sample")
    assert sample_response.status_code == 200

    features = sample_response.get_json()["features"].copy()
    features[FEATURES[0]] = float("nan")

    response = client.post("/api/predict", json=features)

    assert response.status_code == 400
    assert "must be finite" in response.get_json()["error"]


# --------------------------------------------------
# /api/samples - sample list
# --------------------------------------------------

def test_samples_endpoint_returns_list(client):
    response = client.get("/api/samples")

    assert response.status_code == 200

    data = response.get_json()

    assert "count" in data
    assert "records" in data
    assert data["count"] > 0
    assert len(data["records"]) == data["count"]


def test_samples_records_have_required_metadata(client):
    response = client.get("/api/samples")
    data = response.get_json()

    for record in data["records"]:
        assert "case" in record
        assert "run" in record
        assert "measured_VB_mm" in record
        assert "has_measured_wear" in record
        assert isinstance(record["case"], int)
        assert isinstance(record["run"], int)
        assert isinstance(record["has_measured_wear"], bool)


def test_samples_contains_records_with_and_without_wear(client):
    response = client.get("/api/samples")
    data = response.get_json()

    with_wear = [r for r in data["records"] if r["has_measured_wear"]]
    without_wear = [r for r in data["records"] if not r["has_measured_wear"]]

    assert len(with_wear) > 0, "Expected some records with measured wear"
    assert len(without_wear) > 0, "Expected some records without measured wear"


# --------------------------------------------------
# /api/sample?case=X&run=Y - specific sample
# --------------------------------------------------

def test_sample_with_case_and_run_returns_specific_record(client):
    response = client.get("/api/sample?case=1&run=1")

    assert response.status_code == 200

    data = response.get_json()

    assert data["case"] == 1
    assert data["run"] == 1
    assert data["feature_count"] == 50
    assert isinstance(data["features"], dict)
    assert set(data["features"].keys()) == set(FEATURES)


def test_sample_different_case_returns_different_features(client):
    r1 = client.get("/api/sample?case=1&run=1")
    r2 = client.get("/api/sample?case=2&run=1")

    assert r1.status_code == 200
    assert r2.status_code == 200

    d1 = r1.get_json()
    d2 = r2.get_json()

    # Different cases should produce different feature vectors
    assert d1["case"] != d2["case"] or d1["run"] != d2["run"]
    assert d1["features"] != d2["features"]


def test_sample_invalid_case_run_returns_404(client):
    response = client.get("/api/sample?case=9999&run=9999")

    assert response.status_code == 404

    data = response.get_json()
    assert "error" in data


def test_sample_missing_run_param_returns_400(client):
    response = client.get("/api/sample?case=1")

    assert response.status_code == 400

    data = response.get_json()
    assert "error" in data


def test_sample_non_integer_params_returns_400(client):
    response = client.get("/api/sample?case=abc&run=xyz")

    assert response.status_code == 400

    data = response.get_json()
    assert "error" in data


def test_sample_with_null_vb_returns_null(client):
    """Records without a measured VB should have null, not 0."""
    samples_response = client.get("/api/samples")
    data = samples_response.get_json()

    null_records = [r for r in data["records"] if not r["has_measured_wear"]]

    if null_records:
        r = null_records[0]
        response = client.get(f"/api/sample?case={r['case']}&run={r['run']}")

        assert response.status_code == 200

        sample_data = response.get_json()
        assert sample_data["measured_VB_mm"] is None


def test_default_sample_still_works(client):
    """The original /api/sample endpoint (no params) still works."""
    response = client.get("/api/sample")

    assert response.status_code == 200

    data = response.get_json()

    assert data["feature_count"] == 50
    assert isinstance(data["features"], dict)
