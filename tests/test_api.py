
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
