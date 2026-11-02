"""API and HTTP integration tests for FastAPI serving layer (app.py)."""

from datetime import datetime, timezone
import pytest
from starlette.testclient import TestClient

from realtime_inference.app import create_app


@pytest.fixture(scope="module")
def client():
    """Module-scoped FastAPI TestClient running startup and shutdown lifespans."""
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_index_endpoint(client):
    """Verify index route returns service overview and available endpoints."""
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["service"] == "realtime-inference"
    assert data["status"] == "online"
    assert "predict" in data["endpoints"]


def test_health_endpoints(client):
    """Verify /health and /healthz endpoints return 200 and healthy status."""
    for path in ["/health", "/healthz"]:
        resp = client.get(path)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["model_version"].startswith("v")


def test_ready_endpoints(client):
    """Verify /ready and /readyz report readiness for traffic routing."""
    for path in ["/ready", "/readyz"]:
        resp = client.get(path)
        assert resp.status_code == 200
        data = resp.json()
        assert data["ready"] is True
        assert data["model_loaded"] is True


def test_predict_single_with_micro_batcher(client, valid_low_risk_payload_dict):
    """Verify single transaction prediction via dynamic micro-batcher."""
    resp = client.post("/predict", json=valid_low_risk_payload_dict)
    assert resp.status_code == 200
    data = resp.json()
    assert data["request_id"] == "req-safe-001"
    assert data["decision"] == "ALLOW"
    assert data["risk_score"] < 0.25
    assert data["confidence"] > 0.70
    assert "inference_latency_ms" in data


def test_predict_single_direct_sync(client, valid_high_risk_payload_dict):
    """Verify single transaction prediction bypassing micro-batcher with use_batcher=false."""
    resp = client.post("/predict?use_batcher=false", json=valid_high_risk_payload_dict)
    assert resp.status_code == 200
    data = resp.json()
    assert data["request_id"] == "req-fraud-999"
    assert data["decision"] == "BLOCK"
    assert data["risk_score"] > 0.75


def test_predict_rejects_negative_amount(client, valid_low_risk_payload_dict):
    """Verify boundary violation (negative amount) returns HTTP 422."""
    bad_payload = dict(valid_low_risk_payload_dict)
    bad_payload["features"] = dict(valid_low_risk_payload_dict["features"])
    bad_payload["features"]["amount"] = -10.0

    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422
    assert "details" in resp.json()


def test_predict_rejects_extra_fields(client, valid_low_risk_payload_dict):
    """Verify injected forbidden fields return HTTP 422."""
    bad_payload = dict(valid_low_risk_payload_dict)
    bad_payload["features"] = dict(valid_low_risk_payload_dict["features"])
    bad_payload["features"]["injected_exploit"] = "attack"

    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422


def test_predict_rejects_string_for_float(client, valid_low_risk_payload_dict):
    """Verify strict type enforcement rejects string numbers in JSON."""
    bad_payload = dict(valid_low_risk_payload_dict)
    bad_payload["features"] = dict(valid_low_risk_payload_dict["features"])
    bad_payload["features"]["amount"] = "25.50"

    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422


def test_predict_batch_endpoint(client, valid_low_risk_payload_dict, valid_high_risk_payload_dict):
    """Verify explicit batch submission endpoint /predict/batch."""
    batch_req = {
        "batch_id": "api-batch-001",
        "items": [
            valid_low_risk_payload_dict,
            valid_high_risk_payload_dict,
        ],
    }

    resp = client.post("/predict/batch", json=batch_req)
    assert resp.status_code == 200
    data = resp.json()
    assert data["batch_id"] == "api-batch-001"
    assert data["total_items"] == 2
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["decision"] == "ALLOW"
    assert data["predictions"][1]["decision"] == "BLOCK"


def test_stats_endpoint(client):
    """Verify /stats endpoint returns operational batcher telemetry."""
    resp = client.get("/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "queue_size" in data
    assert "metrics" in data
    assert data["metrics"]["total_requests"] >= 1


def test_model_info_endpoint(client):
    """Verify /model endpoint reports active model details and circuit breaker."""
    resp = client.get("/model")
    assert resp.status_code == 200
    data = resp.json()
    assert "version" in data
    assert "active_model_class" in data
    assert data["is_loaded"] is True
    assert data["fallback_enabled"] is True


def test_model_reload_endpoints(client, valid_low_risk_payload_dict):
    """Verify POST /model/reload hot-swaps active models with zero downtime."""
    # 1. Hot reload to linear model
    resp = client.post("/model/reload?target=linear")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["model"]["version"] == "v1.0.0"

    # Verify predict uses v1.0.0
    pred1 = client.post("/predict", json=valid_low_risk_payload_dict)
    assert pred1.status_code == 200
    assert pred1.json()["model_version"] == "v1.0.0"

    # 2. Hot reload back to ONNX GBDT model
    resp2 = client.post("/model/reload?target=onnx")
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["status"] == "success"
    assert data2["model"]["version"] == "v2.0.0-gbdt"

    # Verify predict uses v2.0.0-gbdt
    pred2 = client.post("/predict", json=valid_low_risk_payload_dict)
    assert pred2.status_code == 200
    assert pred2.json()["model_version"] == "v2.0.0-gbdt"

    # 3. Reject invalid hot reload target
    bad_reload = client.post("/model/reload?target=non_existent_weights.file")
    assert bad_reload.status_code == 400
