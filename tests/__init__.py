from __future__ import annotations

from fastapi.testclient import TestClient

from realtime_inference.api import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "predictive_maintenance" in payload["domains"]


def test_predict_endpoint() -> None:
    payload = {
        "domain": "predictive_maintenance",
        "payload": {
            "temperature_c": 118.0,
            "vibration_mm_s": 18.5,
            "power_draw_kw": 480.0,
            "pressure_bar": 165.0,
            "rpm": 13000.0,
        },
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["domain"] == "predictive_maintenance"
    assert body["decision"] in {"NORMAL", "WARNING", "CRITICAL"}


def test_metrics_endpoint() -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert "summary" in body
    assert len(body["domains"]) >= 1


def test_predict_batch_endpoint() -> None:
    payload = {
        "domain": "fraud_anomaly",
        "payloads": [
            {
                "amount": 6420.0,
                "account_age_days": 90,
                "transaction_velocity_1h": 42,
                "external_risk_score": 0.84,
                "device_trust_score": 28.0,
                "geo_distance_km": 1800.0,
            }
        ],
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total"] == 1
    assert body["results"][0]["domain"] == "fraud_anomaly"
