from __future__ import annotations

from realtime_inference.model_registry import MultiDomainModelRegistry


def test_registry_predicts_expected_domain() -> None:
    registry = MultiDomainModelRegistry(artifact_dir="/tmp/realtime_inference_registry_test")
    result = registry.predict(
        "predictive_maintenance",
        {
            "temperature_c": 118.0,
            "vibration_mm_s": 18.5,
            "power_draw_kw": 480.0,
            "pressure_bar": 165.0,
            "rpm": 13000.0,
        },
    )
    assert result["domain"] == "predictive_maintenance"
    assert result["decision"] in {"NORMAL", "WARNING", "CRITICAL"}
    assert 0.0 <= result["score"] <= 1.0
    assert 0.0 <= result["confidence"] <= 1.0


def test_registry_retraining_updates_version() -> None:
    registry = MultiDomainModelRegistry(artifact_dir="/tmp/realtime_inference_registry_retrain_test")
    original_version = registry.model_versions["fraud_anomaly"]
    rows = [
        {
            "amount": 5000.0,
            "account_age_days": 20,
            "transaction_velocity_1h": 80,
            "external_risk_score": 0.96,
            "device_trust_score": 12.0,
            "geo_distance_km": 3000.0,
            "label": "BLOCK",
        }
    ]
    metrics = registry.retrain_domain("fraud_anomaly", new_rows=rows)
    assert metrics["domain"] == "fraud_anomaly"
    assert registry.model_versions["fraud_anomaly"] != original_version
    assert registry.models["fraud_anomaly"].version == registry.model_versions["fraud_anomaly"]
