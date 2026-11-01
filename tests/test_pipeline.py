"""End-to-end integration tests for RealtimeInferenceEngine (inference_engine.py)."""

from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from inference_engine import RealtimeInferenceEngine
from schemas import (
    BatchInferenceInputPayload,
    BatchInferenceOutputPayload,
    InferenceInputPayload,
    InferenceOutputPayload,
)


def test_engine_predict_from_dict(inference_engine, valid_low_risk_payload_dict):
    """Verify raw dictionary payload is strictly validated and processed."""
    output = inference_engine.predict(valid_low_risk_payload_dict)

    assert isinstance(output, InferenceOutputPayload)
    assert output.request_id == "req-safe-001"
    assert output.model_version == "v1.0.0"
    assert output.decision == "ALLOW"
    assert 0.0 <= output.risk_score <= 1.0
    assert output.inference_latency_ms >= 0.0
    assert sum(output.probabilities.values()) == pytest.approx(1.0, rel=1e-3)


def test_engine_predict_from_pydantic_instance(inference_engine, valid_input_payload):
    """Verify already-validated Pydantic instance works seamlessly."""
    output = inference_engine.predict(valid_input_payload)

    assert isinstance(output, InferenceOutputPayload)
    assert output.request_id == valid_input_payload.request_id
    assert output.decision == "ALLOW"


def test_engine_blocks_high_risk_transaction(inference_engine, valid_high_risk_payload_dict):
    """Verify high-risk transaction input results in BLOCK decision."""
    output = inference_engine.predict(valid_high_risk_payload_dict)

    assert output.decision == "BLOCK"
    assert output.risk_score > 0.70
    assert output.probabilities["BLOCK"] > 0.50


def test_engine_rejects_malformed_input_before_inference(inference_engine):
    """Ensure invalid payloads raise ValidationError without calling inference."""
    invalid_payload = {
        "request_id": "bad-payload",
        "client_id": "test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": {
            "amount": -50.0,  # Negative amount violates schema constraint
            "account_age_days": 10,
            "transaction_velocity_1h": 1,
            "external_risk_score": 0.2,
            "device_trust_score": 90.0,
            "geo_distance_km": 1.0,
        },
    }

    with pytest.raises(ValidationError):
        inference_engine.predict(invalid_payload)


def test_engine_rejects_payload_with_extra_fields(inference_engine, valid_low_risk_payload_dict):
    """Ensure payloads with injected unknown fields fail validation."""
    malicious_payload = dict(valid_low_risk_payload_dict)
    malicious_payload["features"]["injected_bypass_flag"] = True

    with pytest.raises(ValidationError):
        inference_engine.predict(malicious_payload)


def test_engine_predict_batch(inference_engine, valid_low_risk_payload_dict, valid_high_risk_payload_dict):
    """Verify batch inference coordinates validation and vectorized execution."""
    batch_payload = {
        "batch_id": "batch-101",
        "items": [
            valid_low_risk_payload_dict,
            valid_high_risk_payload_dict,
        ],
    }

    batch_output = inference_engine.predict_batch(batch_payload)

    assert isinstance(batch_output, BatchInferenceOutputPayload)
    assert batch_output.batch_id == "batch-101"
    assert batch_output.total_items == 2
    assert len(batch_output.predictions) == 2

    # Check item 1 (low risk)
    assert batch_output.predictions[0].request_id == valid_low_risk_payload_dict["request_id"]
    assert batch_output.predictions[0].decision == "ALLOW"

    # Check item 2 (high risk)
    assert batch_output.predictions[1].request_id == valid_high_risk_payload_dict["request_id"]
    assert batch_output.predictions[1].decision == "BLOCK"

    assert batch_output.total_batch_latency_ms >= 0.0
