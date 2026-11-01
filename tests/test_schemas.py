"""Unit tests for Pydantic V2 strict schemas (schemas.py)."""

from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from schemas import (
    BatchInferenceInputPayload,
    BatchInferenceOutputPayload,
    InferenceInputPayload,
    InferenceOutputPayload,
    TransactionFeatures,
)


def test_valid_transaction_features():
    """Verify standard valid feature dictionary passes without error."""
    features = TransactionFeatures(
        amount=150.0,
        account_age_days=100,
        transaction_velocity_1h=2,
        external_risk_score=0.25,
        device_trust_score=75.0,
        geo_distance_km=15.0,
    )
    arr = features.to_array()
    assert len(arr) == 6
    assert arr[0] == 150.0
    assert features.feature_names() == [
        "amount",
        "account_age_days",
        "transaction_velocity_1h",
        "external_risk_score",
        "device_trust_score",
        "geo_distance_km",
    ]


def test_strict_type_enforcement():
    """Verify strict mode rejects string-to-float or loose coercions."""
    with pytest.raises(ValidationError) as exc:
        TransactionFeatures(
            amount="150.0",  # String instead of float in strict mode
            account_age_days=100,
            transaction_velocity_1h=2,
            external_risk_score=0.25,
            device_trust_score=75.0,
            geo_distance_km=15.0,
        )
    assert "Input should be a valid number" in str(exc.value) or "float" in str(exc.value).lower()


def test_forbid_extra_fields():
    """Verify that unexpected extra payload fields trigger ValidationError."""
    payload_data = {
        "amount": 100.0,
        "account_age_days": 50,
        "transaction_velocity_1h": 1,
        "external_risk_score": 0.1,
        "device_trust_score": 90.0,
        "geo_distance_km": 5.0,
        "unauthorized_extra_feature": 12345,  # Extra forbidden field
    }
    with pytest.raises(ValidationError) as exc:
        TransactionFeatures.model_validate(payload_data)
    assert "Extra inputs are not permitted" in str(exc.value)


def test_numeric_boundary_validations():
    """Verify range limits (negative amount, negative age, out-of-bounds scores)."""
    # Negative amount
    with pytest.raises(ValidationError):
        TransactionFeatures(
            amount=-10.0,
            account_age_days=100,
            transaction_velocity_1h=2,
            external_risk_score=0.25,
            device_trust_score=75.0,
            geo_distance_km=15.0,
        )

    # Zero amount (must be strictly gt=0.0)
    with pytest.raises(ValidationError):
        TransactionFeatures(
            amount=0.0,
            account_age_days=100,
            transaction_velocity_1h=2,
            external_risk_score=0.25,
            device_trust_score=75.0,
            geo_distance_km=15.0,
        )

    # Risk score > 1.0
    with pytest.raises(ValidationError):
        TransactionFeatures(
            amount=100.0,
            account_age_days=100,
            transaction_velocity_1h=2,
            external_risk_score=1.5,
            device_trust_score=75.0,
            geo_distance_km=15.0,
        )

    # Device trust score > 100.0
    with pytest.raises(ValidationError):
        TransactionFeatures(
            amount=100.0,
            account_age_days=100,
            transaction_velocity_1h=2,
            external_risk_score=0.5,
            device_trust_score=105.0,
            geo_distance_km=15.0,
        )


def test_reject_nan_and_inf():
    """Verify that NaN and Inf are rejected for float features."""
    with pytest.raises(ValidationError) as exc:
        TransactionFeatures(
            amount=float("nan"),
            account_age_days=100,
            transaction_velocity_1h=2,
            external_risk_score=0.25,
            device_trust_score=75.0,
            geo_distance_km=15.0,
        )
    assert "finite" in str(exc.value).lower() or "nan" in str(exc.value).lower()

    with pytest.raises(ValidationError) as exc:
        TransactionFeatures(
            amount=float("inf"),
            account_age_days=100,
            transaction_velocity_1h=2,
            external_risk_score=0.25,
            device_trust_score=75.0,
            geo_distance_km=15.0,
        )
    assert "finite" in str(exc.value).lower() or "inf" in str(exc.value).lower()


def test_frozen_immutability():
    """Ensure validated instances cannot be mutated at runtime."""
    features = TransactionFeatures(
        amount=100.0,
        account_age_days=50,
        transaction_velocity_1h=1,
        external_risk_score=0.1,
        device_trust_score=90.0,
        geo_distance_km=5.0,
    )
    with pytest.raises(ValidationError):
        features.amount = 200.0  # type: ignore


def test_inference_input_payload_validation(valid_low_risk_payload_dict):
    """Verify full request envelope validation."""
    payload = InferenceInputPayload.model_validate(valid_low_risk_payload_dict)
    assert payload.request_id == "req-safe-001"
    assert payload.client_id == "payment_gateway"
    assert payload.features.amount == 25.50

    # Missing required field
    corrupted = dict(valid_low_risk_payload_dict)
    del corrupted["request_id"]
    with pytest.raises(ValidationError):
        InferenceInputPayload.model_validate(corrupted)


def test_inference_output_payload_probability_sum_check():
    """Verify output schema validates probability distribution sums to 1.0."""
    now = datetime.now(timezone.utc)
    # Valid output
    valid_out = InferenceOutputPayload(
        request_id="req-1",
        model_version="v1.0.0",
        decision="ALLOW",
        risk_score=0.05,
        confidence=0.92,
        probabilities={"ALLOW": 0.92, "REVIEW": 0.05, "BLOCK": 0.03},
        inference_latency_ms=0.12,
        processed_at=now,
    )
    assert valid_out.decision == "ALLOW"

    # Probabilities do not sum to 1.0
    with pytest.raises(ValidationError) as exc:
        InferenceOutputPayload(
            request_id="req-1",
            model_version="v1.0.0",
            decision="ALLOW",
            risk_score=0.05,
            confidence=0.92,
            probabilities={"ALLOW": 0.50, "REVIEW": 0.10, "BLOCK": 0.10},  # Sum = 0.70 != 1.0
            inference_latency_ms=0.12,
            processed_at=now,
        )
    assert "Probabilities must sum to 1.0" in str(exc.value)


def test_batch_payload_constraints(valid_low_risk_payload_dict):
    """Verify batch size constraints (minimum 1, maximum 1000)."""
    single = InferenceInputPayload.model_validate(valid_low_risk_payload_dict)

    # Empty batch
    with pytest.raises(ValidationError):
        BatchInferenceInputPayload(batch_id="b-0", items=[])

    # Valid batch of 2
    batch = BatchInferenceInputPayload(batch_id="b-1", items=[single, single])
    assert len(batch.items) == 2
