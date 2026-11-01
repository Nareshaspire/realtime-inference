"""Pytest fixtures for unit and integration testing."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
import pytest

from models.wrapper import LightweightInferenceWrapper
from inference_engine import RealtimeInferenceEngine
from schemas import InferenceInputPayload, TransactionFeatures


@pytest.fixture(scope="session")
def model_weights_path() -> Path:
    """Return the path to pre-trained model weights."""
    return Path(__file__).resolve().parent.parent / "models" / "weights" / "model_weights.npz"


@pytest.fixture
def model_wrapper(model_weights_path: Path) -> LightweightInferenceWrapper:
    """Provide an initialized in-memory model wrapper."""
    return LightweightInferenceWrapper(weights_path=model_weights_path)


@pytest.fixture
def inference_engine(model_wrapper: LightweightInferenceWrapper) -> RealtimeInferenceEngine:
    """Provide an initialized real-time inference engine."""
    return RealtimeInferenceEngine(model_wrapper=model_wrapper)


@pytest.fixture
def valid_low_risk_payload_dict() -> Dict[str, Any]:
    """Valid payload representative of a regular, low-risk transaction."""
    return {
        "request_id": "req-safe-001",
        "client_id": "payment_gateway",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": {
            "amount": 25.50,
            "account_age_days": 730,
            "transaction_velocity_1h": 1,
            "external_risk_score": 0.05,
            "device_trust_score": 98.0,
            "geo_distance_km": 1.5,
        },
    }


@pytest.fixture
def valid_high_risk_payload_dict() -> Dict[str, Any]:
    """Valid payload representative of an anomalous, high-risk transaction."""
    return {
        "request_id": "req-fraud-999",
        "client_id": "payment_gateway",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": {
            "amount": 4999.00,
            "account_age_days": 1,
            "transaction_velocity_1h": 45,
            "external_risk_score": 0.98,
            "device_trust_score": 5.0,
            "geo_distance_km": 8500.0,
        },
    }


@pytest.fixture
def valid_input_payload(valid_low_risk_payload_dict: Dict[str, Any]) -> InferenceInputPayload:
    """Pre-validated Pydantic model instance."""
    return InferenceInputPayload.model_validate(valid_low_risk_payload_dict)
