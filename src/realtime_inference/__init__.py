"""Realtime Inference package with high-performance in-memory wrapper and strict Pydantic schemas."""

from realtime_inference.schemas import (
    BatchInferenceInputPayload,
    BatchInferenceOutputPayload,
    InferenceInputPayload,
    InferenceOutputPayload,
    TransactionFeatures,
)
from realtime_inference.models.base import BaseInferenceModel, DecisionType, ModelPredictionResult
from realtime_inference.models.wrapper import LightweightInferenceWrapper
from realtime_inference.inference_engine import RealtimeInferenceEngine

__all__ = [
    "BatchInferenceInputPayload",
    "BatchInferenceOutputPayload",
    "InferenceInputPayload",
    "InferenceOutputPayload",
    "TransactionFeatures",
    "BaseInferenceModel",
    "DecisionType",
    "ModelPredictionResult",
    "LightweightInferenceWrapper",
    "RealtimeInferenceEngine",
]


def main() -> None:
    """CLI entry point demonstrating inference engine execution."""
    from datetime import datetime, timezone

    engine = RealtimeInferenceEngine()
    sample_payload = {
        "request_id": "demo-req-001",
        "client_id": "web-checkout-gateway",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": {
            "amount": 49.95,
            "account_age_days": 420,
            "transaction_velocity_1h": 1,
            "external_risk_score": 0.08,
            "device_trust_score": 96.0,
            "geo_distance_km": 3.2,
        },
    }

    output = engine.predict(sample_payload)
    print("Inference executed successfully:")
    print(f"Request ID: {output.request_id}")
    print(f"Decision:   {output.decision}")
    print(f"Risk Score: {output.risk_score:.4f}")
    print(f"Confidence: {output.confidence:.4f}")
    print(f"Latency:    {output.inference_latency_ms:.3f} ms")


if __name__ == "__main__":
    main()
