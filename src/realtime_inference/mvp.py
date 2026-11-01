"""Realtime Inference package with high-performance in-memory wrapper and strict Pydantic schemas."""

from realtime_inference.models.base import BaseInferenceModel, DecisionType, ModelPredictionResult
from realtime_inference.models.wrapper import LightweightInferenceWrapper
from realtime_inference.mvp import MultiDomainPrediction, MultiDomainRealtimeMVP, demo
from realtime_inference.inference_engine import RealtimeInferenceEngine
from realtime_inference.schemas import (
    BatchInferenceInputPayload,
    BatchInferenceOutputPayload,
    InferenceInputPayload,
    InferenceOutputPayload,
    TransactionFeatures,
)

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
    "MultiDomainPrediction",
    "MultiDomainRealtimeMVP",
    "demo",
]


def main() -> None:
    """CLI entry point demonstrating the multi-domain real-time MVP."""
    demo()


if __name__ == "__main__":
    main()

