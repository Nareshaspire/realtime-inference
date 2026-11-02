"""Root re-export for BaseInferenceModel and result contracts."""

from realtime_inference.models.base import (
    BaseInferenceModel,
    DecisionType,
    ModelPredictionResult,
)

__all__ = [
    "BaseInferenceModel",
    "DecisionType",
    "ModelPredictionResult",
]
