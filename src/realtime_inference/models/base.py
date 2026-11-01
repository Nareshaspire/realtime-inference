"""Abstract Base Model interface for in-memory lightweight inference wrappers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple
import numpy as np


DecisionType = Literal["ALLOW", "REVIEW", "BLOCK"]


@dataclass(frozen=True)
class ModelPredictionResult:
    """Raw prediction result directly from the in-memory inference wrapper."""

    decision: DecisionType
    risk_score: float
    confidence: float
    probabilities: Dict[DecisionType, float]
    raw_latency_ns: int


class BaseInferenceModel(ABC):
    """Abstract contract for high-performance in-memory inference models."""

    @abstractmethod
    def load_weights(self) -> None:
        """Load and cache pre-trained weights in-memory."""
        pass

    @abstractmethod
    def predict_one(self, feature_vector: List[float] | np.ndarray) -> ModelPredictionResult:
        """Perform zero-allocation single item prediction."""
        pass

    @abstractmethod
    def predict_batch(self, feature_matrix: np.ndarray) -> List[ModelPredictionResult]:
        """Perform vectorized SIMD batch prediction."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Return semantic model version."""
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if weights are loaded and locked in-memory."""
        pass
