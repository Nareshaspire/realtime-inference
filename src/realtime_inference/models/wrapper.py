"""High-performance in-memory lightweight inference wrapper."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from realtime_inference.models.base import BaseInferenceModel, DecisionType, ModelPredictionResult


class LightweightInferenceWrapper(BaseInferenceModel):
    """Zero-overhead in-memory inference wrapper executing vectorized predictions.

    Features:
    - Pre-trained weights loaded into contiguous C-memory once at startup.
    - Write-protection (writeable=False) ensuring thread safety under concurrent requests.
    - Vectorized, SIMD-aligned matrix algebra with numerically stable softmax.
    - Microsecond execution latency profiling.
    """

    def __init__(self, weights_path: Optional[Union[str, Path]] = None) -> None:
        if weights_path is None:
            weights_path = Path(__file__).resolve().parent / "weights" / "model_weights.npz"

        self._weights_path = Path(weights_path)
        self._is_loaded: bool = False
        self._version: str = "unknown"
        self._feature_names: List[str] = []
        self._classes: List[str] = []

        # Contiguous in-memory parameters
        self._weights: Optional[np.ndarray] = None
        self._bias: Optional[np.ndarray] = None
        self._means: Optional[np.ndarray] = None
        self._scales: Optional[np.ndarray] = None

        self.load_weights()

    def load_weights(self) -> None:
        """Load pre-trained weights from disk into fast in-memory buffers."""
        if not self._weights_path.is_file():
            raise FileNotFoundError(
                f"Model weights file not found at: {self._weights_path}. "
                "Ensure weights are generated or provided."
            )

        with np.load(self._weights_path) as archive:
            required_keys = {
                "weights",
                "bias",
                "feature_names",
                "feature_means",
                "feature_scales",
                "classes",
                "version",
            }
            missing_keys = required_keys - set(archive.files)
            if missing_keys:
                raise ValueError(f"Corrupt weights archive. Missing required keys: {missing_keys}")

            raw_weights = archive["weights"]
            raw_bias = archive["bias"]
            raw_means = archive["feature_means"]
            raw_scales = archive["feature_scales"]

            # Integrity verification
            if np.isnan(raw_weights).any() or np.isinf(raw_weights).any():
                raise ValueError("Weights tensor contains NaN or Inf values")
            if (raw_scales <= 0).any():
                raise ValueError("Feature scales must be strictly positive")

            # Store in contiguous C-order float32 arrays
            self._weights = np.ascontiguousarray(raw_weights, dtype=np.float32)
            self._bias = np.ascontiguousarray(raw_bias, dtype=np.float32)
            self._means = np.ascontiguousarray(raw_means, dtype=np.float32)
            self._scales = np.ascontiguousarray(raw_scales, dtype=np.float32)

            # Freeze in memory to guarantee thread-safe read-only operations
            self._weights.flags.writeable = False
            self._bias.flags.writeable = False
            self._means.flags.writeable = False
            self._scales.flags.writeable = False

            self._feature_names = [str(name) for name in archive["feature_names"]]
            self._classes = [str(cls_name) for cls_name in archive["classes"]]
            self._version = str(archive["version"])

        self._is_loaded = True

    @property
    def version(self) -> str:
        return self._version

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_names)

    @property
    def classes(self) -> List[str]:
        return list(self._classes)

    def _normalize_features(self, x: np.ndarray) -> np.ndarray:
        """Apply z-score feature standardization in-memory."""
        assert self._means is not None and self._scales is not None
        return (x - self._means) / self._scales

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax activation over the last dimension."""
        max_logits = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    def predict_one(self, feature_vector: Union[List[float], np.ndarray]) -> ModelPredictionResult:
        """Execute ultra-low latency single-item prediction.

        Parameters
        ----------
        feature_vector : List[float] or np.ndarray
            Ordered numeric feature values.

        Returns
        -------
        ModelPredictionResult
            Inference output containing decision, calibrated risk score, confidence, and latency.
        """
        if not self._is_loaded:
            raise RuntimeError("Model weights are not loaded into memory")

        t_start = time.perf_counter_ns()

        arr = np.asarray(feature_vector, dtype=np.float32)
        if arr.ndim != 1 or arr.shape[0] != self._weights.shape[0]:
            raise ValueError(
                f"Expected 1D feature array of size {self._weights.shape[0]}, got shape {arr.shape}"
            )

        norm_features = (arr - self._means) / self._scales
        logits = np.dot(norm_features, self._weights) + self._bias

        # Softmax for 1D logits
        max_logit = np.max(logits)
        exp_logits = np.exp(logits - max_logit)
        probs = exp_logits / np.sum(exp_logits)

        # Map classes: ["ALLOW", "REVIEW", "BLOCK"]
        prob_dict: Dict[DecisionType, float] = {
            "ALLOW": float(probs[0]),
            "REVIEW": float(probs[1]),
            "BLOCK": float(probs[2]),
        }

        # Weighted composite risk score
        risk_score = float(np.clip(probs[2] * 1.0 + probs[1] * 0.5, 0.0, 1.0))

        # Argmax decision & confidence
        pred_idx = int(np.argmax(probs))
        decision: DecisionType = self._classes[pred_idx]  # type: ignore
        confidence = float(probs[pred_idx])

        t_end = time.perf_counter_ns()

        return ModelPredictionResult(
            decision=decision,
            risk_score=risk_score,
            confidence=confidence,
            probabilities=prob_dict,
            raw_latency_ns=t_end - t_start,
        )

    def predict_batch(self, feature_matrix: np.ndarray) -> List[ModelPredictionResult]:
        """Execute vectorized SIMD batch inference across multiple feature records.

        Parameters
        ----------
        feature_matrix : np.ndarray
            Contiguous 2D matrix of shape (batch_size, num_features).

        Returns
        -------
        List[ModelPredictionResult]
            List of prediction results matching input order.
        """
        if not self._is_loaded:
            raise RuntimeError("Model weights are not loaded into memory")

        t_start = time.perf_counter_ns()

        mat = np.asarray(feature_matrix, dtype=np.float32)
        if mat.ndim != 2 or mat.shape[1] != self._weights.shape[0]:
            raise ValueError(
                f"Expected 2D matrix of shape (N, {self._weights.shape[0]}), got {mat.shape}"
            )

        norm_features = (mat - self._means) / self._scales
        logits = np.dot(norm_features, self._weights) + self._bias
        probs = self._softmax(logits)

        # Batch latency average attribution
        t_end = time.perf_counter_ns()
        batch_size = mat.shape[0]
        avg_latency_ns = (t_end - t_start) // max(1, batch_size)

        results: List[ModelPredictionResult] = []
        pred_indices = np.argmax(probs, axis=1)

        for i in range(batch_size):
            p = probs[i]
            idx = pred_indices[i]
            prob_dict: Dict[DecisionType, float] = {
                "ALLOW": float(p[0]),
                "REVIEW": float(p[1]),
                "BLOCK": float(p[2]),
            }
            risk_score = float(np.clip(p[2] * 1.0 + p[1] * 0.5, 0.0, 1.0))
            decision: DecisionType = self._classes[idx]  # type: ignore
            confidence = float(p[idx])

            results.append(
                ModelPredictionResult(
                    decision=decision,
                    risk_score=risk_score,
                    confidence=confidence,
                    probabilities=prob_dict,
                    raw_latency_ns=avg_latency_ns,
                )
            )

        return results
