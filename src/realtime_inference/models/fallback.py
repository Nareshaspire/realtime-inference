"""Deterministic heuristic rule-based fallback classifier for circuit breaker protection."""

from __future__ import annotations

import time
from typing import Dict, List, Union
import numpy as np

from realtime_inference.models.base import BaseInferenceModel, DecisionType, ModelPredictionResult


class HeuristicFallbackClassifier(BaseInferenceModel):
    """Rule-based deterministic classifier used as an automated circuit breaker.

    Engaged when active model weights fail or throw runtime inference exceptions,
    guaranteeing zero dropped transactions and predictable risk bounds.
    """

    def __init__(self, version: str = "v0.0.1-fallback") -> None:
        self._version = version
        self._is_loaded = True
        self._classes = ["ALLOW", "REVIEW", "BLOCK"]
        self._feature_names = [
            "amount",
            "account_age_days",
            "transaction_velocity_1h",
            "external_risk_score",
            "device_trust_score",
            "geo_distance_km",
        ]

    def load_weights(self) -> None:
        """Heuristic rules require no disk weights."""
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

    def _evaluate_rules(self, vec: List[float] | np.ndarray) -> tuple[DecisionType, float, Dict[DecisionType, float]]:
        """Evaluate deterministic tiered risk thresholds."""
        amount = float(vec[0])
        account_age_days = float(vec[1])
        velocity = float(vec[2])
        ext_risk = float(vec[3])
        device_trust = float(vec[4])
        distance = float(vec[5])

        # High-risk BLOCK conditions
        if (
            amount >= 5000.0
            or ext_risk >= 0.85
            or device_trust <= 15.0
            or velocity >= 25.0
            or (account_age_days <= 2 and amount >= 1000.0)
        ):
            decision: DecisionType = "BLOCK"
            risk_score = 0.95
            probs = {"ALLOW": 0.02, "REVIEW": 0.08, "BLOCK": 0.90}
            return decision, risk_score, probs

        # Medium-risk REVIEW conditions
        if (
            amount >= 1500.0
            or ext_risk >= 0.50
            or device_trust <= 40.0
            or velocity >= 8.0
            or distance >= 4000.0
        ):
            decision = "REVIEW"
            risk_score = 0.55
            probs = {"ALLOW": 0.15, "REVIEW": 0.75, "BLOCK": 0.10}
            return decision, risk_score, probs

        # Low-risk ALLOW
        decision = "ALLOW"
        risk_score = 0.05
        probs = {"ALLOW": 0.92, "REVIEW": 0.06, "BLOCK": 0.02}
        return decision, risk_score, probs

    def predict_one(self, feature_vector: Union[List[float], np.ndarray]) -> ModelPredictionResult:
        t0 = time.perf_counter_ns()
        vec = np.asarray(feature_vector, dtype=np.float32)
        if vec.shape[0] != 6:
            raise ValueError(f"Expected 6 features, got {vec.shape[0]}")

        decision, risk_score, probs = self._evaluate_rules(vec)
        t_end = time.perf_counter_ns()

        return ModelPredictionResult(
            decision=decision,
            risk_score=risk_score,
            confidence=probs[decision],
            probabilities=probs,
            raw_latency_ns=t_end - t0,
        )

    def predict_batch(self, feature_matrix: np.ndarray) -> List[ModelPredictionResult]:
        t0 = time.perf_counter_ns()
        mat = np.asarray(feature_matrix, dtype=np.float32)
        if mat.ndim != 2 or mat.shape[1] != 6:
            raise ValueError(f"Expected 2D matrix (N, 6), got {mat.shape}")

        batch_size = mat.shape[0]
        results: List[ModelPredictionResult] = []

        for i in range(batch_size):
            decision, risk_score, probs = self._evaluate_rules(mat[i])
            results.append(
                ModelPredictionResult(
                    decision=decision,
                    risk_score=risk_score,
                    confidence=probs[decision],
                    probabilities=probs,
                    raw_latency_ns=0,
                )
            )

        t_end = time.perf_counter_ns()
        avg_latency = (t_end - t0) // max(1, batch_size)
        return [
            ModelPredictionResult(
                decision=r.decision,
                risk_score=r.risk_score,
                confidence=r.confidence,
                probabilities=r.probabilities,
                raw_latency_ns=avg_latency,
            )
            for r in results
        ]
