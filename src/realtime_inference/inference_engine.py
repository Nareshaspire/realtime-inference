"""High-performance real-time inference engine integrating strict validation and in-memory scoring."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
import numpy as np
from pydantic import ValidationError

from realtime_inference.models.wrapper import LightweightInferenceWrapper
from realtime_inference.schemas import (
    BatchInferenceInputPayload,
    BatchInferenceOutputPayload,
    InferenceInputPayload,
    InferenceOutputPayload,
    TransactionFeatures,
)


class RealtimeInferenceEngine:
    """End-to-end inference orchestrator.

    Pipeline:
    1. Strict input validation against Pydantic schema (rejects invalid types/missing/extra fields).
    2. Zero-overhead numerical conversion to contiguous vector.
    3. Low-latency in-memory model inference execution.
    4. Strict output validation before returning response to callers.
    """

    def __init__(self, model_wrapper: Optional[LightweightInferenceWrapper] = None) -> None:
        self.model = model_wrapper if model_wrapper is not None else LightweightInferenceWrapper()

    def predict(
        self,
        payload: Union[Dict[str, Any], InferenceInputPayload],
    ) -> InferenceOutputPayload:
        """Validate input payload, run in-memory inference, and return validated output.

        Raises
        ------
        ValidationError
            If the incoming payload violates strict feature bounds, types, or extra-field rules.
        """
        # Step 1: Strict input schema validation
        if isinstance(payload, InferenceInputPayload):
            validated_input = payload
        else:
            validated_input = InferenceInputPayload.model_validate(payload)

        # Step 2: Feature extraction into ordered numeric array
        feature_vector = validated_input.features.to_array()

        # Step 3: Fast in-memory model inference
        t_start = time.perf_counter_ns()
        result = self.model.predict_one(feature_vector)
        t_end = time.perf_counter_ns()
        latency_ms = (t_end - t_start) / 1_000_000.0

        # Step 4: Strict output validation
        output = InferenceOutputPayload(
            request_id=validated_input.request_id,
            model_version=self.model.version,
            decision=result.decision,
            risk_score=result.risk_score,
            confidence=result.confidence,
            probabilities=result.probabilities,
            inference_latency_ms=round(latency_ms, 4),
            processed_at=datetime.now(timezone.utc),
        )

        return output

    def predict_batch(
        self,
        batch_payload: Union[Dict[str, Any], BatchInferenceInputPayload],
    ) -> BatchInferenceOutputPayload:
        """Process batch of real-time payloads with SIMD vectorized inference."""
        t_batch_start = time.perf_counter_ns()

        if isinstance(batch_payload, BatchInferenceInputPayload):
            validated_batch = batch_payload
        else:
            validated_batch = BatchInferenceInputPayload.model_validate(batch_payload)

        # Vectorized feature matrix extraction
        raw_rows = [item.features.to_array() for item in validated_batch.items]
        matrix = np.ascontiguousarray(raw_rows, dtype=np.float32)

        # Fast in-memory batch prediction
        model_results = self.model.predict_batch(matrix)

        now = datetime.now(timezone.utc)
        predictions: List[InferenceOutputPayload] = []

        for item, res in zip(validated_batch.items, model_results):
            single_out = InferenceOutputPayload(
                request_id=item.request_id,
                model_version=self.model.version,
                decision=res.decision,
                risk_score=res.risk_score,
                confidence=res.confidence,
                probabilities=res.probabilities,
                inference_latency_ms=round(res.raw_latency_ns / 1_000_000.0, 4),
                processed_at=now,
            )
            predictions.append(single_out)

        t_batch_end = time.perf_counter_ns()
        total_latency_ms = (t_batch_end - t_batch_start) / 1_000_000.0

        return BatchInferenceOutputPayload(
            batch_id=validated_batch.batch_id,
            total_items=len(predictions),
            predictions=predictions,
            total_batch_latency_ms=round(total_latency_ms, 4),
        )
