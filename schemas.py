"""Strict input and output validation schemas for real-time inference using Pydantic V2."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class StrictBaseSchema(BaseModel):
    """Base schema enforcing strict type constraints, frozen immutability, and zero extra fields."""

    model_config = ConfigDict(
        extra="forbid",            # Reject unknown fields to avoid silent payload corruption
        strict=True,               # Enforce strict types (no int -> str or str -> float coercion)
        frozen=True,               # Immutable data model for safe zero-copy multithreading
        validate_assignment=True,  # Re-validate if attributes are somehow reassigned
        str_strip_whitespace=True, # Remove unexpected leading/trailing whitespace
    )


class TransactionFeatures(StrictBaseSchema):
    """Payload features required for real-time risk assessment model inference."""

    amount: Annotated[
        float,
        Field(
            gt=0.0,
            le=1_000_000.0,
            description="Transaction monetary amount. Must be strictly positive and under 1,000,000.",
            examples=[149.99],
        ),
    ]
    account_age_days: Annotated[
        int,
        Field(
            ge=0,
            le=36500,
            description="Age of the user account in days. Must be non-negative.",
            examples=[180],
        ),
    ]
    transaction_velocity_1h: Annotated[
        int,
        Field(
            ge=0,
            le=500,
            description="Number of transactions initiated by this account in the last 60 minutes.",
            examples=[3],
        ),
    ]
    external_risk_score: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="External risk score normalized between 0.0 and 1.0.",
            examples=[0.12],
        ),
    ]
    device_trust_score: Annotated[
        float,
        Field(
            ge=0.0,
            le=100.0,
            description="Device trust evaluation score between 0.0 (untrusted) and 100.0 (fully trusted).",
            examples=[85.0],
        ),
    ]
    geo_distance_km: Annotated[
        float,
        Field(
            ge=0.0,
            le=40075.0,
            description="Distance in kilometers between current transaction location and billing residence.",
            examples=[12.4],
        ),
    ]

    @field_validator("amount", "external_risk_score", "device_trust_score", "geo_distance_km", mode="after")
    @classmethod
    def validate_finite_floats(cls, value: float) -> float:
        """Reject NaN and infinite floating-point values."""
        if math.isnan(value) or math.isinf(value):
            raise ValueError("Floating point values must be finite and not NaN/Inf")
        return value

    def to_array(self) -> List[float]:
        """Convert features to an ordered list for vectorized model input."""
        return [
            float(self.amount),
            float(self.account_age_days),
            float(self.transaction_velocity_1h),
            float(self.external_risk_score),
            float(self.device_trust_score),
            float(self.geo_distance_km),
        ]

    @classmethod
    def feature_names(cls) -> List[str]:
        """Return the strict feature ordering expected by the model weights."""
        return [
            "amount",
            "account_age_days",
            "transaction_velocity_1h",
            "external_risk_score",
            "device_trust_score",
            "geo_distance_km",
        ]


class InferenceInputPayload(StrictBaseSchema):
    """Real-time incoming request envelope containing metadata and validated feature vectors."""

    request_id: Annotated[
        str,
        Field(
            min_length=1,
            max_length=64,
            description="Unique transaction or request tracking identifier.",
            examples=["req_8f19da37-4d10-449a"],
        ),
    ]
    client_id: Annotated[
        str,
        Field(
            min_length=1,
            max_length=64,
            description="Identifier of the client service or tenant submitting the payload.",
            examples=["checkout_service_v2"],
        ),
    ]
    timestamp: Annotated[
        datetime,
        Field(
            description="ISO-8601 UTC timestamp of request generation.",
        ),
    ]
    features: Annotated[
        TransactionFeatures,
        Field(
            description="Strictly validated feature set for risk scoring.",
        ),
    ]

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, value: Any) -> datetime:
        """Parse ISO-8601 string or accept datetime before strict validation."""
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError as e:
                raise ValueError(f"Invalid ISO-8601 timestamp string: {value}") from e
        return value



class InferenceOutputPayload(StrictBaseSchema):
    """Strictly validated prediction response output from the inference wrapper."""

    request_id: Annotated[
        str,
        Field(
            description="Mirrored request ID corresponding to the input payload.",
        ),
    ]
    model_version: Annotated[
        str,
        Field(
            min_length=1,
            description="Semantic version of the pre-trained in-memory model weights used.",
            examples=["v1.2.0"],
        ),
    ]
    decision: Annotated[
        Literal["ALLOW", "REVIEW", "BLOCK"],
        Field(
            description="Categorical decision determined by model scoring and operational thresholds.",
        ),
    ]
    risk_score: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Calibrated probability of risk/fraud (0.0 = completely safe, 1.0 = fraudulent).",
        ),
    ]
    confidence: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Confidence score in the predicted categorical decision.",
        ),
    ]
    probabilities: Annotated[
        Dict[Literal["ALLOW", "REVIEW", "BLOCK"], float],
        Field(
            description="Multi-class probability distribution across all decision outcomes.",
        ),
    ]
    inference_latency_ms: Annotated[
        float,
        Field(
            ge=0.0,
            description="Total model evaluation time measured in milliseconds.",
        ),
    ]
    processed_at: Annotated[
        datetime,
        Field(
            description="ISO-8601 UTC timestamp when prediction was finalized.",
        ),
    ]

    @field_validator("processed_at", mode="before")
    @classmethod
    def parse_processed_at(cls, value: Any) -> datetime:
        """Parse ISO-8601 string or accept datetime before strict validation."""
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError as e:
                raise ValueError(f"Invalid ISO-8601 timestamp string: {value}") from e
        return value


    @field_validator("risk_score", "confidence", "inference_latency_ms", mode="after")
    @classmethod
    def validate_finite_output_floats(cls, value: float) -> float:
        """Ensure all output metrics are finite numbers."""
        if math.isnan(value) or math.isinf(value):
            raise ValueError("Output numerical metrics must be finite and not NaN/Inf")
        return value

    @model_validator(mode="after")
    def validate_probabilities_sum(self) -> "InferenceOutputPayload":
        """Verify that output probability distribution sums to 1.0 within numerical epsilon."""
        prob_sum = sum(self.probabilities.values())
        if not (0.99 <= prob_sum <= 1.01):
            raise ValueError(f"Probabilities must sum to 1.0 (+/- 0.01), got {prob_sum:.4f}")
        return self


class BatchInferenceInputPayload(StrictBaseSchema):
    """High-throughput batch inference request envelope."""

    batch_id: Annotated[
        str,
        Field(
            min_length=1,
            max_length=64,
            description="Unique identifier for the batch submission.",
        ),
    ]
    items: Annotated[
        List[InferenceInputPayload],
        Field(
            min_length=1,
            max_length=1000,
            description="List of individual payload envelopes. Capped at 1,000 for strict latency SLOs.",
        ),
    ]


class BatchInferenceOutputPayload(StrictBaseSchema):
    """Strictly validated response envelope for batch inference."""

    batch_id: Annotated[
        str,
        Field(
            description="Mirrored batch identifier.",
        ),
    ]
    total_items: Annotated[
        int,
        Field(
            ge=1,
            description="Total count of successfully processed items.",
        ),
    ]
    predictions: Annotated[
        List[InferenceOutputPayload],
        Field(
            description="Ordered list of predictions matching the input batch sequence.",
        ),
    ]
    total_batch_latency_ms: Annotated[
        float,
        Field(
            ge=0.0,
            description="End-to-end batch processing duration in milliseconds.",
        ),
    ]
