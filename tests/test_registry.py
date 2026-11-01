from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from realtime_inference.model_registry import MultiDomainModelRegistry
from realtime_inference.streaming import InMemoryStreamProcessor, RedisStreamProcessor


class PredictionRequest(BaseModel):
    domain: str = Field(..., description="Domain name to score, e.g. predictive_maintenance")
    payload: Dict[str, Any] = Field(..., description="Feature dictionary for the selected domain")


class BatchPredictionRequest(BaseModel):
    domain: str = Field(..., description="Domain name to score")
    payloads: List[Dict[str, Any]] = Field(..., min_length=1, description="List of payloads")


class PredictionResponse(BaseModel):
    domain: str
    decision: str
    score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    recommended_action: str
    explanation: str
    model_version: str
    model_name: str


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    total: int


class RetrainRequest(BaseModel):
    domain: str = Field(..., description="Domain to retrain")
    new_rows: List[Dict[str, Any]] = Field(default_factory=list, description="New training records to merge")
    force: bool = Field(default=False, description="Force retraining even when no new rows are supplied")


class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    domains: List[str]
    timestamp: str


class MetricsResponse(BaseModel):
    registry_version: str
    backend: str
    domains: List[str]
    model_versions: Dict[str, str]
    summary: Dict[str, Dict[str, float]]


registry = MultiDomainModelRegistry()
backend_name = os.getenv("REALTIME_STREAM_BACKEND", "memory").lower()
stream_processor = RedisStreamProcessor(registry=registry) if backend_name == "redis" else InMemoryStreamProcessor(registry=registry)
app = FastAPI(title="Realtime Multi-Domain Inference API", version="0.1.0")


@app.get("/", response_model=Dict[str, str])
def root() -> Dict[str, str]:
    return {"message": "Realtime Multi-Domain Inference API"}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        models_loaded=len(registry.models),
        domains=registry.list_domains(),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    summary = registry.evaluate_all()
    return MetricsResponse(
        registry_version="realtime-inference-v1.0.0",
        backend=backend_name,
        domains=registry.list_domains(),
        model_versions=registry.get_model_versions(),
        summary={domain: {key: float(value) for key, value in metrics.items()} for domain, metrics in summary.items()},
    )


@app.get("/domains")
def list_domains() -> Dict[str, List[str]]:
    return {"domains": registry.list_domains()}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    try:
        result = registry.predict(payload.domain, payload.payload)
        return PredictionResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchPredictionRequest) -> BatchPredictionResponse:
    try:
        results = registry.predict_batch(payload.domain, payload.payloads)
        return BatchPredictionResponse(results=[PredictionResponse(**item) for item in results], total=len(results))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/retrain", response_model=Dict[str, Any])
def retrain(payload: RetrainRequest) -> Dict[str, Any]:
    try:
        metrics = registry.retrain_domain(payload.domain, payload.new_rows, force=payload.force)
        return {"status": "retrained", "domain": payload.domain, "metrics": metrics}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/stream")
def stream_event(domain: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    event = stream_processor.emit(domain, payload)
    output = registry.predict(domain, payload)
    return {"event_id": event.event_id, "domain": domain, "result": output}


@app.get("/stream/history")
def stream_history() -> Dict[str, Any]:
    recent = list(stream_processor.history)
    return {"events": recent, "count": len(recent)}


__all__ = [
    "app",
    "PredictionRequest",
    "BatchPredictionRequest",
    "PredictionResponse",
    "BatchPredictionResponse",
    "RetrainRequest",
    "HealthResponse",
    "MetricsResponse",
    "registry",
    "stream_processor",
]
