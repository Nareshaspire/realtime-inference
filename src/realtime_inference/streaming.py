from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException

from realtime_inference.model_registry import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    MetricsResponse,
    PredictionRequest,
    PredictionResponse,
    RetrainRequest,
    MultiDomainModelRegistry,
)
from realtime_inference.streaming import InMemoryStreamProcessor, RedisStreamProcessor

registry = MultiDomainModelRegistry()
backend_name = os.getenv("REALTIME_STREAM_BACKEND", "memory").lower()
stream_processor = RedisStreamProcessor() if backend_name == "redis" else InMemoryStreamProcessor(registry=registry)

app = FastAPI(title="Realtime Multi-Domain Inference API", version="0.1.0")


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
        model_versions=registry.model_versions,
        summary={domain: {k: float(v) for k, v in metrics.items()} for domain, metrics in summary.items()},
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
    result = registry.predict(domain, payload)
    return {"event_id": event.event_id, "domain": domain, "result": result}


@app.get("/stream/history")
def stream_history() -> Dict[str, Any]:
    latest = list(stream_processor.history)
    return {"events": latest, "count": len(latest)}


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Realtime Multi-Domain Inference API"}


__all__ = ["app", "registry", "stream_processor"]

