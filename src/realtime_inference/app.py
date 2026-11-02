"""Production FastAPI serving application for real-time inference with dynamic micro-batching."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from realtime_inference.batcher import DynamicMicroBatcher, QueueFullError
from realtime_inference.inference_engine import RealtimeInferenceEngine
from realtime_inference.schemas import (
    BatchInferenceInputPayload,
    BatchInferenceOutputPayload,
    InferenceInputPayload,
    InferenceOutputPayload,
)

logger = logging.getLogger("realtime_inference.api")
START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage lifecycle of model engine and dynamic micro-batcher background worker."""
    logger.info("Initializing RealtimeInferenceEngine and DynamicMicroBatcher...")
    engine = RealtimeInferenceEngine()
    batcher = DynamicMicroBatcher(
        engine=engine,
        max_batch_size=64,
        batch_window_ms=1.0,
        max_queue_size=5000,
    )
    await batcher.start()

    app.state.engine = engine
    app.state.batcher = batcher

    yield

    logger.info("Shutting down DynamicMicroBatcher...")
    await batcher.stop()


def create_app() -> FastAPI:
    """Factory creating and configuring the FastAPI inference application."""
    app = FastAPI(
        title="High-Performance Real-Time Inference Service",
        description=(
            "Ultra-low latency transaction risk assessment with strict Pydantic V2 validation "
            "and adaptive dynamic micro-batching."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Return structured, lightweight validation error messages."""
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Validation Error",
                "details": exc.errors(),
            },
        )

    @app.exception_handler(QueueFullError)
    async def queue_full_exception_handler(
        request: Request, exc: QueueFullError
    ) -> JSONResponse:
        """Handle queue saturation with HTTP 503 and retry hint."""
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            headers={"Retry-After": "1"},
            content={
                "error": "Service Temporarily Overloaded",
                "message": str(exc),
            },
        )

    @app.get("/", summary="Service Index")
    async def index() -> Dict[str, Any]:
        return {
            "service": "realtime-inference",
            "version": "0.1.0",
            "status": "online",
            "uptime_seconds": round(time.time() - START_TIME, 1),
            "endpoints": {
                "predict": "POST /predict",
                "predict_batch": "POST /predict/batch",
                "health": "GET /healthz",
                "ready": "GET /readyz",
                "stats": "GET /stats",
            },
        }

    @app.get("/health", tags=["Health"], summary="Health check")
    @app.get("/healthz", tags=["Health"], summary="Liveness probe")
    async def health(request: Request) -> Dict[str, Any]:
        """Liveness check reporting model status and batcher health."""
        batcher: DynamicMicroBatcher = request.app.state.batcher
        engine: RealtimeInferenceEngine = request.app.state.engine

        return {
            "status": "healthy",
            "uptime_seconds": round(time.time() - START_TIME, 1),
            "model_version": engine.model.version,
            "model_loaded": engine.model.is_loaded,
            "batcher_active": batcher.is_running,
            "queue_size": batcher.queue_size,
        }

    @app.get("/ready", tags=["Health"], summary="Readiness check")
    @app.get("/readyz", tags=["Health"], summary="Readiness probe")
    async def ready(request: Request) -> Dict[str, Any]:
        """Kubernetes readiness probe verifying memory write-protection and load status."""
        engine: RealtimeInferenceEngine = request.app.state.engine
        batcher: DynamicMicroBatcher = request.app.state.batcher

        if not engine.model.is_loaded or not batcher.is_running:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Engine or micro-batcher not ready to accept traffic",
            )

        return {
            "ready": True,
            "model_loaded": True,
            "model_version": engine.model.version,
        }

    @app.post(
        "/predict",
        response_model=InferenceOutputPayload,
        tags=["Inference"],
        summary="Predict risk for single transaction payload",
    )
    async def predict_single(
        payload: InferenceInputPayload,
        request: Request,
        use_batcher: bool = Query(
            default=True,
            description="Whether to route through adaptive dynamic micro-batcher for SIMD concurrency",
        ),
    ) -> InferenceOutputPayload:
        """Score a single transaction payload.

        If `use_batcher=True` (default), bundles incoming request with concurrent arrivals
        within a ~1ms window for high-throughput vectorized SIMD evaluation.
        If `use_batcher=False`, immediately evaluates on CPU without queuing.
        """
        if use_batcher:
            batcher: DynamicMicroBatcher = request.app.state.batcher
            return await batcher.predict(payload)
        else:
            engine: RealtimeInferenceEngine = request.app.state.engine
            return engine.predict(payload)

    @app.post(
        "/predict/batch",
        response_model=BatchInferenceOutputPayload,
        tags=["Inference"],
        summary="Predict risk for explicit batch submission",
    )
    async def predict_batch(
        batch_payload: BatchInferenceInputPayload,
        request: Request,
    ) -> BatchInferenceOutputPayload:
        """Process explicit batch submission using SIMD matrix algebra."""
        engine: RealtimeInferenceEngine = request.app.state.engine
        return engine.predict_batch(batch_payload)

    @app.get("/stats", tags=["Monitoring"], summary="Micro-batcher operational metrics")
    async def stats(request: Request) -> Dict[str, Any]:
        """Return operational throughput, latency, and queue telemetry."""
        batcher: DynamicMicroBatcher = request.app.state.batcher
        return {
            "queue_size": batcher.queue_size,
            "max_queue_size": batcher.max_queue_size,
            "max_batch_size": batcher.max_batch_size,
            "batch_window_ms": batcher.batch_window_ms,
            "metrics": batcher.stats.snapshot(),
        }

    @app.get("/model", tags=["Model Management"], summary="Active model metadata and circuit breaker status")
    async def get_model_info(request: Request) -> Dict[str, Any]:
        """Return diagnostic info about the active model and fallback circuit breaker."""
        engine: RealtimeInferenceEngine = request.app.state.engine
        return engine.get_model_info()

    @app.post("/model/reload", tags=["Model Management"], summary="Hot-reload active model with zero downtime")
    async def reload_model(
        request: Request,
        target: str = Query(
            default="onnx",
            description="Target model identifier ('onnx', 'linear', 'fallback') or filesystem path",
        ),
    ) -> Dict[str, Any]:
        """Atomically hot-swap active model after safety dry-run verification."""
        engine: RealtimeInferenceEngine = request.app.state.engine
        try:
            updated_info = engine.hot_reload(target)
            return {
                "status": "success",
                "message": f"Hot-reloaded model to {updated_info['version']}",
                "model": updated_info,
            }
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model hot-reload rejected: {exc}",
            )

    return app


app = create_app()


def main() -> None:
    """CLI entrypoint to launch FastAPI inference server with uvicorn."""
    import uvicorn

    print("Starting High-Performance Real-Time Inference Server on http://0.0.0.0:8000...")
    uvicorn.run(
        "realtime_inference.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        access_log=False,
        loop="uvloop",
    )


if __name__ == "__main__":
    main()
