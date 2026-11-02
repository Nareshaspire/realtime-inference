"""Adaptive dynamic micro-batcher for real-time vectorized inference."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
import numpy as np

from realtime_inference.inference_engine import RealtimeInferenceEngine
from realtime_inference.schemas import (
    InferenceInputPayload,
    InferenceOutputPayload,
)

logger = logging.getLogger("realtime_inference.batcher")


class QueueFullError(Exception):
    """Raised when the micro-batcher queue exceeds maximum capacity under backpressure."""
    pass


@dataclass
class BatchItem:
    """Internal envelope tracking an enqueued request and its awaiting future."""

    payload: InferenceInputPayload
    future: asyncio.Future[InferenceOutputPayload]
    enqueued_at_ns: int


@dataclass
class BatcherStats:
    """Real-time operational metrics for dynamic batcher throughput profiling."""

    total_requests: int = 0
    total_batches: int = 0
    total_dropped: int = 0
    sum_batch_sizes: int = 0
    sum_wait_time_ms: float = 0.0
    sum_batch_eval_ms: float = 0.0

    @property
    def avg_batch_size(self) -> float:
        return (self.sum_batch_sizes / self.total_batches) if self.total_batches > 0 else 0.0

    @property
    def avg_wait_time_ms(self) -> float:
        return (self.sum_wait_time_ms / self.total_requests) if self.total_requests > 0 else 0.0

    @property
    def avg_batch_eval_ms(self) -> float:
        return (self.sum_batch_eval_ms / self.total_batches) if self.total_batches > 0 else 0.0

    def snapshot(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "total_batches": self.total_batches,
            "total_dropped": self.total_dropped,
            "avg_batch_size": round(self.avg_batch_size, 2),
            "avg_wait_time_ms": round(self.avg_wait_time_ms, 3),
            "avg_batch_eval_ms": round(self.avg_batch_eval_ms, 3),
        }


class DynamicMicroBatcher:
    """Async dynamic micro-batcher accumulating concurrent single requests into SIMD batches.

    Parameters
    ----------
    engine : RealtimeInferenceEngine
        Underlying real-time inference engine holding pre-loaded model weights.
    max_batch_size : int, default 64
        Maximum number of items bundled into a single vectorized inference call.
    batch_window_ms : float, default 1.0
        Maximum time window in milliseconds to wait for additional items before executing.
    max_queue_size : int, default 5000
        Maximum queue threshold. Requests exceeding this are dropped with QueueFullError.
    """

    def __init__(
        self,
        engine: Optional[RealtimeInferenceEngine] = None,
        max_batch_size: int = 64,
        batch_window_ms: float = 1.0,
        max_queue_size: int = 5000,
    ) -> None:
        self.engine = engine if engine is not None else RealtimeInferenceEngine()
        self.max_batch_size = max(1, max_batch_size)
        self.batch_window_ms = max(0.1, batch_window_ms)
        self.max_queue_size = max_queue_size

        self._queue: asyncio.Queue[BatchItem] = asyncio.Queue(maxsize=self.max_queue_size)
        self._worker_task: Optional[asyncio.Task[None]] = None
        self._stop_event = asyncio.Event()
        self._stats = BatcherStats()
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def stats(self) -> BatcherStats:
        return self._stats

    async def start(self) -> None:
        """Start the background batch processing loop."""
        if self._is_running:
            return
        self._stop_event.clear()
        self._is_running = True
        self._worker_task = asyncio.create_task(self._batch_worker(), name="micro-batcher-worker")
        logger.info(
            f"DynamicMicroBatcher started (max_batch_size={self.max_batch_size}, "
            f"batch_window={self.batch_window_ms}ms, max_queue={self.max_queue_size})"
        )

    async def stop(self) -> None:
        """Gracefully drain pending requests and shut down the background worker."""
        if not self._is_running:
            return
        self._is_running = False
        self._stop_event.set()

        # Drain any remaining requests in queue
        await self._drain_remaining()

        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        logger.info("DynamicMicroBatcher stopped successfully")

    async def predict(
        self,
        payload: Union[Dict[str, Any], InferenceInputPayload],
    ) -> InferenceOutputPayload:
        """Enqueue payload for micro-batched prediction and await resolution.

        Raises
        ------
        QueueFullError
            If queue capacity is saturated.
        ValidationError
            If the payload violates Pydantic schema constraints.
        """
        # Validate immediately before enqueuing so invalid requests fail fast
        if isinstance(payload, InferenceInputPayload):
            validated = payload
        else:
            validated = InferenceInputPayload.model_validate(payload)

        if not self._is_running:
            # Fallback to direct synchronous execution if batcher is stopped
            return self.engine.predict(validated)

        loop = asyncio.get_running_loop()
        future: asyncio.Future[InferenceOutputPayload] = loop.create_future()
        item = BatchItem(
            payload=validated,
            future=future,
            enqueued_at_ns=time.perf_counter_ns(),
        )

        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            self._stats.total_dropped += 1
            raise QueueFullError("Micro-batcher queue is saturated (backpressure limit reached)")

        return await future

    async def _batch_worker(self) -> None:
        """Continuously aggregate pending items and dispatch vectorized matrix batches."""
        while not self._stop_event.is_set():
            try:
                # Wait for the first request in the batch
                try:
                    first_item = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                batch: List[BatchItem] = [first_item]
                deadline = time.perf_counter() + (self.batch_window_ms / 1000.0)

                # Accumulate subsequent items until max_batch_size or window expiry
                while len(batch) < self.max_batch_size:
                    remaining_timeout = deadline - time.perf_counter()
                    if remaining_timeout <= 0:
                        break
                    try:
                        next_item = await asyncio.wait_for(
                            self._queue.get(),
                            timeout=remaining_timeout,
                        )
                        batch.append(next_item)
                    except asyncio.TimeoutError:
                        break

                # Execute vectorized batch
                self._dispatch_batch(batch)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Error in batcher worker loop: {exc}", exc_info=True)

    def _dispatch_batch(self, batch: List[BatchItem]) -> None:
        """Execute vectorized prediction across items and resolve their futures."""
        if not batch:
            return

        t_start = time.perf_counter_ns()
        now_utc = datetime.now(timezone.utc)
        batch_size = len(batch)

        try:
            # Vectorized feature matrix extraction
            raw_matrix = np.ascontiguousarray(
                [item.payload.features.to_array() for item in batch],
                dtype=np.float32,
            )

            # SIMD in-memory batch prediction (with registry fallback protection)
            model_results = self.engine.registry.predict_batch(raw_matrix)
            t_eval_end = time.perf_counter_ns()
            eval_duration_ms = (t_eval_end - t_start) / 1_000_000.0

            # Resolve individual futures
            for item, res in zip(batch, model_results):
                total_latency_ms = (t_eval_end - item.enqueued_at_ns) / 1_000_000.0
                out = InferenceOutputPayload(
                    request_id=item.payload.request_id,
                    model_version=self.engine.model.version,
                    decision=res.decision,
                    risk_score=res.risk_score,
                    confidence=res.confidence,
                    probabilities=res.probabilities,
                    inference_latency_ms=round(total_latency_ms, 4),
                    processed_at=now_utc,
                )
                if not item.future.cancelled():
                    item.future.set_result(out)

            # Update metrics
            self._stats.total_batches += 1
            self._stats.total_requests += batch_size
            self._stats.sum_batch_sizes += batch_size
            self._stats.sum_batch_eval_ms += eval_duration_ms
            wait_time_total = sum((t_start - it.enqueued_at_ns) / 1_000_000.0 for it in batch)
            self._stats.sum_wait_time_ms += wait_time_total

        except Exception as exc:
            logger.error(f"Failed to execute batch inference: {exc}", exc_info=True)
            for item in batch:
                if not item.future.cancelled():
                    item.future.set_exception(exc)

    async def _drain_remaining(self) -> None:
        """Process any straggler requests remaining in the queue during shutdown."""
        remaining_items: List[BatchItem] = []
        while not self._queue.empty():
            try:
                remaining_items.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        if remaining_items:
            logger.info(f"Draining {len(remaining_items)} remaining requests during batcher shutdown")
            self._dispatch_batch(remaining_items)
