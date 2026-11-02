"""Unit and concurrency tests for DynamicMicroBatcher (batcher.py)."""

import asyncio
from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from batcher import DynamicMicroBatcher, QueueFullError
from inference_engine import RealtimeInferenceEngine
from schemas import InferenceInputPayload


@pytest.fixture
def batcher():
    """Fixture providing a fresh DynamicMicroBatcher with short batch window for testing."""
    engine = RealtimeInferenceEngine()
    b = DynamicMicroBatcher(
        engine=engine,
        max_batch_size=16,
        batch_window_ms=5.0,
        max_queue_size=200,
    )
    return b


@pytest.mark.asyncio
async def test_batcher_lifecycle(batcher):
    """Test start and graceful stop of the micro-batcher."""
    assert not batcher.is_running
    await batcher.start()
    assert batcher.is_running
    await batcher.stop()
    assert not batcher.is_running


@pytest.mark.asyncio
async def test_batcher_single_predict(batcher, valid_low_risk_payload_dict):
    """Test single item prediction through running batcher."""
    await batcher.start()
    try:
        output = await batcher.predict(valid_low_risk_payload_dict)
        assert output.request_id == "req-safe-001"
        assert output.decision == "ALLOW"
        assert output.risk_score < 0.25
        assert batcher.stats.total_requests == 1
        assert batcher.stats.total_batches == 1
    finally:
        await batcher.stop()


@pytest.mark.asyncio
async def test_batcher_concurrent_requests_batching(batcher):
    """Verify concurrent requests are grouped into vectorized batches."""
    await batcher.start()
    num_requests = 30

    async def send_req(i: int):
        payload = {
            "request_id": f"req-concurrent-{i:03d}",
            "client_id": "pytest-worker",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "features": {
                "amount": 20.0 + i,
                "account_age_days": 100 + i,
                "transaction_velocity_1h": 1,
                "external_risk_score": 0.05,
                "device_trust_score": 90.0,
                "geo_distance_km": 5.0,
            },
        }
        return await batcher.predict(payload)

    try:
        results = await asyncio.gather(*(send_req(i) for i in range(num_requests)))
        assert len(results) == num_requests
        for i, res in enumerate(results):
            assert res.request_id == f"req-concurrent-{i:03d}"
            assert res.decision in ["ALLOW", "REVIEW", "BLOCK"]

        # Ensure batching occurred (batches should be fewer than total requests)
        assert batcher.stats.total_requests == num_requests
        assert batcher.stats.total_batches < num_requests
        assert batcher.stats.avg_batch_size > 1.0
    finally:
        await batcher.stop()


@pytest.mark.asyncio
async def test_batcher_queue_saturation_backpressure():
    """Verify QueueFullError is raised when the queue capacity is exceeded."""
    engine = RealtimeInferenceEngine()
    # Create batcher with queue capacity of 2 that does NOT start worker
    tight_batcher = DynamicMicroBatcher(engine=engine, max_queue_size=2)
    tight_batcher._is_running = True  # mark running so it enqueues without processing

    payload = {
        "request_id": "test-q",
        "client_id": "test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": {
            "amount": 50.0,
            "account_age_days": 100,
            "transaction_velocity_1h": 1,
            "external_risk_score": 0.1,
            "device_trust_score": 90.0,
            "geo_distance_km": 5.0,
        },
    }

    # First 2 enqueues succeed
    t1 = asyncio.create_task(tight_batcher.predict(payload))
    t2 = asyncio.create_task(tight_batcher.predict(payload))
    await asyncio.sleep(0.01)

    # 3rd enqueue must trigger QueueFullError
    with pytest.raises(QueueFullError):
        await tight_batcher.predict(payload)

    t1.cancel()
    t2.cancel()


@pytest.mark.asyncio
async def test_batcher_rejects_invalid_schema_fast(batcher):
    """Verify invalid payloads fail validation immediately without entering queue."""
    await batcher.start()
    try:
        invalid_payload = {
            "request_id": "bad-req",
            "client_id": "test",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "features": {
                "amount": -50.0,  # Negative amount violates schema constraint
                "account_age_days": 100,
                "transaction_velocity_1h": 1,
                "external_risk_score": 0.1,
                "device_trust_score": 90.0,
                "geo_distance_km": 5.0,
            },
        }
        with pytest.raises(ValidationError):
            await batcher.predict(invalid_payload)

        # Ensure queue remained empty
        assert batcher.stats.total_requests == 0
    finally:
        await batcher.stop()
