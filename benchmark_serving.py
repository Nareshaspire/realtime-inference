"""High-concurrency serving benchmark evaluating dynamic micro-batcher vs direct execution."""

import asyncio
import time
from datetime import datetime, timezone
from typing import List
import numpy as np
from httpx import ASGITransport, AsyncClient

from realtime_inference.app import create_app
from realtime_inference.batcher import DynamicMicroBatcher
from realtime_inference.inference_engine import RealtimeInferenceEngine


def make_sample_payload(idx: int) -> dict:
    return {
        "request_id": f"bench-req-{idx:05d}",
        "client_id": "load-tester",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": {
            "amount": 49.99 + (idx % 100),
            "account_age_days": 180 + (idx % 365),
            "transaction_velocity_1h": (idx % 5) + 1,
            "external_risk_score": 0.12,
            "device_trust_score": 92.0,
            "geo_distance_km": 8.5,
        },
    }


async def benchmark_direct_batcher(
    concurrency_levels: List[int] = [10, 50, 100],
    total_requests: int = 1_000,
) -> None:
    print("=" * 72)
    print("   ADAPTIVE MICRO-BATCHER CONCURRENCY BENCHMARK (In-Process)")
    print("=" * 72)

    engine = RealtimeInferenceEngine()
    batcher = DynamicMicroBatcher(
        engine=engine,
        max_batch_size=64,
        batch_window_ms=1.0,
    )
    await batcher.start()

    try:
        # Warm-up
        for i in range(100):
            await batcher.predict(make_sample_payload(i))

        for concurrency in concurrency_levels:
            batcher._stats.total_requests = 0
            batcher._stats.total_batches = 0
            batcher._stats.sum_batch_sizes = 0

            sem = asyncio.Semaphore(concurrency)
            latencies_ns: List[int] = []

            async def worker(idx: int):
                async with sem:
                    payload = make_sample_payload(idx)
                    t0 = time.perf_counter_ns()
                    res = await batcher.predict(payload)
                    t1 = time.perf_counter_ns()
                    latencies_ns.append(t1 - t0)
                    return res

            t_start = time.perf_counter()
            await asyncio.gather(*(worker(i) for i in range(total_requests)))
            total_duration = time.perf_counter() - t_start

            latencies_ms = np.array(latencies_ns) / 1_000_000.0
            throughput = total_requests / total_duration

            print(f"\nConcurrency Level: {concurrency} workers ({total_requests:,} total requests)")
            print(f"   Throughput:      {throughput:,.0f} req/sec")
            print(f"   p50 Latency:     {np.percentile(latencies_ms, 50):.3f} ms")
            print(f"   p95 Latency:     {np.percentile(latencies_ms, 95):.3f} ms")
            print(f"   p99 Latency:     {np.percentile(latencies_ms, 99):.3f} ms")
            print(f"   Mean Latency:    {np.mean(latencies_ms):.3f} ms")
            print(f"   Batches Formed:  {batcher.stats.total_batches}")
            print(f"   Avg Batch Size:  {batcher.stats.avg_batch_size:.1f} items/batch")

    finally:
        await batcher.stop()

    print("=" * 72)


async def benchmark_http_api(total_requests: int = 500, concurrency: int = 25) -> None:
    print("\n" + "=" * 72)
    print("   FASTAPI HTTP SERVING BENCHMARK (ASGI in-memory transport)")
    print("=" * 72)

    app = create_app()
    # Use lifespan context manager for FastAPI
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        # Start lifespan explicitly
        async with app.router.lifespan_context(app):
            # 1. Direct sync single endpoint
            t0 = time.perf_counter()
            for i in range(100):
                resp = await client.post("/predict?use_batcher=false", json=make_sample_payload(i))
                assert resp.status_code == 200
            direct_dur = time.perf_counter() - t0
            print(f"\nSequential Direct HTTP Requests (100 reqs):")
            print(f"   Throughput:  {100 / direct_dur:,.0f} req/sec")
            print(f"   Avg Latency: {(direct_dur / 100) * 1000.0:.3f} ms")

            # 2. Concurrent HTTP requests through Dynamic Micro-Batcher
            sem = asyncio.Semaphore(concurrency)
            latencies_ms: List[float] = []

            async def send_http_req(i: int):
                async with sem:
                    t_req_start = time.perf_counter()
                    resp = await client.post("/predict?use_batcher=true", json=make_sample_payload(i))
                    assert resp.status_code == 200
                    latencies_ms.append((time.perf_counter() - t_req_start) * 1000.0)

            t0 = time.perf_counter()
            await asyncio.gather(*(send_http_req(i) for i in range(total_requests)))
            batch_dur = time.perf_counter() - t0
            throughput = total_requests / batch_dur

            print(f"\nConcurrent HTTP via Micro-Batcher ({total_requests} reqs @ {concurrency} concurrency):")
            print(f"   Throughput:  {throughput:,.0f} req/sec")
            print(f"   p50 Latency: {np.percentile(latencies_ms, 50):.3f} ms")
            print(f"   p95 Latency: {np.percentile(latencies_ms, 95):.3f} ms")
            print(f"   p99 Latency: {np.percentile(latencies_ms, 99):.3f} ms")
            print(f"   Mean Latency:{np.mean(latencies_ms):.3f} ms")

    print("=" * 72)


async def main() -> None:
    await benchmark_direct_batcher(concurrency_levels=[10, 50, 100], total_requests=2_000)
    await benchmark_http_api(total_requests=500, concurrency=25)


if __name__ == "__main__":
    asyncio.run(main())
