"""High-performance benchmarking script for real-time inference and schema validation."""

import time
from datetime import datetime, timezone
import numpy as np

from inference_engine import RealtimeInferenceEngine
from models.wrapper import LightweightInferenceWrapper
from schemas import InferenceInputPayload, TransactionFeatures


def benchmark_suite(iterations: int = 10_000) -> None:
    print("=" * 70)
    print("   REAL-TIME INFERENCE & STRICT VALIDATION BENCHMARK SUITE")
    print("=" * 70)

    model = LightweightInferenceWrapper()
    engine = RealtimeInferenceEngine(model_wrapper=model)

    raw_payload = {
        "request_id": "bench-req-001",
        "client_id": "latency-tester",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": {
            "amount": 125.50,
            "account_age_days": 180,
            "transaction_velocity_1h": 2,
            "external_risk_score": 0.15,
            "device_trust_score": 90.0,
            "geo_distance_km": 12.0,
        },
    }

    # Warm-up phase
    for _ in range(500):
        engine.predict(raw_payload)

    # 1. Pure In-Memory Model Inference (Zero Validation Overhead)
    feature_vector = [125.50, 180.0, 2.0, 0.15, 90.0, 12.0]
    latencies_model_ns = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        model.predict_one(feature_vector)
        latencies_model_ns.append(time.perf_counter_ns() - t0)

    latencies_model_us = np.array(latencies_model_ns) / 1_000.0

    print(f"\n1. In-Memory Model Only (Vectorized Matrix Mul + Softmax, {iterations:,} runs):")
    print(f"   p50 Latency: {np.percentile(latencies_model_us, 50):.2f} µs")
    print(f"   p95 Latency: {np.percentile(latencies_model_us, 95):.2f} µs")
    print(f"   p99 Latency: {np.percentile(latencies_model_us, 99):.2f} µs")
    print(f"   Mean:        {np.mean(latencies_model_us):.2f} µs")
    print(f"   Throughput:  {1_000_000.0 / np.mean(latencies_model_us):,.0f} inferences/sec (single thread)")

    # 2. Strict Input Validation Only
    latencies_val_ns = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        InferenceInputPayload.model_validate(raw_payload)
        latencies_val_ns.append(time.perf_counter_ns() - t0)

    latencies_val_us = np.array(latencies_val_ns) / 1_000.0
    print(f"\n2. Strict Pydantic V2 Input Validation ({iterations:,} runs):")
    print(f"   p50 Latency: {np.percentile(latencies_val_us, 50):.2f} µs")
    print(f"   p95 Latency: {np.percentile(latencies_val_us, 95):.2f} µs")
    print(f"   p99 Latency: {np.percentile(latencies_val_us, 99):.2f} µs")
    print(f"   Mean:        {np.mean(latencies_val_us):.2f} µs")

    # 3. Full End-to-End Pipeline (Input Validation -> Inference -> Output Validation)
    latencies_e2e_ns = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        engine.predict(raw_payload)
        latencies_e2e_ns.append(time.perf_counter_ns() - t0)

    latencies_e2e_us = np.array(latencies_e2e_ns) / 1_000.0
    print(f"\n3. Full End-to-End Pipeline (Validate In -> Infer -> Validate Out, {iterations:,} runs):")
    print(f"   p50 Latency: {np.percentile(latencies_e2e_us, 50):.2f} µs")
    print(f"   p95 Latency: {np.percentile(latencies_e2e_us, 95):.2f} µs")
    print(f"   p99 Latency: {np.percentile(latencies_e2e_us, 99):.2f} µs")
    print(f"   Mean:        {np.mean(latencies_e2e_us):.2f} µs")
    print(f"   Throughput:  {1_000_000.0 / np.mean(latencies_e2e_us):,.0f} requests/sec (single thread)")

    # 4. Batch SIMD Vectorized Inference (Batch Size = 500)
    batch_size = 500
    features_batch = np.tile(feature_vector, (batch_size, 1)).astype(np.float32)
    batch_latencies_ms = []
    for _ in range(1_000):
        t0 = time.perf_counter_ns()
        model.predict_batch(features_batch)
        batch_latencies_ms.append((time.perf_counter_ns() - t0) / 1_000_000.0)

    batch_latencies_ms = np.array(batch_latencies_ms)
    mean_batch_ms = np.mean(batch_latencies_ms)
    batch_throughput = (batch_size * 1_000.0) / mean_batch_ms

    print(f"\n4. Batch Vectorized SIMD Inference (Batch Size: {batch_size}):")
    print(f"   Mean Batch Duration: {mean_batch_ms:.3f} ms")
    print(f"   Per-Item Latency:    {(mean_batch_ms / batch_size) * 1_000.0:.2f} µs")
    print(f"   Batch Throughput:    {batch_throughput:,.0f} inferences/sec")
    print("=" * 70)


if __name__ == "__main__":
    benchmark_suite()
