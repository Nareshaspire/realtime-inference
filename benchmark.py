"""High-performance benchmarking script for real-time inference, ONNX Runtime, and schema validation."""

import time
from datetime import datetime, timezone
import numpy as np

from inference_engine import RealtimeInferenceEngine
from models.onnx_wrapper import ONNXInferenceWrapper
from models.wrapper import LightweightInferenceWrapper
from schemas import InferenceInputPayload, TransactionFeatures


def benchmark_suite(iterations: int = 10_000) -> None:
    print("=" * 72)
    print("   REAL-TIME INFERENCE BENCHMARK: LINEAR VS ONNX GBDT VS E2E")
    print("=" * 72)

    linear_model = LightweightInferenceWrapper()
    onnx_model = ONNXInferenceWrapper()
    engine = RealtimeInferenceEngine(model_wrapper=onnx_model)

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

    feature_vector = [125.50, 180.0, 2.0, 0.15, 90.0, 12.0]

    # Warm-up phase
    for _ in range(500):
        linear_model.predict_one(feature_vector)
        onnx_model.predict_one(feature_vector)
        engine.predict(raw_payload)

    # 1. Pure In-Memory Linear Model
    latencies_linear_ns = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        linear_model.predict_one(feature_vector)
        latencies_linear_ns.append(time.perf_counter_ns() - t0)

    lat_linear_us = np.array(latencies_linear_ns) / 1_000.0
    print(f"\n1. In-Memory Linear Model (C-order matrix mul + softmax, {iterations:,} runs):")
    print(f"   p50 Latency: {np.percentile(lat_linear_us, 50):.2f} µs")
    print(f"   p95 Latency: {np.percentile(lat_linear_us, 95):.2f} µs")
    print(f"   p99 Latency: {np.percentile(lat_linear_us, 99):.2f} µs")
    print(f"   Mean:        {np.mean(lat_linear_us):.2f} µs")
    print(f"   Throughput:  {1_000_000.0 / np.mean(lat_linear_us):,.0f} inf/sec")

    # 2. Compiled ONNX GBDT Model (Tree Ensemble)
    latencies_onnx_ns = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        onnx_model.predict_one(feature_vector)
        latencies_onnx_ns.append(time.perf_counter_ns() - t0)

    lat_onnx_us = np.array(latencies_onnx_ns) / 1_000.0
    print(f"\n2. Compiled ONNX Runtime GBDT (C++ tree traversal, {iterations:,} runs):")
    print(f"   p50 Latency: {np.percentile(lat_onnx_us, 50):.2f} µs")
    print(f"   p95 Latency: {np.percentile(lat_onnx_us, 95):.2f} µs")
    print(f"   p99 Latency: {np.percentile(lat_onnx_us, 99):.2f} µs")
    print(f"   Mean:        {np.mean(lat_onnx_us):.2f} µs")
    print(f"   Throughput:  {1_000_000.0 / np.mean(lat_onnx_us):,.0f} inf/sec")

    # 3. Strict Input Validation Only
    latencies_val_ns = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        InferenceInputPayload.model_validate(raw_payload)
        latencies_val_ns.append(time.perf_counter_ns() - t0)

    lat_val_us = np.array(latencies_val_ns) / 1_000.0
    print(f"\n3. Strict Pydantic V2 Input Validation ({iterations:,} runs):")
    print(f"   p50 Latency: {np.percentile(lat_val_us, 50):.2f} µs")
    print(f"   p95 Latency: {np.percentile(lat_val_us, 95):.2f} µs")
    print(f"   p99 Latency: {np.percentile(lat_val_us, 99):.2f} µs")
    print(f"   Mean:        {np.mean(lat_val_us):.2f} µs")

    # 4. Full End-to-End Pipeline with ONNX Model
    latencies_e2e_ns = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        engine.predict(raw_payload)
        latencies_e2e_ns.append(time.perf_counter_ns() - t0)

    lat_e2e_us = np.array(latencies_e2e_ns) / 1_000.0
    print(f"\n4. Full E2E Pipeline (Validate In -> ONNX Infer -> Validate Out, {iterations:,} runs):")
    print(f"   p50 Latency: {np.percentile(lat_e2e_us, 50):.2f} µs")
    print(f"   p95 Latency: {np.percentile(lat_e2e_us, 95):.2f} µs")
    print(f"   p99 Latency: {np.percentile(lat_e2e_us, 99):.2f} µs")
    print(f"   Mean:        {np.mean(lat_e2e_us):.2f} µs")
    print(f"   Throughput:  {1_000_000.0 / np.mean(lat_e2e_us):,.0f} req/sec (single thread)")

    # 5. Batch SIMD Vectorized ONNX Inference (Batch Size = 500)
    batch_size = 500
    features_batch = np.tile(feature_vector, (batch_size, 1)).astype(np.float32)
    batch_latencies_ms = []
    for _ in range(1_000):
        t0 = time.perf_counter_ns()
        onnx_model.predict_batch(features_batch)
        batch_latencies_ms.append((time.perf_counter_ns() - t0) / 1_000_000.0)

    batch_lat_ms = np.array(batch_latencies_ms)
    mean_batch_ms = np.mean(batch_lat_ms)
    batch_throughput = (batch_size * 1_000.0) / mean_batch_ms

    print(f"\n5. Batch Vectorized ONNX SIMD Inference (Batch Size: {batch_size}):")
    print(f"   Mean Batch Duration: {mean_batch_ms:.3f} ms")
    print(f"   Per-Item Latency:    {(mean_batch_ms / batch_size) * 1_000.0:.2f} µs")
    print(f"   Batch Throughput:    {batch_throughput:,.0f} inferences/sec")
    print("=" * 72)


if __name__ == "__main__":
    benchmark_suite()
