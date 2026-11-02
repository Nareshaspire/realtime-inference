"""Unit and concurrency tests for ONNXInferenceWrapper (models/onnx_wrapper.py)."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pytest

from realtime_inference.models.onnx_wrapper import ONNXInferenceWrapper


@pytest.fixture
def onnx_model():
    """Return an initialized ONNXInferenceWrapper."""
    weights_path = Path(__file__).resolve().parent.parent / "models" / "weights" / "risk_model_gbdt.onnx"
    return ONNXInferenceWrapper(model_path=weights_path)


def test_onnx_wrapper_loads_weights(onnx_model):
    """Verify ONNX model session loads and exposes correct metadata."""
    assert onnx_model.is_loaded is True
    assert onnx_model.version == "v2.0.0-gbdt"
    assert len(onnx_model.feature_names) == 6
    assert onnx_model.classes == ["ALLOW", "REVIEW", "BLOCK"]


def test_onnx_predict_one_low_risk(onnx_model):
    """Verify ONNX prediction on typical legitimate transaction."""
    # amount: 35.0, age: 600, velocity: 1, ext_risk: 0.05, device_trust: 95.0, distance: 2.0
    safe_features = [35.0, 600.0, 1.0, 0.05, 95.0, 2.0]
    result = onnx_model.predict_one(safe_features)

    assert result.decision == "ALLOW"
    assert result.risk_score < 0.20
    assert result.confidence > 0.80
    assert result.raw_latency_ns > 0
    assert pytest.approx(sum(result.probabilities.values()), rel=1e-3) == 1.0


def test_onnx_predict_one_high_risk(onnx_model):
    """Verify ONNX prediction on anomalous fraud transaction."""
    # amount: 5000.0, age: 1, velocity: 45, ext_risk: 0.95, device_trust: 5.0, distance: 9000.0
    fraud_features = [5000.0, 1.0, 45.0, 0.95, 5.0, 9000.0]
    result = onnx_model.predict_one(fraud_features)

    assert result.decision == "BLOCK"
    assert result.risk_score > 0.80
    assert result.confidence > 0.80
    assert pytest.approx(sum(result.probabilities.values()), rel=1e-3) == 1.0


def test_onnx_predict_batch_parity_with_predict_one(onnx_model):
    """Verify vectorized ONNX batch prediction produces identical results to single-item predictions."""
    vectors = [
        [35.0, 600.0, 1.0, 0.05, 95.0, 2.0],
        [300.0, 90.0, 4.0, 0.35, 65.0, 250.0],
        [5000.0, 1.0, 45.0, 0.95, 5.0, 9000.0],
    ]
    batch_matrix = np.array(vectors, dtype=np.float32)
    batch_results = onnx_model.predict_batch(batch_matrix)

    assert len(batch_results) == 3
    for vec, batch_res in zip(vectors, batch_results):
        single_res = onnx_model.predict_one(vec)
        assert batch_res.decision == single_res.decision
        assert pytest.approx(batch_res.risk_score, rel=1e-3) == single_res.risk_score
        assert pytest.approx(batch_res.confidence, rel=1e-3) == single_res.confidence


def test_onnx_shape_mismatch_rejection(onnx_model):
    """Verify shape mismatch raises ValueError."""
    with pytest.raises(ValueError, match="Expected 6 features"):
        onnx_model.predict_one([10.0, 20.0, 30.0])  # Only 3 features


def test_onnx_multithreaded_concurrency(onnx_model):
    """Verify thread-safe concurrent execution across threads in ONNX Runtime."""
    safe_vec = [35.0, 600.0, 1.0, 0.05, 95.0, 2.0]

    def worker(_):
        return onnx_model.predict_one(safe_vec)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(worker, range(100)))

    assert len(results) == 100
    for r in results:
        assert r.decision == "ALLOW"
