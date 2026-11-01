"""Unit tests for in-memory model wrapper (models/wrapper.py)."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pytest

from models.wrapper import LightweightInferenceWrapper


def test_wrapper_loads_weights_in_memory(model_wrapper):
    """Verify weights load correctly and cache properties."""
    assert model_wrapper.is_loaded is True
    assert model_wrapper.version == "v1.0.0"
    assert len(model_wrapper.feature_names) == 6
    assert model_wrapper.classes == ["ALLOW", "REVIEW", "BLOCK"]


def test_weights_in_memory_write_protection(model_wrapper):
    """Ensure in-memory arrays are strictly read-only for thread safety."""
    assert model_wrapper._weights.flags.writeable is False
    assert model_wrapper._bias.flags.writeable is False
    assert model_wrapper._means.flags.writeable is False
    assert model_wrapper._scales.flags.writeable is False

    with pytest.raises(ValueError, match="read-only"):
        model_wrapper._weights[0, 0] = 999.0


def test_predict_one_low_risk(model_wrapper):
    """Verify inference output on typical legitimate transaction features."""
    # amount: 30.0, age: 700, velocity: 1, ext_risk: 0.05, device_trust: 95.0, distance: 2.0
    safe_features = [30.0, 700.0, 1.0, 0.05, 95.0, 2.0]
    result = model_wrapper.predict_one(safe_features)

    assert result.decision == "ALLOW"
    assert result.risk_score < 0.25
    assert result.confidence > 0.70
    assert result.raw_latency_ns > 0
    assert pytest.approx(sum(result.probabilities.values()), rel=1e-4) == 1.0


def test_predict_one_high_risk(model_wrapper):
    """Verify inference output on fraud transaction features."""
    # amount: 5000.0, age: 1, velocity: 50, ext_risk: 0.95, device_trust: 5.0, distance: 9000.0
    fraud_features = [5000.0, 1.0, 50.0, 0.95, 5.0, 9000.0]
    result = model_wrapper.predict_one(fraud_features)

    assert result.decision == "BLOCK"
    assert result.risk_score > 0.75
    assert result.confidence > 0.70
    assert pytest.approx(sum(result.probabilities.values()), rel=1e-4) == 1.0


def test_predict_batch_parity_with_predict_one(model_wrapper):
    """Ensure batch vectorized inference produces identical results to sequential predict_one."""
    vectors = [
        [30.0, 700.0, 1.0, 0.05, 95.0, 2.0],
        [400.0, 60.0, 5.0, 0.40, 60.0, 350.0],
        [5000.0, 1.0, 50.0, 0.95, 5.0, 9000.0],
    ]

    single_results = [model_wrapper.predict_one(v) for v in vectors]
    batch_results = model_wrapper.predict_batch(np.array(vectors, dtype=np.float32))

    assert len(batch_results) == len(single_results)
    for single, batch in zip(single_results, batch_results):
        assert single.decision == batch.decision
        assert pytest.approx(single.risk_score, abs=1e-5) == batch.risk_score
        assert pytest.approx(single.confidence, abs=1e-5) == batch.confidence
        for key in ["ALLOW", "REVIEW", "BLOCK"]:
            assert pytest.approx(single.probabilities[key], abs=1e-5) == batch.probabilities[key]


def test_predict_one_shape_mismatch(model_wrapper):
    """Verify passing incorrect dimension raises clear ValueError."""
    with pytest.raises(ValueError, match="Expected 1D feature array of size 6"):
        model_wrapper.predict_one([1.0, 2.0, 3.0])  # Only 3 features instead of 6


def test_predict_batch_dimension_mismatch(model_wrapper):
    """Verify passing 1D array to predict_batch raises ValueError."""
    with pytest.raises(ValueError, match="Expected 2D matrix"):
        model_wrapper.predict_batch(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


def test_missing_weights_file(tmp_path):
    """Verify FileNotFoundError if weights path does not exist."""
    non_existent = tmp_path / "missing_weights.npz"
    with pytest.raises(FileNotFoundError, match="Model weights file not found"):
        LightweightInferenceWrapper(weights_path=non_existent)


def test_corrupted_weights_nan(tmp_path):
    """Verify loading weights with NaN or inf triggers validation error."""
    corrupt_file = tmp_path / "corrupt.npz"
    np.savez_compressed(
        corrupt_file,
        weights=np.array([[np.nan, 0.0, 0.0]], dtype=np.float32),
        bias=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        feature_names=np.array(["f1"], dtype=str),
        feature_means=np.array([0.0], dtype=np.float32),
        feature_scales=np.array([1.0], dtype=np.float32),
        classes=np.array(["A", "B", "C"], dtype=str),
        version=np.array("v0", dtype=str),
    )
    with pytest.raises(ValueError, match="NaN or Inf"):
        LightweightInferenceWrapper(weights_path=corrupt_file)


def test_concurrent_multithreaded_inference(model_wrapper):
    """Verify thread-safety when multiple worker threads execute predictions simultaneously."""
    sample = [50.0, 200.0, 2.0, 0.15, 85.0, 10.0]
    expected = model_wrapper.predict_one(sample)

    def worker_predict(_):
        return model_wrapper.predict_one(sample)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(worker_predict, range(50)))

    assert len(results) == 50
    for res in results:
        assert res.decision == expected.decision
        assert res.risk_score == expected.risk_score
        assert res.confidence == expected.confidence
