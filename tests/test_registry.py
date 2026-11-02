"""Unit tests for ModelRegistry, zero-downtime hot reloading, and circuit breaker fallback."""

import pytest

from realtime_inference.models.fallback import HeuristicFallbackClassifier
from realtime_inference.models.registry import ModelRegistry, ModelReloadError
from realtime_inference.models.wrapper import LightweightInferenceWrapper


def test_registry_initialization():
    """Verify registry initializes with active model and provides info."""
    registry = ModelRegistry()
    info = registry.get_info()

    assert info["is_loaded"] is True
    assert "version" in info
    assert info["fallback_enabled"] is True
    assert info["fallback_triggered_count"] == 0


def test_registry_zero_downtime_hot_reload():
    """Verify hot reloading between linear and ONNX models on the fly."""
    registry = ModelRegistry()

    # Hot reload to linear model
    info1 = registry.hot_reload("linear")
    assert info1["active_model_class"] == "LightweightInferenceWrapper"
    assert info1["version"] == "v1.0.0"

    # Make prediction on linear model
    safe_features = [30.0, 700.0, 1.0, 0.05, 95.0, 2.0]
    res1 = registry.predict_one(safe_features)
    assert res1.decision == "ALLOW"

    # Hot reload back to ONNX GBDT model
    info2 = registry.hot_reload("onnx")
    assert info2["active_model_class"] == "ONNXInferenceWrapper"
    assert info2["version"] == "v2.0.0-gbdt"

    # Make prediction on ONNX model
    res2 = registry.predict_one(safe_features)
    assert res2.decision == "ALLOW"


def test_registry_hot_reload_rejection_preserves_active_model():
    """Verify failed dry-run on corrupt candidate model preserves active model untouched."""
    registry = ModelRegistry()
    initial_version = registry.active_model.version

    # Attempt hot reload with invalid non-existent path
    with pytest.raises(ModelReloadError):
        registry.hot_reload("/tmp/non_existent_model_weights.onnx")

    # Active model must remain completely untouched
    assert registry.active_model.version == initial_version


def test_registry_circuit_breaker_fallback():
    """Verify that when an active model throws an exception, the circuit breaker engages."""
    # Create a faulty model mock that always raises RuntimeError
    class BrokenModel(LightweightInferenceWrapper):
        def predict_one(self, vec):
            raise RuntimeError("Corrupt tensor evaluation failure")

    broken = BrokenModel()
    registry = ModelRegistry(default_model=broken, fallback_enabled=True)

    # Prediction should not crash; it should fall back to HeuristicFallbackClassifier
    safe_features = [30.0, 700.0, 1.0, 0.05, 95.0, 2.0]
    res = registry.predict_one(safe_features)

    assert res.decision == "ALLOW"
    assert registry.fallback_triggered_count == 1

    # Verify high-risk transaction through fallback
    fraud_features = [6000.0, 1.0, 50.0, 0.95, 5.0, 9000.0]
    res_fraud = registry.predict_one(fraud_features)
    assert res_fraud.decision == "BLOCK"
    assert registry.fallback_triggered_count == 2
