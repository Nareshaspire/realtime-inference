"""Thread-safe model registry supporting zero-downtime hot reloading and circuit breaker fallback."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from realtime_inference.models.base import BaseInferenceModel, DecisionType, ModelPredictionResult
from realtime_inference.models.fallback import HeuristicFallbackClassifier
from realtime_inference.models.onnx_wrapper import ONNXInferenceWrapper
from realtime_inference.models.wrapper import LightweightInferenceWrapper

logger = logging.getLogger("realtime_inference.registry")


class ModelReloadError(Exception):
    """Raised when candidate model fails validation dry-run during hot-reload."""
    pass


class ModelRegistry:
    """Thread-safe model container enabling atomic hot-swapping and circuit breaker protection.

    Parameters
    ----------
    default_model : Optional[BaseInferenceModel]
        Initial active model. Defaults to ONNXInferenceWrapper if available, else LightweightInferenceWrapper.
    fallback_enabled : bool, default True
        Whether to route requests to HeuristicFallbackClassifier if model execution faults.
    """

    def __init__(
        self,
        default_model: Optional[BaseInferenceModel] = None,
        fallback_enabled: bool = True,
    ) -> None:
        self._lock = threading.RLock()
        self._fallback = HeuristicFallbackClassifier()
        self._fallback_enabled = fallback_enabled
        self._fallback_triggered_count = 0

        if default_model is not None:
            self._active_model = default_model
        else:
            # Default to ONNX model if present, otherwise fallback to in-memory linear wrapper
            onnx_path = Path(__file__).resolve().parent / "weights" / "risk_model_gbdt.onnx"
            if onnx_path.is_file():
                self._active_model = ONNXInferenceWrapper(model_path=onnx_path)
            else:
                self._active_model = LightweightInferenceWrapper()

    @property
    def active_model(self) -> BaseInferenceModel:
        with self._lock:
            return self._active_model

    @property
    def fallback_enabled(self) -> bool:
        return self._fallback_enabled

    @fallback_enabled.setter
    def fallback_enabled(self, val: bool) -> None:
        self._fallback_enabled = bool(val)

    @property
    def fallback_triggered_count(self) -> int:
        return self._fallback_triggered_count

    def get_info(self) -> Dict[str, Any]:
        """Return diagnostic metadata about active model and circuit breaker state."""
        with self._lock:
            return {
                "active_model_class": self._active_model.__class__.__name__,
                "version": self._active_model.version,
                "is_loaded": self._active_model.is_loaded,
                "classes": self._active_model.classes,
                "feature_names": self._active_model.feature_names,
                "fallback_enabled": self._fallback_enabled,
                "fallback_triggered_count": self._fallback_triggered_count,
            }

    def hot_reload(self, target: Union[str, Path, BaseInferenceModel]) -> Dict[str, Any]:
        """Atomically hot-swap active model with candidate after dry-run verification.

        Parameters
        ----------
        target : str, Path, or BaseInferenceModel
            Model instance, path to .onnx/.npz file, or alias ("onnx", "linear", "fallback").

        Returns
        -------
        Dict[str, Any]
            Updated model status information.

        Raises
        ------
        ModelReloadError
            If candidate model fails load or dry-run inference.
        """
        # Step 1: Instantiate candidate model without holding lock
        try:
            if isinstance(target, BaseInferenceModel):
                candidate = target
            elif isinstance(target, (str, Path)):
                target_str = str(target).lower()
                if target_str in ("onnx", "gbdt"):
                    candidate = ONNXInferenceWrapper()
                elif target_str in ("linear", "numpy"):
                    candidate = LightweightInferenceWrapper()
                elif target_str == "fallback":
                    candidate = self._fallback
                else:
                    path_obj = Path(target)
                    if path_obj.suffix == ".onnx":
                        candidate = ONNXInferenceWrapper(model_path=path_obj)
                    elif path_obj.suffix == ".npz":
                        candidate = LightweightInferenceWrapper(weights_path=path_obj)
                    else:
                        raise ValueError(f"Unsupported model specification: {target}")
            else:
                raise TypeError(f"Invalid target type for hot_reload: {type(target)}")

            if not candidate.is_loaded:
                candidate.load_weights()

            # Step 2: Safety dry-run inference
            test_vector = [50.0, 180.0, 2.0, 0.15, 90.0, 10.0]
            dry_run_res = candidate.predict_one(test_vector)
            if dry_run_res.decision not in candidate.classes:
                raise ValueError(f"Dry run produced invalid decision: {dry_run_res.decision}")

        except Exception as exc:
            logger.error(f"Hot-reload failed safety verification: {exc}", exc_info=True)
            raise ModelReloadError(f"Candidate model failed verification: {exc}") from exc

        # Step 3: Atomic pointer swap under lock
        with self._lock:
            old_version = self._active_model.version
            self._active_model = candidate
            logger.info(f"Zero-downtime hot-reload completed: {old_version} -> {candidate.version}")

        return self.get_info()

    def predict_one(self, feature_vector: Union[List[float], np.ndarray]) -> ModelPredictionResult:
        """Route prediction to active model with automatic fallback handling."""
        with self._lock:
            model = self._active_model

        try:
            return model.predict_one(feature_vector)
        except Exception as exc:
            if self._fallback_enabled:
                self._fallback_triggered_count += 1
                logger.warning(
                    f"Active model '{model.version}' faulted ({exc}); engaging heuristic fallback circuit breaker"
                )
                return self._fallback.predict_one(feature_vector)
            raise

    def predict_batch(self, feature_matrix: np.ndarray) -> List[ModelPredictionResult]:
        """Route batch prediction to active model with automatic fallback handling."""
        with self._lock:
            model = self._active_model

        try:
            return model.predict_batch(feature_matrix)
        except Exception as exc:
            if self._fallback_enabled:
                self._fallback_triggered_count += len(feature_matrix)
                logger.warning(
                    f"Active model '{model.version}' faulted in batch ({exc}); engaging heuristic fallback"
                )
                return self._fallback.predict_batch(feature_matrix)
            raise
