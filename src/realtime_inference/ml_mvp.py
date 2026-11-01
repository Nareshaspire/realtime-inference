from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from pydantic import Field

from realtime_inference.schemas import StrictBaseSchema


@dataclass
class DomainModelBundle:
    """Container for a trained domain model and metadata."""

    classifier: Any
    labels: List[str]
    feature_names: List[str]
    model_name: str


class MultiDomainMLPrediction(StrictBaseSchema):
    """Unified output schema for trained multi-domain inference."""

    domain: str = Field(..., min_length=1, description="Problem domain being scored.")
    decision: str = Field(..., min_length=1, description="Predicted class.")
    score: float = Field(..., ge=0.0, le=1.0, description="Probability or severity score.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence.")
    recommended_action: str = Field(..., min_length=1, description="Operational action to take.")
    explanation: str = Field(..., min_length=1, description="Human-readable model summary.")
    model_name: str = Field(..., min_length=1, description="Model used for inference.")


class PredictiveMaintenanceFeatures(StrictBaseSchema):
    temperature_c: float = Field(..., ge=0.0, le=200.0)
    vibration_mm_s: float = Field(..., ge=0.0, le=50.0)
    power_draw_kw: float = Field(..., ge=0.0, le=1000.0)
    pressure_bar: float = Field(..., ge=0.0, le=300.0)
    rpm: float = Field(..., ge=0.0, le=20000.0)


class FraudAnomalyFeatures(StrictBaseSchema):
    amount: float = Field(..., gt=0.0, le=1_000_000.0)
    account_age_days: int = Field(..., ge=0, le=36500)
    transaction_velocity_1h: int = Field(..., ge=0, le=500)
    external_risk_score: float = Field(..., ge=0.0, le=1.0)
    device_trust_score: float = Field(..., ge=0.0, le=100.0)
    geo_distance_km: float = Field(..., ge=0.0, le=40075.0)


class SensorStreamFeatures(StrictBaseSchema):
    baseline: float = Field(..., ge=0.0, le=1000.0)
    variance: float = Field(..., ge=0.0, le=1000.0)
    peak_value: float = Field(..., ge=0.0, le=1000.0)
    slope: float = Field(..., ge=-100.0, le=100.0)
    duty_cycle: float = Field(..., ge=0.0, le=1.0)
    error_count: int = Field(..., ge=0, le=1000)


class RealTimeRiskFeatures(StrictBaseSchema):
    request_severity: float = Field(..., ge=0.0, le=1.0)
    latency_ms: float = Field(..., ge=0.0, le=10000.0)
    anomaly_count: int = Field(..., ge=0, le=1000)
    auth_risk: float = Field(..., ge=0.0, le=1.0)
    historical_failures: int = Field(..., ge=0, le=1000)
    revenue_impact: float = Field(..., ge=0.0, le=1_000_000.0)


class TradingSignalFeatures(StrictBaseSchema):
    price_volatility: float = Field(..., ge=0.0, le=1.0)
    momentum: float = Field(..., ge=-1.0, le=1.0)
    spread_ratio: float = Field(..., ge=0.0, le=1.0)
    volume_ratio: float = Field(..., ge=0.0, le=10.0)
    trend_score: float = Field(..., ge=-1.0, le=1.0)
    open_interest: float = Field(..., ge=0.0, le=1.0)


class QualityInspectionFeatures(StrictBaseSchema):
    defect_rate: float = Field(..., ge=0.0, le=1.0)
    temperature_c: float = Field(..., ge=0.0, le=200.0)
    pressure_bar: float = Field(..., ge=0.0, le=300.0)
    vibration_mm_s: float = Field(..., ge=0.0, le=50.0)
    throughput: float = Field(..., ge=0.0, le=10000.0)
    yield_pct: float = Field(..., ge=0.0, le=100.0)


class MultiDomainMLMVP:
    """ML-based version of the multi-domain real-time inference MVP."""

    DOMAIN_ORDER = [
        "predictive_maintenance",
        "fraud_anomaly",
        "sensor_stream_classification",
        "real_time_risk_scoring",
        "trading_signal_detection",
        "quality_inspection",
    ]

    _FEATURES = {
        "predictive_maintenance": [
            "temperature_c",
            "vibration_mm_s",
            "power_draw_kw",
            "pressure_bar",
            "rpm",
        ],
        "fraud_anomaly": [
            "amount",
            "account_age_days",
            "transaction_velocity_1h",
            "external_risk_score",
            "device_trust_score",
            "geo_distance_km",
        ],
        "sensor_stream_classification": [
            "baseline",
            "variance",
            "peak_value",
            "slope",
            "duty_cycle",
            "error_count",
        ],
        "real_time_risk_scoring": [
            "request_severity",
            "latency_ms",
            "anomaly_count",
            "auth_risk",
            "historical_failures",
            "revenue_impact",
        ],
        "trading_signal_detection": [
            "price_volatility",
            "momentum",
            "spread_ratio",
            "volume_ratio",
            "trend_score",
            "open_interest",
        ],
        "quality_inspection": [
            "defect_rate",
            "temperature_c",
            "pressure_bar",
            "vibration_mm_s",
            "throughput",
            "yield_pct",
        ],
    }

    _LABELS = {
        "predictive_maintenance": ["NORMAL", "WARNING", "CRITICAL"],
        "fraud_anomaly": ["ALLOW", "REVIEW", "BLOCK"],
        "sensor_stream_classification": ["NORMAL", "WARNING", "FAULT"],
        "real_time_risk_scoring": ["ALLOW", "REVIEW", "BLOCK"],
        "trading_signal_detection": ["BUY", "HOLD", "SELL"],
        "quality_inspection": ["PASS", "ADJUST", "REJECT"],
    }

    _ACTIONS = {
        "predictive_maintenance": {
            "NORMAL": "Continue operation with monitoring",
            "WARNING": "Schedule maintenance inspection",
            "CRITICAL": "Stop machine and inspect immediately",
        },
        "fraud_anomaly": {
            "ALLOW": "Proceed normally",
            "REVIEW": "Send to manual review",
            "BLOCK": "Reject and escalate to fraud team",
        },
        "sensor_stream_classification": {
            "NORMAL": "Continue monitoring",
            "WARNING": "Inspect sensor calibration",
            "FAULT": "Trigger maintenance or isolate sensor",
        },
        "real_time_risk_scoring": {
            "ALLOW": "Proceed with standard handling",
            "REVIEW": "Queue for human or second-stage validation",
            "BLOCK": "Reject and protect downstream services",
        },
        "trading_signal_detection": {
            "BUY": "Add to long position",
            "HOLD": "Wait for confirmation",
            "SELL": "Reduce or short exposure",
        },
        "quality_inspection": {
            "PASS": "Release to next stage",
            "ADJUST": "Tune process parameters and recheck",
            "REJECT": "Reject the batch and isolate production line",
        },
    }

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed
        self.models: Dict[str, DomainModelBundle] = {}
        self._train_all_models()

    def _train_all_models(self) -> None:
        for domain in self.DOMAIN_ORDER:
            X, y = self._generate_training_data(domain, n_samples=1200, seed=self.random_seed + len(self.models))
            rf = self._train_random_forest(X, y, seed=self.random_seed)
            logit = self._train_logistic_regression(X, y, seed=self.random_seed)

            chosen = self._select_best_model(rf, logit, X, y)
            self.models[domain] = DomainModelBundle(
                classifier=chosen,
                labels=self._LABELS[domain],
                feature_names=self._FEATURES[domain],
                model_name=chosen.__class__.__name__,
            )

    def _select_best_model(self, rf_model: Any, logit_model: Any, X: np.ndarray, y: np.ndarray) -> Any:
        from sklearn.metrics import accuracy_score

        rf_score = accuracy_score(y, rf_model.predict(X))
        logit_score = accuracy_score(y, logit_model.predict(X))
        return rf_model if rf_score >= logit_score else logit_model

    def _train_random_forest(self, X: np.ndarray, y: np.ndarray, seed: int) -> Any:
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=seed,
            class_weight="balanced",
        )
        model.fit(X, y)
        return model

    def _train_logistic_regression(self, X: np.ndarray, y: np.ndarray, seed: int) -> Any:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, multi_class="auto", solver="lbfgs")),
            ]
        )
        pipeline.fit(X, y)
        return pipeline

    def _generate_training_data(self, domain: str, n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)

        if domain == "predictive_maintenance":
            X = np.column_stack(
                [
                    rng.uniform(40.0, 160.0, n_samples),
                    rng.uniform(1.0, 30.0, n_samples),
                    rng.uniform(60.0, 700.0, n_samples),
                    rng.uniform(50.0, 240.0, n_samples),
                    rng.uniform(1000.0, 18000.0, n_samples),
                ]
            )
            risk = (
                0.30 * (X[:, 0] / 160.0)
                + 0.25 * (X[:, 1] / 30.0)
                + 0.20 * (X[:, 2] / 700.0)
                + 0.15 * (X[:, 3] / 240.0)
                + 0.10 * (X[:, 4] / 18000.0)
            )
            y = np.where(risk < 0.35, 0, np.where(risk < 0.70, 1, 2))
            return X.astype(np.float32), y.astype(int)

        if domain == "fraud_anomaly":
            X = np.column_stack(
                [
                    rng.uniform(10.0, 8000.0, n_samples),
                    rng.integers(1, 3650, n_samples),
                    rng.integers(0, 120, n_samples),
                    rng.uniform(0.0, 1.0, n_samples),
                    rng.uniform(0.0, 100.0, n_samples),
                    rng.uniform(0.0, 40000.0, n_samples),
                ]
            )
            risk = (
                0.35 * (X[:, 0] / 8000.0)
                + 0.15 * (1.0 - np.clip(X[:, 1] / 3650.0, 0, 1))
                + 0.20 * (X[:, 2] / 120.0)
                + 0.15 * X[:, 3]
                + 0.10 * (1.0 - np.clip(X[:, 4] / 100.0, 0, 1))
                + 0.05 * (X[:, 5] / 40000.0)
            )
            y = np.where(risk < 0.35, 0, np.where(risk < 0.70, 1, 2))
            return X.astype(np.float32), y.astype(int)

        if domain == "sensor_stream_classification":
            X = np.column_stack(
                [
                    rng.uniform(50.0, 500.0, n_samples),
                    rng.uniform(0.0, 600.0, n_samples),
                    rng.uniform(100.0, 900.0, n_samples),
                    rng.uniform(-50.0, 50.0, n_samples),
                    rng.uniform(0.0, 1.0, n_samples),
                    rng.integers(0, 60, n_samples),
                ]
            )
            risk = (
                0.30 * (X[:, 1] / 600.0)
                + 0.25 * (X[:, 2] / 900.0)
                + 0.20 * (np.abs(X[:, 3]) / 50.0)
                + 0.15 * X[:, 4]
                + 0.10 * (X[:, 5] / 60.0)
            )
            y = np.where(risk < 0.35, 0, np.where(risk < 0.70, 1, 2))
            return X.astype(np.float32), y.astype(int)

        if domain == "real_time_risk_scoring":
            X = np.column_stack(
                [
                    rng.uniform(0.0, 1.0, n_samples),
                    rng.uniform(0.0, 2000.0, n_samples),
                    rng.integers(0, 80, n_samples),
                    rng.uniform(0.0, 1.0, n_samples),
                    rng.integers(0, 25, n_samples),
                    rng.uniform(0.0, 500000.0, n_samples),
                ]
            )
            risk = (
                0.25 * X[:, 0]
                + 0.20 * (X[:, 1] / 2000.0)
                + 0.20 * (X[:, 2] / 80.0)
                + 0.20 * X[:, 3]
                + 0.15 * (X[:, 4] / 25.0)
            )
            y = np.where(risk < 0.35, 0, np.where(risk < 0.70, 1, 2))
            return X.astype(np.float32), y.astype(int)

        if domain == "trading_signal_detection":
            X = np.column_stack(
                [
                    rng.uniform(0.0, 1.0, n_samples),
                    rng.uniform(-1.0, 1.0, n_samples),
                    rng.uniform(0.0, 1.0, n_samples),
                    rng.uniform(0.0, 6.0, n_samples),
                    rng.uniform(-1.0, 1.0, n_samples),
                    rng.uniform(0.0, 1.0, n_samples),
                ]
            )
            trend = 0.30 * X[:, 0] + 0.25 * np.clip((X[:, 1] + 1.0) / 2.0, 0, 1) + 0.20 * X[:, 2] + 0.15 * (X[:, 3] / 6.0) + 0.10 * np.clip((X[:, 4] + 1.0) / 2.0, 0, 1)
            y = np.where(trend < 0.40, 0, np.where(trend < 0.70, 1, 2))
            return X.astype(np.float32), y.astype(int)

        if domain == "quality_inspection":
            X = np.column_stack(
                [
                    rng.uniform(0.0, 1.0, n_samples),
                    rng.uniform(30.0, 180.0, n_samples),
                    rng.uniform(60.0, 250.0, n_samples),
                    rng.uniform(1.0, 30.0, n_samples),
                    rng.uniform(1000.0, 10000.0, n_samples),
                    rng.uniform(20.0, 100.0, n_samples),
                ]
            )
            risk = (
                0.30 * X[:, 0]
                + 0.20 * (X[:, 1] / 180.0)
                + 0.20 * (X[:, 2] / 250.0)
                + 0.15 * (X[:, 3] / 30.0)
                + 0.15 * (1.0 - (X[:, 5] / 100.0))
            )
            y = np.where(risk < 0.35, 0, np.where(risk < 0.70, 1, 2))
            return X.astype(np.float32), y.astype(int)

        raise ValueError(f"Unsupported domain: {domain}")

    def _as_feature_vector(self, domain: str, payload: Dict[str, Any]) -> np.ndarray:
        feature_names = self._FEATURES[domain]
        row = [payload[name] for name in feature_names]
        return np.asarray(row, dtype=np.float32).reshape(1, -1)

    def predict(self, domain: str, payload: Dict[str, Any]) -> MultiDomainMLPrediction:
        normalized_domain = domain.strip().lower().replace(" ", "_")
        if normalized_domain not in self.models:
            raise ValueError(f"Unsupported domain: {domain}")

        model_bundle = self.models[normalized_domain]
        X = self._as_feature_vector(normalized_domain, payload)
        pred = model_bundle.classifier.predict(X)[0]
        probabilities = model_bundle.classifier.predict_proba(X)[0]
        predicted_label = model_bundle.labels[int(pred)]
        confidence = float(np.max(probabilities))
        score = float(np.clip(probabilities[int(pred)], 0.0, 1.0))

        recommended_action = self._ACTIONS[normalized_domain].get(predicted_label, "Review the output")
        return MultiDomainMLPrediction(
            domain=normalized_domain,
            decision=predicted_label,
            score=score,
            confidence=confidence,
            recommended_action=recommended_action,
            explanation=f"Model '{model_bundle.model_name}' predicted {predicted_label} with {confidence:.2f} confidence.",
            model_name=model_bundle.model_name,
        )

    def predict_batch(self, domain: str, rows: Iterable[Dict[str, Any]]) -> List[MultiDomainMLPrediction]:
        values = list(rows)
        if not values:
            return []
        return [self.predict(domain, row) for row in values]


def demo() -> None:
    app = MultiDomainMLMVP()
    examples = {
        "predictive_maintenance": {
            "temperature_c": 118.0,
            "vibration_mm_s": 18.5,
            "power_draw_kw": 480.0,
            "pressure_bar": 165.0,
            "rpm": 13000.0,
        },
        "fraud_anomaly": {
            "amount": 6420.0,
            "account_age_days": 90,
            "transaction_velocity_1h": 42,
            "external_risk_score": 0.84,
            "device_trust_score": 28.0,
            "geo_distance_km": 1800.0,
        },
        "sensor_stream_classification": {
            "baseline": 120.0,
            "variance": 260.0,
            "peak_value": 760.0,
            "slope": 44.0,
            "duty_cycle": 0.82,
            "error_count": 13,
        },
        "real_time_risk_scoring": {
            "request_severity": 0.88,
            "latency_ms": 980.0,
            "anomaly_count": 18,
            "auth_risk": 0.76,
            "historical_failures": 12,
            "revenue_impact": 240000.0,
        },
        "trading_signal_detection": {
            "price_volatility": 0.72,
            "momentum": 0.68,
            "spread_ratio": 0.35,
            "volume_ratio": 2.8,
            "trend_score": 0.63,
            "open_interest": 0.84,
        },
        "quality_inspection": {
            "defect_rate": 0.81,
            "temperature_c": 140.0,
            "pressure_bar": 180.0,
            "vibration_mm_s": 24.0,
            "throughput": 8500.0,
            "yield_pct": 68.0,
        },
    }

    for domain, payload in examples.items():
        result = app.predict(domain, payload)
        print(f"[{domain}] decision={result.decision} score={result.score:.2f} confidence={result.confidence:.2f} model={result.model_name}")
        print(f"  action: {result.recommended_action}")
        print(f"  summary: {result.explanation}")


__all__ = [
    "MultiDomainMLPrediction",
    "MultiDomainMLMVP",
    "PredictiveMaintenanceFeatures",
    "FraudAnomalyFeatures",
    "SensorStreamFeatures",
    "RealTimeRiskFeatures",
    "TradingSignalFeatures",
    "QualityInspectionFeatures",
    "demo",
]

