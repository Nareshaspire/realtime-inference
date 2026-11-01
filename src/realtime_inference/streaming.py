from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelEvaluation:
    domain: str
    model_name: str
    version: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    feature_names: List[str]
    labels: List[str]


@dataclass
class TrainedModelBundle:
    model: Any
    metadata: ModelEvaluation


class MultiDomainModelRegistry:
    """Load trained models from disk or train them if they do not yet exist."""

    DOMAIN_ORDER = [
        "predictive_maintenance",
        "fraud_anomaly",
        "sensor_stream_classification",
        "real_time_risk_scoring",
        "trading_signal_detection",
        "quality_inspection",
    ]

    FEATURE_ORDER = {
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

    LABELS = {
        "predictive_maintenance": ["NORMAL", "WARNING", "CRITICAL"],
        "fraud_anomaly": ["ALLOW", "REVIEW", "BLOCK"],
        "sensor_stream_classification": ["NORMAL", "WARNING", "FAULT"],
        "real_time_risk_scoring": ["ALLOW", "REVIEW", "BLOCK"],
        "trading_signal_detection": ["BUY", "HOLD", "SELL"],
        "quality_inspection": ["PASS", "ADJUST", "REJECT"],
    }

    ACTIONS = {
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

    def __init__(self, artifact_dir: str | None = None, random_seed: int = 42) -> None:
        self.random_seed = random_seed
        self.artifact_dir = Path(artifact_dir) if artifact_dir else Path(__file__).resolve().parent / "artifacts"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, TrainedModelBundle] = {}
        self.versions: Dict[str, str] = {}
        self._load_or_train_all()

    def _load_or_train_all(self) -> None:
        for domain in self.DOMAIN_ORDER:
            if self._artifact_exists(domain):
                artifact = self._load_model_from_disk(domain)
                self.models[domain] = artifact
                self.versions[domain] = artifact.metadata.version
            else:
                model_bundle = self._train_and_save_model(domain)
                self.models[domain] = model_bundle
                self.versions[domain] = model_bundle.metadata.version

    def _artifact_exists(self, domain: str) -> bool:
        domain_dir = self.artifact_dir / domain
        return (domain_dir / "model.joblib").exists() and (domain_dir / "metadata.json").exists()

    def _model_version(self, domain: str) -> str:
        return f"{domain}-v1.0.0"

    def _generate_training_data(self, domain: str, n_samples: int = 1500) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.random_seed + hash(domain) % 1000)

        if domain == "predictive_maintenance":
            X = np.column_stack([
                rng.uniform(40.0, 160.0, n_samples),
                rng.uniform(1.0, 30.0, n_samples),
                rng.uniform(60.0, 700.0, n_samples),
                rng.uniform(50.0, 240.0, n_samples),
                rng.uniform(1000.0, 18000.0, n_samples),
            ])
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
            X = np.column_stack([
                rng.uniform(10.0, 8000.0, n_samples),
                rng.integers(1, 3650, n_samples),
                rng.integers(0, 120, n_samples),
                rng.uniform(0.0, 1.0, n_samples),
                rng.uniform(0.0, 100.0, n_samples),
                rng.uniform(0.0, 40000.0, n_samples),
            ])
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
            X = np.column_stack([
                rng.uniform(50.0, 500.0, n_samples),
                rng.uniform(0.0, 600.0, n_samples),
                rng.uniform(100.0, 900.0, n_samples),
                rng.uniform(-50.0, 50.0, n_samples),
                rng.uniform(0.0, 1.0, n_samples),
                rng.integers(0, 60, n_samples),
            ])
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
            X = np.column_stack([
                rng.uniform(0.0, 1.0, n_samples),
                rng.uniform(0.0, 2000.0, n_samples),
                rng.integers(0, 80, n_samples),
                rng.uniform(0.0, 1.0, n_samples),
                rng.integers(0, 25, n_samples),
                rng.uniform(0.0, 500000.0, n_samples),
            ])
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
            X = np.column_stack([
                rng.uniform(0.0, 1.0, n_samples),
                rng.uniform(-1.0, 1.0, n_samples),
                rng.uniform(0.0, 1.0, n_samples),
                rng.uniform(0.0, 6.0, n_samples),
                rng.uniform(-1.0, 1.0, n_samples),
                rng.uniform(0.0, 1.0, n_samples),
            ])
            trend = (
                0.30 * X[:, 0]
                + 0.25 * np.clip((X[:, 1] + 1.0) / 2.0, 0, 1)
                + 0.20 * X[:, 2]
                + 0.15 * (X[:, 3] / 6.0)
                + 0.10 * np.clip((X[:, 4] + 1.0) / 2.0, 0, 1)
            )
            y = np.where(trend < 0.40, 0, np.where(trend < 0.70, 1, 2))
            return X.astype(np.float32), y.astype(int)

        if domain == "quality_inspection":
            X = np.column_stack([
                rng.uniform(0.0, 1.0, n_samples),
                rng.uniform(30.0, 180.0, n_samples),
                rng.uniform(60.0, 250.0, n_samples),
                rng.uniform(1.0, 30.0, n_samples),
                rng.uniform(1000.0, 10000.0, n_samples),
                rng.uniform(20.0, 100.0, n_samples),
            ])
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

    def _train_model(self, domain: str, X: np.ndarray, y: np.ndarray) -> Any:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=self.random_seed)

        rf = RandomForestClassifier(
            n_estimators=350,
            max_depth=None,
            min_samples_leaf=2,
            random_state=self.random_seed,
            class_weight="balanced",
        )
        rf.fit(X_train, y_train)

        pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, multi_class="auto", solver="lbfgs")),
            ]
        )
        pipe.fit(X_train, y_train)

        rf_score = f1_score(y_test, rf.predict(X_test), average="weighted")
        pipe_score = f1_score(y_test, pipe.predict(X_test), average="weighted")
        best_model = rf if rf_score >= pipe_score else pipe
        return best_model

    def _evaluate_model(self, domain: str, model: Any, X: np.ndarray, y: np.ndarray) -> ModelEvaluation:
        preds = model.predict(X)
        metrics = {
            "accuracy": accuracy_score(y, preds),
            "precision": precision_score(y, preds, average="weighted", zero_division=0),
            "recall": recall_score(y, preds, average="weighted", zero_division=0),
            "f1": f1_score(y, preds, average="weighted", zero_division=0),
        }
        version = self._model_version(domain)
        return ModelEvaluation(
            domain=domain,
            model_name=model.__class__.__name__,
            version=version,
            accuracy=float(metrics["accuracy"]),
            precision=float(metrics["precision"]),
            recall=float(metrics["recall"]),
            f1=float(metrics["f1"]),
            feature_names=self.FEATURE_ORDER[domain],
            labels=self.LABELS[domain],
        )

    def _train_and_save_model(self, domain: str) -> TrainedModelBundle:
        X, y = self._generate_training_data(domain)
        model = self._train_model(domain, X, y)
        evaluation = self._evaluate_model(domain, model, X, y)
        domain_dir = self.artifact_dir / domain
        domain_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, domain_dir / "model.joblib")
        (domain_dir / "metadata.json").write_text(json.dumps({
            "domain": evaluation.domain,
            "model_name": evaluation.model_name,
            "version": evaluation.version,
            "accuracy": evaluation.accuracy,
            "precision": evaluation.precision,
            "recall": evaluation.recall,
            "f1": evaluation.f1,
            "feature_names": evaluation.feature_names,
            "labels": evaluation.labels,
        }, indent=2))
        return TrainedModelBundle(model=model, metadata=evaluation)

    def _load_model_from_disk(self, domain: str) -> TrainedModelBundle:
        domain_dir = self.artifact_dir / domain
        model = joblib.load(domain_dir / "model.joblib")
        metadata_payload = json.loads((domain_dir / "metadata.json").read_text())
        evaluation = ModelEvaluation(
            domain=metadata_payload["domain"],
            model_name=metadata_payload["model_name"],
            version=metadata_payload["version"],
            accuracy=float(metadata_payload["accuracy"]),
            precision=float(metadata_payload["precision"]),
            recall=float(metadata_payload["recall"]),
            f1=float(metadata_payload["f1"]),
            feature_names=list(metadata_payload["feature_names"]),
            labels=list(metadata_payload["labels"]),
        )
        return TrainedModelBundle(model=model, metadata=evaluation)

    def _resolve_domain(self, domain: str) -> str:
        key = domain.strip().lower().replace(" ", "_")
        for valid in self.DOMAIN_ORDER:
            if key == valid:
                return valid
        return key

    def _as_feature_vector(self, domain: str, payload: Dict[str, Any]) -> np.ndarray:
        ordered = []
        for name in self.FEATURE_ORDER[domain]:
            if name not in payload:
                raise ValueError(f"Missing required feature '{name}' for domain '{domain}'")
            ordered.append(float(payload[name]))
        return np.asarray(ordered, dtype=np.float32).reshape(1, -1)

    def get_model_version(self, domain: str) -> str:
        normalized_domain = self._resolve_domain(domain)
        return self.versions.get(normalized_domain, self._model_version(normalized_domain))

    def predict(self, domain: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        normalized_domain = self._resolve_domain(domain)
        if normalized_domain not in self.models:
            raise ValueError(f"Unsupported domain: {domain}")

        model_bundle = self.models[normalized_domain]
        X = self._as_feature_vector(normalized_domain, payload)
        probabilities = model_bundle.model.predict_proba(X)[0]
        pred_index = int(np.argmax(probabilities))
        label = model_bundle.metadata.labels[pred_index]
        confidence = float(probabilities[pred_index])
        score = float(probabilities[pred_index])
        action = self.ACTIONS.get(normalized_domain, {}).get(label, "Review the output")

        return {
            "domain": normalized_domain,
            "decision": label,
            "score": round(score, 4),
            "confidence": round(confidence, 4),
            "recommended_action": action,
            "explanation": f"Model '{model_bundle.metadata.model_name}' predicted {label} with confidence {confidence:.2f}.",
            "model_version": model_bundle.metadata.version,
            "model_name": model_bundle.metadata.model_name,
        }

    def predict_batch(self, domain: str, rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.predict(domain, row) for row in rows]

    def list_available_domains(self) -> List[str]:
        return list(self.DOMAIN_ORDER)

    def evaluate_all(self) -> Dict[str, ModelEvaluation]:
        report: Dict[str, ModelEvaluation] = {}
        for domain in self.DOMAIN_ORDER:
            X, y = self._generate_training_data(domain)
            bundle = self.models[domain]
            report[domain] = self._evaluate_model(domain, bundle.model, X, y)
        return report


__all__ = [
    "MultiDomainModelRegistry",
    "ModelEvaluation",
]

