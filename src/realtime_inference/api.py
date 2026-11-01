from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    domain: str = Field(..., description="Domain name to score, e.g. predictive_maintenance")
    payload: Dict[str, Any] = Field(..., description="Feature dictionary for the selected domain")


class BatchPredictionRequest(BaseModel):
    domain: str = Field(..., description="Domain name to score")
    payloads: List[Dict[str, Any]] = Field(..., min_length=1, description="List of payloads")


class PredictionResponse(BaseModel):
    domain: str
    decision: str
    score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    recommended_action: str
    explanation: str
    model_version: str
    model_name: str


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    total: int


class RetrainRequest(BaseModel):
    domain: str = Field(..., description="Domain to retrain")
    new_rows: List[Dict[str, Any]] = Field(default_factory=list, description="New training records that should be merged")
    force: bool = Field(default=False, description="Force retraining even if no new data was supplied")


class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    domains: List[str]
    timestamp: str


class MetricsResponse(BaseModel):
    registry_version: str
    backend: str
    domains: List[str]
    model_versions: Dict[str, str]
    summary: Dict[str, Dict[str, float]]


@dataclass
class DomainModelBundle:
    model: Any
    model_name: str
    labels: List[str]
    feature_names: List[str]
    version: str
    metadata: Dict[str, Any]


class MultiDomainModelRegistry:
    """Loads, evaluates, and retrains multi-domain models using joblib artifacts."""

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

    def __init__(self, artifact_dir: str | None = None, random_seed: int = 42):
        import joblib
        import json
        from pathlib import Path

        self.random_seed = random_seed
        self.artifact_dir = Path(artifact_dir) if artifact_dir else Path(__file__).resolve().parent / "artifacts"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, DomainModelBundle] = {}
        self.model_versions: Dict[str, str] = {}
        self._load_or_train_all()

    def _resolve_domain(self, domain: str) -> str:
        key = domain.strip().lower().replace(" ", "_")
        for valid in self.DOMAIN_ORDER:
            if key == valid:
                return valid
        return key

    def _make_version(self, domain: str) -> str:
        return f"{self._resolve_domain(domain)}-v1.0.0"

    def _artifact_path(self, domain: str) -> Path:
        return self.artifact_dir / self._resolve_domain(domain)

    def _artifact_exists(self, domain: str) -> bool:
        domain_dir = self._artifact_path(domain)
        return (domain_dir / "model.joblib").exists() and (domain_dir / "metadata.json").exists()

    def _generate_training_data(self, domain: str, n_samples: int = 1500):
        domain_name = self._resolve_domain(domain)
        rng = np.random.default_rng(sum(ord(ch) for ch in domain_name) + self.random_seed)

        if domain_name == "predictive_maintenance":
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

        if domain_name == "fraud_anomaly":
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

        if domain_name == "sensor_stream_classification":
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

        if domain_name == "real_time_risk_scoring":
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

        if domain_name == "trading_signal_detection":
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

        if domain_name == "quality_inspection":
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

        raise ValueError(f"Unsupported domain: {domain_name}")

    def _train_model(self, domain: str, X: np.ndarray, y: np.ndarray):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_seed
        )

        rf = RandomForestClassifier(
            n_estimators=350,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=self.random_seed,
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
        logit_score = f1_score(y_test, pipe.predict(X_test), average="weighted")
        return rf if rf_score >= logit_score else pipe

    def _evaluate_model(self, domain: str, model: Any, X: np.ndarray, y: np.ndarray):
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        preds = model.predict(X)
        return {
            "domain": self._resolve_domain(domain),
            "model_name": model.__class__.__name__,
            "version": self._make_version(domain),
            "accuracy": float(accuracy_score(y, preds)),
            "precision": float(precision_score(y, preds, average="weighted", zero_division=0)),
            "recall": float(recall_score(y, preds, average="weighted", zero_division=0)),
            "f1": float(f1_score(y, preds, average="weighted", zero_division=0)),
            "feature_names": self.FEATURE_ORDER[self._resolve_domain(domain)],
            "labels": self.LABELS[self._resolve_domain(domain)],
        }

    def _save_model(self, domain: str, model: Any, metrics: Dict[str, Any]) -> None:
        import joblib
        import json

        domain_dir = self._artifact_path(domain)
        domain_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, domain_dir / "model.joblib")
        (domain_dir / "metadata.json").write_text(json.dumps(metrics, indent=2))

    def _load_model(self, domain: str):
        import joblib
        import json

        domain_dir = self._artifact_path(domain)
        model = joblib.load(domain_dir / "model.joblib")
        metrics = json.loads((domain_dir / "metadata.json").read_text())
        return model, metrics

    def _load_or_train_all(self) -> None:
        for domain in self.DOMAIN_ORDER:
            if self._artifact_exists(domain):
                model, metrics = self._load_model(domain)
                self.models[domain] = DomainModelBundle(
                    model=model,
                    model_name=metrics["model_name"],
                    labels=metrics["labels"],
                    feature_names=metrics["feature_names"],
                    version=metrics["version"],
                    metadata=metrics,
                )
                self.model_versions[domain] = metrics["version"]
            else:
                X, y = self._generate_training_data(domain)
                model = self._train_model(domain, X, y)
                metrics = self._evaluate_model(domain, model, X, y)
                self._save_model(domain, model, metrics)
                self.models[domain] = DomainModelBundle(
                    model=model,
                    model_name=metrics["model_name"],
                    labels=metrics["labels"],
                    feature_names=metrics["feature_names"],
                    version=metrics["version"],
                    metadata=metrics,
                )
                self.model_versions[domain] = metrics["version"]

    def get_domain_metrics(self) -> Dict[str, Dict[str, float]]:
        return {domain: {k: float(v) for k, v in self.models[domain].metadata.items() if k in {"accuracy", "precision", "recall", "f1"}} for domain in self.models}

    def list_domains(self) -> List[str]:
        return list(self.DOMAIN_ORDER)

    def _as_feature_vector(self, domain: str, payload: Dict[str, Any]) -> np.ndarray:
        normalized = self._resolve_domain(domain)
        ordered = []
        for name in self.FEATURE_ORDER[normalized]:
            if name not in payload:
                raise ValueError(f"Missing feature '{name}' for domain '{normalized}'")
            ordered.append(float(payload[name]))
        return np.asarray(ordered, dtype=np.float32).reshape(1, -1)

    def predict(self, domain: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        normalized = self._resolve_domain(domain)
        bundle = self.models[normalized]
        X = self._as_feature_vector(normalized, payload)
        probs = bundle.model.predict_proba(X)[0]
        pred_index = int(np.argmax(probs))
        label = bundle.labels[pred_index]
        confidence = float(probs[pred_index])
        action = self.ACTIONS.get(normalized, {}).get(label, "Review the output")
        return {
            "domain": normalized,
            "decision": label,
            "score": float(confidence),
            "confidence": float(confidence),
            "recommended_action": action,
            "explanation": f"Model '{bundle.model_name}' predicted {label} with confidence {confidence:.2f}.",
            "model_version": bundle.version,
            "model_name": bundle.model_name,
        }

    def predict_batch(self, domain: str, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.predict(domain, row) for row in rows]

    def retrain_domain(self, domain: str, new_rows: Sequence[Dict[str, Any]] | None = None, force: bool = False) -> Dict[str, Any]:
        normalized = self._resolve_domain(domain)
        X, y = self._generate_training_data(normalized)

        if new_rows:
            rows = list(new_rows)
            training_features = self.FEATURE_ORDER[normalized]
            label_lookup = {label.lower(): idx for idx, label in enumerate(self.LABELS[normalized])}
            augment_X = []
            augment_y = []
            for row in rows:
                vector = []
                for feature_name in training_features:
                    if feature_name not in row:
                        raise ValueError(f"Missing feature '{feature_name}' for retraining domain '{normalized}'")
                    vector.append(float(row[feature_name]))
                label_value = row.get("label") or row.get("target") or row.get("decision")
                if label_value is None:
                    raise ValueError(f"Retraining row for {normalized} is missing a target label")
                label_key = str(label_value).strip().upper()
                if label_key not in {label.upper() for label in self.LABELS[normalized]}:
                    raise ValueError(f"Unsupported label '{label_value}' for domain '{normalized}'")
                augment_X.append(vector)
                augment_y.append(label_lookup.get(label_key.lower(), 0))
            if augment_X:
                X = np.vstack([X, np.asarray(augment_X, dtype=np.float32)])
                y = np.concatenate([y, np.asarray(augment_y, dtype=int)])

        model = self._train_model(normalized, X, y)
        metrics = self._evaluate_model(normalized, model, X, y)
        self._save_model(normalized, model, metrics)
        self.models[normalized] = DomainModelBundle(
            model=model,
            model_name=metrics["model_name"],
            labels=metrics["labels"],
            feature_names=metrics["feature_names"],
            version=metrics["version"],
            metadata=metrics,
        )
        self.model_versions[normalized] = metrics["version"]
        return metrics

    def evaluate_all(self) -> Dict[str, Dict[str, float]]:
        results = {}
        for domain in self.DOMAIN_ORDER:
            X, y = self._generate_training_data(domain)
            metrics = self._evaluate_model(domain, self.models[domain].model, X, y)
            results[domain] = {k: float(v) for k, v in metrics.items() if k in {"accuracy", "precision", "recall", "f1"}}
        return results


__all__ = [
    "PredictionRequest",
    "BatchPredictionRequest",
    "PredictionResponse",
    "BatchPredictionResponse",
    "RetrainRequest",
    "HealthResponse",
    "MetricsResponse",
    "MultiDomainModelRegistry",
]

