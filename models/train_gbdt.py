"""Train and export a calibrated Gradient Boosted Decision Tree (GBDT) model to ONNX format."""

import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType


def generate_synthetic_training_data(n_samples: int = 5_000, random_state: int = 42):
    """Generate realistic synthetic transaction risk dataset."""
    rng = np.random.default_rng(random_state)

    # 6 features:
    # 0: amount ($)
    # 1: account_age_days
    # 2: transaction_velocity_1h
    # 3: external_risk_score (0.0 to 1.0)
    # 4: device_trust_score (0.0 to 100.0)
    # 5: geo_distance_km (0.0 to 40075.0)

    amount = rng.exponential(scale=120.0, size=n_samples) + 1.0
    amount = np.clip(amount, 1.0, 50_000.0)

    account_age_days = rng.exponential(scale=400.0, size=n_samples)
    account_age_days = np.clip(account_age_days, 0, 3650).astype(int)

    transaction_velocity_1h = rng.poisson(lam=2.0, size=n_samples)
    transaction_velocity_1h = np.clip(transaction_velocity_1h, 0, 50).astype(int)

    external_risk_score = rng.beta(a=1.5, b=5.0, size=n_samples)
    external_risk_score = np.clip(external_risk_score, 0.0, 1.0)

    device_trust_score = rng.beta(a=6.0, b=1.5, size=n_samples) * 100.0
    device_trust_score = np.clip(device_trust_score, 0.0, 100.0)

    geo_distance_km = rng.exponential(scale=50.0, size=n_samples)
    geo_distance_km = np.clip(geo_distance_km, 0.0, 20_000.0)

    X = np.column_stack([
        amount,
        account_age_days.astype(np.float32),
        transaction_velocity_1h.astype(np.float32),
        external_risk_score.astype(np.float32),
        device_trust_score.astype(np.float32),
        geo_distance_km.astype(np.float32),
    ]).astype(np.float32)

    # Risk latent scoring for labels:
    # 0 = ALLOW (legitimate), 1 = REVIEW (suspicious), 2 = BLOCK (fraudulent)
    risk_latent = (
        (amount > 2000.0) * 2.0
        + (account_age_days < 7) * 1.5
        + (transaction_velocity_1h > 10) * 2.5
        + (external_risk_score > 0.7) * 3.0
        + (device_trust_score < 30.0) * 2.0
        + (geo_distance_km > 3000.0) * 1.5
        + rng.normal(0, 0.5, size=n_samples)
    )

    y = np.zeros(n_samples, dtype=np.int64)
    y[risk_latent > 1.2] = 1  # REVIEW
    y[risk_latent > 3.0] = 2  # BLOCK

    return X, y


def train_and_export_onnx(output_path: Path | str) -> Path:
    """Train GradientBoostingClassifier and serialize to optimized ONNX format."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    X, y = generate_synthetic_training_data(n_samples=5_000, random_state=42)

    clf = GradientBoostingClassifier(
        n_estimators=30,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )
    clf.fit(X, y)

    # Convert to ONNX with 2D float tensor input shape (None, 6)
    initial_type = [("float_input", FloatTensorType([None, 6]))]
    onnx_model = to_onnx(
        clf,
        initial_types=initial_type,
        options={id(clf): {"zipmap": False}},
        target_opset=12,
    )

    # Save ONNX binary model
    with open(out_file, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # Save companion metadata JSON
    meta_file = out_file.with_suffix(".json")
    metadata = {
        "model_type": "GradientBoostingClassifier",
        "runtime": "onnxruntime",
        "version": "v2.0.0-gbdt",
        "feature_names": [
            "amount",
            "account_age_days",
            "transaction_velocity_1h",
            "external_risk_score",
            "device_trust_score",
            "geo_distance_km",
        ],
        "classes": ["ALLOW", "REVIEW", "BLOCK"],
        "num_trees": 30,
        "max_depth": 4,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "input_name": "float_input",
    }
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Exported GBDT ONNX model to {out_file} ({out_file.stat().st_size:,} bytes)")
    print(f"Exported metadata to {meta_file}")
    return out_file


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "weights" / "risk_model_gbdt.onnx"
    train_and_export_onnx(target)
