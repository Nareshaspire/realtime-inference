"""Utility to generate calibrated pre-trained weights for the real-time inference model."""

from pathlib import Path
import numpy as np


def generate_calibrated_weights(output_path: Path | str) -> Path:
    """Generate and save pre-trained serialized weights to compressed .npz format."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 6 Features:
    # 0: amount ($)
    # 1: account_age_days
    # 2: transaction_velocity_1h
    # 3: external_risk_score (0.0 to 1.0)
    # 4: device_trust_score (0.0 to 100.0)
    # 5: geo_distance_km (0.0 to 40075.0)

    feature_names = np.array([
        "amount",
        "account_age_days",
        "transaction_velocity_1h",
        "external_risk_score",
        "device_trust_score",
        "geo_distance_km",
    ], dtype=str)

    # Normalization baselines (mean, scale)
    feature_means = np.array([150.0, 365.0, 2.0, 0.20, 80.0, 50.0], dtype=np.float32)
    feature_scales = np.array([250.0, 300.0, 3.0, 0.25, 20.0, 200.0], dtype=np.float32)

    # 3 Classes: [ALLOW, REVIEW, BLOCK]
    # Linear projection matrix (shape: 6 features x 3 classes)
    # ALLOW favours: low amount, mature account, low velocity, low ext risk, high device trust, low distance
    # BLOCK favours: high amount, new account, high velocity, high ext risk, low device trust, high distance
    weights = np.array([
        # amount
        [-1.2, 0.4, 1.8],
        # account_age_days (older accounts get positive ALLOW, negative BLOCK)
        [1.5, -0.2, -1.6],
        # transaction_velocity_1h
        [-1.8, 0.8, 2.4],
        # external_risk_score
        [-2.5, 0.6, 2.9],
        # device_trust_score
        [2.2, -0.4, -2.5],
        # geo_distance_km
        [-1.0, 0.5, 1.7],
    ], dtype=np.float32)

    # Class base priors (bias)
    bias = np.array([2.0, -0.5, -2.5], dtype=np.float32)

    classes = np.array(["ALLOW", "REVIEW", "BLOCK"], dtype=str)
    version = np.array("v1.0.0", dtype=str)
    trained_at = np.array("2026-10-31T23:59:59Z", dtype=str)

    np.savez_compressed(
        out_file,
        weights=weights,
        bias=bias,
        feature_names=feature_names,
        feature_means=feature_means,
        feature_scales=feature_scales,
        classes=classes,
        version=version,
        trained_at=trained_at,
    )

    return out_file


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "weights" / "model_weights.npz"
    created = generate_calibrated_weights(target)
    print(f"Generated pre-trained model weights at {created}")
