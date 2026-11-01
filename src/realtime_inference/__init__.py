from __future__ import annotations

from typing import Any, Dict

from pydantic import Field

from realtime_inference.schemas import StrictBaseSchema


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


class MultiDomainPrediction(StrictBaseSchema):
    """Unified output schema for all example real-time use cases."""

    domain: str = Field(..., min_length=1, description="Problem domain being scored.")
    decision: str = Field(..., min_length=1, description="Classified outcome.")
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized risk or severity score.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the final decision.")
    recommended_action: str = Field(..., min_length=1, description="Operational action to take.")
    explanation: str = Field(..., min_length=1, description="Human-readable summary.")


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


class MultiDomainRealtimeMVP:
    """Simple MVP that demonstrates six real-time problem types under one app pattern."""

    def predict(self, domain: str, payload: Dict[str, Any]) -> MultiDomainPrediction:
        normalized_domain = domain.strip().lower().replace(" ", "_")

        if normalized_domain == "predictive_maintenance":
            return self._predict_predictive_maintenance(payload)
        if normalized_domain in {"fraud", "fraud_anomaly", "anomaly_detection"}:
            return self._predict_fraud_anomaly(payload)
        if normalized_domain in {"sensor_stream", "sensor_stream_classification", "sensor_anomaly"}:
            return self._predict_sensor_stream(payload)
        if normalized_domain in {"real_time_risk", "risk_scoring", "risk"}:
            return self._predict_real_time_risk(payload)
        if normalized_domain in {"trading", "trading_signal_detection", "signal_detection"}:
            return self._predict_trading_signal(payload)
        if normalized_domain in {"quality", "quality_inspection", "production_quality"}:
            return self._predict_quality_inspection(payload)

        raise ValueError(f"Unsupported domain: {domain}")

    def _predict_predictive_maintenance(self, payload: Dict[str, Any]) -> MultiDomainPrediction:
        features = PredictiveMaintenanceFeatures.model_validate(payload)
        temp_score = clamp(features.temperature_c / 120.0)
        vibration_score = clamp(features.vibration_mm_s / 20.0)
        power_score = clamp(features.power_draw_kw / 500.0)
        pressure_score = clamp(features.pressure_bar / 180.0)
        rpm_score = clamp((features.rpm / 12000.0) * 0.8)

        score = clamp(
            0.30 * temp_score + 0.25 * vibration_score + 0.20 * power_score + 0.15 * pressure_score + 0.10 * rpm_score
        )

        if score > 0.8:
            decision = "CRITICAL"
            action = "Stop machine and inspect immediately"
            explanation = "Temperature, vibration, and load all indicate high failure probability in the next minutes."
        elif score > 0.55:
            decision = "WARNING"
            action = "Schedule maintenance inspection"
            explanation = "The machine is trending toward instability and should be reviewed before the next operating cycle."
        else:
            decision = "NORMAL"
            action = "Continue operation with monitoring"
            explanation = "System remains within expected operating bounds and no immediate intervention is required."

        confidence = clamp(0.55 + score * 0.45)
        return MultiDomainPrediction(
            domain="predictive_maintenance",
            decision=decision,
            score=score,
            confidence=confidence,
            recommended_action=action,
            explanation=explanation,
        )

    def _predict_fraud_anomaly(self, payload: Dict[str, Any]) -> MultiDomainPrediction:
        features = FraudAnomalyFeatures.model_validate(payload)
        risk = (
            0.35 * clamp(features.amount / 5000.0)
            + 0.15 * clamp((36500 - features.account_age_days) / 36500.0)
            + 0.20 * clamp(features.transaction_velocity_1h / 50.0)
            + 0.15 * clamp(features.external_risk_score)
            + 0.10 * clamp(1 - (features.device_trust_score / 100.0))
            + 0.05 * clamp(features.geo_distance_km / 2500.0)
        )
        score = clamp(risk)

        if score > 0.8:
            decision = "BLOCK"
            action = "Reject and escalate to fraud team"
            explanation = "High-risk transaction pattern with abnormal velocity and weak trust signals."
        elif score > 0.5:
            decision = "REVIEW"
            action = "Send to manual review"
            explanation = "Behavior is suspicious enough to warrant a second look, but not enough to block immediately."
        else:
            decision = "ALLOW"
            action = "Proceed normally"
            explanation = "Signals remain within acceptable thresholds for normal transaction behavior."

        confidence = clamp(0.5 + score * 0.5)
        return MultiDomainPrediction(
            domain="fraud_anomaly",
            decision=decision,
            score=score,
            confidence=confidence,
            recommended_action=action,
            explanation=explanation,
        )

    def _predict_sensor_stream(self, payload: Dict[str, Any]) -> MultiDomainPrediction:
        features = SensorStreamFeatures.model_validate(payload)
        anomaly_score = (
            0.30 * clamp(features.variance / 300.0)
            + 0.25 * clamp(features.peak_value / 900.0)
            + 0.20 * clamp(abs(features.slope) / 50.0)
            + 0.15 * clamp(features.duty_cycle)
            + 0.10 * clamp(features.error_count / 100.0)
        )
        score = clamp(anomaly_score)

        if score > 0.75:
            decision = "FAULT"
            action = "Trigger maintenance or isolate sensor"
            explanation = "The signal stream shows sustained abnormal deviations and elevated fault indicators."
        elif score > 0.45:
            decision = "WARNING"
            action = "Inspect sensor calibration"
            explanation = "Some deviation is present; a calibration or health check is recommended."
        else:
            decision = "NORMAL"
            action = "Continue monitoring"
            explanation = "Sensor stream is stable and consistent with expected operating behavior."

        confidence = clamp(0.6 + score * 0.4)
        return MultiDomainPrediction(
            domain="sensor_stream_classification",
            decision=decision,
            score=score,
            confidence=confidence,
            recommended_action=action,
            explanation=explanation,
        )

    def _predict_real_time_risk(self, payload: Dict[str, Any]) -> MultiDomainPrediction:
        features = RealTimeRiskFeatures.model_validate(payload)
        score = clamp(
            0.25 * features.request_severity
            + 0.20 * clamp(features.latency_ms / 1000.0)
            + 0.20 * clamp(features.anomaly_count / 50.0)
            + 0.20 * clamp(features.auth_risk)
            + 0.15 * clamp(features.historical_failures / 20.0)
        )

        if score > 0.8:
            decision = "BLOCK"
            action = "Reject and protect downstream services"
            explanation = "This request has unusually high severity and multiple risk indicators."
        elif score > 0.5:
            decision = "REVIEW"
            action = "Queue for human or second-stage validation"
            explanation = "The request is uncertain and should be handled by a higher-friction workflow."
        else:
            decision = "ALLOW"
            action = "Proceed with standard handling"
            explanation = "The request is within acceptable operational thresholds."

        confidence = clamp(0.55 + score * 0.45)
        return MultiDomainPrediction(
            domain="real_time_risk_scoring",
            decision=decision,
            score=score,
            confidence=confidence,
            recommended_action=action,
            explanation=explanation,
        )

    def _predict_trading_signal(self, payload: Dict[str, Any]) -> MultiDomainPrediction:
        features = TradingSignalFeatures.model_validate(payload)
        score = clamp(
            0.25 * clamp(features.price_volatility)
            + 0.25 * clamp((features.momentum + 1.0) / 2.0)
            + 0.20 * clamp(features.spread_ratio)
            + 0.15 * clamp(features.volume_ratio / 5.0)
            + 0.15 * clamp((features.trend_score + 1.0) / 2.0)
        )

        if score > 0.7 and features.momentum > 0.2:
            decision = "BUY"
            action = "Add to long position"
            explanation = "Momentum and trend are aligned, and the market structure supports a bullish trade."
        elif score > 0.7 and features.momentum < -0.2:
            decision = "SELL"
            action = "Reduce or short exposure"
            explanation = "Momentum and trend indicate weakening conditions and downside risk."
        else:
            decision = "HOLD"
            action = "Wait for confirmation"
            explanation = "Market conditions are mixed and do not justify a directional move yet."

        confidence = clamp(0.6 + score * 0.4)
        return MultiDomainPrediction(
            domain="trading_signal_detection",
            decision=decision,
            score=score,
            confidence=confidence,
            recommended_action=action,
            explanation=explanation,
        )

    def _predict_quality_inspection(self, payload: Dict[str, Any]) -> MultiDomainPrediction:
        features = QualityInspectionFeatures.model_validate(payload)
        defect_score = (
            0.30 * clamp(features.defect_rate)
            + 0.20 * clamp(features.temperature_c / 150.0)
            + 0.20 * clamp(features.pressure_bar / 200.0)
            + 0.15 * clamp(features.vibration_mm_s / 30.0)
            + 0.15 * clamp(1.0 - (features.yield_pct / 100.0))
        )
        score = clamp(defect_score)

        if score > 0.8:
            decision = "REJECT"
            action = "Reject the batch and isolate production line"
            explanation = "Material or process quality has drifted too far from the accepted operating window."
        elif score > 0.5:
            decision = "ADJUST"
            action = "Tune process parameters and recheck"
            explanation = "The batch is acceptable but close to specification limits; adjustment is recommended."
        else:
            decision = "PASS"
            action = "Release to next stage"
            explanation = "The batch is within expected quality and process tolerances."

        confidence = clamp(0.6 + score * 0.4)
        return MultiDomainPrediction(
            domain="quality_inspection",
            decision=decision,
            score=score,
            confidence=confidence,
            recommended_action=action,
            explanation=explanation,
        )


def demo() -> None:
    app = MultiDomainRealtimeMVP()
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
        print(f"[{domain}] decision={result.decision} score={result.score:.2f} confidence={result.confidence:.2f}")
        print(f"  action: {result.recommended_action}")
        print(f"  summary: {result.explanation}")


if __name__ == "__main__":
    demo()


__all__ = [
    "MultiDomainPrediction",
    "MultiDomainRealtimeMVP",
    "demo",
]
""