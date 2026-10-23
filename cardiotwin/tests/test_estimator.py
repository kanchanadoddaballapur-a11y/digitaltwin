"""Unit tests for the UKF adaptive estimator and divergence engine."""

import pytest
import numpy as np

from cardiotwin.twin.estimator import CardiovascularUKF
from cardiotwin.twin.divergence import DivergenceEngine
from cardiotwin.twin.model import fast_windkessel_beat
from cardiotwin.twin.explain import ExplainabilityEngine
from cardiotwin.twin.whatif import simulate_whatif


def test_estimator_convergence_on_noiseless_data():
    """Verify UKF converges toward true physiological parameters on noiseless observations."""
    true_state = np.array([1.30, 1.20, 75.0, 80.0])  # [Rp, C, SV, HR]
    hr, sbp, dbp = fast_windkessel_beat(true_state[0], true_state[1], 0.038, true_state[2], true_state[3])
    obs = np.array([hr, sbp, dbp])

    # Start with off-target initial state
    ukf = CardiovascularUKF(initial_state=[0.90, 1.60, 60.0, 65.0])

    for _ in range(60):
        est = ukf.step(obs)

    # State should track observations tightly
    pred_y = est.y_pred
    assert np.allclose(pred_y, obs, atol=2.0), f"Observation prediction mismatch: {pred_y} vs {obs}"
    assert np.abs(est.HR - true_state[3]) < 2.0, "HR must converge to true value"


def test_missing_observation_handling():
    """Verify UKF safely handles NaNs without crashing and maintains state."""
    ukf = CardiovascularUKF(initial_state=[1.10, 1.50, 70.0, 72.0])
    # Step with normal data
    est1 = ukf.step(np.array([72.0, 120.0, 80.0]))
    prior_state = est1.state_mean.copy()

    # Step with NaN missing data
    est_nan = ukf.step(np.array([np.nan, 120.0, 80.0]))
    assert est_nan.is_missing_obs is True
    assert np.all(np.isfinite(est_nan.state_mean))

    # Another NaN step
    est_none = ukf.step(None)
    assert est_none.is_missing_obs is True
    assert np.all(np.isfinite(est_none.state_cov))


def test_divergence_score_rises_after_injected_anomaly():
    """Verify normalized innovation divergence score surges when an acute anomaly occurs."""
    ukf = CardiovascularUKF(initial_state=[1.10, 1.50, 70.0, 72.0])
    engine = DivergenceEngine(persistence_threshold=2)

    # Run 10 nominal steps to stabilize
    nominal_obs = np.array([72.0, 120.0, 80.0])
    for _ in range(10):
        est = ukf.step(nominal_obs)
        res = engine.evaluate(est)

    baseline_score = res.raw_divergence_score
    assert baseline_score < 3.0, f"Baseline score should be normal, got {baseline_score}"

    # Sudden acute hypertensive anomaly: SBP jumps to 180, DBP jumps to 115
    anomaly_obs = np.array([72.0, 185.0, 118.0])
    est_anom = ukf.step(anomaly_obs)
    res_anom = engine.evaluate(est_anom)

    assert res_anom.raw_divergence_score > baseline_score + 2.0, (
        f"Divergence score should sharply rise upon anomaly. Was {baseline_score}, now {res_anom.raw_divergence_score}"
    )
    assert res_anom.status in ["Watch", "Significant"]


def test_explainability_identifies_primary_parameter_shift():
    """Verify explainability engine detects which parameter underwent significant shift."""
    ukf = CardiovascularUKF(initial_state=[1.10, 1.50, 70.0, 72.0])
    engine = DivergenceEngine()
    explainer = ExplainabilityEngine({"Rp": 1.10, "C": 1.50, "SV": 70.0, "HR": 72.0})

    # High pressure observations driven by high resistance
    obs_high_svr = np.array([72.0, 160.0, 105.0])
    for _ in range(8):
        est = ukf.step(obs_high_svr)
        div = engine.evaluate(est)

    explanation = explainer.explain(est, div)
    assert len(explanation.shifts) == 4
    assert explanation.summary_text != ""
    assert explanation.confidence_label in ["High", "Moderate", "Low"]


def test_whatif_scenario_simulation():
    """Verify What-If simulations produce expected hemodynamic directional changes."""
    ukf = CardiovascularUKF(initial_state=[1.10, 1.50, 70.0, 72.0])
    est = ukf.step(np.array([72.0, 120.0, 80.0]))

    # Test Exercise scenario (HR up, SBP up, Rp down)
    outcome_ex = simulate_whatif(est, scenario_key="exercise")
    assert outcome_ex.perturbed_metrics.hr > outcome_ex.baseline_metrics.hr
    assert outcome_ex.delta_sbp > 0.0

    # Test Vasodilator scenario (SBP and DBP down)
    outcome_vaso = simulate_whatif(est, scenario_key="vasodilator")
    assert outcome_vaso.delta_sbp < 0.0
    assert outcome_vaso.delta_dbp < 0.0
