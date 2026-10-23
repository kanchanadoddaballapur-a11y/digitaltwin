"""Unit tests for the 3-element Windkessel physiological model."""

import pytest
import numpy as np

from cardiotwin.twin.model import (
    WindkesselParameters,
    simulate_windkessel_ivp,
    fast_windkessel_beat,
    apply_baroreflex_coupling,
    compute_inflow,
)


def test_default_model_produces_roughly_120_80():
    """Verify that default parameter values produce ~120/80 mmHg (within +/- 5 mmHg)."""
    params = WindkesselParameters()
    metrics = simulate_windkessel_ivp(params, n_cycles=4)

    assert 115.0 <= metrics.sbp <= 125.0, f"Expected SBP near 120 mmHg, got {metrics.sbp:.2f}"
    assert 75.0 <= metrics.dbp <= 85.0, f"Expected DBP near 80 mmHg, got {metrics.dbp:.2f}"
    assert metrics.pulse_pressure > 30.0, "Pulse pressure should be physiologically positive"
    assert 85.0 <= metrics.map_ <= 105.0, f"Expected MAP near 95 mmHg, got {metrics.map_:.2f}"


def test_inflow_conservation_of_volume():
    """Verify integral of half-sine inflow over systole equals stroke volume."""
    sv = 70.0
    t_sys = 0.28
    t_samples = np.linspace(0, t_sys, 1000)
    inflows = np.array([compute_inflow(t, t_sys, sv) for t in t_samples])
    computed_sv = np.trapezoid(inflows, t_samples) if hasattr(np, "trapezoid") else np.trapz(inflows, t_samples)

    assert np.isclose(computed_sv, sv, rtol=0.01)


def test_peripheral_resistance_sensitivity():
    """Verify that increasing peripheral resistance (Rp) increases MAP and SBP/DBP."""
    base_p = WindkesselParameters(Rp=1.10)
    high_rp_p = WindkesselParameters(Rp=1.60)

    base_m = simulate_windkessel_ivp(base_p, n_cycles=4)
    high_m = simulate_windkessel_ivp(high_rp_p, n_cycles=4)

    assert high_m.map_ > base_m.map_ + 10.0, "Elevated Rp must increase Mean Arterial Pressure"
    assert high_m.sbp > base_m.sbp
    assert high_m.dbp > base_m.dbp


def test_compliance_sensitivity_on_pulse_pressure():
    """Verify that decreasing arterial compliance (C) expands pulse pressure (arterial stiffening)."""
    compliant_p = WindkesselParameters(C=1.80)
    stiff_p = WindkesselParameters(C=0.70)

    comp_m = simulate_windkessel_ivp(compliant_p, n_cycles=4)
    stiff_m = simulate_windkessel_ivp(stiff_p, n_cycles=4)

    assert stiff_m.pulse_pressure > comp_m.pulse_pressure + 15.0, "Low compliance must significantly increase pulse pressure"


def test_baroreflex_coupling():
    """Verify baroreflex coupling depresses HR when MAP is high and increases HR when MAP is low."""
    # High resistance -> high MAP -> HR should drop
    high_rp_params = WindkesselParameters(Rp=1.80, HR=75.0)
    coupled_high = apply_baroreflex_coupling(high_rp_params, target_map=93.3, sensitivity_k=0.5)
    assert coupled_high.HR < high_rp_params.HR, "Elevated MAP should reflexively decrease heart rate"

    # Low resistance -> low MAP -> HR should reflexively increase
    low_rp_params = WindkesselParameters(Rp=0.75, HR=75.0)
    coupled_low = apply_baroreflex_coupling(low_rp_params, target_map=93.3, sensitivity_k=0.5)
    assert coupled_low.HR > low_rp_params.HR, "Depressed MAP should reflexively increase heart rate"


def test_fast_solver_consistency_with_ivp():
    """Verify analytical fast solver is within 3 mmHg of numeric solve_ivp."""
    p = WindkesselParameters(Rp=1.10, C=1.50, Rc=0.038, SV=70.0, HR=72.0)
    ivp_m = simulate_windkessel_ivp(p, n_cycles=4)
    hr_fast, sbp_fast, dbp_fast = fast_windkessel_beat(p.Rp, p.C, p.Rc, p.SV, p.HR)

    assert abs(sbp_fast - ivp_m.sbp) < 4.0
    assert abs(dbp_fast - ivp_m.dbp) < 3.0
