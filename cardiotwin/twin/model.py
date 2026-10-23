"""Physiological 3-Element Windkessel Model with Baroreflex Coupling.

This module implements:
1. 3-element Windkessel model (Rc, Rp, C) integrated with scipy.integrate.solve_ivp.
2. Inflow ejection waveform (half-sine) driven by HR, SV, and systolic fraction.
3. Beat metrics: Systolic BP (SBP), Diastolic BP (DBP), Mean Arterial Pressure (MAP),
   Pulse Pressure (PP), and full continuous pressure waveform P(t).
4. Realistic default parameters yielding ~120/80 mmHg baseline pressure.
5. Baroreflex coupling mechanism modulating heart rate based on MAP error from setpoint.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.integrate import solve_ivp


@dataclasses.dataclass(frozen=True)
class WindkesselParameters:
    """Parameters for the 3-element Windkessel physiological model.

    Attributes:
        Rp: Peripheral resistance (mmHg * s / ml). Normal: ~0.8 - 1.4
        C: Arterial compliance (ml / mmHg). Normal: ~1.0 - 2.0
        Rc: Aortic characteristic resistance (mmHg * s / ml). Normal: ~0.04 - 0.12
        SV: Stroke volume (ml). Normal: ~50 - 100
        HR: Heart rate (beats per minute). Normal: ~50 - 100
        systolic_fraction: Fraction of cardiac cycle spent in systole. Default: 0.33
    """
    Rp: float = 1.10
    C: float = 1.50
    Rc: float = 0.038
    SV: float = 70.0
    HR: float = 72.0
    systolic_fraction: float = 0.33

    def to_vector(self) -> np.ndarray:
        """Return parameter array [Rp, C, Rc, SV, HR]."""
        return np.array([self.Rp, self.C, self.Rc, self.SV, self.HR], dtype=np.float64)

    @classmethod
    def from_vector(cls, vec: Union[List[float], np.ndarray], systolic_fraction: float = 0.33) -> WindkesselParameters:
        """Create parameters from array [Rp, C, Rc, SV, HR] or [Rp, C, SV, HR]."""
        arr = np.asarray(vec, dtype=np.float64)
        if len(arr) == 4:
            # Assumed [Rp, C, SV, HR] with default Rc=0.038
            return cls(Rp=float(arr[0]), C=float(arr[1]), Rc=0.038, SV=float(arr[2]), HR=float(arr[3]), systolic_fraction=systolic_fraction)
        elif len(arr) == 5:
            return cls(Rp=float(arr[0]), C=float(arr[1]), Rc=float(arr[2]), SV=float(arr[3]), HR=float(arr[4]), systolic_fraction=systolic_fraction)
        else:
            raise ValueError(f"Expected 4 or 5 parameters, got {len(arr)}")


# Realistic physiological parameter bounds [min, max]
PARAMETER_BOUNDS: Dict[str, Tuple[float, float]] = {
    "Rp": (0.35, 2.80),    # peripheral resistance (vasodilation to severe hypertension)
    "C": (0.30, 3.20),     # arterial compliance (stiff elderly to compliant youth)
    "Rc": (0.02, 0.25),    # aortic impedance
    "SV": (25.0, 160.0),   # stroke volume (heart failure to athletic stroke)
    "HR": (38.0, 195.0),   # heart rate (bradycardia to maximal tachycardia)
}


@dataclasses.dataclass
class BeatMetrics:
    """Cardiovascular hemodynamic outputs for a cardiac cycle."""
    sbp: float             # Systolic Blood Pressure (mmHg)
    dbp: float             # Diastolic Blood Pressure (mmHg)
    map_: float            # Mean Arterial Pressure (mmHg)
    pulse_pressure: float  # Pulse Pressure (mmHg)
    hr: float              # Heart rate (bpm)
    sv: float              # Stroke volume (ml)
    co: float              # Cardiac output (L/min)
    time: np.ndarray       # Time vector over single cycle [0, T] in seconds
    pressure: np.ndarray   # Continuous aortic pressure waveform P(t) in mmHg
    flow: np.ndarray       # Continuous aortic inflow waveform Qin(t) in ml/s


def compute_inflow(t_rel: float, T_sys: float, SV: float) -> float:
    """Compute half-sine aortic inflow Qin(t) during cardiac cycle relative to beat onset.

    Args:
        t_rel: Time relative to beat start (seconds), 0 <= t_rel < T.
        T_sys: Systolic duration (seconds).
        SV: Stroke volume (ml).

    Returns:
        Qin: Instantaneous volume inflow (ml/s).
    """
    if 0.0 <= t_rel < T_sys:
        # Half-sine waveform: integral from 0 to T_sys is (2/pi) * Q_peak * T_sys = SV
        Q_peak = (np.pi * SV) / (2.0 * T_sys)
        return float(Q_peak * np.sin(np.pi * t_rel / T_sys))
    return 0.0


def simulate_windkessel_ivp(
    params: WindkesselParameters,
    n_cycles: int = 4,
    points_per_cycle: int = 120,
) -> BeatMetrics:
    """Simulate 3-element Windkessel model using scipy.integrate.solve_ivp.

    Solves the compliance chamber ODE:
        dPd/dt = (Qin(t) - Pd(t)/Rp) / C
    with aortic pressure:
        P(t) = Pd(t) + Rc * Qin(t)

    Multiple cycles are integrated to reach asymptotic limit cycle. The final
    cycle is extracted for report metrics.

    Args:
        params: Windkessel model parameters.
        n_cycles: Number of cycles to run for periodic steady state (default 4).
        points_per_cycle: Number of discrete evaluation points in the final beat.

    Returns:
        BeatMetrics containing SBP, DBP, MAP, PP, and time/waveform series.
    """
    hr = max(30.0, float(params.HR))
    T = 60.0 / hr
    T_sys = float(params.systolic_fraction) * T
    Rp = max(0.2, float(params.Rp))
    C = max(0.2, float(params.C))
    Rc = max(0.01, float(params.Rc))
    SV = max(10.0, float(params.SV))

    tau = Rp * C
    omega = np.pi / T_sys
    Q0 = (np.pi * SV) / (2.0 * T_sys)

    denom = (1.0 / tau) ** 2 + omega ** 2
    A = (Q0 / C) * (1.0 / tau) / denom
    B = -(Q0 / C) * omega / denom

    exp_dia = np.exp(-(T - T_sys) / tau)
    exp_all = np.exp(-T / tau)
    c1 = -B * (1.0 + exp_dia) / max(1e-7, (1.0 - exp_all))
    p0_periodic = B + c1

    def ode_system(t: float, y: np.ndarray) -> np.ndarray:
        qin = compute_inflow(t, T_sys, SV)
        pd = y[0]
        dpd_dt = (qin - (pd / Rp)) / C
        return np.array([dpd_dt], dtype=np.float64)

    t_eval_final = np.linspace(0.0, T, points_per_cycle)

    sol = solve_ivp(
        fun=ode_system,
        t_span=(0.0, T),
        y0=np.array([p0_periodic]),
        method="RK45",
        t_eval=t_eval_final,
        rtol=1e-5,
        atol=1e-5,
    )

    pd_final = sol.y[0]
    t_final = sol.t

    # Calculate aortic pressure: P(t) = Pd(t) + Rc * Qin(t)
    qin_final = np.array([compute_inflow(tb, T_sys, SV) for tb in t_final])
    p_aortic = pd_final + Rc * qin_final

    sbp = float(np.max(p_aortic))
    dbp = float(np.min(p_aortic))
    map_ = float(np.trapezoid(p_aortic, t_final) / T) if hasattr(np, "trapezoid") else float(np.trapz(p_aortic, t_final) / T)
    pp = sbp - dbp
    co = (SV * hr) / 1000.0  # L/min

    return BeatMetrics(
        sbp=sbp,
        dbp=dbp,
        map_=map_,
        pulse_pressure=pp,
        hr=hr,
        sv=SV,
        co=co,
        time=t_final,
        pressure=p_aortic,
        flow=qin_final,
    )


def fast_windkessel_beat(
    Rp: float,
    C: float,
    Rc: float,
    SV: float,
    HR: float,
    systolic_fraction: float = 0.33,
    n_points: int = 100,
) -> Tuple[float, float, float]:
    """Analytical / high-speed evaluation of Windkessel beat metrics (HR, SBP, DBP).

    Ideal for fast iterative evaluation in Kalman filter sigma points.
    During diastole: Qin(t) = 0 => Pd(t) decays exponentially: Pd(t) = Pd(Tsys) * exp(-(t-Tsys)/(Rp*C)).
    During systole: Qin(t) = Q0 * sin(omega * t). The linear 1st-order ODE has exact closed-form solution:
        Pd(t) = [Q0 / C] * [ (1/tau)*sin(omega*t) - omega*cos(omega*t) ] / [ (1/tau)^2 + omega^2 ] + K * exp(-t/tau)
    Enforcing periodicity Pd(0) = Pd(T) gives closed-form exact periodic limit cycle.

    Returns:
        (HR, SBP, DBP)
    """
    hr = max(35.0, float(HR))
    T = 60.0 / hr
    T_sys = systolic_fraction * T
    Rp = max(0.2, float(Rp))
    C = max(0.2, float(C))
    Rc = max(0.01, float(Rc))
    SV = max(10.0, float(SV))

    tau = Rp * C
    omega = np.pi / T_sys
    Q0 = (np.pi * SV) / (2.0 * T_sys)

    # Particular solution coefficients for Pd_p(t) = A*sin(w*t) + B*cos(w*t)
    # Pd' + (1/tau)*Pd = (Q0/C)*sin(w*t)
    denom = (1.0 / tau) ** 2 + omega ** 2
    A = (Q0 / C) * (1.0 / tau) / denom
    B = -(Q0 / C) * omega / denom

    # At t=0: Pd_p(0) = B
    # At t=Tsys: Pd_p(Tsys) = A*sin(pi) + B*cos(pi) = -B
    # In systole: Pd_sys(t) = A*sin(w*t) + B*cos(w*t) + C1*exp(-t/tau)
    # At t=Tsys: Pd_sys(Tsys) = -B + C1*exp(-Tsys/tau)
    # In diastole (t in [Tsys, T]): Pd_dia(t) = Pd_sys(Tsys) * exp(-(t - Tsys)/tau)
    # At t=T: Pd_dia(T) = (-B + C1*exp(-Tsys/tau)) * exp(-(T - Tsys)/tau)
    # Enforcing continuity at beat boundary Pd_sys(0) = Pd_dia(T):
    # B + C1 = (-B + C1*exp(-Tsys/tau)) * exp(-(T - Tsys)/tau)
    # B + C1 = -B*exp(-(T-Tsys)/tau) + C1*exp(-T/tau)
    # C1 * (1 - exp(-T/tau)) = -B * (1 + exp(-(T-Tsys)/tau))
    # C1 = -B * (1 + exp(-(T-Tsys)/tau)) / (1 - exp(-T/tau))

    exp_dia = np.exp(-(T - T_sys) / tau)
    exp_all = np.exp(-T / tau)
    c1 = -B * (1.0 + exp_dia) / max(1e-7, (1.0 - exp_all))

    # Evaluate waveform at discrete samples to extract exact peak (SBP) and trough (DBP)
    t_sys_grid = np.linspace(0.0, T_sys, n_points // 2)
    pd_sys = A * np.sin(omega * t_sys_grid) + B * np.cos(omega * t_sys_grid) + c1 * np.exp(-t_sys_grid / tau)
    qin_sys = Q0 * np.sin(omega * t_sys_grid)
    p_sys = pd_sys + Rc * qin_sys

    pd_end_sys = -B + c1 * np.exp(-T_sys / tau)
    t_dia_grid = np.linspace(T_sys, T, n_points // 2)
    pd_dia = pd_end_sys * np.exp(-(t_dia_grid - T_sys) / tau)
    p_dia = pd_dia  # Qin is 0 during diastole

    p_full = np.concatenate([p_sys, p_dia])
    sbp = float(np.max(p_full))
    dbp = float(np.min(p_full))
    return hr, sbp, dbp


def apply_baroreflex_coupling(
    params: WindkesselParameters,
    target_map: float = 93.3,
    sensitivity_k: float = 0.35,
) -> WindkesselParameters:
    """Apply baroreflex feedback coupling.

    If MAP rises above setpoint, HR reflexively decreases to buffer pressure.
    If MAP falls below setpoint, HR reflexively increases.

    Args:
        params: Current baseline Windkessel parameters.
        target_map: Baroreceptor setpoint in mmHg (default 93.3 mmHg).
        sensitivity_k: Baroreflex gain in bpm / mmHg (default 0.35).

    Returns:
        Updated WindkesselParameters with modulated HR.
    """
    # Evaluate current uncoupled hemodynamics
    current_beat = simulate_windkessel_ivp(params, n_cycles=3, points_per_cycle=60)
    map_error = current_beat.map_ - target_map

    # HR adjustment with physiological saturation
    d_hr = -sensitivity_k * map_error
    new_hr = float(np.clip(params.HR + d_hr, 45.0, 180.0))

    return dataclasses.replace(params, HR=new_hr)
