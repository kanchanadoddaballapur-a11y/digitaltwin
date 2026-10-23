"""Adaptive Unscented Kalman Filter (UKF) for Cardiovascular Digital Twin.

Estimates unobserved physiological parameters:
    State: x = [Rp, C, SV, HR]^T
from non-invasive hemodynamic observations:
    Observation: y = [HR, SBP, DBP]^T

Features:
- Van der Merwe scaled unscented transform formulation
- Physiological bounds enforcement on sigma points and state updates
- Robust handling of missing observations (skipping measurement update)
- Posterior parameter covariance and confidence interval tracking
- High-efficiency forward observation model via closed-form Windkessel solver
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .model import fast_windkessel_beat


@dataclasses.dataclass
class TwinEstimate:
    """Estimated physiological state with posterior uncertainty at time step k."""
    time_step: int
    state_mean: np.ndarray       # [Rp, C, SV, HR]
    state_cov: np.ndarray        # 4x4 covariance matrix
    y_pred: np.ndarray           # Predicted observations [HR, SBP, DBP]
    innovation: np.ndarray       # y_obs - y_pred (zeros if observation missing)
    innovation_cov: np.ndarray   # S_k (3x3 innovation covariance)
    is_missing_obs: bool         # True if observation was missing/NaN

    @property
    def Rp(self) -> float:
        return float(self.state_mean[0])

    @property
    def C(self) -> float:
        return float(self.state_mean[1])

    @property
    def SV(self) -> float:
        return float(self.state_mean[2])

    @property
    def HR(self) -> float:
        return float(self.state_mean[3])

    def get_std_errors(self) -> Dict[str, float]:
        """Return posterior standard deviations for each state parameter."""
        diag = np.maximum(0.0, np.diag(self.state_cov))
        stds = np.sqrt(diag)
        return {
            "Rp": float(stds[0]),
            "C": float(stds[1]),
            "SV": float(stds[2]),
            "HR": float(stds[3]),
        }

    def get_confidence_intervals(self, z_score: float = 1.96) -> Dict[str, Tuple[float, float]]:
        """Return 95% confidence intervals [lower, upper] for each state parameter."""
        stds = self.get_std_errors()
        return {
            "Rp": (self.Rp - z_score * stds["Rp"], self.Rp + z_score * stds["Rp"]),
            "C": (self.C - z_score * stds["C"], self.C + z_score * stds["C"]),
            "SV": (self.SV - z_score * stds["SV"], self.SV + z_score * stds["SV"]),
            "HR": (self.HR - z_score * stds["HR"], self.HR + z_score * stds["HR"]),
        }


class CardiovascularUKF:
    """Unscented Kalman Filter for physiological state estimation."""

    STATE_NAMES = ["Rp", "C", "SV", "HR"]
    OBS_NAMES = ["HR", "SBP", "DBP"]

    def __init__(
        self,
        initial_state: Optional[Union[List[float], np.ndarray]] = None,
        initial_cov: Optional[np.ndarray] = None,
        q_diag: Optional[Union[List[float], np.ndarray]] = None,
        r_diag: Optional[Union[List[float], np.ndarray]] = None,
        alpha: float = 0.08,
        beta: float = 2.0,
        kappa: float = 0.0,
        Rc_fixed: float = 0.038,
    ) -> None:
        """Initialize Cardiovascular UKF.

        Args:
            initial_state: [Rp, C, SV, HR] (defaults to [1.10, 1.50, 70.0, 72.0])
            initial_cov: Initial 4x4 state covariance matrix
            q_diag: Diagonal process noise variances [q_Rp, q_C, q_SV, q_HR]
            r_diag: Diagonal measurement noise variances [r_HR, r_SBP, r_DBP]
            alpha: Spread of sigma points (default 0.08)
            beta: Distribution parameter (default 2.0 for Gaussian)
            kappa: Secondary scaling parameter (default 0.0)
            Rc_fixed: Fixed aortic resistance (default 0.038 mmHg*s/ml)
        """
        self.n_dim = 4
        self.m_dim = 3
        self.Rc_fixed = Rc_fixed

        # Default initial state: healthy baseline
        if initial_state is None:
            self.x = np.array([1.10, 1.50, 70.0, 72.0], dtype=np.float64)
        else:
            self.x = np.asarray(initial_state, dtype=np.float64).copy()

        # Default initial covariance
        if initial_cov is None:
            self.P = np.diag([0.15**2, 0.25**2, 8.0**2, 5.0**2]).astype(np.float64)
        else:
            self.P = np.asarray(initial_cov, dtype=np.float64).copy()

        # Process noise covariance Q
        if q_diag is None:
            # Subtle physiological drift per minute
            self.Q = np.diag([0.015**2, 0.015**2, 0.65**2, 0.85**2]).astype(np.float64)
        else:
            self.Q = np.diag(q_diag).astype(np.float64)

        # Measurement noise covariance R: sensor uncertainty
        if r_diag is None:
            self.R = np.diag([1.5**2, 2.5**2, 2.0**2]).astype(np.float64)
        else:
            self.R = np.diag(r_diag).astype(np.float64)

        # UKF weights
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambda_param = (alpha ** 2) * (self.n_dim + kappa) - self.n_dim

        # Calculate unscented weights
        n = self.n_dim
        lam = self.lambda_param
        w_m = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)), dtype=np.float64)
        w_c = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)), dtype=np.float64)
        w_m[0] = lam / (n + lam)
        w_c[0] = (lam / (n + lam)) + (1.0 - alpha ** 2 + beta)
        self.W_m = w_m
        self.W_c = w_c

        self.step_count = 0

    def _generate_sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Generate 2n+1 sigma points using scaled unscented transform."""
        n = self.n_dim
        lam = self.lambda_param
        scale = n + lam

        # Ensure positive-definiteness via eigenvalue clipping
        P_sym = 0.5 * (P + P.T)
        evals, evecs = np.linalg.eigh(P_sym)
        evals = np.maximum(evals, 1e-9)
        P_clean = evecs @ np.diag(evals) @ evecs.T

        try:
            L = np.linalg.cholesky(scale * P_clean)
        except np.linalg.LinAlgError:
            L = np.diag(np.sqrt(np.maximum(1e-9, np.diag(scale * P_clean))))

        sigma_points = np.zeros((2 * n + 1, n), dtype=np.float64)
        sigma_points[0] = x

        for i in range(n):
            col = L[:, i]
            sigma_points[i + 1] = x + col
            sigma_points[n + i + 1] = x - col

        # Physiological clamping of sigma points to prevent unphysical ODE parameters
        sigma_points[:, 0] = np.clip(sigma_points[:, 0], 0.25, 3.0)   # Rp
        sigma_points[:, 1] = np.clip(sigma_points[:, 1], 0.20, 3.5)   # C
        sigma_points[:, 2] = np.clip(sigma_points[:, 2], 20.0, 180.0) # SV
        sigma_points[:, 3] = np.clip(sigma_points[:, 3], 35.0, 210.0) # HR

        return sigma_points

    def _observation_model(self, x_vec: np.ndarray) -> np.ndarray:
        """Map parameter vector [Rp, C, SV, HR] to observation [HR, SBP, DBP]."""
        rp = float(x_vec[0])
        c = float(x_vec[1])
        sv = float(x_vec[2])
        hr = float(x_vec[3])

        pred_hr, pred_sbp, pred_dbp = fast_windkessel_beat(
            Rp=rp,
            C=c,
            Rc=self.Rc_fixed,
            SV=sv,
            HR=hr,
        )
        return np.array([pred_hr, pred_sbp, pred_dbp], dtype=np.float64)

    def step(self, y_obs: Optional[Union[List[float], np.ndarray]]) -> TwinEstimate:
        """Execute one UKF predict & update cycle.

        Args:
            y_obs: Observed vector [HR, SBP, DBP]. Can be None or contain NaNs for missing data.

        Returns:
            TwinEstimate with updated state, covariance, predictions, and innovations.
        """
        self.step_count += 1
        n = self.n_dim
        m = self.m_dim

        # 1. TIME UPDATE (PREDICT)
        # Random-walk state propagation: f(x) = x
        sigmas_k_minus_1 = self._generate_sigma_points(self.x, self.P)
        sigmas_pred = sigmas_k_minus_1.copy()

        # Prior state mean
        x_pred = np.sum(self.W_m[:, None] * sigmas_pred, axis=0)

        # Prior covariance
        diff_x = sigmas_pred - x_pred[None, :]
        P_pred = np.zeros((n, n), dtype=np.float64)
        for i in range(2 * n + 1):
            P_pred += self.W_c[i] * np.outer(diff_x[i], diff_x[i])
        P_pred += self.Q

        # Regenerate sigma points around predicted state
        sigmas_k = self._generate_sigma_points(x_pred, P_pred)

        # 2. OBSERVATION PREDICTION
        gammas = np.zeros((2 * n + 1, m), dtype=np.float64)
        for i in range(2 * n + 1):
            gammas[i] = self._observation_model(sigmas_k[i])

        y_pred = np.sum(self.W_m[:, None] * gammas, axis=0)

        # Innovation covariance S_k
        diff_y = gammas - y_pred[None, :]
        S_k = np.zeros((m, m), dtype=np.float64)
        for i in range(2 * n + 1):
            S_k += self.W_c[i] * np.outer(diff_y[i], diff_y[i])
        S_k += self.R

        # Cross-covariance P_xy
        diff_x_k = sigmas_k - x_pred[None, :]
        P_xy = np.zeros((n, m), dtype=np.float64)
        for i in range(2 * n + 1):
            P_xy += self.W_c[i] * np.outer(diff_x_k[i], diff_y[i])

        # 3. MEASUREMENT UPDATE OR MISSING DATA HANDLING
        is_missing = False
        if y_obs is None:
            is_missing = True
        else:
            y_arr = np.asarray(y_obs, dtype=np.float64)
            if len(y_arr) != 3 or np.any(np.isnan(y_arr)) or np.any(np.isinf(y_arr)):
                is_missing = True

        if is_missing:
            # Skip measurement update, keep prior prediction
            self.x = x_pred
            self.P = P_pred
            innovation = np.zeros(m, dtype=np.float64)
        else:
            y_meas = np.asarray(y_obs, dtype=np.float64)
            innovation = y_meas - y_pred

            # Kalman gain K = P_xy * inv(S_k)
            try:
                K = P_xy @ np.linalg.inv(S_k)
            except np.linalg.LinAlgError:
                K = P_xy @ np.linalg.pinv(S_k)

            # State update
            x_updated = x_pred + K @ innovation

            # Physiological clipping on posterior state
            x_updated[0] = np.clip(x_updated[0], 0.35, 2.8)   # Rp
            x_updated[1] = np.clip(x_updated[1], 0.30, 3.2)   # C
            x_updated[2] = np.clip(x_updated[2], 25.0, 160.0) # SV
            x_updated[3] = np.clip(x_updated[3], 38.0, 195.0) # HR
            self.x = x_updated

            # Covariance update: P = P_pred - K * S_k * K^T
            P_updated = P_pred - K @ S_k @ K.T
            # Ensure symmetry and positive-definiteness
            P_updated = 0.5 * (P_updated + P_updated.T)
            np.fill_diagonal(P_updated, np.maximum(1e-7, np.diag(P_updated)))
            self.P = P_updated

        return TwinEstimate(
            time_step=self.step_count,
            state_mean=self.x.copy(),
            state_cov=self.P.copy(),
            y_pred=y_pred.copy(),
            innovation=innovation.copy(),
            innovation_cov=S_k.copy(),
            is_missing_obs=is_missing,
        )
