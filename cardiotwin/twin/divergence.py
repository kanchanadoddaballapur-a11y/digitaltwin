"""Digital Twin Divergence Engine & Uncertainty Ensemble.

Computes normalized innovation Mahalanobis distance between real-time observations
and the twin's predictive state, performs statistical hypothesis testing using
chi-square quantiles, applies persistence filtering to suppress false alarms,
runs a 50-member Monte Carlo posterior ensemble for predictive bands, and runs
an IsolationForest secondary anomaly detector for dual-model verification.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest

from .estimator import TwinEstimate
from .model import fast_windkessel_beat


@dataclasses.dataclass
class DivergenceResult:
    """Diagnostic evaluation of digital twin divergence at a given time step."""
    time_step: int
    raw_divergence_score: float     # Mahalanobis distance sqrt(v^T * S^-1 * v)
    smoothed_score: float           # Exponentially filtered divergence score
    status: str                     # "Normal", "Watch", "Significant"
    consecutive_significant: int    # Count of persistent elevated steps
    p_value: float                  # Chi-square p-value for hypothesis test
    signal_contributions: Dict[str, float]  # Contribution of [HR, SBP, DBP] to divergence
    ensemble_bands: Dict[str, Tuple[float, float, float]]  # (5th, median, 95th) for HR, SBP, DBP
    isolation_forest_anomaly: bool  # Secondary ML detector flag
    detector_agreement: bool        # True if UKF & IsolationForest agree


class DivergenceEngine:
    """Tracks divergence between observations and physiological digital twin."""

    def __init__(
        self,
        persistence_threshold: int = 3,
        watch_threshold: float = 2.50,         # ~chi2(3) 90th percentile sqrt
        significant_threshold: float = 3.40,   # ~chi2(3) 99th percentile sqrt
        smoothing_factor: float = 0.25,
        ensemble_size: int = 50,
        random_seed: int = 42,
    ) -> None:
        """Initialize divergence detection engine.

        Args:
            persistence_threshold: Required consecutive steps to flag 'Significant'
            watch_threshold: Lower threshold for warning
            significant_threshold: Upper threshold for critical divergence
            smoothing_factor: EMA factor for smoothed score
            ensemble_size: Monte Carlo sample count for predictive bands
            random_seed: Reproducibility seed
        """
        self.persistence_threshold = persistence_threshold
        self.watch_threshold = watch_threshold
        self.significant_threshold = significant_threshold
        self.smoothing_factor = smoothing_factor
        self.ensemble_size = ensemble_size
        self.rng = np.random.default_rng(random_seed)

        self.consecutive_elevated = 0
        self.smoothed_score: Optional[float] = None
        self.history_scores: List[float] = []

        # Secondary anomaly detector (IsolationForest)
        # Pre-warmed with nominal residual distributions
        self.iso_forest = IsolationForest(
            n_estimators=60,
            contamination=0.08,
            random_state=random_seed,
        )
        self.residual_buffer: List[np.ndarray] = []
        self._init_isolation_forest()

    def _init_isolation_forest(self) -> None:
        """Pre-train Isolation Forest with nominal expected residuals."""
        nominal_residuals = self.rng.normal(
            loc=[0.0, 0.0, 0.0],
            scale=[1.6, 2.8, 2.2],
            size=(250, 3),
        )
        # Features: [res_hr, res_sbp, res_dbp, abs(res_sbp), abs(res_dbp)]
        feats = np.column_stack([
            nominal_residuals,
            np.abs(nominal_residuals[:, 1]),
            np.abs(nominal_residuals[:, 2]),
        ])
        self.iso_forest.fit(feats)

    def compute_ensemble_bands(
        self,
        estimate: TwinEstimate,
        Rc_fixed: float = 0.038,
    ) -> Dict[str, Tuple[float, float, float]]:
        """Sample 50 parameter vectors from posterior covariance to create predictive bands.

        Returns:
            Dict mapping signal name -> (5th_percentile, 50th_percentile, 95th_percentile)
        """
        mean = estimate.state_mean
        cov = estimate.state_cov

        # Symmetrize and regularize covariance
        cov_clean = 0.5 * (cov + cov.T) + 1e-6 * np.eye(len(mean))

        try:
            samples = self.rng.multivariate_normal(mean, cov_clean, size=self.ensemble_size)
        except Exception:
            # Fallback to diagonal perturbation
            stds = np.sqrt(np.maximum(1e-6, np.diag(cov_clean)))
            samples = mean[None, :] + self.rng.normal(0.0, 1.0, size=(self.ensemble_size, len(mean))) * stds[None, :]

        # Clamp parameter samples to physiological domain
        samples[:, 0] = np.clip(samples[:, 0], 0.30, 2.8)   # Rp
        samples[:, 1] = np.clip(samples[:, 1], 0.25, 3.2)   # C
        samples[:, 2] = np.clip(samples[:, 2], 25.0, 160.0) # SV
        samples[:, 3] = np.clip(samples[:, 3], 38.0, 195.0) # HR

        sim_hr = np.zeros(self.ensemble_size)
        sim_sbp = np.zeros(self.ensemble_size)
        sim_dbp = np.zeros(self.ensemble_size)

        for i in range(self.ensemble_size):
            h, s, d = fast_windkessel_beat(
                Rp=samples[i, 0],
                C=samples[i, 1],
                Rc=Rc_fixed,
                SV=samples[i, 2],
                HR=samples[i, 3],
            )
            sim_hr[i] = h
            sim_sbp[i] = s
            sim_dbp[i] = d

        bands = {
            "HR": (float(np.percentile(sim_hr, 5)), float(np.median(sim_hr)), float(np.percentile(sim_hr, 95))),
            "SBP": (float(np.percentile(sim_sbp, 5)), float(np.median(sim_sbp)), float(np.percentile(sim_sbp, 95))),
            "DBP": (float(np.percentile(sim_dbp, 5)), float(np.median(sim_dbp)), float(np.percentile(sim_dbp, 95))),
        }
        return bands

    def evaluate(self, estimate: TwinEstimate, compute_ensemble: bool = False) -> DivergenceResult:
        """Evaluate divergence between observations and digital twin state."""
        if estimate.is_missing_obs:
            # Missing sensor observation: score 0, maintain normal
            return DivergenceResult(
                time_step=estimate.time_step,
                raw_divergence_score=0.0,
                smoothed_score=self.smoothed_score if self.smoothed_score is not None else 0.0,
                status="Normal",
                consecutive_significant=0,
                p_value=1.0,
                signal_contributions={"HR": 0.0, "SBP": 0.0, "DBP": 0.0},
                ensemble_bands=self.compute_ensemble_bands(estimate) if compute_ensemble else {
                    "HR": (estimate.y_pred[0] - 3.0, estimate.y_pred[0], estimate.y_pred[0] + 3.0),
                    "SBP": (estimate.y_pred[1] - 5.0, estimate.y_pred[1], estimate.y_pred[1] + 5.0),
                    "DBP": (estimate.y_pred[2] - 4.0, estimate.y_pred[2], estimate.y_pred[2] + 4.0),
                },
                isolation_forest_anomaly=False,
                detector_agreement=True,
            )

        v = estimate.innovation  # [v_HR, v_SBP, v_DBP]
        S = estimate.innovation_cov  # 3x3 covariance

        # Mahalanobis distance calculation: d_M^2 = v^T * S^-1 * v
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        d_sq = float(np.dot(v, np.dot(S_inv, v)))
        d_sq = max(0.0, d_sq)
        raw_score = float(np.sqrt(d_sq))

        # Chi-square cumulative probability with 3 degrees of freedom
        p_val = float(1.0 - stats.chi2.cdf(d_sq, df=3))

        # Exponential moving average smoothing
        if self.smoothed_score is None:
            self.smoothed_score = raw_score
        else:
            self.smoothed_score = (self.smoothing_factor * raw_score + (1.0 - self.smoothing_factor) * self.smoothed_score)

        self.history_scores.append(raw_score)

        # Per-signal contribution decomposition
        # Component contribution: v_i * (S_inv * v)_i / total
        v_transformed = np.dot(S_inv, v)
        contributions_raw = v * v_transformed
        contributions_abs = np.maximum(0.0, contributions_raw)
        sum_c = np.sum(contributions_abs)
        if sum_c > 1e-6:
            norm_contrib = (contributions_abs / sum_c) * 100.0
        else:
            norm_contrib = np.array([33.33, 33.33, 33.34])

        signal_contrib = {
            "HR": float(norm_contrib[0]),
            "SBP": float(norm_contrib[1]),
            "DBP": float(norm_contrib[2]),
        }

        # Secondary Isolation Forest Anomaly Detection
        feat_vector = np.array([[v[0], v[1], v[2], abs(v[1]), abs(v[2])]])
        iso_pred = self.iso_forest.predict(feat_vector)[0]  # +1 normal, -1 anomaly
        is_iso_anomaly = bool(iso_pred == -1)

        # Periodically retrain Isolation Forest on ongoing background residuals (excluding outliers)
        if raw_score < self.watch_threshold:
            self.residual_buffer.append(v)
            if len(self.residual_buffer) >= 100 and len(self.residual_buffer) % 200 == 0:
                buf_arr = np.array(self.residual_buffer[-100:])
                buf_feats = np.column_stack([buf_arr, np.abs(buf_arr[:, 1]), np.abs(buf_arr[:, 2])])
                try:
                    self.iso_forest.fit(buf_feats)
                except Exception:
                    pass

        # Classification with Persistence Filter
        if raw_score >= self.significant_threshold:
            self.consecutive_elevated += 1
        elif raw_score < self.watch_threshold:
            self.consecutive_elevated = max(0, self.consecutive_elevated - 1)

        # Determine status
        if self.consecutive_elevated >= self.persistence_threshold or (raw_score > 4.5 and self.consecutive_elevated >= 2):
            status = "Significant"
        elif raw_score >= self.watch_threshold:
            status = "Watch"
        else:
            status = "Normal"

        # Detector agreement: Both classify as abnormal or both classify as normal
        is_ukf_anomalous = (status in ["Watch", "Significant"])
        detector_agreement = bool(is_ukf_anomalous == is_iso_anomaly)

        if compute_ensemble:
            bands = self.compute_ensemble_bands(estimate)
        else:
            s_diag = np.sqrt(np.maximum(1e-4, np.diag(S)))
            bands = {
                "HR": (float(estimate.y_pred[0] - 1.96 * s_diag[0]), float(estimate.y_pred[0]), float(estimate.y_pred[0] + 1.96 * s_diag[0])),
                "SBP": (float(estimate.y_pred[1] - 1.96 * s_diag[1]), float(estimate.y_pred[1]), float(estimate.y_pred[1] + 1.96 * s_diag[1])),
                "DBP": (float(estimate.y_pred[2] - 1.96 * s_diag[2]), float(estimate.y_pred[2]), float(estimate.y_pred[2] + 1.96 * s_diag[2])),
            }

        return DivergenceResult(
            time_step=estimate.time_step,
            raw_divergence_score=round(raw_score, 2),
            smoothed_score=round(self.smoothed_score, 2),
            status=status,
            consecutive_significant=self.consecutive_elevated,
            p_value=round(p_val, 4),
            signal_contributions=signal_contrib,
            ensemble_bands=bands,
            isolation_forest_anomaly=is_iso_anomaly,
            detector_agreement=detector_agreement,
        )
