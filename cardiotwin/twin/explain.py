"""Physiological Explainability Engine for CardioTwin X.

Analyzes divergence events by comparing posterior digital twin state estimates
against the patient's established baseline, calculating normalized parameter
shifts (Z-scores and percentage deltas), decomposing signal innovations,
and synthesizing clinical-grade physiological explanations.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .estimator import TwinEstimate
from .divergence import DivergenceResult


PARAMETER_METADATA = {
    "Rp": {
        "name": "Peripheral Resistance (Rp)",
        "unit": "mmHg·s/ml",
        "physiology": "Arteriolar vascular tone / Systemic Vascular Resistance (SVR)",
        "increase_causes": "Acute vasoconstriction, sympathetic activation, cold exposure, high sodium, or vasopressor effect",
        "decrease_causes": "Arteriolar vasodilation, exercise hyperemia, heat, or vasodilatory medication",
    },
    "C": {
        "name": "Arterial Compliance (C)",
        "unit": "ml/mmHg",
        "physiology": "Elastic buffering capacity of the aorta and large conduit arteries",
        "increase_causes": "Acute vascular destiffening or measurement fluctuation",
        "decrease_causes": "Arterial stiffening, structural remodeling, loss of elastin, or hypertensive stress",
    },
    "SV": {
        "name": "Stroke Volume (SV)",
        "unit": "ml",
        "physiology": "Blood volume ejected per ventricular contraction",
        "increase_causes": "Elevated preload, inotropic enhancement, or post-exercise filling",
        "decrease_causes": "Hypovolemia, dehydration, impaired venous return, or shortened diastolic filling during tachycardia",
    },
    "HR": {
        "name": "Heart Rate (HR)",
        "unit": "bpm",
        "physiology": "SA nodal chronotropy and autonomic balance",
        "increase_causes": "Sympathetic activation, paroxysmal tachycardia, fever, or pain",
        "decrease_causes": "Vagal activation, baroreflex buffering, or bradycardia",
    },
}


@dataclasses.dataclass
class ParameterShift:
    """Detailed shift analysis for a single physiological parameter."""
    param_key: str
    param_name: str
    unit: str
    baseline_value: float
    current_value: float
    delta_pct: float
    z_score: float
    posterior_std: float


@dataclasses.dataclass
class DivergenceExplanation:
    """Synthesized clinical explanation for a digital twin divergence."""
    time_step: int
    status: str
    summary_text: str
    detailed_text: str
    confidence_label: str  # "High", "Moderate", "Low"
    primary_driver: str
    primary_driver_name: str
    shifts: List[ParameterShift]
    signal_contributions: Dict[str, float]
    recommended_clinical_action: str


class ExplainabilityEngine:
    """Explains why digital twin diverges from observed hemodynamics."""

    def __init__(self, baseline_params: Dict[str, float]) -> None:
        """Initialize with patient baseline parameters.

        Args:
            baseline_params: Dict with 'Rp', 'C', 'SV', 'HR'.
        """
        self.baseline = {
            "Rp": float(baseline_params.get("Rp", 1.10)),
            "C": float(baseline_params.get("C", 1.50)),
            "SV": float(baseline_params.get("SV", 70.0)),
            "HR": float(baseline_params.get("HR", 72.0)),
        }
        # Baseline population standard deviations for Z-score normalization
        self.baseline_stds = {
            "Rp": 0.12,
            "C": 0.16,
            "SV": 6.5,
            "HR": 4.5,
        }

    def update_baseline(self, new_baseline: Dict[str, float]) -> None:
        """Update patient baseline."""
        self.baseline.update(new_baseline)

    def explain(
        self,
        estimate: TwinEstimate,
        divergence: DivergenceResult,
    ) -> DivergenceExplanation:
        """Generate human-readable physiological explanation of twin divergence.

        Args:
            estimate: Current state posterior and covariance
            divergence: Current divergence result from DivergenceEngine

        Returns:
            DivergenceExplanation object
        """
        curr_vals = {
            "Rp": estimate.Rp,
            "C": estimate.C,
            "SV": estimate.SV,
            "HR": estimate.HR,
        }
        post_stds = estimate.get_std_errors()

        shifts: List[ParameterShift] = []
        max_z = -1.0
        primary_key = "Rp"

        for key in ["Rp", "C", "SV", "HR"]:
            base_v = self.baseline[key]
            curr_v = curr_vals[key]
            p_std = post_stds[key]
            b_std = self.baseline_stds[key]

            delta_pct = ((curr_v - base_v) / max(1e-5, base_v)) * 100.0
            # Total variance = baseline dispersion + posterior state estimation variance
            sigma_total = np.sqrt(b_std ** 2 + p_std ** 2)
            z_score = (curr_v - base_v) / max(1e-5, sigma_total)

            shift_obj = ParameterShift(
                param_key=key,
                param_name=PARAMETER_METADATA[key]["name"],
                unit=PARAMETER_METADATA[key]["unit"],
                baseline_value=round(base_v, 3),
                current_value=round(curr_v, 3),
                delta_pct=round(delta_pct, 1),
                z_score=round(z_score, 2),
                posterior_std=round(p_std, 3),
            )
            shifts.append(shift_obj)

            if abs(z_score) > max_z:
                max_z = abs(z_score)
                primary_key = key

        # Posterior confidence score based on trace of state covariance
        cov_trace = float(np.trace(estimate.state_cov))
        if cov_trace < 15.0:
            confidence_label = "High"
        elif cov_trace < 45.0:
            confidence_label = "Moderate"
        else:
            confidence_label = "Low"

        # Construct explanation strings
        prim_meta = PARAMETER_METADATA[primary_key]
        prim_shift = next(s for s in shifts if s.param_key == primary_key)
        direction_word = "increased" if prim_shift.delta_pct > 0 else "decreased"
        dir_cause = prim_meta["increase_causes"] if prim_shift.delta_pct > 0 else prim_meta["decrease_causes"]

        if divergence.status == "Normal":
            summary_text = "Hemodynamic observations remain congruent with the digital twin baseline."
            detailed_text = (
                f"All physiological parameters are within expected ranges. "
                f"Minor fluctuations in {prim_meta['name']} ({prim_shift.delta_pct:+.1f}%, {prim_shift.z_score:+.1f}σ) "
                f"reflect nominal autonomic tone."
            )
            recommendation = "Continue passive telemetry monitoring. No clinical intervention indicated."
        elif divergence.status == "Watch":
            summary_text = (
                f"Mild hemodynamic divergence detected: {prim_meta['name']} has {direction_word} "
                f"by {abs(prim_shift.delta_pct):.1f}% ({prim_shift.z_score:+.1f}σ)."
            )
            detailed_text = (
                f"Digital twin innovation distance is elevated (d_M = {divergence.raw_divergence_score:.2f}). "
                f"Primary divergence driver is {prim_meta['name']} shifting from {prim_shift.baseline_value} "
                f"to {prim_shift.current_value} {prim_shift.unit}. "
                f"Common etiologies include: {dir_cause}."
            )
            recommendation = "Maintain elevated monitoring frequency. Observe for persistence or autonomic normalization."
        else:  # Significant
            summary_text = (
                f"⚠️ Significant physiological divergence: {prim_meta['name']} {direction_word} "
                f"by {abs(prim_shift.delta_pct):.1f}% ({prim_shift.z_score:+.1f}σ over baseline)."
            )
            detailed_text = (
                f"Digital twin state diverges persistently from observations (d_M = {divergence.raw_divergence_score:.2f}, "
                f"p < {divergence.p_value:.4f}). "
                f"The estimated shift in {prim_meta['name']} represents a {abs(prim_shift.delta_pct):.1f}% change. "
                f"Physiological mechanism: {prim_meta['physiology']}. "
                f"Likely physiological triggers: {dir_cause}."
            )
            recommendation = "Prompt clinical review recommended. Correlate with patient symptoms and medication log."

        return DivergenceExplanation(
            time_step=estimate.time_step,
            status=divergence.status,
            summary_text=summary_text,
            detailed_text=detailed_text,
            confidence_label=confidence_label,
            primary_driver=primary_key,
            primary_driver_name=prim_meta["name"],
            shifts=shifts,
            signal_contributions=divergence.signal_contributions,
            recommended_clinical_action=recommendation,
        )
