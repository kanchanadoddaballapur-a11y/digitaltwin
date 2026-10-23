"""CardioTwin X: Adaptive Cardiovascular Digital Twin Streamlit Application.

Baby Pink Aesthetic | Personalized Lifestyle Questionnaire | SVG Anatomical Working Human Heart | Active Reminders for Checkup, Walking & Tablets
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Automatically ensure workspace root and cardiotwin directory are in sys.path
_current_file = Path(__file__).resolve()
_cardiotwin_dir = _current_file.parent
_workspace_root = _cardiotwin_dir.parent

for _p in [str(_workspace_root), str(_cardiotwin_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import textwrap
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components


def clean_html(raw_html: str) -> str:
    """Strips all leading and trailing whitespace from every line of an HTML string.
    Guarantees no line starts with 4+ spaces, preventing Streamlit's Markdown parser
    from converting HTML lines into verbatim pre/code blocks.
    """
    return "\n".join(line.strip() for line in raw_html.splitlines() if line.strip())

# Twin package imports (with fallback for both root and cardiotwin execution)
try:
    from cardiotwin.twin.data import (
        PATIENT_PRESETS,
        PatientProfile,
        generate_synthetic_patient_timeseries,
        get_patient_profile,
    )
    from cardiotwin.twin.divergence import DivergenceEngine, DivergenceResult
    from cardiotwin.twin.estimator import CardiovascularUKF, TwinEstimate
    from cardiotwin.twin.explain import ExplainabilityEngine
    from cardiotwin.twin.model import (
        PARAMETER_BOUNDS,
        WindkesselParameters,
        fast_windkessel_beat,
        simulate_windkessel_ivp,
    )
    from cardiotwin.twin.whatif import (
        PRESET_SCENARIOS,
        ScenarioOutcome,
        simulate_whatif,
    )
except ModuleNotFoundError:
    from twin.data import (
        PATIENT_PRESETS,
        PatientProfile,
        generate_synthetic_patient_timeseries,
        get_patient_profile,
    )
    from twin.divergence import DivergenceEngine, DivergenceResult
    from twin.estimator import CardiovascularUKF, TwinEstimate
    from twin.explain import ExplainabilityEngine
    from twin.model import (
        PARAMETER_BOUNDS,
        WindkesselParameters,
        fast_windkessel_beat,
        simulate_windkessel_ivp,
    )
    from twin.whatif import (
        PRESET_SCENARIOS,
        ScenarioOutcome,
        simulate_whatif,
    )

# Set Streamlit Page Config
st.set_page_config(
    page_title="CardioTwin X | Your Personal Working Human Heart Twin",
    page_icon="💖",
    layout="wide",
    initial_sidebar_state="expanded",
)

DISCLAIMER_TEXT = (
    "💖 **RESEARCH & DEMONSTRATION PROTOTYPE** — "
    "Designed to illustrate biological cardiovascular mechanics and care tracking using mathematical simulations. "
    "Not a medical diagnostic device. Always consult your doctor for medical concerns."
)

# Global Baby Pink Custom CSS
st.markdown(
    textwrap.dedent(
        """
        <style>
        .stApp {
            background-color: #fff5f7 !important;
            color: #701a75 !important;
        }
        .disclaimer-banner {
            background: linear-gradient(90deg, #fdf2f8, #fce7f3);
            color: #9d174d;
            padding: 10px 18px;
            border-radius: 12px;
            border-left: 5px solid #f472b6;
            font-size: 0.88rem;
            margin-bottom: 16px;
            box-shadow: 0 2px 8px rgba(244, 114, 182, 0.15);
        }
        .pink-card {
            background: #ffffff;
            border: 2px solid #fbcfe8;
            border-radius: 16px;
            padding: 18px;
            box-shadow: 0 4px 15px rgba(244, 114, 182, 0.12);
            margin-bottom: 16px;
        }
        .notification-bar {
            background: linear-gradient(90deg, #fff0f5, #fce7f3);
            border: 1.5px solid #f472b6;
            border-radius: 12px;
            padding: 12px 18px;
            margin-bottom: 14px;
            box-shadow: 0 3px 10px rgba(244,114,182,0.18);
        }
        .badge-normal {
            background-color: #d1fae5;
            color: #065f46;
            padding: 5px 12px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 0.88rem;
            display: inline-block;
            border: 1px solid #a7f3d0;
        }
        .badge-watch {
            background-color: #fef3c7;
            color: #92400e;
            padding: 5px 12px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 0.88rem;
            display: inline-block;
            border: 1px solid #fde68a;
        }
        .badge-significant {
            background-color: #ffe4e6;
            color: #be123c;
            padding: 5px 12px;
            border-radius: 20px;
            font-weight: 800;
            font-size: 0.88rem;
            display: inline-block;
            border: 1px solid #fecdd3;
        }
        div.stButton > button:first-child {
            background: linear-gradient(135deg, #f472b6, #ec4899);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 8px 18px;
            font-weight: 600;
            box-shadow: 0 4px 10px rgba(236, 72, 153, 0.25);
        }
        div.stButton > button:first-child:hover {
            background: linear-gradient(135deg, #ec4899, #db2777);
            color: white;
            transform: translateY(-1px);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #fce7f3;
            padding: 6px;
            border-radius: 14px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            color: #831843;
            font-weight: 600;
            padding: 8px 16px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ffffff !important;
            color: #db2777 !important;
            box-shadow: 0 2px 8px rgba(219, 39, 119, 0.15);
        }
        section[data-testid="stSidebar"] {
            background-color: #fdf2f8 !important;
            border-right: 2px solid #fbcfe8 !important;
        }
        </style>
        """
    ),
    unsafe_allow_html=True,
)

# Initialize Session State
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "patient_name" not in st.session_state:
    st.session_state.patient_name = "Sarah Jenkins"
if "patient_age" not in st.session_state:
    st.session_state.patient_age = 42
if "recent_sbp" not in st.session_state:
    st.session_state.recent_sbp = 120.0
if "recent_dbp" not in st.session_state:
    st.session_state.recent_dbp = 80.0
if "is_smoker" not in st.session_state:
    st.session_state.is_smoker = False
if "has_alcohol" not in st.session_state:
    st.session_state.has_alcohol = False
if "doctor_info" not in st.session_state:
    st.session_state.doctor_info = {"name": "Dr. Emily Chen, MD", "phone": "(555) 382-9102", "hospital": "Metro Heart Clinic"}
if "family_members" not in st.session_state:
    st.session_state.family_members = [
        {"name": "David Jenkins", "relation": "Spouse", "phone": "(555) 382-9199", "alert": True},
    ]
if "tablets_list" not in st.session_state:
    st.session_state.tablets_list = [
        {"name": "Lisinopril (Blood Pressure)", "dose": "10 mg", "time": "Morning", "taken": True},
        {"name": "Aspirin (Cardio Protection)", "dose": "81 mg", "time": "Afternoon", "taken": False},
    ]
if "meals_list" not in st.session_state:
    st.session_state.meals_list = [
        {"meal": "Breakfast", "food": "Oatmeal with berries", "weight_g": 250, "sodium_mg": 125, "status": "Logged"},
    ]
if "walk_sessions" not in st.session_state:
    st.session_state.walk_sessions = [
        {"date": "Today", "duration": "30 mins", "steps": "3,000 steps", "distance": "2.1 km", "status": "Completed 🟢"},
    ]


# ==============================================================================
# PATIENT SIGN-IN PAGE
# ==============================================================================
def render_patient_signin_page() -> None:
    st.markdown(
        f"""
        <div class="disclaimer-banner">
            {DISCLAIMER_TEXT}
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2.4, 1])

    with col2:
        st.markdown(
            """
            <div style="text-align: center; padding: 20px 10px 10px 10px;">
                <svg width="120" height="120" viewBox="0 0 200 200" style="filter: drop-shadow(0 8px 16px rgba(244, 114, 182, 0.4));">
                    <defs>
                        <linearGradient id="signinHeartGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" stop-color="#f472b6" />
                            <stop offset="60%" stop-color="#e11d48" />
                            <stop offset="100%" stop-color="#9f1239" />
                        </linearGradient>
                    </defs>
                    <path d="M 100 45 C 80 15, 130 15, 120 45" fill="none" stroke="#fbcfe8" stroke-width="14" stroke-linecap="round" />
                    <path d="M 100 178 C 30 120, 20 75, 60 50 C 85 35, 105 55, 110 65 C 115 55, 135 35, 160 50 C 200 75, 190 120, 100 178 Z"
                          fill="url(#signinHeartGrad)" />
                </svg>
                <h1 style="color: #831843; font-size: 2.4rem; font-weight: 800; margin: 10px 0 2px 0;">
                    CardioTwin <span style="color: #ec4899;">X</span>
                </h1>
                <p style="color: #9d174d; font-size: 1.1rem; font-weight: 600; margin-bottom: 20px;">
                    Your Personal Working Human Heart Twin 💖
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container():
            st.markdown(
                """
                <div class="pink-card">
                    <h3 style="color: #831843; margin-top: 0; font-size: 1.25rem;">🌸 Patient Setup & Lifestyle Questionnaire</h3>
                    <p style="color: #701a75; font-size: 0.92rem; line-height: 1.5;">
                        To build your personalized <strong>biologically working human heart twin</strong>, please enter your 
                        recent blood pressure check values and lifestyle details below.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.form("patient_signin_form"):
                p_name = st.text_input("Your Full Name:", value=st.session_state.patient_name)
                p_age = st.number_input("Your Age:", min_value=18, max_value=110, value=int(st.session_state.patient_age))
                
                st.markdown("#### 🩸 Recent Blood Pressure Check")
                col_bp1, col_bp2 = st.columns(2)
                with col_bp1:
                    r_sbp = st.number_input("Top SBP (Systolic, e.g. 120):", min_value=80, max_value=220, value=int(st.session_state.recent_sbp))
                with col_bp2:
                    r_dbp = st.number_input("Bottom DBP (Diastolic, e.g. 80):", min_value=40, max_value=140, value=int(st.session_state.recent_dbp))

                st.markdown("#### 🌿 Lifestyle & Habits")
                smoker_yn = st.selectbox("Are you a smoker?", ["No", "Yes"])
                alcohol_yn = st.selectbox("Have you consumed alcoholic beverages recently?", ["No", "Yes"])

                submitted = st.form_submit_button("🚀 Launch My Working Human Heart Twin", use_container_width=True)
                if submitted:
                    st.session_state.authenticated = True
                    st.session_state.patient_name = p_name
                    st.session_state.patient_age = int(p_age)
                    st.session_state.recent_sbp = float(r_sbp)
                    st.session_state.recent_dbp = float(r_dbp)
                    st.session_state.is_smoker = (smoker_yn == "Yes")
                    st.session_state.has_alcohol = (alcohol_yn == "Yes")
                    st.rerun()


if not st.session_state.authenticated:
    render_patient_signin_page()
    st.stop()


# ==============================================================================
# DATA CACHING & PERSONALIZED SIMULATION
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_cached_patient_data(preset_key: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    return generate_synthetic_patient_timeseries(
        preset_key=preset_key,
        duration_hours=24,
        resolution_min=1,
        random_seed=42,
        inject_anomalies=True,
    )


@st.cache_data(show_spinner=False)
def load_personalized_simulation(
    recent_sbp: float,
    recent_dbp: float,
    is_smoker: bool,
    has_alcohol: bool,
    age: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if recent_sbp >= 140:
        preset_key = "hypertensive"
    elif age >= 65 or (recent_sbp - recent_dbp) >= 60:
        preset_key = "elderly"
    else:
        preset_key = "healthy"

    df, meta = generate_synthetic_patient_timeseries(
        preset_key=preset_key,
        duration_hours=24,
        resolution_min=1,
        random_seed=42,
        inject_anomalies=True,
    )
    profile = get_patient_profile(preset_key)
    base_p = profile.base_params

    adjusted_Rp = base_p.Rp * (1.15 if is_smoker else 1.0) * (1.05 if has_alcohol else 1.0)
    adjusted_C = base_p.C * (0.85 if is_smoker else 1.0)

    ukf = CardiovascularUKF(
        initial_state=[adjusted_Rp, adjusted_C, base_p.SV, base_p.HR],
        Rc_fixed=base_p.Rc,
    )
    divergence_engine = DivergenceEngine(persistence_threshold=3)

    estimates_records = []
    divergence_records = []

    for i in range(len(df)):
        row = df.iloc[i]
        obs = [
            row["hr_obs"],
            row["sbp_obs"] + (recent_sbp - 120.0),
            row["dbp_obs"] + (recent_dbp - 80.0),
        ]
        est = ukf.step(obs)
        div = divergence_engine.evaluate(est, compute_ensemble=False)

        stds = est.get_std_errors()
        estimates_records.append({
            "step": i,
            "time_min": row["time_min"],
            "timestamp": row["timestamp"],
            "Rp": est.Rp,
            "C": est.C,
            "SV": est.SV,
            "HR": est.HR,
            "Rp_std": stds["Rp"],
            "C_std": stds["C"],
            "SV_std": stds["SV"],
            "HR_std": stds["HR"],
            "y_pred_hr": est.y_pred[0],
            "y_pred_sbp": est.y_pred[1],
            "y_pred_dbp": est.y_pred[2],
            "is_missing": est.is_missing_obs,
        })

        divergence_records.append({
            "step": i,
            "time_min": row["time_min"],
            "timestamp": row["timestamp"],
            "score": div.raw_divergence_score,
            "smoothed_score": div.smoothed_score,
            "status": div.status,
            "consecutive": div.consecutive_significant,
            "p_value": div.p_value,
            "contrib_hr": div.signal_contributions["HR"],
            "contrib_sbp": div.signal_contributions["SBP"],
            "contrib_dbp": div.signal_contributions["DBP"],
            "bands_hr_low": div.ensemble_bands["HR"][0],
            "bands_hr_high": div.ensemble_bands["HR"][2],
            "bands_sbp_low": div.ensemble_bands["SBP"][0],
            "bands_sbp_high": div.ensemble_bands["SBP"][2],
            "bands_dbp_low": div.ensemble_bands["DBP"][0],
            "bands_dbp_high": div.ensemble_bands["DBP"][2],
            "iso_anomaly": div.isolation_forest_anomaly,
            "agreement": div.detector_agreement,
            "anomaly_event": row["anomaly_event"],
        })

    return estimates_records, divergence_records


est_records, div_records = load_personalized_simulation(
    recent_sbp=st.session_state.recent_sbp,
    recent_dbp=st.session_state.recent_dbp,
    is_smoker=st.session_state.is_smoker,
    has_alcohol=st.session_state.has_alcohol,
    age=st.session_state.patient_age,
)

df, meta = load_cached_patient_data("healthy")

if "current_view_step" not in st.session_state:
    st.session_state.current_view_step = 600

cur_idx = min(st.session_state.current_view_step, len(est_records) - 1)
cur_df_row = df.iloc[cur_idx]
cur_est = est_records[cur_idx]
cur_div = div_records[cur_idx]


# ==============================================================================
# SIDEBAR: CARE CIRCLE
# ==============================================================================
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; padding: 4px 0 10px 0;">
            <span style="font-size: 2rem;">🫀</span>
            <div>
                <span style="font-size: 1.2rem; font-weight: 800; color: #831843;">CardioTwin <span style="color: #ec4899;">X</span></span><br/>
                <span style="font-size: 0.78rem; color: #9d174d;">Personal Heart Twin</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style="background:#ffffff; border:1px solid #fbcfe8; border-radius:12px; padding:12px; margin-bottom:12px;">
            <div style="font-size:0.75rem; color:#9d174d;">Patient Profile:</div>
            <div style="font-weight:700; color:#831843; font-size:1.05rem;">💖 {st.session_state.patient_name}</div>
            <div style="font-size:0.82rem; color:#701a75;">
                Age: {st.session_state.patient_age} | Recent BP: <strong>{int(st.session_state.recent_sbp)}/{int(st.session_state.recent_dbp)}</strong><br/>
                Smoker: {'Yes ⚠️' if st.session_state.is_smoker else 'No 🟢'} | Alcohol: {'Yes ⚠️' if st.session_state.has_alcohol else 'No 🟢'}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🔄 Retake Lifestyle & BP Check", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

    st.markdown("---")
    st.markdown("### 👨‍⚕️ Care Circle (Doctor & Family)")

    doc = st.session_state.doctor_info
    st.markdown(
        f"""
        <div style="background:#fff0f5; padding:10px; border-radius:10px; border:1px solid #fbcfe8; margin-bottom:8px; font-size:0.85rem; color:#831843;">
            <strong>🩺 Doctor:</strong> {doc['name']}<br/>
            <strong>Clinic Ph:</strong> {doc['phone']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    for idx, fam in enumerate(st.session_state.family_members):
        st.markdown(
            f"""
            <div style="background:#fff0f5; padding:8px 10px; border-radius:10px; border:1px solid #fbcfe8; margin-bottom:6px; font-size:0.85rem; color:#701a75;">
                <strong>👤 {fam['name']} ({fam['relation']})</strong><br/>
                📞 {fam['phone']}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("➕ Add Family Member"):
        with st.form("sidebar_add_fam_form"):
            f_name = st.text_input("Name:")
            f_rel = st.text_input("Relationship (e.g. Spouse, Son):")
            f_ph = st.text_input("Phone:")
            if st.form_submit_button("Save Family Member"):
                if f_name:
                    st.session_state.family_members.append({"name": f_name, "relation": f_rel, "phone": f_ph, "alert": True})
                    st.success("Family member added!")
                    st.rerun()


# ==============================================================================
# WORKING ANATOMICAL HUMAN HEART (SVG ANIMATED & ROBUST)
# ==============================================================================
def render_biological_working_heart(
    hr_val: float,
    sbp_val: float,
    dbp_val: float,
    sv_val: float,
    status: str,
    is_smoker: bool,
    has_alcohol: bool,
) -> None:
    beat_sec = max(0.35, min(2.0, 60.0 / max(35.0, hr_val)))

    if sbp_val >= 140 or sbp_val > 135:
        heart_color = "#e11d48"
        status_title = "⚠️ LEFT VENTRICLE WORKING UNDER HIGH PRESSURE"
        bio_desc = "Your heart is pumping forcefully against tighter arteries. Nicotine or alcohol can temporarily constrict blood vessels, requiring higher squeezing force."
        status_color = "#e11d48"
    elif hr_val >= 90:
        heart_color = "#f97316"
        status_title = "⚡ RAPID PULSE PUMPING"
        bio_desc = "Your SA node is firing rapidly. The heart chambers are filling quickly and contracting with shorter resting pauses."
        status_color = "#ea580c"
    else:
        heart_color = "#db2777"
        status_title = "💖 BIOLOGICALLY BALANCED PUMPING"
        bio_desc = "Your heart chambers are expanding and contracting rhythmically. Oxygenated blood flows efficiently through the aorta."
        status_color = "#059669"

    st.markdown(
        clean_html(
            f"""
            <div class="pink-card" style="border: 2px solid #fbcfe8; background: linear-gradient(135deg, #ffffff, #fff5f8); padding: 22px;">
                <div style="text-align: center; margin-bottom: 12px;">
                    <span style="background: {status_color}; color: white; border-radius: 20px; padding: 4px 14px; font-weight: 700; font-size: 0.85rem;">
                        BIOLOGICAL WORKING HUMAN HEART TWIN (YOUR LIVE VIRTUAL ORGAN)
                    </span>
                    <h3 style="margin: 8px 0 4px 0; color: #831843;">{status_title}</h3>
                    <p style="color: #701a75; font-size: 0.95rem; max-width: 720px; margin: 0 auto; line-height: 1.5;">
                        {bio_desc}
                    </p>
                </div>
            """
        ),
        unsafe_allow_html=True,
    )

    col_heart, col_metrics = st.columns([1, 1.2])

    with col_heart:
        components.html(
            f"""
            <!DOCTYPE html>
            <html>
            <head>
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    background: transparent;
                    font-family: system-ui, -apple-system, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }}
                @keyframes anatomicalBeat {{
                    0% {{ transform: scale(1.0); }}
                    15% {{ transform: scale(1.16); filter: drop-shadow(0 0 18px {heart_color}); }}
                    30% {{ transform: scale(0.95); }}
                    42% {{ transform: scale(1.08); filter: drop-shadow(0 0 12px {heart_color}); }}
                    60% {{ transform: scale(1.0); }}
                    100% {{ transform: scale(1.0); }}
                }}
                .anatomical-beating-heart {{
                    animation: anatomicalBeat {beat_sec}s cubic-bezier(0.28, 0.84, 0.42, 1.1) infinite;
                    transform-origin: center 110px;
                }}
            </style>
            </head>
            <body>
            <div style="text-align: center; background: #fff0f5; padding: 12px 16px; border-radius: 16px; border: 2px solid #fbcfe8; box-sizing: border-box; width: 100%;">
                <svg width="190" height="190" viewBox="0 0 220 220">
                    <defs>
                        <linearGradient id="humanHeartGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" stop-color="#f472b6" />
                            <stop offset="50%" stop-color="{heart_color}" />
                            <stop offset="100%" stop-color="#881337" />
                        </linearGradient>
                        <linearGradient id="aortaArch" x1="0%" y1="100%" x2="100%" y2="0%">
                            <stop offset="0%" stop-color="#fbcfe8" />
                            <stop offset="100%" stop-color="#fda4af" />
                        </linearGradient>
                    </defs>
                    <path d="M 70 40 L 70 80" fill="none" stroke="#93c5fd" stroke-width="12" stroke-linecap="round" />
                    <path d="M 115 52 C 105 16, 152 16, 142 52" fill="none" stroke="url(#aortaArch)" stroke-width="15" stroke-linecap="round" />
                    
                    <g class="anatomical-beating-heart">
                        <path d="M 110 188 C 38 132, 28 82, 68 52 C 92 36, 112 58, 118 72 C 124 58, 144 36, 168 52 C 208 82, 198 132, 110 188 Z"
                              fill="url(#humanHeartGrad)" stroke="#500724" stroke-width="3" stroke-linejoin="round" />
                        <path d="M 110 72 Q 98 128 110 182" fill="none" stroke="rgba(255,255,255,0.4)" stroke-width="3" stroke-linecap="round" />
                        <circle cx="110" cy="74" r="7" fill="#ffffff" opacity="0.85" />
                        <ellipse cx="110" cy="182" rx="5" ry="3" fill="#fda4af" opacity="0.9" />
                    </g>
                </svg>
                <div style="font-weight: 700; color: {status_color}; font-size: 1.05rem; margin-top: 4px;">
                    ⚡ {hr_val:.0f} Beats / Min ({'Active' if hr_val > 75 else 'Resting'})
                </div>
            </div>
            </body>
            </html>
            """,
            height=260,
        )

    with col_metrics:
        st.markdown("#### 🧬 How Your Heart is Working Right Now")
        mc1, mc2 = st.columns(2)
        with mc1:
            st.metric("Top Squeeze (SBP)", f"{sbp_val:.0f} mmHg", delta="Arterial Pressure")
            st.metric("Heart Pulse", f"{hr_val:.0f} bpm", delta="Beats / Min")
        with mc2:
            st.metric("Bottom Rest (DBP)", f"{dbp_val:.0f} mmHg", delta="Vascular Tone")
            st.metric("Pump Volume", f"{sv_val:.0f} ml", delta="Stroke Volume")

        st.markdown(
            clean_html(
                f"""
                <div style="background:#fff0f5; padding:14px; border-radius:12px; border:1px solid #fbcfe8; margin-top:14px; font-size:0.9rem; color:#701a75;">
                    <strong>Biological Status:</strong> Live simulation calibrated with your recent check values. 
                    {'Smoking habits active ⚠️' if is_smoker else 'No smoking 🟢'} | {'Alcohol consumed ⚠️' if has_alcohol else 'No alcohol 🟢'}
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# ACTIVE NOTIFICATIONS & CARE REMINDERS BANNER
# ==============================================================================
st.markdown(
    clean_html(
        """
        <div class="notification-bar">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.4rem;">🔔</span>
                    <div>
                        <strong style="color: #831843; font-size: 1rem;">Active Care Reminders & Notifications:</strong>
                        <div style="font-size: 0.9rem; color: #701a75; margin-top: 2px;">
                            💊 <strong>Tablets:</strong> Take Aspirin (81 mg) afternoon dose &nbsp;|&nbsp; 
                            🚶 <strong>Walking:</strong> Daily 30-min heart walk &nbsp;|&nbsp; 
                            🩺 <strong>Checkup:</strong> Dr. Chen appointment on Nov 14
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
    ),
    unsafe_allow_html=True,
)


# ==============================================================================
# MAIN TABS
# ==============================================================================
status_badge_html = (
    '<span class="badge-significant">🔴 HIGH STRAIN / CHECK DOCTOR</span>'
    if cur_div['status'] == 'Significant'
    else (
        '<span class="badge-watch">🟡 WATCH CLOSELY / SLIGHT STRAIN</span>'
        if cur_div['status'] == 'Watch'
        else '<span class="badge-normal">🟢 HEALTHY & BALANCED</span>'
    )
)

st.markdown(
    clean_html(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; background:#ffffff; padding:14px 20px; border-radius:14px; margin-bottom:14px; border:2px solid #fbcfe8; box-shadow:0 2px 10px rgba(244,114,182,0.12);">
            <div>
                <span style="font-size:1.4rem; font-weight:800; color:#831843;">CardioTwin <span style="color:#ec4899;">X</span></span>
                <span style="color:#9d174d; margin-left:14px; font-weight:600;">Patient: <strong style="color:#db2777;">{st.session_state.patient_name}</strong></span>
            </div>
            <div style="display:flex; align-items:center; gap:10px; margin-top:4px;">
                <span style="color:#831843; font-weight:600;">Heart Status:</span>
                {status_badge_html}
            </div>
        </div>
        """
    ),
    unsafe_allow_html=True,
)

# Render Biological Working Heart
render_biological_working_heart(
    hr_val=cur_est['y_pred_hr'],
    sbp_val=cur_est['y_pred_sbp'],
    dbp_val=cur_est['y_pred_dbp'],
    sv_val=cur_est['SV'],
    status=cur_div['status'],
    is_smoker=st.session_state.is_smoker,
    has_alcohol=st.session_state.has_alcohol,
)

tab_twin, tab_pills, tab_walk, tab_whatif, tab_pitch = st.tabs([
    "🫀 Live Human Heart & Vitals",
    "💊 Tablets & Meals (Auto Sodium)",
    "🚶 Walking & Exercise",
    "🩺 Doctor & What-If Sandbox",
    "🏆 About CardioTwin X (Judge Pitch)",
])


# ==========================================
# TAB 1: LIVE HUMAN HEART & VITALS
# ==========================================
with tab_twin:
    st.markdown("### 📈 Live Blood Pressure & Heart Rate Trends")
    st.caption("Clean and simple visualization of your recent blood pressure check values and pulse speed.")

    view_window = 180
    start_step = max(0, cur_idx - view_window)
    sub_est = est_records[start_step : cur_idx + 1]
    sub_time = [r["time_min"] for r in sub_est]

    fig_bp = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.6, 0.4],
        subplot_titles=[
            "Top Squeeze Pressure (SBP) vs Bottom Rest Pressure (DBP)",
            "Heart Pulse Speed (Beats per Minute)",
        ],
    )

    fig_bp.add_trace(
        go.Scatter(
            x=sub_time,
            y=[r["y_pred_sbp"] for r in sub_est],
            mode="lines",
            line=dict(width=3, color="#e11d48"),
            name="Top Squeeze Pressure (SBP)",
        ),
        row=1, col=1,
    )
    fig_bp.add_trace(
        go.Scatter(
            x=sub_time,
            y=[r["y_pred_dbp"] for r in sub_est],
            mode="lines",
            line=dict(width=3, color="#2563eb"),
            name="Bottom Rest Pressure (DBP)",
        ),
        row=1, col=1,
    )

    fig_bp.add_trace(
        go.Scatter(
            x=sub_time,
            y=[r["y_pred_hr"] for r in sub_est],
            mode="lines",
            line=dict(width=3, color="#059669"),
            name="Heart Pulse (BPM)",
        ),
        row=2, col=1,
    )

    fig_bp.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#fff5f8",
    )
    fig_bp.update_yaxes(title_text="Pressure (mmHg)", row=1, col=1, gridcolor="#fce7f3")
    fig_bp.update_yaxes(title_text="Pulse (bpm)", row=2, col=1, gridcolor="#fce7f3")
    fig_bp.update_xaxes(title_text="Time of Day", row=2, col=1, gridcolor="#fce7f3")

    st.plotly_chart(fig_bp, use_container_width=True)


# ==========================================
# TAB 2: TABLETS & MEALS (AUTO SODIUM)
# ==========================================
with tab_pills:
    st.markdown("### 💊 Tablets & Meals (Automatic Sodium Weight Calculation)")
    st.caption("Log meals by entering **weight in grams**. Sodium is computed automatically.")

    tc1, tc2 = st.columns(2)

    with tc1:
        st.markdown("#### 💊 Prescribed Tablets")
        for idx, tab in enumerate(st.session_state.tablets_list):
            col_t1, col_t2 = st.columns([3, 1])
            with col_t1:
                st.markdown(
                    f"""
                    <div style="background:#ffffff; padding:10px 14px; border-radius:10px; border:1px solid #fbcfe8; margin-bottom:8px;">
                        <strong>{tab['name']}</strong> ({tab['dose']}) — <em>{tab['time']}</em><br/>
                        Status: <strong style="color: {'#059669' if tab['taken'] else '#d97706'};">{'Taken ✅' if tab['taken'] else 'Pending ⏰'}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_t2:
                if not tab["taken"]:
                    if st.button("Take 💊", key=f"take_tab_{idx}"):
                        st.session_state.tablets_list[idx]["taken"] = True
                        st.rerun()

    with tc2:
        st.markdown("#### 🥗 Meal Logger (Weight in Grams)")
        for meal in st.session_state.meals_list:
            st.markdown(
                f"""
                <div style="background:#ffffff; padding:10px 14px; border-radius:10px; border:1px solid #fbcfe8; margin-bottom:8px;">
                    <strong>🍽️ {meal['meal']}</strong>: {meal['food']}<br/>
                    Weight: <strong>{meal['weight_g']} g</strong> | Computed Sodium: <code style="color:#e11d48; font-weight:700;">{meal['sodium_mg']} mg</code>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("➕ Log Meal by Weight"):
            with st.form("meal_weight_form"):
                m_type = st.selectbox("Meal Type:", ["Breakfast", "Lunch", "Dinner", "Snack"])
                m_desc = st.text_input("Food Item / Description:")
                m_weight = st.number_input("Meal Weight (grams):", min_value=10, max_value=2000, value=300, step=25)
                m_processed = st.checkbox("Processed / Restaurant Meal (higher sodium)", value=False)
                
                if st.form_submit_button("Calculate Sodium & Log Meal"):
                    multiplier = 1.4 if m_processed else 0.5
                    computed_sodium = int(m_weight * multiplier)
                    st.session_state.meals_list.append({
                        "meal": m_type,
                        "food": m_desc,
                        "weight_g": int(m_weight),
                        "sodium_mg": computed_sodium,
                        "status": "Logged",
                    })
                    st.success(f"Meal logged! Computed sodium: {computed_sodium} mg.")
                    st.rerun()


# ==========================================
# TAB 3: WALKING & EXERCISE
# ==========================================
with tab_walk:
    st.markdown("### 🚶 Walking & Exercise Session Tracker")
    st.caption("Record daily walks. Steps are estimated automatically from duration.")

    wc1, wc2 = st.columns([2, 1])

    with wc1:
        st.markdown("#### Logged Walking Sessions")
        for session in st.session_state.walk_sessions:
            st.markdown(
                f"""
                <div class="pink-card" style="background:#ffffff;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <h4 style="margin:0 0 4px 0; color:#831843;">👟 Outdoor Walking Session</h4>
                            <p style="margin:0; color:#701a75; font-size:0.92rem;">
                                Duration: <strong>{session['duration']}</strong> | Auto Steps: <strong>{session['steps']}</strong> | Distance: <strong>{session['distance']}</strong>
                            </p>
                        </div>
                        <span class="badge-normal">{session['status']}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("➕ Record New Walk"):
            with st.form("walk_form"):
                w_dur = st.selectbox("Walking Duration:", ["15 mins", "30 mins", "45 mins", "60 mins"])
                w_dist = st.text_input("Distance:", value="2.0 km")
                if st.form_submit_button("Log Walk"):
                    mins = int(w_dur.split()[0])
                    auto_steps = f"{mins * 100:,} steps"
                    st.session_state.walk_sessions.append({
                        "date": "Today",
                        "duration": w_dur,
                        "steps": auto_steps,
                        "distance": w_dist,
                        "status": "Completed 🟢",
                    })
                    st.success("Walk recorded successfully!")
                    st.rerun()

    with wc2:
        st.markdown("#### 💡 Heart Walking Advice")
        st.markdown(
            """
            <div style="background:#fff0f5; padding:14px; border-radius:12px; border:1px solid #fbcfe8; font-size:0.88rem; color:#701a75; line-height:1.5;">
                Regular walking reduces arterial stiffness and strengthens your heart muscle naturally.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==========================================
# TAB 4: DOCTOR & WHAT-IF
# ==========================================
with tab_whatif:
    st.markdown("### 🩺 Doctor Consultations & What-If Simulation")
    st.caption("Review notes or test lifestyle scenarios.")

    dc1, dc2 = st.columns(2)

    with dc1:
        st.markdown("#### 🩺 Doctor Consultation Notes")
        doc = st.session_state.doctor_info
        st.markdown(
            f"""
            <div class="pink-card">
                <h4 style="color: #831843; margin-top:0;">{doc['name']}</h4>
                <p style="color: #701a75; margin-bottom:8px;">
                    <strong>Clinic Ph:</strong> {doc['phone']}<br/>
                    <strong>Next Checkup:</strong> November 14, 2026
                </p>
                <div style="background:#fff5f8; padding:10px; border-radius:8px; border:1px solid #fbcfe8; font-size:0.88rem;">
                    <strong>Latest Doctor Note:</strong><br/>
                    <em>"Recent BP check values integrated into heart twin. Maintain walking and low-sodium nutrition."</em>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with dc2:
        st.markdown("#### 🧪 What-If Life Simulator")
        scen_options = {
            "exercise": "🏃 Brisk Walk",
            "stress": "⚡ Strong Coffee / Stress",
            "high_sodium": "🧂 High Sodium Meal",
            "vasodilator": "💊 Blood Pressure Meds",
        }
        chosen_scen = st.selectbox("Test scenario:", options=list(scen_options.keys()), format_func=lambda k: scen_options[k])

        curr_twin_obj = TwinEstimate(
            time_step=cur_idx,
            state_mean=np.array([cur_est["Rp"], cur_est["C"], cur_est["SV"], cur_est["HR"]]),
            state_cov=np.diag([cur_est["Rp_std"]**2, cur_est["C_std"]**2, cur_est["SV_std"]**2, cur_est["HR_std"]**2]),
            y_pred=np.array([cur_est["y_pred_hr"], cur_est["y_pred_sbp"], cur_est["y_pred_dbp"]]),
            innovation=np.zeros(3),
            innovation_cov=np.eye(3),
            is_missing_obs=False,
        )
        outcome = simulate_whatif(current_estimate=curr_twin_obj, scenario_key=chosen_scen)

        st.markdown(
            f"""
            <div style="background:#fff0f5; padding:12px; border-radius:10px; border:1px solid #fbcfe8; font-size:0.88rem; color:#701a75;">
                <strong>Effect:</strong> {outcome.scenario.clinical_rationale}<br/>
                <strong>Top SBP Change:</strong> {outcome.delta_sbp:+.1f} mmHg
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==========================================
# TAB 5: JUDGE PITCH
# ==========================================
with tab_pitch:
    st.markdown("### 🏆 About CardioTwin X — The Big Idea & Judge Pitch")
    st.markdown(
        """
        <div class="pink-card" style="background: linear-gradient(135deg, #ffffff, #fff5f8); border: 2px solid #fbcfe8; padding: 24px;">
            <h2 style="color: #831843; margin-top: 0; font-size: 1.8rem;">
                💡 One-Line Pitch for Judges
            </h2>
            <blockquote style="font-size: 1.2rem; font-weight: 700; color: #db2777; border-left: 5px solid #f472b6; padding-left: 16px; margin: 16px 0;">
                "Instead of just showing a patient's numbers, we build a living virtual copy of their cardiovascular system that spots meaningful changes, explains why, and simulates what could happen next."
            </blockquote>
            
            <hr style="border: 1px solid #fbcfe8; margin: 20px 0;"/>

            <h3 style="color: #831843;">🎮 The Big Idea: Like a Video Game Character for Your Heart</h3>
            <p style="color: #701a75; font-size: 1.05rem; line-height: 1.6;">
                Imagine a game character that's built to behave exactly like you. Every time your real body gives new data 
                (heart rate, blood pressure), the game character adjusts itself to match you. That's our <strong>digital twin</strong>.
            </p>

            <h3 style="color: #831843; margin-top: 24px;">❓ What Problem Does It Solve?</h3>
            <p style="color: #701a75; font-size: 1.05rem; line-height: 1.6;">
                Normal health apps only show numbers: <em>"your BP is 130/85."</em> Many AI tools only say <em>"you have a 30% risk of heart disease."</em> 
                Neither explains <em>why</em> things are changing or <em>what would happen if</em> something changed. CardioTwin X does both.
            </p>

            <h3 style="color: #831843; margin-top: 24px;">⚙️ How It Works, in 4 Steps</h3>
            <ol style="color: #701a75; font-size: 1.05rem; line-height: 1.7; padding-left: 20px;">
                <li><strong>Build the virtual heart:</strong> A math model treats your blood vessels like a pipe system. Pressure depends on how hard the heart pumps, how narrow or stiff the pipes are, and how fast the heart beats.</li>
                <li><strong>Keep it updated:</strong> Each new reading nudges the model's settings (like pipe narrowness) so the virtual heart keeps matching the real one. This is the <strong>Kalman filter</strong>.</li>
                <li><strong>Compare real vs. virtual:</strong> The twin predicts what your readings <em>should</em> be. If your real reading is far from the prediction, something unusual may be happening. This is the <strong>Divergence Engine</strong>, the main feature. It also shows how <em>sure</em> the system is, preventing false alarms.</li>
                <li><strong>Explain and test "what ifs":</strong> It tells you in plain words what changed (<em>"blood vessel resistance went up 18%"</em>) and lets you test scenarios like exercise, stress, or salt intake to see how your virtual heart reacts.</li>
            </ol>

            <h3 style="color: #831843; margin-top: 24px;">✨ Why It's Different</h3>
            <ul style="color: #701a75; font-size: 1.05rem; line-height: 1.6; padding-left: 20px;">
                <li><strong>Normal dashboard:</strong> "Here's your number."</li>
                <li><strong>Normal AI:</strong> "Here's your risk."</li>
                <li><strong>CardioTwin X:</strong> "Here's what your heart is doing, why it's different from expected, and what might happen next."</li>
            </ul>

            <div style="background: #fff0f5; padding: 14px; border-radius: 12px; border: 1px solid #fbcfe8; margin-top: 20px; font-size: 0.9rem; color: #9d174d;">
                ⚠️ <strong>Important Research Note:</strong> CardioTwin X uses simulated (synthetic) data and is built as a research and demonstration prototype. It is not a clinical diagnostic device.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
