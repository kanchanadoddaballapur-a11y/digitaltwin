# CardioTwin X: Adaptive Cardiovascular Digital Twin

> **⚠️ RESEARCH & DEMONSTRATION PROTOTYPE USING SYNTHETIC DATA**  
> *Not a clinically validated diagnostic system. All physiology models and data are simulated.*

---

## 1. Overview

**CardioTwin X** is an in-silico adaptive cardiovascular digital twin implemented in Python. It pairs a **mechanistic 3-element Windkessel lumped-parameter hemodynamic model** with an **Unscented Kalman Filter (UKF)** state observer to track unobservable internal vascular properties (peripheral resistance $R_p$, arterial compliance $C$, stroke volume $SV$, and heart rate $HR$) in real time from non-invasive blood pressure and pulse telemetry.

When the patient's observed physiology diverges from the digital twin's forecast, a statistical **Mahalanobis divergence engine** flags acute events with persistence filtering, while an **Explainability Engine** decomposes the signal into parametric shifts (Z-scores and percentage changes) to provide clear clinical physiological rationales. An interactive **What-If simulation lab** enables hypothesis testing and prospective interventions with propagated posterior parameter uncertainty.

---

## 2. System Architecture

```
                       ┌────────────────────────────────────────────────────────┐
                       │          Non-Invasive Telemetry (1-min step)           │
                       │   • Heart Rate (HR)   • Systolic BP (SBP)              │
                       │   • Diastolic BP (DBP)• ECG: RR, HRV-SDNN, QT-like     │
                       └───────────────────────────┬────────────────────────────┘
                                                   │ Observation y_k = [HR, SBP, DBP]^T
                                                   ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│                                ADAPTIVE DIGITAL TWIN CORE                                     │
│                                                                                               │
│   ┌────────────────────────────────┐                 ┌────────────────────────────────────┐   │
│   │   Unscented Kalman Filter      │                 │  3-Element Windkessel (WK3) Model  │   │
│   │   • State x = [Rp, C, SV, HR]  │  Forward h(x)   │  • ODE: dPd/dt = (Qin - Pd/Rp)/C   │   │
│   │   • Scaled Unscented Transform ├────────────────►│  • Aortic P(t) = Pd(t) + Rc*Qin(t)  │   │
│   │   • Posterior Covariance P_k   │                 │  • Half-sine inflow Qin(t)         │   │
│   │   • Missing Telemetry Bypassing│◄────────────────┤  • Baroreflex coupling loop        │   │
│   └──────────────┬─────────────────┘  Prediction y_k └────────────────────────────────────┘   │
│                  │                                                                            │
│                  │ Innovation v_k = y_k - y_pred, Innovation Covariance S_k                   │
│                  ▼                                                                            │
│   ┌───────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                         Digital Twin Divergence Engine                                │   │
│   │   • Mahalanobis Distance: d_M = sqrt(v_k^T * S_k^-1 * v_k)                            │   │
│   │   • Chi-Square Hypothesis Test (df=3) + Persistence Filtering (N >= 3 steps)          │   │
│   │   • 50-Member Monte Carlo Posterior Predictive Ensemble (95% uncertainty ribbons)     │   │
│   │   • Secondary Anomaly Detector: Isolation Forest on residual vectors (Dual ML)        │   │
│   └──────────────┬────────────────────────────────────────────────────────────────────────┘   │
└──────────────────┼────────────────────────────────────────────────────────────────────────────┘
                   │
         ┌─────────┴───────────────────────┬──────────────────────────────────┐
         ▼                                 ▼                                  ▼
┌──────────────────┐             ┌─────────────────────┐            ┌────────────────────┐
│  Explainability  │             │     What-If Lab     │            │  3D Vessel Canvas  │
│  • Z-scores / %  │             │  • Exercise, Stress │            │  • Interactive 3D  │
│  • Signal Contrib│             │  • Sodium, Vasodil. │            │    Aortic tube     │
│  • Natural Lang  │             │  • Before vs After  │            │  • Color mapped to │
│    Physiology    │             │    Waveform Overlay │            │    systolic strain │
└──────────────────┘             └─────────────────────┘            └────────────────────┘
```

---

## 3. Core Modules & Methodology

### 3.1 Physiological Model (`cardiotwin/twin/model.py`)
- **3-Element Windkessel (WK3):** Consists of peripheral vascular resistance $R_p$, systemic arterial compliance $C$, and aortic characteristic resistance $R_c$.
- **Governing Equations:**
  $$\frac{d P_d(t)}{dt} = \frac{Q_{in}(t) - \frac{P_d(t)}{R_p}}{C}$$
  $$P(t) = P_d(t) + R_c Q_{in}(t)$$
  Where $P_d(t)$ is compliance chamber pressure, $Q_{in}(t)$ is ventricular volume ejection into the ascending aorta, and $P(t)$ is total aortic root pressure.
- **Inflow Waveform:** Modeled as a half-sine during systolic ejection duration $T_{sys} = 0.33 \cdot T$:
  $$Q_{in}(t) = \frac{\pi \cdot SV}{2 T_{sys}} \sin\left(\frac{\pi t}{T_{sys}}\right), \quad 0 \le t \le T_{sys}$$
  $$Q_{in}(t) = 0, \quad T_{sys} < t \le T$$
  Integrated numerically with `scipy.integrate.solve_ivp` with exact periodic boundary matching.
- **Baroreflex Coupling:** Autonomic reflex modulating heart rate based on error from baseline MAP setpoint:
  $$HR_{coupled} = HR_{nominal} - k_{baro} \cdot (MAP - MAP_{target})$$

### 3.2 Synthetic Data Generator (`cardiotwin/twin/data.py`)
- Simulates 24-hour hemodynamic time series at 1-minute resolution (1,440 data points).
- **Physiological Variations:** 24h circadian sinus rhythm, afternoon mental stress episode (~13:30), evening aerobic exercise session (~17:15), and Gaussian sensor noise.
- **ECG-Derived Features:** Beat-to-beat RR interval (ms), HRV-SDNN (ms), and Bazett-scaled QT interval (ms).
- **Injected Anomalies:** Paroxysmal tachycardia (07:45), sensor telemetry dropout (11:15), and acute vascular resistance surge (20:05).
- **3 Patient Presets:**
  1. *Healthy*: 42yo normotensive ($BP \approx 118/78$ mmHg, $C = 1.50$, $R_p = 1.10$).
  2. *Hypertensive*: 58yo essential hypertension ($BP \approx 146/94$ mmHg, $C = 0.95$, $R_p = 1.48$).
  3. *Elderly*: 74yo vascular stiffening / isolated systolic hypertension ($BP \approx 156/72$ mmHg, $C = 0.62$, $R_p = 1.35$, wide pulse pressure).

### 3.3 Adaptive State Estimator (`cardiotwin/twin/estimator.py`)
- **Unscented Kalman Filter (UKF):** Van der Merwe scaled unscented transform estimating hidden state $\mathbf{x} = [R_p, C, SV, HR]^T$ from observation vector $\mathbf{y} = [HR, SBP, DBP]^T$.
- **Sigma Points:** Enforces physiological parameter boundaries to ensure numerical stability during forward ODE evaluation.
- **Missing Data Handling:** If incoming telemetry contains `NaN` (sensor dropout), the filter seamlessly bypasses the measurement update step, advancing state via process prediction and expanding posterior covariance.

### 3.4 Divergence Engine & Dual ML Consensus (`cardiotwin/twin/divergence.py`)
- **Normalized Innovation (Mahalanobis Distance):**
  $$d_M = \sqrt{\mathbf{v}_k^T \mathbf{S}_k^{-1} \mathbf{v}_k}$$
  where $\mathbf{v}_k = \mathbf{y}_k - \hat{\mathbf{y}}_{k|k-1}$ is the innovation vector and $\mathbf{S}_k$ is the innovation covariance.
- **Persistence Filter:** Flags *Significant* divergence only if $d_M \ge 3.40$ ($\chi^2_3$ 99th percentile) persists for $N \ge 3$ consecutive steps, eliminating transient motion artifacts.
- **Predictive Bands:** 50-member Monte Carlo sample drawn from posterior covariance $\mathcal{N}(\hat{\mathbf{x}}_k, \mathbf{P}_k)$ creates 95% confidence envelopes for forward blood pressure.
- **Secondary Anomaly Detector:** An `IsolationForest` (scikit-learn) trained on residual vectors runs in parallel; agreement between UKF and IsolationForest indicates high-confidence genuine pathology.

### 3.5 Explainability & What-If Simulations (`twin/explain.py`, `twin/whatif.py`)
- Computes parameter shift Z-scores ($\sigma$) and percentage changes over patient baseline.
- Produces plain-English clinical explanations and per-signal innovation contribution breakdowns.
- Simulates acute interventions (exercise, stress, sodium excess, vasodilators, arterial stiffening, custom sliders) with before-and-after waveform comparisons.

---

## 4. Key Assumptions & Limitations

1. **Synthetic Data Only:** All telemetry, waveforms, and clinical profiles are computationally generated. The application is intended for research prototyping, algorithmic validation, and demonstration.
2. **Simplified Lumped-Parameter Physiology:** The 3-element Windkessel model treats the arterial tree as a single compliance chamber and resistive outlet. It captures systemic pressure dynamics accurately but does not model distributed wave reflections or spatial pressure gradients.
3. **Circular In-Silico Validation:** Synthetic observations are generated from known equations and estimated via an observer grounded in the same family of differential equations. Real clinical physiology exhibits non-stationary nonlinear complexities not fully captured in low-order ODEs.
4. **Autonomous Autonomic Reflexes:** Baroreflex is modeled as a first-order proportional feedback loop on heart rate, omitting renal renin-angiotensin-aldosterone long-term volume control.

---

## 5. How to Run

### Installation
```bash
pip install -r requirements.txt
```

### Running Unit Tests
```bash
pytest
```
*All 11 unit tests verify baseline blood pressure calibration, ODE volume conservation, parameter sensitivities, baroreflex coupling, UKF convergence, missing telemetry handling, divergence detection, and what-if simulation stability.*

### Launching the Streamlit Application
```bash
streamlit run cardiotwin/app.py
```
Or from the project root:
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501` (or port 3000 in AI Studio).

---

## 6. Suggested Future Work
- **Clinical Data Validation:** Validate UKF state convergence against clinical ambulatory arterial pressure waveforms from public benchmark datasets (e.g. MIMIC-III, PTB-XL, BIDMC PPG/BP database).
- **Wearable Sensor Integration:** Ingest continuous photoplethysmography (PPG) pulse transit time (PTT) streams from commercial wearables.
- **Physics-Informed Neural Networks (PINNs):** Couple deep neural networks with Windkessel conservation laws for real-time patient-specific arterial compliance mapping.
- **Distributed 1D Hemodynamic Modeling:** Extend the lumped compliance chamber into a multibranch 1D wave-propagation network for aortic bifurcation and carotid artery analysis.
