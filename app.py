"""
╔══════════════════════════════════════════════════════════════════╗
║           CreditWise — Loan Approval Prediction System           ║
║                                                                  ║
║  Features:                                                       ║
║  • Matches exact feature pipeline from Creditwise_loanSystem.ipynb║
║  • Feature engineering: DTI_Ratio_sq, Credit_Score_sq           ║
║  • Risk score breakdown with visual gauge                        ║
║  • Per-factor analysis with color-coded feedback                 ║
║  • Premium dark-themed UI with custom CSS                        ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be the very first st call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CreditWise | Loan Approval System",
    page_icon="🏦",
    layout="wide",          # wide layout for a dashboard feel
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS — premium dark-blue banking theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root palette ── */
:root {
    --bg-deep:    #0a0f1e;
    --bg-card:    #111827;
    --bg-hover:   #1a2236;
    --accent:     #4f8ef7;
    --accent-glow:#4f8ef740;
    --green:      #22c55e;
    --red:        #ef4444;
    --yellow:     #f59e0b;
    --text-main:  #e8eaf0;
    --text-muted: #8b9ab0;
    --border:     #1e2d45;
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: var(--bg-deep) !important;
    color: var(--text-main) !important;
}

/* ── Streamlit chrome overrides ── */
.stApp { background: var(--bg-deep) !important; }
header[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding: 2rem 3rem !important; max-width: 1200px; }

/* ── Card wrapper ── */
.cw-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
    transition: border-color 0.2s;
}
.cw-card:hover { border-color: var(--accent); }

/* ── Section heading ── */
.cw-section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 1rem;
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0d1b35 0%, #112244 50%, #0a1830 100%);
    border: 1px solid #1e3a6e;
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, #4f8ef720 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1.15;
    margin: 0 0 0.5rem;
    background: linear-gradient(90deg, #e8eaf0, #4f8ef7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-size: 1rem;
    color: var(--text-muted);
    margin: 0;
}

/* ── Input labels ── */
label { color: var(--text-muted) !important; font-size: 0.85rem !important; }

/* ── Streamlit input widgets ── */
.stNumberInput input,
.stTextInput input,
.stSelectbox > div > div {
    background: #0d1527 !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-main) !important;
}
.stSlider > div { color: var(--text-main) !important; }

/* ── Predict button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #2563eb, #4f8ef7) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px #4f8ef740 !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px #4f8ef760 !important;
}

/* ── Result boxes ── */
.result-approved {
    background: linear-gradient(135deg, #052e16, #064e27);
    border: 1px solid #16a34a;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-rejected {
    background: linear-gradient(135deg, #2a0a0a, #3d0d0d);
    border: 1px solid #dc2626;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    margin: 0.5rem 0;
}
.result-sub { color: var(--text-muted); font-size: 0.9rem; }

/* ── Gauge container ── */
.gauge-wrap {
    display: flex; flex-direction: column; align-items: center;
    margin: 1rem 0;
}

/* ── Factor badge ── */
.factor-good  { color: #22c55e; }
.factor-warn  { color: #f59e0b; }
.factor-bad   { color: #ef4444; }
.factor-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.55rem 0;
    border-bottom: 1px solid #1a2236;
    font-size: 0.88rem;
}
.factor-label { color: var(--text-muted); }

/* ── Metric tiles ── */
.metric-tile {
    background: #0d1527;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
}
.metric-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 2px;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HELPER — load model artefacts (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model_assets():
    """
    Load the four saved artefacts produced by the notebook:
      models/loan_model.pkl  — trained classifier  (LogisticRegression / NaiveBayes)
      models/scaler.pkl      — StandardScaler fitted on X_train
      models/ohe.pkl         — OneHotEncoder fitted on categorical columns
      models/columns.pkl     — ordered list of feature columns the model expects
    Returns (model, scaler, ohe, columns) or (None, …) on missing files.
    """
    paths = {
        "model":   "models/loan_model.pkl",
        "scaler":  "models/scaler.pkl",
        "ohe":     "models/ohe.pkl",
        "columns": "models/columns.pkl",
    }
    missing = [k for k, v in paths.items() if not os.path.exists(v)]
    if missing:
        return None, None, None, None

    model   = joblib.load(paths["model"])
    scaler  = joblib.load(paths["scaler"])
    ohe     = joblib.load(paths["ohe"])
    columns = joblib.load(paths["columns"])
    return model, scaler, ohe, columns


# ─────────────────────────────────────────────
#  HELPER — build a human-readable risk summary
# ─────────────────────────────────────────────
def evaluate_risk_factors(income, co_income, loan_amount, loan_term,
                           credit_score, existing_loans, dti, savings,
                           collateral, employment_status):
    """
    Heuristic factor analysis so users understand *why* they got the result.
    Returns a list of (label, status, comment) tuples.
    status ∈ {"good", "warn", "bad"}
    """
    factors = []

    # 1. Credit score bands
    if credit_score >= 750:
        factors.append(("Credit Score", "good", f"{credit_score} — Excellent"))
    elif credit_score >= 680:
        factors.append(("Credit Score", "warn", f"{credit_score} — Moderate"))
    else:
        factors.append(("Credit Score", "bad",  f"{credit_score} — Below threshold"))

    # 2. DTI ratio (lower is better; dataset range 0.10–0.60)
    if dti <= 0.25:
        factors.append(("Debt-to-Income Ratio", "good", f"{dti:.2f} — Low debt burden"))
    elif dti <= 0.45:
        factors.append(("Debt-to-Income Ratio", "warn", f"{dti:.2f} — Manageable"))
    else:
        factors.append(("Debt-to-Income Ratio", "bad",  f"{dti:.2f} — High debt burden"))

    # 3. Savings buffer
    monthly_emi = loan_amount / max(loan_term, 1)
    months_covered = savings / max(monthly_emi, 1)
    if months_covered >= 6:
        factors.append(("Savings Buffer", "good", f"Covers ~{months_covered:.0f} EMIs"))
    elif months_covered >= 3:
        factors.append(("Savings Buffer", "warn", f"Covers ~{months_covered:.0f} EMIs"))
    else:
        factors.append(("Savings Buffer", "bad",  f"Covers ~{months_covered:.0f} EMIs"))

    # 4. Collateral coverage
    coverage = collateral / max(loan_amount, 1)
    if coverage >= 1.5:
        factors.append(("Collateral Coverage", "good", f"{coverage:.1f}× loan amount"))
    elif coverage >= 0.8:
        factors.append(("Collateral Coverage", "warn", f"{coverage:.1f}× loan amount"))
    else:
        factors.append(("Collateral Coverage", "bad",  f"{coverage:.1f}× loan amount"))

    # 5. Existing loans
    if existing_loans == 0:
        factors.append(("Existing Loans", "good", "No current obligations"))
    elif existing_loans <= 2:
        factors.append(("Existing Loans", "warn", f"{existing_loans} active loan(s)"))
    else:
        factors.append(("Existing Loans", "bad",  f"{existing_loans} active loans — high load"))

    # 6. Employment
    if employment_status == "Employed":
        factors.append(("Employment Status", "good", "Stable employment"))
    elif employment_status == "Self-Employed":
        factors.append(("Employment Status", "warn", "Self-employed — income variability"))
    else:
        factors.append(("Employment Status", "bad",  "Unemployed — income risk"))

    return factors


# ─────────────────────────────────────────────
#  HELPER — SVG semi-circle gauge widget
# ─────────────────────────────────────────────
def render_gauge(probability_approved: float):
    """
    Renders an SVG arc gauge showing approval probability 0–100 %.
    Color transitions red → yellow → green.
    """
    pct   = probability_approved * 100
    # Arc math: 180° semi-circle from left to right
    angle = pct / 100 * 180          # degrees swept
    rad   = np.radians(180 - angle)  # map to SVG coords
    cx, cy, r = 110, 100, 80

    # End-point of the colored arc
    ex = cx + r * np.cos(np.radians(180 - angle))
    ey = cy - r * np.sin(np.radians(180 - angle))

    # Arc color
    if pct >= 65:
        arc_color = "#22c55e"
    elif pct >= 40:
        arc_color = "#f59e0b"
    else:
        arc_color = "#ef4444"

    large_arc = 1 if angle > 180 else 0

    gauge_svg = f"""
    <svg viewBox="0 0 220 120" width="220" height="120"
         xmlns="http://www.w3.org/2000/svg" style="display:block;margin:auto">
      <!-- Background track -->
      <path d="M 30 100 A 80 80 0 0 1 190 100"
            fill="none" stroke="#1e2d45" stroke-width="14" stroke-linecap="round"/>
      <!-- Colored arc -->
      <path d="M 30 100 A 80 80 0 {large_arc} 1 {ex:.2f} {ey:.2f}"
            fill="none" stroke="{arc_color}" stroke-width="14" stroke-linecap="round"/>
      <!-- Needle -->
      <line x1="{cx}" y1="{cy}"
            x2="{cx + 68*np.cos(np.radians(180 - angle)):.2f}"
            y2="{cy - 68*np.sin(np.radians(180 - angle)):.2f}"
            stroke="#e8eaf0" stroke-width="2.5" stroke-linecap="round"/>
      <circle cx="{cx}" cy="{cy}" r="5" fill="#e8eaf0"/>
      <!-- Labels -->
      <text x="22" y="116" font-size="9" fill="#8b9ab0" font-family="DM Sans">0%</text>
      <text x="100" y="22" font-size="9" fill="#8b9ab0" text-anchor="middle" font-family="DM Sans">50%</text>
      <text x="195" y="116" font-size="9" fill="#8b9ab0" text-anchor="end" font-family="DM Sans">100%</text>
      <!-- Centre value -->
      <text x="{cx}" y="{cy+30}" font-size="20" font-weight="700"
            fill="{arc_color}" text-anchor="middle" font-family="Syne">{pct:.0f}%</text>
    </svg>
    """
    st.markdown(gauge_svg, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD ASSETS
# ─────────────────────────────────────────────
model, scaler, ohe, model_columns = load_model_assets()


# ─────────────────────────────────────────────
#  HERO BANNER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-title">🏦 CreditWise</div>
  <p class="hero-sub">
    ML-powered loan eligibility analysis — instant, data-driven, transparent.
  </p>
</div>
""", unsafe_allow_html=True)

# Model missing warning
if model is None:
    st.error(
        "**Model files not found.** "
        "Please ensure the following exist relative to `app.py`:\n"
        "- `models/loan_model.pkl`\n"
        "- `models/scaler.pkl`\n"
        "- `models/ohe.pkl`\n"
        "- `models/columns.pkl`"
    )
    st.stop()


# ─────────────────────────────────────────────
#  INPUT FORM  (three columns × two sections)
# ─────────────────────────────────────────────
# ── Section 1: Applicant profile ──────────────
st.markdown('<div class="cw-card">', unsafe_allow_html=True)
st.markdown('<div class="cw-section-title">👤 Applicant Profile</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input(
        "Age", min_value=21, max_value=59, value=35,
        help="Applicant's current age (dataset range: 21–59)."
    )
    gender = st.selectbox(
        "Gender", ["Male", "Female"],
        help="Gender as recorded in the application."
    )
with col2:
    marital_status = st.selectbox(
        "Marital Status", ["Single", "Married", "Divorced"],
        help="Current marital status of the applicant."
    )
    dependents = st.number_input(
        "Number of Dependents", min_value=0, max_value=3, value=1,
        help="Children / family members financially dependent on the applicant."
    )
with col3:
    education_level = st.selectbox(
        "Education Level",
        ["Graduate", "Post-Graduate", "Under-Graduate", "Doctorate"],
        help="Highest academic qualification."
    )
    employment_status = st.selectbox(
        "Employment Status", ["Employed", "Self-Employed", "Unemployed"],
        help="Current employment type."
    )

st.markdown('</div>', unsafe_allow_html=True)

# ── Section 2: Financial profile ──────────────
st.markdown('<div class="cw-card">', unsafe_allow_html=True)
st.markdown('<div class="cw-section-title">💰 Financial Profile</div>', unsafe_allow_html=True)

col4, col5, col6 = st.columns(3)
with col4:
    applicant_income = st.number_input(
        "Applicant Income (₹/month)", min_value=2009, max_value=19988,
        value=10000, step=500,
        help="Monthly gross income of the applicant."
    )
    coapplicant_income = st.number_input(
        "Co-Applicant Income (₹/month)", min_value=0, max_value=9996,
        value=3000, step=500,
        help="Monthly income of a co-applicant (0 if none)."
    )
with col5:
    savings = st.number_input(
        "Savings / Bank Balance (₹)", min_value=65, max_value=19996,
        value=8000, step=1000,
        help="Total liquid savings in bank accounts."
    )
    collateral_value = st.number_input(
        "Collateral Value (₹)", min_value=36, max_value=49954,
        value=20000, step=1000,
        help="Market value of asset pledged as collateral."
    )
with col6:
    credit_score = st.slider(
        "Credit Score", min_value=550, max_value=799, value=680, step=1,
        help="CIBIL / credit bureau score (dataset range: 550–799)."
    )
    existing_loans = st.number_input(
        "Existing Active Loans", min_value=0, max_value=4, value=1,
        help="Number of currently outstanding loans."
    )

st.markdown('</div>', unsafe_allow_html=True)

# ── Section 3: Loan details ────────────────────
st.markdown('<div class="cw-card">', unsafe_allow_html=True)
st.markdown('<div class="cw-section-title">📋 Loan Details</div>', unsafe_allow_html=True)

col7, col8, col9 = st.columns(3)
with col7:
    loan_amount = st.number_input(
        "Loan Amount (₹)", min_value=1015, max_value=39995,
        value=15000, step=1000,
        help="Principal amount being requested."
    )
    loan_term = st.selectbox(
        "Loan Term (months)", [12, 24, 36, 48, 60, 72, 84],
        index=3,
        help="Repayment period in months."
    )
with col8:
    loan_purpose = st.selectbox(
        "Loan Purpose",
        ["Home", "Education", "Personal", "Business", "Medical", "Auto", "Other"],
        help="Primary reason for taking the loan."
    )
    employer_category = st.selectbox(
        "Employer Category",
        ["Government", "Private", "Public Sector", "NGO", "Self"],
        help="Type of employer / work organisation."
    )
with col9:
    property_area = st.selectbox(
        "Property Area",
        ["Urban", "Rural", "Semi-Urban"],
        help="Location type of the applicant's residence."
    )
    # Auto-compute DTI for display, user can override
    auto_dti = round(loan_amount / max(applicant_income + coapplicant_income, 1), 3)
    dti_ratio = st.number_input(
        "Debt-to-Income Ratio (auto-computed)",
        min_value=0.10, max_value=0.60,
        value=float(np.clip(auto_dti, 0.10, 0.60)),
        step=0.01, format="%.2f",
        help="loan_amount ÷ (applicant + co-applicant income). Auto-filled; adjust if needed."
    )

st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  PREDICT BUTTON
# ─────────────────────────────────────────────
_, btn_col, _ = st.columns([2, 3, 2])
with btn_col:
    run_prediction = st.button("⚡ Analyse My Application", use_container_width=True)


# ─────────────────────────────────────────────
#  PREDICTION LOGIC
# ─────────────────────────────────────────────
if run_prediction:

    # ── 1. Build raw DataFrame matching notebook columns ──────────
    raw = pd.DataFrame({
        "Applicant_Income":    [applicant_income],
        "Coapplicant_Income":  [coapplicant_income],
        "Employment_Status":   [employment_status],
        "Age":                 [age],
        "Marital_Status":      [marital_status],
        "Dependents":          [float(dependents)],
        "Credit_Score":        [float(credit_score)],
        "Existing_Loans":      [float(existing_loans)],
        "DTI_Ratio":           [dti_ratio],
        "Savings":             [savings],
        "Collateral_Value":    [collateral_value],
        "Loan_Amount":         [loan_amount],
        "Loan_Term":           [float(loan_term)],
        "Loan_Purpose":        [loan_purpose],
        "Property_Area":       [property_area],
        "Education_Level":     [education_level],
        "Gender":              [gender],
        "Employer_Category":   [employer_category],
    })

    # ── 2. Feature Engineering (mirrors notebook cells 36) ────────
    # The notebook drops raw Credit_Score & DTI_Ratio and uses their squares
    raw["DTI_Ratio_sq"]    = raw["DTI_Ratio"]    ** 2
    raw["Credit_Score_sq"] = raw["Credit_Score"] ** 2

    # Drop the raw versions that were removed before model training
    raw.drop(columns=["DTI_Ratio", "Credit_Score"], inplace=True)

    # ── 3. One-Hot Encoding for categorical columns ───────────────
    # Columns as used in the notebook
    categorical_cols = [
        "Employment_Status", "Marital_Status", "Loan_Purpose",
        "Property_Area", "Education_Level", "Gender", "Employer_Category"
    ]

    try:
        encoded_arr = ohe.transform(raw[categorical_cols])
        encoded_df  = pd.DataFrame(
            encoded_arr,
            columns=ohe.get_feature_names_out(categorical_cols)
        )
    except Exception as e:
        st.error(f"Encoding error — check that your OHE artefact matches the input categories.\n\n`{e}`")
        st.stop()

    # ── 4. Combine numeric + encoded features ─────────────────────
    numeric_df = raw.drop(columns=categorical_cols)
    final_df   = pd.concat([numeric_df.reset_index(drop=True),
                             encoded_df.reset_index(drop=True)], axis=1)

    # Align to the exact column order the model was trained on
    # (fill_value=0 handles any OHE categories not present in this single row)
    final_df = final_df.reindex(columns=model_columns, fill_value=0)

    # ── 5. Scale ─────────────────────────────────────────────────
    final_scaled = scaler.transform(final_df)

    # ── 6. Predict ───────────────────────────────────────────────
    prediction   = model.predict(final_scaled)[0]           # "Yes" / "No" or 1 / 0
    proba        = model.predict_proba(final_scaled)[0]     # [P(class0), P(class1)]

    # Determine which class index represents approval
    classes = list(model.classes_)
    # Support both string ("Yes"/"No") and int (1/0) label formats
    if "Yes" in classes:
        approved_idx = classes.index("Yes")
    else:
        approved_idx = 0   # class 0 = approved in original notebook encoding

    prob_approved = proba[approved_idx]
    is_approved   = (prediction == "Yes") or (prediction == 1 and approved_idx == 0)

    # ── 7. Risk factor heuristics ─────────────────────────────────
    factors = evaluate_risk_factors(
        income=applicant_income,
        co_income=coapplicant_income,
        loan_amount=loan_amount,
        loan_term=loan_term,
        credit_score=credit_score,
        existing_loans=existing_loans,
        dti=dti_ratio,
        savings=savings,
        collateral=collateral_value,
        employment_status=employment_status,
    )

    # ─────────────────────────────────────────────
    #  RESULTS LAYOUT
    # ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="cw-section-title" style="font-size:0.9rem">📊 Analysis Results</div>',
                unsafe_allow_html=True)

    res_col, gauge_col, factor_col = st.columns([1.6, 1.2, 1.6])

    # ── Verdict card ─────────────────────────────
    with res_col:
        if is_approved:
            st.markdown(f"""
            <div class="result-approved">
              <div style="font-size:3rem">✅</div>
              <div class="result-title" style="color:#22c55e">Approved</div>
              <div class="result-sub">
                Your application meets the lending criteria.<br>
                Proceed to the next step with your bank.
              </div>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f"""
            <div class="result-rejected">
              <div style="font-size:3rem">❌</div>
              <div class="result-title" style="color:#ef4444">Rejected</div>
              <div class="result-sub">
                The profile doesn't meet current thresholds.<br>
                Review the factor analysis to improve eligibility.
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Quick metric tiles below the verdict
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""
            <div class="metric-tile">
              <div class="metric-value" style="color:#4f8ef7">{prob_approved*100:.1f}%</div>
              <div class="metric-label">Approval Probability</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            risk_label = "Low" if dti_ratio < 0.30 else ("Medium" if dti_ratio < 0.50 else "High")
            risk_color = "#22c55e" if risk_label=="Low" else ("#f59e0b" if risk_label=="Medium" else "#ef4444")
            st.markdown(f"""
            <div class="metric-tile">
              <div class="metric-value" style="color:{risk_color}">{risk_label}</div>
              <div class="metric-label">DTI Risk Level</div>
            </div>""", unsafe_allow_html=True)

    # ── Gauge ─────────────────────────────────────
    with gauge_col:
        st.markdown('<div class="cw-card" style="text-align:center">', unsafe_allow_html=True)
        st.markdown('<div class="cw-section-title" style="text-align:center">Approval Score</div>',
                    unsafe_allow_html=True)
        render_gauge(prob_approved)
        st.markdown(f"""
        <p style="color:var(--text-muted);font-size:0.8rem;text-align:center;margin-top:0.5rem">
          Model confidence: <b style="color:var(--text-main)">{max(proba)*100:.1f}%</b>
        </p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Factor breakdown ───────────────────────────
    with factor_col:
        st.markdown('<div class="cw-card">', unsafe_allow_html=True)
        st.markdown('<div class="cw-section-title">Risk Factor Analysis</div>',
                    unsafe_allow_html=True)

        icon_map  = {"good": "✅", "warn": "⚠️", "bad": "❌"}
        class_map = {"good": "factor-good", "warn": "factor-warn", "bad": "factor-bad"}

        for label, status, comment in factors:
            icon  = icon_map[status]
            cls   = class_map[status]
            st.markdown(f"""
            <div class="factor-row">
              <span class="factor-label">{label}</span>
              <span class="{cls}">{icon} {comment}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Tips if rejected ──────────────────────────
    if not is_approved:
        st.markdown("---")
        st.markdown('<div class="cw-section-title">💡 Improvement Tips</div>', unsafe_allow_html=True)

        bad_factors = [f for f in factors if f[1] == "bad"]
        tips_map = {
            "Credit Score":          "Pay outstanding dues on time. Avoid multiple credit enquiries. Aim for 700+ before reapplying.",
            "Debt-to-Income Ratio":  "Reduce existing debt or increase income. A DTI below 0.35 is the safe zone.",
            "Savings Buffer":        "Build a buffer of at least 6 monthly EMIs in your savings account.",
            "Collateral Coverage":   "Offer additional assets as security, or apply for a smaller loan amount.",
            "Existing Loans":        "Close at least one existing loan before applying to reduce your debt load.",
            "Employment Status":     "Provide 2+ years of income-tax returns / audited P&L if self-employed. Secure employment before applying.",
        }

        tip_cols = st.columns(min(len(bad_factors), 3))
        for i, (label, _, _) in enumerate(bad_factors):
            with tip_cols[i % 3]:
                tip_text = tips_map.get(label, "Consult a financial advisor for tailored guidance.")
                st.markdown(f"""
                <div class="cw-card" style="min-height:100px">
                  <div class="cw-section-title" style="color:#f59e0b">{label}</div>
                  <p style="font-size:0.83rem;color:var(--text-muted);margin:0">{tip_text}</p>
                </div>
                """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<hr>
<p style="text-align:center;color:#8b9ab0;font-size:0.78rem;margin-top:0.5rem">
  CreditWise &nbsp;·&nbsp; Powered by Logistic Regression / Naïve Bayes
  &nbsp;·&nbsp; Trained on 950 records &nbsp;·&nbsp; For demonstration only
</p>
""", unsafe_allow_html=True)