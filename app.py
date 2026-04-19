import os

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EMPLOYMENT_OPTIONS = ["Contract", "Salaried", "Self-employed", "Unemployed"]
MARITAL_OPTIONS = ["Married", "Single"]
LOAN_PURPOSE_OPTIONS = ["Business", "Car", "Education", "Home", "Personal"]
PROPERTY_AREA_OPTIONS = ["Rural", "Semiurban", "Urban"]
EDUCATION_OPTIONS = ["Graduate", "Not Graduate"]
GENDER_OPTIONS = ["Female", "Male"]
EMPLOYER_OPTIONS = ["Business", "Government", "MNC", "Private", "Unemployed"]
LOAN_TERM_OPTIONS = [12, 24, 36, 48, 60, 72, 84]

DEFAULTS = {
    "Applicant_Income": 10548,
    "Coapplicant_Income": 5206,
    "Age": 40,
    "Dependents": 1,
    "Credit_Score": 678,
    "Existing_Loans": 2,
    "DTI_Ratio": 0.34,
    "Savings": 9880,
    "Collateral_Value": 24321,
    "Loan_Amount": 21210,
    "Loan_Term": 48,
}

@st.cache_resource
def load_artifacts():
    return {
        "model": joblib.load(os.path.join(BASE_DIR, "loan_model.pkl")),
        "scaler": joblib.load(os.path.join(BASE_DIR, "scaler.pkl")),
        "feature_columns": joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl")),
    }


@st.cache_data
def load_reference_data():
    return pd.read_csv(os.path.join(BASE_DIR, "loan_approval_data.csv"))


def encode_education(level: str) -> int:
    return 0 if level == "Graduate" else 1


def build_feature_frame(form_data: dict, feature_columns) -> pd.DataFrame:
    features = {column: 0 for column in feature_columns}

    features.update(
        {
            "Applicant_Income": form_data["Applicant_Income"],
            "Coapplicant_Income": form_data["Coapplicant_Income"],
            "Age": form_data["Age"],
            "Dependents": form_data["Dependents"],
            "Existing_Loans": form_data["Existing_Loans"],
            "Savings": form_data["Savings"],
            "Collateral_Value": form_data["Collateral_Value"],
            "Loan_Amount": form_data["Loan_Amount"],
            "Loan_Term": form_data["Loan_Term"],
            "Education_Level": encode_education(form_data["Education_Level"]),
            "DTI_Ratio_sq": form_data["DTI_Ratio"] ** 2,
            "Credit_Score_sq": form_data["Credit_Score"] ** 2,
        }
    )

    encoded_columns = {
        "Employment_Status": {
            "Salaried": "Employment_Status_Salaried",
            "Self-employed": "Employment_Status_Self-employed",
            "Unemployed": "Employment_Status_Unemployed",
        },
        "Marital_Status": {"Single": "Marital_Status_Single"},
        "Loan_Purpose": {
            "Car": "Loan_Purpose_Car",
            "Education": "Loan_Purpose_Education",
            "Home": "Loan_Purpose_Home",
            "Personal": "Loan_Purpose_Personal",
        },
        "Property_Area": {
            "Semiurban": "Property_Area_Semiurban",
            "Urban": "Property_Area_Urban",
        },
        "Gender": {"Male": "Gender_Male"},
        "Employer_Category": {
            "Government": "Employer_Category_Government",
            "MNC": "Employer_Category_MNC",
            "Private": "Employer_Category_Private",
            "Unemployed": "Employer_Category_Unemployed",
        },
    }

    for field_name, mapping in encoded_columns.items():
        selected_value = form_data[field_name]
        column_name = mapping.get(selected_value)
        if column_name in features:
            features[column_name] = 1

    return pd.DataFrame([features]).reindex(columns=feature_columns, fill_value=0)


def percentile_of_value(series: pd.Series, value: float) -> float:
    clean_series = series.dropna()
    if clean_series.empty:
        return 0.0
    return float((clean_series <= value).mean() * 100)


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def build_analysis(form_data: dict, probability: float, reference_df: pd.DataFrame) -> dict:
    total_income = form_data["Applicant_Income"] + form_data["Coapplicant_Income"]
    income_to_loan = safe_ratio(total_income, form_data["Loan_Amount"])
    savings_to_loan = safe_ratio(form_data["Savings"], form_data["Loan_Amount"])
    collateral_cover = safe_ratio(form_data["Collateral_Value"], form_data["Loan_Amount"])
    monthly_loan_load = safe_ratio(form_data["Loan_Amount"], form_data["Loan_Term"])

    percentiles = {
        "Income": percentile_of_value(
            reference_df["Applicant_Income"] + reference_df["Coapplicant_Income"], total_income
        ),
        "Credit Score": percentile_of_value(reference_df["Credit_Score"], form_data["Credit_Score"]),
        "Savings": percentile_of_value(reference_df["Savings"], form_data["Savings"]),
        "Loan Amount": percentile_of_value(reference_df["Loan_Amount"], form_data["Loan_Amount"]),
        "DTI Ratio": percentile_of_value(reference_df["DTI_Ratio"], form_data["DTI_Ratio"]),
    }

    strengths = []
    watchouts = []

    if form_data["Credit_Score"] >= reference_df["Credit_Score"].quantile(0.75):
        strengths.append("Credit score is in the stronger range of the training data.")
    elif form_data["Credit_Score"] <= reference_df["Credit_Score"].quantile(0.25):
        watchouts.append("Credit score sits in the lower quartile of past applicants.")

    if form_data["DTI_Ratio"] <= reference_df["DTI_Ratio"].quantile(0.25):
        strengths.append("Debt-to-income ratio is conservative compared with the dataset.")
    elif form_data["DTI_Ratio"] >= reference_df["DTI_Ratio"].quantile(0.75):
        watchouts.append("Debt-to-income ratio is elevated relative to most approved profiles.")

    if total_income >= (
        reference_df["Applicant_Income"] + reference_df["Coapplicant_Income"]
    ).quantile(0.5):
        strengths.append("Household income is above the median applicant profile.")
    else:
        watchouts.append("Household income is below the dataset median, which can reduce cushion.")

    if savings_to_loan >= 0.5:
        strengths.append("Savings create a healthy reserve against the requested loan amount.")
    elif savings_to_loan < 0.2:
        watchouts.append("Savings buffer is thin relative to the requested loan amount.")

    if collateral_cover >= 1.0:
        strengths.append("Collateral fully covers the requested loan amount.")
    elif collateral_cover < 0.6:
        watchouts.append("Collateral coverage is light for the requested loan size.")

    if form_data["Existing_Loans"] >= 3:
        watchouts.append("Existing loan count is already high.")
    elif form_data["Existing_Loans"] == 0:
        strengths.append("No existing loans improves repayment flexibility.")

    if form_data["Employment_Status"] == "Salaried":
        strengths.append("Salaried employment is typically viewed as stable in the model inputs.")
    if form_data["Employment_Status"] == "Unemployed":
        watchouts.append("Unemployed status is a clear negative signal.")

    if probability >= 0.8:
        confidence_label = "High confidence"
    elif probability >= 0.6:
        confidence_label = "Moderate confidence"
    else:
        confidence_label = "Low confidence"

    return {
        "total_income": total_income,
        "income_to_loan": income_to_loan,
        "savings_to_loan": savings_to_loan,
        "collateral_cover": collateral_cover,
        "monthly_loan_load": monthly_loan_load,
        "percentiles": percentiles,
        "strengths": strengths[:4],
        "watchouts": watchouts[:4],
        "confidence_label": confidence_label,
    }


def inject_styles():
    st.markdown(
        """
        <style>
        :root {
            --bg: #edf3fa;
            --panel: rgba(255, 255, 255, 0.92);
            --panel-strong: #ffffff;
            --ink: #162033;
            --muted: #5f6c84;
            --line: rgba(23, 37, 61, 0.10);
            --line-strong: rgba(23, 37, 61, 0.16);
            --accent: #2f6fed;
            --accent-soft: rgba(47, 111, 237, 0.12);
            --navy: #13233f;
            --navy-soft: #1b3158;
            --success: #0f8a5f;
            --success-soft: rgba(15, 138, 95, 0.10);
            --danger: #cb4f3f;
            --danger-soft: rgba(203, 79, 63, 0.10);
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(47, 111, 237, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(15, 138, 95, 0.10), transparent 22%),
                linear-gradient(180deg, #f4f8fc 0%, #e9eff7 48%, #eef3fa 100%);
            color: var(--ink);
        }
        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        h1, h2, h3 {
            color: var(--ink);
            font-family: "Avenir Next", "Segoe UI", sans-serif;
            letter-spacing: -0.02em;
            font-weight: 700;
        }
        h4 {
            color: var(--ink);
            font-family: "Avenir Next", "Segoe UI", sans-serif;
            letter-spacing: -0.02em;
            font-weight: 650;
        }
        .hero-card, .panel-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 26px;
            box-shadow: 0 22px 50px rgba(20, 36, 64, 0.08);
        }
        .hero-card {
            background:
                linear-gradient(135deg, rgba(22, 32, 51, 0.98), rgba(27, 49, 88, 0.95)),
                linear-gradient(180deg, rgba(47, 111, 237, 0.18), transparent);
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 1.8rem 1.9rem;
            margin-bottom: 1.4rem;
            position: relative;
            overflow: hidden;
            box-shadow: 0 24px 60px rgba(17, 27, 46, 0.24);
        }
        .panel-card {
            padding: 1.15rem 1.2rem;
            margin-bottom: 1rem;
            background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(248,251,255,0.94));
        }
        .eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 0.74rem;
            color: rgba(198, 220, 255, 0.78);
            font-weight: 700;
        }
        .hero-title {
            color: #ffffff;
            font-size: 2.3rem;
            line-height: 1.05;
            margin: 0.35rem 0 0.6rem 0;
        }
        .hero-copy {
            color: rgba(232, 239, 251, 0.82);
            font-size: 1rem;
            line-height: 1.6;
            margin: 0;
            max-width: 760px;
        }
        .kpi-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
            margin-top: 1.25rem;
        }
        .kpi-chip {
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            backdrop-filter: blur(8px);
        }
        .kpi-label {
            color: rgba(212, 225, 247, 0.72);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .kpi-value {
            color: #ffffff;
            font-size: 1.35rem;
            font-weight: 700;
            margin-top: 0.2rem;
        }
        .section-label {
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--accent);
            margin-bottom: 0.5rem;
        }
        .callout {
            border-radius: 24px;
            padding: 1.15rem 1.2rem;
            margin-bottom: 1.1rem;
            border: 1px solid transparent;
            box-shadow: 0 16px 34px rgba(20, 36, 64, 0.08);
        }
        .callout.success {
            background: linear-gradient(180deg, rgba(15, 138, 95, 0.10), rgba(255,255,255,0.96));
            border-color: rgba(15, 138, 95, 0.16);
        }
        .callout.danger {
            background: linear-gradient(180deg, rgba(203, 79, 63, 0.10), rgba(255,255,255,0.96));
            border-color: rgba(203, 79, 63, 0.16);
        }
        .callout-title {
            font-weight: 700;
            font-size: 1.1rem;
            color: var(--ink);
        }
        .callout-copy {
            margin-top: 0.35rem;
            color: var(--muted);
        }
        .pill {
            display: inline-block;
            margin-right: 0.4rem;
            margin-bottom: 0.5rem;
            padding: 0.48rem 0.78rem;
            border-radius: 999px;
            background: linear-gradient(180deg, #ffffff, #f7faff);
            border: 1px solid var(--line-strong);
            font-size: 0.88rem;
            color: var(--ink);
            box-shadow: 0 8px 16px rgba(20, 36, 64, 0.04);
        }
        .small-note {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.6;
        }
        .form-heading {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin: 0 0 0.8rem 0;
        }
        .form-heading .title {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--ink);
            letter-spacing: -0.02em;
        }
        .form-heading .sub {
            font-size: 0.92rem;
            color: var(--muted);
            max-width: 420px;
            text-align: right;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.9rem;
            margin: 0.2rem 0 1rem 0;
        }
        .metric-card {
            background: linear-gradient(180deg, #ffffff 0%, #f6f9fd 100%);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1rem 1.05rem;
            min-height: 112px;
            box-shadow: 0 18px 36px rgba(20, 36, 64, 0.07);
        }
        .metric-card .metric-label {
            color: var(--muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
        }
        .metric-card .metric-value {
            color: var(--ink);
            font-size: 1.7rem;
            line-height: 1.1;
            margin-top: 0.55rem;
            font-weight: 750;
            letter-spacing: -0.03em;
        }
        .metric-card .metric-foot {
            margin-top: 0.55rem;
            color: var(--muted);
            font-size: 0.86rem;
        }
        .review-card {
            background: linear-gradient(180deg, #ffffff 0%, #f7faff 100%);
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 1.15rem 1.2rem;
            box-shadow: 0 18px 34px rgba(20, 36, 64, 0.06);
            height: 100%;
        }
        .review-title {
            color: var(--ink);
            font-size: 1.4rem;
            font-weight: 750;
            letter-spacing: -0.02em;
            margin-bottom: 0.9rem;
        }
        .review-list {
            margin: 0;
            padding-left: 1.15rem;
            color: var(--ink);
        }
        .review-list li {
            margin-bottom: 0.8rem;
            line-height: 1.65;
        }
        div[data-testid="stForm"] {
            background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(247,250,255,0.96));
            border: 1px solid var(--line);
            border-radius: 28px;
            padding: 1.15rem 1.15rem 0.4rem 1.15rem;
            box-shadow: 0 22px 48px rgba(20, 36, 64, 0.10);
        }
        label, .stMarkdown p, .stCaption, .stText {
            color: var(--ink);
        }
        div[data-testid="stNumberInput"] label,
        div[data-testid="stSelectbox"] label,
        div[data-testid="stTextInput"] label,
        div[data-testid="stMarkdownContainer"] p {
            color: var(--ink) !important;
        }
        div[data-baseweb="input"],
        div[data-baseweb="select"] > div,
        div[data-baseweb="base-input"] {
            background: #f8fbff !important;
            border: 1px solid var(--line-strong) !important;
            border-radius: 16px !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.8);
        }
        div[data-baseweb="input"] input,
        div[data-baseweb="select"] input {
            color: var(--ink) !important;
            font-weight: 600 !important;
        }
        div[data-baseweb="select"] svg,
        div[data-baseweb="input"] svg {
            fill: var(--muted) !important;
        }
        div[data-testid="stForm"] button[kind="secondaryFormSubmit"] {
            background: linear-gradient(135deg, var(--accent), #1e58cb) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 16px !important;
            min-height: 3.2rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.01em;
            box-shadow: 0 16px 32px rgba(47, 111, 237, 0.24);
        }
        div[data-testid="stForm"] button[kind="secondaryFormSubmit"]:hover {
            background: linear-gradient(135deg, #245fd8, #174cad) !important;
        }
        div[data-testid="stProgressBar"] > div > div {
            background: linear-gradient(90deg, var(--accent), #18a36b) !important;
            border-radius: 999px !important;
        }
        div[data-testid="stProgressBar"] > div {
            background: rgba(22, 32, 51, 0.12) !important;
            border-radius: 999px !important;
            height: 0.6rem !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.55rem;
            margin-bottom: 0.7rem;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.86);
            border-radius: 999px;
            padding: 0.5rem 1.05rem;
            border: 1px solid var(--line);
            color: var(--muted);
            font-weight: 650;
        }
        .stTabs [aria-selected="true"] {
            background: var(--navy) !important;
            color: #ffffff !important;
            border-color: var(--navy) !important;
            box-shadow: 0 12px 24px rgba(19, 35, 63, 0.18);
        }
        [data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 18px;
            overflow: hidden;
            box-shadow: 0 14px 28px rgba(20, 36, 64, 0.05);
        }
        @media (max-width: 900px) {
            .kpi-strip,
            .metric-grid {
                grid-template-columns: 1fr;
            }
            .form-heading {
                display: block;
            }
            .form-heading .sub {
                max-width: none;
                text-align: left;
                margin-top: 0.4rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(reference_df: pd.DataFrame):
    approval_rate = (reference_df["Loan_Approved"] == "Yes").mean() * 100
    median_credit = int(reference_df["Credit_Score"].median())
    median_loan = int(reference_df["Loan_Amount"].median())

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="eyebrow">CreditWise Decision Studio</div>
            <div class="hero-title">Loan Approval Dashboard</div>
            <p class="hero-copy">
                Run an applicant through the trained model, benchmark their profile against the
                reference dataset, and review a deeper underwriting-style summary instead of only
                a binary decision.
            </p>
            <div class="kpi-strip">
                <div class="kpi-chip">
                    <div class="kpi-label">Historical Approval Rate</div>
                    <div class="kpi-value">{approval_rate:.1f}%</div>
                </div>
                <div class="kpi-chip">
                    <div class="kpi-label">Median Credit Score</div>
                    <div class="kpi-value">{median_credit}</div>
                </div>
                <div class="kpi-chip">
                    <div class="kpi-label">Median Loan Amount</div>
                    <div class="kpi-value">{median_loan:,}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_notes(reference_df: pd.DataFrame):
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Benchmarks</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="small-note">
            Stronger profiles in this dataset generally combine a credit score above
            <b>{int(reference_df["Credit_Score"].quantile(0.75))}</b>,
            a DTI ratio below <b>{reference_df["DTI_Ratio"].quantile(0.25):.2f}</b>,
            and savings above <b>{int(reference_df["Savings"].quantile(0.50)):,}</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">What Changed</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="small-note">
            This version adds profile benchmarking, affordability ratios, confidence labeling,
            and a clearer review of strengths and watchouts around the prediction.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_result_banner(prediction: int, probability: float, confidence_label: str):
    approved = prediction == 1
    variant = "success" if approved else "danger"
    title = "Loan Approved" if approved else "Loan Rejected"
    copy = (
        f"The model estimates a {probability * 100:.1f}% approval probability. "
        f"Current signal strength: {confidence_label}."
    )
    st.markdown(
        f"""
        <div class="callout {variant}">
            <div class="callout-title">{title}</div>
            <div class="callout-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(analysis: dict, probability: float):
    metrics = [
        ("Approval Probability", f"{probability * 100:.1f}%", "Model confidence outcome"),
        ("Total Household Income", f"{analysis['total_income']:,.0f}", "Applicant plus coapplicant"),
        ("Collateral Coverage", f"{analysis['collateral_cover']:.2f}x", "Collateral to loan ratio"),
        ("Savings Buffer", f"{analysis['savings_to_loan']:.2f}x", "Savings relative to loan"),
    ]
    metric_html = "".join(
        [
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-foot">{foot}</div>
            </div>
            """
            for label, value, foot in metrics
        ]
    )
    st.markdown(f'<div class="metric-grid">{metric_html}</div>', unsafe_allow_html=True)


def render_signal_pills(strengths: list[str], watchouts: list[str]):
    st.markdown('<div class="section-label">Signal Summary</div>', unsafe_allow_html=True)
    if strengths:
        st.markdown("**Strengths**")
        st.markdown(
            "".join([f'<span class="pill">{item}</span>' for item in strengths]),
            unsafe_allow_html=True,
        )
    if watchouts:
        st.markdown("**Watchouts**")
        st.markdown(
            "".join([f'<span class="pill">{item}</span>' for item in watchouts]),
            unsafe_allow_html=True,
        )
    if not strengths and not watchouts:
        st.info("No standout strengths or risks were triggered for this profile.")


def render_review_card(title: str, items: list[str]):
    items_html = "".join([f"<li>{item}</li>" for item in items])
    st.markdown(
        f"""
        <div class="review-card">
            <div class="review-title">{title}</div>
            <ul class="review-list">{items_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="Loan Approval System", page_icon="💳", layout="wide")
    inject_styles()

    try:
        artifacts = load_artifacts()
        reference_df = load_reference_data()
    except Exception as exc:
        st.error(f"Unable to load required files: {exc}")
        st.stop()

    feature_columns = artifacts["feature_columns"]
    model = artifacts["model"]
    scaler = artifacts["scaler"]

    render_hero(reference_df)

    layout_left, layout_right = st.columns([1.5, 0.8], gap="large")

    with layout_right:
        render_sidebar_notes(reference_df)

    with layout_left:
        st.markdown(
            """
            <div class="form-heading">
                <div class="title">Applicant Profile</div>
                <div class="sub">
                    Enter the borrower details below to generate a cleaner underwriting summary
                    with benchmarks, affordability signals, and decision confidence.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.form("loan_prediction_form"):
            financial_col, profile_col = st.columns(2, gap="large")

            with financial_col:
                st.markdown("#### Financial Snapshot")
                applicant_income = st.number_input(
                    "Applicant Income",
                    min_value=0,
                    value=DEFAULTS["Applicant_Income"],
                    step=100,
                )
                coapplicant_income = st.number_input(
                    "Coapplicant Income",
                    min_value=0,
                    value=DEFAULTS["Coapplicant_Income"],
                    step=100,
                )
                loan_amount = st.number_input(
                    "Loan Amount",
                    min_value=0,
                    value=DEFAULTS["Loan_Amount"],
                    step=100,
                )
                savings = st.number_input(
                    "Savings",
                    min_value=0,
                    value=DEFAULTS["Savings"],
                    step=100,
                )
                collateral_value = st.number_input(
                    "Collateral Value",
                    min_value=0,
                    value=DEFAULTS["Collateral_Value"],
                    step=100,
                )
                dti_ratio = st.number_input(
                    "DTI Ratio",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(DEFAULTS["DTI_Ratio"]),
                    step=0.01,
                    format="%.2f",
                )

            with profile_col:
                st.markdown("#### Profile Details")
                age = st.number_input("Age", min_value=18, max_value=100, value=DEFAULTS["Age"])
                dependents = st.number_input(
                    "Dependents", min_value=0, max_value=10, value=DEFAULTS["Dependents"]
                )
                credit_score = st.number_input(
                    "Credit Score", min_value=300, max_value=900, value=DEFAULTS["Credit_Score"]
                )
                existing_loans = st.number_input(
                    "Existing Loans",
                    min_value=0,
                    max_value=20,
                    value=DEFAULTS["Existing_Loans"],
                )
                loan_term = st.selectbox(
                    "Loan Term (months)",
                    LOAN_TERM_OPTIONS,
                    index=LOAN_TERM_OPTIONS.index(DEFAULTS["Loan_Term"]),
                )
                employment_status = st.selectbox(
                    "Employment Status", EMPLOYMENT_OPTIONS, index=1
                )

            category_col1, category_col2, category_col3 = st.columns(3)
            with category_col1:
                marital_status = st.selectbox("Marital Status", MARITAL_OPTIONS)
                education_level = st.selectbox("Education Level", EDUCATION_OPTIONS)
            with category_col2:
                loan_purpose = st.selectbox("Loan Purpose", LOAN_PURPOSE_OPTIONS)
                property_area = st.selectbox("Property Area", PROPERTY_AREA_OPTIONS, index=1)
            with category_col3:
                gender = st.selectbox("Gender", GENDER_OPTIONS, index=1)
                employer_category = st.selectbox("Employer Category", EMPLOYER_OPTIONS, index=3)

            submitted = st.form_submit_button("Run Decision Analysis", use_container_width=True)

    if not submitted:
        return

    form_data = {
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Age": age,
        "Dependents": dependents,
        "Credit_Score": credit_score,
        "Existing_Loans": existing_loans,
        "DTI_Ratio": dti_ratio,
        "Savings": savings,
        "Collateral_Value": collateral_value,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Employment_Status": employment_status,
        "Marital_Status": marital_status,
        "Loan_Purpose": loan_purpose,
        "Property_Area": property_area,
        "Education_Level": education_level,
        "Gender": gender,
        "Employer_Category": employer_category,
    }

    try:
        input_df = build_feature_frame(form_data, feature_columns)
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        approval_probability = float(model.predict_proba(input_scaled)[0][1])
        analysis = build_analysis(form_data, approval_probability, reference_df)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    st.markdown("### Decision Review")
    render_result_banner(prediction, approval_probability, analysis["confidence_label"])
    render_metric_cards(analysis, approval_probability)

    overview_tab, benchmark_tab, input_tab = st.tabs(
        ["Overview", "Benchmarks", "Model Input"]
    )

    with overview_tab:
        top_left, top_right = st.columns([1.1, 0.9], gap="large")
        with top_left:
            st.markdown("#### Confidence Meter")
            st.progress(int(approval_probability * 100))
            st.caption(
                f"Model confidence band: {analysis['confidence_label']} based on the approval probability."
            )
            render_signal_pills(analysis["strengths"], analysis["watchouts"])

        with top_right:
            st.markdown("#### Affordability View")
            st.metric("Income / Loan Multiple", f"{analysis['income_to_loan']:.2f}x")
            st.metric("Base Monthly Loan Load", f"{analysis['monthly_loan_load']:,.0f}")
            st.metric("DTI Ratio", f"{form_data['DTI_Ratio']:.2f}")
            st.metric("Existing Loans", f"{form_data['Existing_Loans']}")

        summary_left, summary_right = st.columns(2, gap="large")
        with summary_left:
            notes = analysis["strengths"] or ["Profile is fairly balanced without major standout positives."]
            render_review_card("Analyst Notes", notes)
        with summary_right:
            review_points = analysis["watchouts"] or [
                "No obvious outlier risks were triggered from the benchmark rules."
            ]
            render_review_card("Review Points", review_points)

    with benchmark_tab:
        st.markdown("#### Dataset Positioning")
        benchmark_df = pd.DataFrame(
            {
                "Metric": list(analysis["percentiles"].keys()),
                "Applicant Percentile": [round(v, 1) for v in analysis["percentiles"].values()],
                "Interpretation": [
                    "Higher than more applicants" if name != "DTI Ratio" else "Higher DTI than more applicants"
                    for name in analysis["percentiles"].keys()
                ],
            }
        )
        st.dataframe(benchmark_df, use_container_width=True, hide_index=True)

        compare_col1, compare_col2 = st.columns(2, gap="large")
        with compare_col1:
            st.markdown("#### Raw Applicant Inputs")
            display_fields = {
                "Total Income": analysis["total_income"],
                "Credit Score": form_data["Credit_Score"],
                "DTI Ratio": form_data["DTI_Ratio"],
                "Savings": form_data["Savings"],
                "Loan Amount": form_data["Loan_Amount"],
                "Collateral Value": form_data["Collateral_Value"],
            }
            st.dataframe(
                pd.DataFrame(
                    {"Metric": list(display_fields.keys()), "Value": list(display_fields.values())}
                ),
                use_container_width=True,
                hide_index=True,
            )

        with compare_col2:
            st.markdown("#### Reference Medians")
            reference_fields = {
                "Total Income": float(
                    (reference_df["Applicant_Income"] + reference_df["Coapplicant_Income"]).median()
                ),
                "Credit Score": float(reference_df["Credit_Score"].median()),
                "DTI Ratio": float(reference_df["DTI_Ratio"].median()),
                "Savings": float(reference_df["Savings"].median()),
                "Loan Amount": float(reference_df["Loan_Amount"].median()),
                "Collateral Value": float(reference_df["Collateral_Value"].median()),
            }
            st.dataframe(
                pd.DataFrame(
                    {"Metric": list(reference_fields.keys()), "Median": list(reference_fields.values())}
                ),
                use_container_width=True,
                hide_index=True,
            )

        st.caption(
            "Percentiles are benchmarked against the historical dataset used in this project. "
            "For DTI ratio, a lower percentile is usually healthier."
        )

    with input_tab:
        st.markdown("#### Engineered Model Features")
        st.dataframe(input_df, use_container_width=True)
        st.caption(
            "This is the exact aligned feature frame sent into the scaler and model after encoding."
        )


if __name__ == "__main__":
    main()
