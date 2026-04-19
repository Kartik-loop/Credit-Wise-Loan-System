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

NUMERIC_REFERENCE_COLUMNS = [
    "Applicant_Income",
    "Coapplicant_Income",
    "Credit_Score",
    "DTI_Ratio",
    "Savings",
    "Collateral_Value",
    "Loan_Amount",
    "Existing_Loans",
    "Age",
]


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
            --bg: #f5efe2;
            --panel: rgba(255, 252, 245, 0.88);
            --ink: #1f2a1f;
            --muted: #5d665a;
            --line: rgba(38, 54, 38, 0.12);
            --gold: #c88f2b;
            --forest: #234b3b;
            --success: #1f7a4c;
            --danger: #a5452d;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(200, 143, 43, 0.18), transparent 34%),
                radial-gradient(circle at top right, rgba(35, 75, 59, 0.20), transparent 28%),
                linear-gradient(180deg, #f6f0e4 0%, #efe7d5 100%);
            color: var(--ink);
        }
        .block-container {
            max-width: 1120px;
            padding-top: 2.2rem;
            padding-bottom: 3rem;
        }
        h1, h2, h3 {
            color: var(--ink);
            font-family: Georgia, "Times New Roman", serif;
            letter-spacing: -0.02em;
        }
        .hero-card, .panel-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 24px;
            box-shadow: 0 18px 40px rgba(54, 50, 36, 0.10);
            backdrop-filter: blur(12px);
        }
        .hero-card {
            padding: 1.6rem 1.8rem;
            margin-bottom: 1.2rem;
        }
        .panel-card {
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
        }
        .eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 0.74rem;
            color: var(--forest);
            font-weight: 700;
        }
        .hero-title {
            font-size: 2.2rem;
            line-height: 1.05;
            margin: 0.35rem 0 0.6rem 0;
        }
        .hero-copy {
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.6;
            margin: 0;
        }
        .kpi-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
            margin-top: 1.25rem;
        }
        .kpi-chip {
            background: rgba(255,255,255,0.72);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.9rem 1rem;
        }
        .kpi-label {
            color: var(--muted);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .kpi-value {
            color: var(--forest);
            font-size: 1.35rem;
            font-weight: 700;
            margin-top: 0.2rem;
        }
        .section-label {
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--forest);
            margin-bottom: 0.5rem;
        }
        .callout {
            border-radius: 22px;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
            border: 1px solid transparent;
        }
        .callout.success {
            background: rgba(31, 122, 76, 0.10);
            border-color: rgba(31, 122, 76, 0.16);
        }
        .callout.danger {
            background: rgba(165, 69, 45, 0.10);
            border-color: rgba(165, 69, 45, 0.16);
        }
        .callout-title {
            font-weight: 700;
            font-size: 1.1rem;
        }
        .callout-copy {
            margin-top: 0.35rem;
            color: var(--muted);
        }
        .pill {
            display: inline-block;
            margin-right: 0.4rem;
            margin-bottom: 0.5rem;
            padding: 0.4rem 0.7rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.72);
            border: 1px solid var(--line);
            font-size: 0.88rem;
        }
        .small-note {
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.5;
        }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.72);
            border: 1px solid var(--line);
            padding: 0.9rem 1rem;
            border-radius: 18px;
        }
        div[data-testid="stForm"] {
            background: rgba(255, 252, 245, 0.74);
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 1rem 1rem 0.2rem 1rem;
            box-shadow: 0 18px 40px rgba(54, 50, 36, 0.08);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.6);
            border-radius: 999px;
            padding: 0.45rem 0.95rem;
            border: 1px solid var(--line);
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
        st.markdown("### Applicant Profile")
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

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Approval Probability", f"{approval_probability * 100:.1f}%")
    metric_col2.metric("Total Household Income", f"{analysis['total_income']:,.0f}")
    metric_col3.metric("Collateral Coverage", f"{analysis['collateral_cover']:.2f}x")
    metric_col4.metric("Savings Buffer", f"{analysis['savings_to_loan']:.2f}x")

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
            st.markdown("#### Analyst Notes")
            notes = analysis["strengths"] or ["Profile is fairly balanced without major standout positives."]
            for note in notes:
                st.write(f"• {note}")
        with summary_right:
            st.markdown("#### Review Points")
            review_points = analysis["watchouts"] or [
                "No obvious outlier risks were triggered from the benchmark rules."
            ]
            for note in review_points:
                st.write(f"• {note}")

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
