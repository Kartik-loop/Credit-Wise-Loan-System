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


st.set_page_config(page_title="Loan Approval System", page_icon="💳", layout="centered")
st.title("Loan Approval System")
st.caption("Fill in the applicant details to estimate loan approval.")

try:
    artifacts = load_artifacts()
except Exception as exc:
    st.error(f"Unable to load model files: {exc}")
    st.stop()

feature_columns = artifacts["feature_columns"]
model = artifacts["model"]
scaler = artifacts["scaler"]

with st.form("loan_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
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
        age = st.number_input("Age", min_value=18, max_value=100, value=DEFAULTS["Age"])
        dependents = st.number_input(
            "Dependents", min_value=0, max_value=10, value=DEFAULTS["Dependents"]
        )
        credit_score = st.number_input(
            "Credit Score", min_value=300, max_value=900, value=DEFAULTS["Credit_Score"]
        )
        dti_ratio = st.number_input(
            "DTI Ratio",
            min_value=0.0,
            max_value=1.0,
            value=float(DEFAULTS["DTI_Ratio"]),
            step=0.01,
            format="%.2f",
        )

    with col2:
        existing_loans = st.number_input(
            "Existing Loans", min_value=0, max_value=20, value=DEFAULTS["Existing_Loans"]
        )
        savings = st.number_input(
            "Savings", min_value=0, value=DEFAULTS["Savings"], step=100
        )
        collateral_value = st.number_input(
            "Collateral Value", min_value=0, value=DEFAULTS["Collateral_Value"], step=100
        )
        loan_amount = st.number_input(
            "Loan Amount", min_value=0, value=DEFAULTS["Loan_Amount"], step=100
        )
        loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60, 72, 84], index=3)
        employment_status = st.selectbox("Employment Status", EMPLOYMENT_OPTIONS, index=1)

    marital_status = st.selectbox("Marital Status", MARITAL_OPTIONS)
    loan_purpose = st.selectbox("Loan Purpose", LOAN_PURPOSE_OPTIONS)
    property_area = st.selectbox("Property Area", PROPERTY_AREA_OPTIONS, index=1)
    education_level = st.selectbox("Education Level", EDUCATION_OPTIONS)
    gender = st.selectbox("Gender", GENDER_OPTIONS, index=1)
    employer_category = st.selectbox("Employer Category", EMPLOYER_OPTIONS, index=3)

    submitted = st.form_submit_button("Predict")

if submitted:
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
        approval_probability = model.predict_proba(input_scaled)[0][1]

        if prediction == 1:
            st.success("Loan Approved")
        else:
            st.error("Loan Rejected")

        st.metric("Approval Probability", f"{approval_probability * 100:.1f}%")
        st.dataframe(input_df, use_container_width=True)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
