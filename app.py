import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. PAGE CONFIG & UI ---
st.set_page_config(page_title="LoanGuard AI", page_icon="🏦", layout="centered")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main { background-color: #f4f7f6; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #1f77b4; color: white; font-weight: bold; }
    .reportview-container .main .block-container { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ASSET LOADING ---
@st.cache_resource
def load_model_assets():
    # Ensure these files are in the same folder as app.py
    model = joblib.load('loan_model.pkl')
    scaler = joblib.load('scaler.pkl')
    ohe = joblib.load('ohe.pkl')
    cols = joblib.load('columns.pkl')
    return model, scaler, ohe, cols

model, scaler, ohe, model_columns = load_model_assets()

# --- 3. HEADER ---
st.title("🏦 CreditWise Loan Approval System")
st.markdown("---")

# --- 4. INPUT FORM ---
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal")
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", 18, 100, 30)
        marital = st.selectbox("Marital Status", ["Single", "Married"])
        edu = st.selectbox("Education", ["Graduate", "Not Graduate"])
        dependents = st.selectbox("Dependents", [0, 1, 2, 3])
        area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    with col2:
        st.subheader("Financial")
        income = st.number_input("Monthly Income ($)", value=12000) # Set high for testing
        co_income = st.number_input("Co-applicant Income ($)", value=5000)
        score = st.slider("Credit Score", 300, 900, 750)
        loan_amt = st.number_input("Loan Amount ($)", value=5000)
        term = st.number_input("Term (Months)", value=48)
        dti = st.slider("DTI Ratio", 0.0, 1.0, 0.2)

# Sidebar for less frequent inputs
st.sidebar.header("Employment Details")
emp_status = st.sidebar.selectbox("Employment", ["Salaried", "Self-employed", "Unemployed"])
emp_cat = st.sidebar.selectbox("Employer Type", ["Private", "Government", "MNC", "Unemployed"])
purpose = st.sidebar.selectbox("Loan Purpose", ["Personal", "Car", "Business", "Home", "Education"])
existing_loans = st.sidebar.number_input("Existing Loans", 0, 10, 0)
savings = st.sidebar.number_input("Savings", value=10000)
collateral = st.sidebar.number_input("Collateral Value", value=20000)

# --- 5. PROCESSING & PREDICTION ---
if st.button("PREDICT APPROVAL STATUS"):
    # A. Label Encoding for Education (Matches Cell 19: Graduate=0, Not=1)
    edu_encoded = 0 if edu == "Graduate" else 1
    
    # B. Create Input DataFrame
    raw_df = pd.DataFrame({
        "Applicant_Income": [float(income)], "Coapplicant_Income": [float(co_income)],
        "Employment_Status": [emp_status], "Age": [float(age)],
        "Marital_Status": [marital], "Dependents": [float(dependents)],
        "Credit_Score": [float(score)], "Existing_Loans": [float(existing_loans)],
        "DTI_Ratio": [float(dti)], "Savings": [float(savings)],
        "Collateral_Value": [float(collateral)], "Loan_Amount": [float(loan_amt)],
        "Loan_Term": [float(term)], "Loan_Purpose": [purpose],
        "Property_Area": [area], "Education_Level": [edu_encoded],
        "Gender": [gender], "Employer_Category": [emp_cat]
    })

    # C. One-Hot Encoding (OHE)
    cat_cols = ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"]
    encoded_cats = ohe.transform(raw_df[cat_cols])
    encoded_df = pd.DataFrame(encoded_cats, columns=ohe.get_feature_names_out(cat_cols))
    
    # Combine
    processed_df = pd.concat([raw_df.drop(columns=cat_cols), encoded_df], axis=1)

    # D. Feature Engineering (Matches Cell 38: squaring and dropping originals)
    processed_df["DTI_Ratio_sq"] = processed_df["DTI_Ratio"]**2
    processed_df["Credit_Score_sq"] = processed_df["Credit_Score"]**2
    processed_df = processed_df.drop(columns=["DTI_Ratio", "Credit_Score"])

    # E. Alignment & Scaling
    processed_df = processed_df[model_columns]
    scaled_data = scaler.transform(processed_df)

    # F. Prediction
    prediction = model.predict(scaled_data)[0]
    probs = model.predict_proba(scaled_data)[0]

    # --- 6. RESULTS DISPLAY ---
    st.markdown("---")
    
    # NOTE: In LabelEncoder, 0 is often 'Approved' and 1 is 'Rejected'
    # Check your notebook: if y_train had 'Approved' first, then 0 = Approved.
    if prediction == 0: 
        st.balloons()
        st.success(f"### ✅ APPROVED")
        st.write(f"**Confidence:** {probs[0]:.2%}")
    else:
        st.error(f"### ❌ REJECTED")
        st.write(f"**Confidence:** {probs[1]:.2%}")

    # --- 7. DEBUG SECTION (Expand to see logic) ---
    with st.expander("See technical details"):
        st.write("Model Class Output:", prediction)
        st.write("Probabilities [Class 0, Class 1]:", probs)
        st.write("Processed Features:", processed_df)