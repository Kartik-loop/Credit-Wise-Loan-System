import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 1. SET PAGE CONFIG ---
st.set_page_config(
    page_title="CreditWise | Loan Approval System",
    page_icon="🏦",
    layout="centered"
)

# --- 2. LOAD ASSETS ---
@st.cache_resource
def load_model_assets():
    # Using os.path.join for better compatibility between Mac/Linux
    model_path = 'models/loan_model.pkl'
    scaler_path = 'models/scaler.pkl'
    ohe_path = 'models/ohe.pkl'
    cols_path = 'models/columns.pkl'
    
    # Error checking to prevent the crash you saw earlier
    if not os.path.exists(model_path):
        st.error(f"Missing file: {model_path}. Ensure it is in the 'models' folder!")
        return None, None, None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    ohe = joblib.load(ohe_path)
    cols = joblib.load(cols_path)
    return model, scaler, ohe, cols

model, scaler, ohe, model_columns = load_model_assets()

# --- 3. UI HEADER ---
st.title("🏦 CreditWise Loan Approval System")
st.markdown("""
Predict your loan eligibility in seconds. This system uses machine learning 
to analyze your financial profile and provide a data-driven decision.
---
""")

if model:
    # --- 4. INPUT SECTION (Better UI Layout) ---
    with st.container():
        st.subheader("Personal & Financial Details")
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=25)
            income = st.number_input("Annual Income (INR)", min_value=0, value=500000)
            loan_amount = st.number_input("Loan Amount Requested", min_value=0, value=100000)

        with col2:
            credit_score = st.slider("Credit Score", 300, 900, 650)
            employment_length = st.number_input("Employment Length (Years)", 0, 50, 2)
            house_ownership = st.selectbox("House Ownership", ["RENT", "MORTGAGE", "OWN"])

    with st.container():
        st.subheader("Loan Specifications")
        col3, col4 = st.columns(2)
        
        with col3:
            loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT"])
        with col4:
            loan_grade = st.selectbox("Loan Grade (A is best)", ["A", "B", "C", "D", "E", "F"])

    # --- 5. PREDICTION LOGIC ---
    if st.button("Analyze Application", use_container_width=True):
        # Create DataFrame from inputs
        input_data = pd.DataFrame({
            'person_age': [age],
            'person_income': [income],
            'person_emp_length': [employment_length],
            'loan_amnt': [loan_amount],
            'cb_person_cred_hist_length': [employment_length], # Approximation
            'person_home_ownership': [house_ownership],
            'loan_intent': [loan_intent],
            'loan_grade': [loan_grade],
            'credit_score': [credit_score]
        })

        # Add Feature Engineering (Square DTI/Credit Score if your notebook used it)
        input_data['dti'] = input_data['loan_amnt'] / input_data['person_income']
        
        # Apply One-Hot Encoding
        categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade']
        encoded_cats = ohe.transform(input_data[categorical_cols])
        encoded_df = pd.DataFrame(encoded_cats, columns=ohe.get_feature_names_out(categorical_cols))
        
        # Combine and align with model columns
        final_df = pd.concat([input_data.drop(categorical_cols, axis=1), encoded_df], axis=1)
        final_df = final_df.reindex(columns=model_columns, fill_value=0)
        
        # Scale
        final_df_scaled = scaler.transform(final_df)
        
        # Predict
        prediction = model.predict(final_df_scaled)[0]
        probability = model.predict_proba(final_df_scaled)[0][1]

        # --- 6. RESULTS DISPLAY ---
        st.write("---")
        if prediction == 1:
            st.error(f"### Result: Loan Rejected ❌")
            st.write(f"Confidence: {probability:.2%}")
        else:
            st.success(f"### Result: Loan Approved ✅")
            st.write(f"Approval Probability: {(1-probability):.2%}")
            st.balloons()