import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="CreditWise | Loan Approval", page_icon="🏦", layout="wide")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('models/loan_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        ohe = joblib.load('models/ohe.pkl')
        cols = joblib.load('models/columns.pkl')
        return model, scaler, ohe, cols
    except Exception as e:
        st.error(f"Error loading files: {e}. Check if they are in the 'models' folder!")
        return None, None, None, None

model, scaler, ohe, model_columns = load_assets()

# --- UI STYLE ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #2e7d32; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏦 CreditWise Loan System")
st.info("Fill in the details below to check loan eligibility.")

if ohe:
    # --- AUTOMATIC CATEGORY SYNC ---
    # This prevents the "Unseen Category" error by pulling exactly what the model knows
    edu_options = list(ohe.categories_[0])
    emp_options = list(ohe.categories_[1])
    prop_options = list(ohe.categories_[2])
    intent_options = list(ohe.categories_[3])

    # --- INPUTS ---
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 100, 25)
        income = st.number_input("Annual Income", value=500000)
        emp_length = st.number_input("Employment Length (Years)", 0, 50, 5)
        education = st.selectbox("Education Level", edu_options)
        employer = st.selectbox("Employer Category", emp_options)

    with col2:
        loan_amount = st.number_input("Loan Amount", value=100000)
        cred_hist = st.number_input("Credit History Length", 0, 50, 3)
        credit_score = st.slider("Credit Score", 300, 900, 700)
        property_area = st.selectbox("Property Area", prop_options)
        loan_intent = st.selectbox("Loan Intent", intent_options)

    # --- PREDICTION ---
    if st.button("Check Eligibility"):
        # 1. Create Input DataFrame
        # IMPORTANT: Names must match your training data column names exactly
        raw_data = pd.DataFrame({
            'person_age': [age],
            'person_income': [income],
            'person_emp_length': [emp_length],
            'loan_amnt': [loan_amount],
            'cb_person_cred_hist_length': [cred_hist],
            'credit_score': [credit_score],
            'education_level': [education],
            'employer_category': [employer],
            'property_area': [property_area],
            'loan_intent': [loan_intent]
        })

        # 2. Feature Engineering (Match your notebook logic)
        raw_data['dti'] = raw_data['loan_amnt'] / raw_data['person_income']

        # 3. One-Hot Encoding
        cat_cols = ['education_level', 'employer_category', 'property_area', 'loan_intent']
        encoded_cats = ohe.transform(raw_data[cat_cols])
        encoded_df = pd.DataFrame(encoded_cats, columns=ohe.get_feature_names_out(cat_cols))

        # 4. Final Processing
        final_df = pd.concat([raw_data.drop(cat_cols, axis=1), encoded_df], axis=1)
        
        # Ensure column order matches training
        final_df = final_df.reindex(columns=model_columns, fill_value=0)
        
        # Scale & Predict
        final_scaled = scaler.transform(final_df)
        pred = model.predict(final_scaled)[0]
        prob = model.predict_proba(final_scaled)[0][1]

        # 5. Output
        st.markdown("---")
        if pred == 0:
            st.success(f"### Result: Approved ✅")
            st.balloons()
        else:
            st.error(f"### Result: Rejected ❌")
        st.write(f"Risk Score: {prob:.2f}")

else:
    st.warning("Please ensure your .pkl files are in the 'models' folder to start.")