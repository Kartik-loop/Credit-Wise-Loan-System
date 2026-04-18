import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -------------------------------
# Load files
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "loan_model.pkl"))
num_imputer = joblib.load(os.path.join(BASE_DIR, "num_imputer.pkl"))
cat_imputer = joblib.load(os.path.join(BASE_DIR, "cat_imputer.pkl"))

# (IMPORTANT: you must have saved scaler + columns)
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
columns = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))

# -------------------------------
# UI
# -------------------------------
st.title("💳 Loan Approval System")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
income = st.number_input("Applicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
credit_history = st.selectbox("Credit History", [1, 0])

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):

    try:
        # Raw input
        input_df = pd.DataFrame([{
            "Gender": gender,
            "Married": married,
            "ApplicantIncome": income,
            "LoanAmount": loan_amount,
            "Credit_History": credit_history
        }])

        # -------------------------------
        # 1. Imputation
        # -------------------------------
        num_cols = input_df.select_dtypes(include=np.number).columns
        cat_cols = input_df.select_dtypes(exclude=np.number).columns

        input_df[num_cols] = num_imputer.transform(input_df[num_cols])
        input_df[cat_cols] = cat_imputer.transform(input_df[cat_cols])

        # -------------------------------
        # 2. Encoding (IMPORTANT)
        # -------------------------------
        input_df = pd.get_dummies(input_df)

        # -------------------------------
        # 3. Align columns
        # -------------------------------
        input_df = input_df.reindex(columns=columns, fill_value=0)

        # -------------------------------
        # 4. Scaling
        # -------------------------------
        input_scaled = scaler.transform(input_df)

        # -------------------------------
        # 5. Prediction
        # -------------------------------
        prediction = model.predict(input_scaled)[0]

        if prediction == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

    except Exception as e:
        st.error(f"Error: {e}")