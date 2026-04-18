import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model and preprocessors
model = joblib.load('loan_model.pkl')
num_imputer = joblib.load('num_imputer.pkl')
cat_imputer = joblib.load('cat_imputer.pkl')
features = joblib.load('feature_columns.pkl')

st.title("💳 Credit-Wise Loan Approval System")
st.write("Enter applicant details below to predict loan eligibility.")

# Create input fields based on your dataset columns
col1, col2 = st.columns(2)

with col1:
    income = st.number_input("Applicant Income", min_value=0)
    co_income = st.number_input("Coapplicant Income", min_value=0)
    age = st.slider("Age", 18, 100, 30)
    credit_score = st.number_input("Credit Score", 300, 900, 650)

with col2:
    loan_amount = st.number_input("Loan Amount Requested", min_value=0)
    loan_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60, 72, 84])
    employment = st.selectbox("Employment Status", ["Salaried", "Self-employed"])
    purpose = st.selectbox("Loan Purpose", ["Personal", "Car", "Business", "Education", "Home"])

# When the user clicks Predict
if st.button("Predict Approval Status"):
    # 1. Prepare the input data into a DataFrame
    # Note: Ensure all features used in training are present here
    input_data = pd.DataFrame([[income, co_income, employment, age, credit_score, loan_amount, loan_term, purpose]], 
                              columns=['Applicant_Income', 'Coapplicant_Income', 'Employment_Status', 
                                       'Age', 'Credit_Score', 'Loan_Amount', 'Loan_Term', 'Loan_Purpose'])
    
    # 2. Apply the same preprocessing (Imputation/Encoding) as your notebook
    # (The following is a simplified logic; ensure it matches your specific NB steps)
    prediction = model.predict(input_data)
    
    if prediction[0] == 'Yes':
        st.success("✅ Loan Approved!")
    else:
        st.error("❌ Loan Rejected.")