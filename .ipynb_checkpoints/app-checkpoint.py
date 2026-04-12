import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page configuration
st.set_page_config(page_title="Credit Wise Loan Approval", layout="wide")

# Load the saved artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('loan_model.pkl')
    scaler = joblib.load('scaler.pkl')
    ohe = joblib.load('ohe.pkl')
    cols = joblib.load('columns.pkl')
    return model, scaler, ohe, cols

model, scaler, ohe, model_columns = load_artifacts()

st.title("🏦 Credit Wise Loan Approval System")
st.write("Enter the applicant's details below to check if their loan will be approved.")

# Create the input form
with st.form("loan_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        app_income = st.number_input("Applicant Income ($)", min_value=0.0, value=5000.0)
        coapp_income = st.number_input("Coapplicant Income ($)", min_value=0.0, value=0.0)
        age = st.number_input("Age", min_value=18.0, max_value=100.0, value=30.0)
        dependents = st.selectbox("Dependents", [0.0, 1.0, 2.0, 3.0])
        credit_score = st.number_input("Credit Score", min_value=300.0, max_value=900.0, value=650.0)
        existing_loans = st.number_input("Number of Existing Loans", min_value=0.0, value=0.0)
        dti_ratio = st.number_input("Debt-to-Income (DTI) Ratio", min_value=0.0, max_value=1.0, value=0.3)

    with col2:
        savings = st.number_input("Savings ($)", min_value=0.0, value=2000.0)
        collateral_value = st.number_input("Collateral Value ($)", min_value=0.0, value=5000.0)
        loan_amount = st.number_input("Loan Amount Requested ($)", min_value=0.0, value=15000.0)
        loan_term = st.number_input("Loan Term (Months)", min_value=12.0, value=36.0)
        
        emp_status = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Unemployed"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married"])
        loan_purpose = st.selectbox("Loan Purpose", ["Personal", "Car", "Business", "Home", "Education"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        emp_category = st.selectbox("Employer Category", ["Private", "Government", "MNC", "Unemployed"])

    submit_button = st.form_submit_button(label="Predict Loan Approval")

# Processing and Prediction
if submit_button:
    # 1. Manual Label Encoding for Education (matches your Jupyter logic)
    edu_encoded = 0 if education == "Graduate" else 1
    
    # 2. Create raw dataframe
    input_data = {
        "Applicant_Income": [app_income],
        "Coapplicant_Income": [coapp_income],
        "Employment_Status": [emp_status],
        "Age": [age],
        "Marital_Status": [marital_status],
        "Dependents": [dependents],
        "Credit_Score": [credit_score],
        "Existing_Loans": [existing_loans],
        "DTI_Ratio": [dti_ratio],
        "Savings": [savings],
        "Collateral_Value": [collateral_value],
        "Loan_Amount": [loan_amount],
        "Loan_Term": [loan_term],
        "Loan_Purpose": [loan_purpose],
        "Property_Area": [property_area],
        "Education_Level": [edu_encoded],
        "Gender": [gender],
        "Employer_Category": [emp_category]
    }
    raw_df = pd.DataFrame(input_data)
    
    # 3. Apply One-Hot Encoding to categorical columns
    cat_cols = ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"]
    encoded_cats = ohe.transform(raw_df[cat_cols])
    encoded_df = pd.DataFrame(encoded_cats, columns=ohe.get_feature_names_out(cat_cols))
    
    # Combine numerical and encoded categorical data
    processed_df = pd.concat([raw_df.drop(columns=cat_cols), encoded_df], axis=1)
    
    # 4. Feature Engineering (Match Notebook Cell 36)
    processed_df["DTI_Ratio_sq"] = processed_df["DTI_Ratio"]**2
    processed_df["Credit_Score_sq"] = processed_df["Credit_Score"]**2
    processed_df = processed_df.drop(columns=["DTI_Ratio", "Credit_Score"])
    
    # 5. Reorder columns to exactly match training data
    processed_df = processed_df[model_columns]
    
    # 6. Apply Standard Scaler
    scaled_data = scaler.transform(processed_df)
    
    # 7. Predict
    prediction = model.predict(scaled_data)
    
    # 8. Display Results
    st.markdown("---")
    if prediction[0] == 1:
        st.success("🎉 **Congratulations! The loan is APPROVED.**")
        st.balloons()
    else:
        st.error("❌ **Sorry, the loan application is REJECTED based on the provided details.**")