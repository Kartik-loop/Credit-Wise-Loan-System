# Credit-Wise Loan Approval System

A machine learning project and Streamlit dashboard for loan approval prediction. The repository covers the full flow from notebook-based model development to a styled web app that accepts applicant details, runs the saved model, and returns a decision review with confidence, benchmark comparisons, and risk notes.

## Overview

This project predicts whether a loan application is likely to be approved using structured applicant data such as:

- income
- employment status
- credit score
- existing loans
- savings
- collateral value
- loan amount and term

The repository includes both:

- a Jupyter notebook for data preparation, feature engineering, training, and artifact export
- a Streamlit application for interactive prediction and analysis

## Current App Features

The Streamlit app does more than a simple yes/no prediction. It now includes:

- a professional dashboard-style UI
- aligned preprocessing using saved training artifacts
- approval probability output
- affordability metrics such as income-to-loan, savings buffer, and collateral coverage
- benchmark comparisons against the historical dataset
- analyst-style strengths and watchouts
- a model input view showing the exact engineered feature frame sent to the classifier

## ML Workflow

The model development flow is centered in `Creditwise_loanSystem-Copy1.ipynb`.

High-level pipeline:

1. Load and inspect the loan dataset.
2. Clean missing values using imputers.
3. Encode categorical variables and transform selected fields.
4. Create engineered features such as squared credit-score and DTI terms.
5. Scale the final feature matrix.
6. Train the classifier and export reusable artifacts.
7. Use the saved artifacts inside the Streamlit app for inference.

## Model Artifacts

The app relies on these saved files:

- `loan_model.pkl` — trained classifier
- `num_imputer.pkl` — numerical imputer
- `cat_imputer.pkl` — categorical imputer
- `scaler.pkl` — fitted scaler used before prediction
- `feature_columns.pkl` — final feature schema expected by the model

These artifacts are loaded directly by `app.py`.

## Repository Structure

```text
Credit-Wise-Loan-System/
├── Creditwise_loanSystem-Copy1.ipynb
├── app.py
├── loan_approval_data.csv
├── loan_model.pkl
├── num_imputer.pkl
├── cat_imputer.pkl
├── scaler.pkl
├── feature_columns.pkl
├── requirements.txt
└── README.md
```

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

## Run Locally

1. Clone the repository.
2. Move into the project folder.
3. Install dependencies.
4. Start the Streamlit app.

```bash
git clone https://github.com/Kartik-loop/Credit-Wise-Loan-System.git
cd Credit-Wise-Loan-System
pip install -r requirements.txt
streamlit run app.py
```

## How Prediction Works In The App

When a user submits the form:

1. The app collects the applicant inputs.
2. Inputs are converted into the same feature layout used during training.
3. The feature row is aligned with `feature_columns.pkl`.
4. The saved scaler transforms the row.
5. The trained model predicts approval and probability.
6. The dashboard renders decision metrics, benchmark positioning, and review notes.

## Why This Project Matters

This project is a good example of practical ML deployment because it combines:

- data preprocessing
- feature engineering
- model persistence
- reproducible inference
- an interactive frontend for end users

It is useful both as a learning project and as a starter template for structured-decision ML apps.

## Author

Built by Kartik Goyal as an end-to-end machine learning and Streamlit project.
