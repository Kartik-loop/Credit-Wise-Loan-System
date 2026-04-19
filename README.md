# Credit-Wise Loan Approval System

An end-to-end supervised machine learning project that predicts whether a loan application is likely to be approved based on applicant details such as income, employment status, credit score, loan amount, and loan purpose.

This project was built to practice the **complete ML workflow** — from data preprocessing and feature handling to model training, evaluation, artifact saving, and deployment using **Streamlit**.

---

## Project Overview

The goal of this project is to build a machine learning system that can assist in **loan approval prediction** using structured applicant data.

The project includes:

- Data loading and exploration
- Data cleaning and preprocessing
- Handling numerical and categorical features
- Model training and comparison
- Saving trained artifacts for reuse
- A Streamlit web app for interactive predictions

This repository is primarily centered around the **Jupyter Notebook workflow**, with the deployed app acting as a lightweight interface for testing the trained model.

---

## Problem Statement

Loan approval decisions depend on multiple applicant-related factors.  
This project aims to predict the approval status of a loan application using machine learning so that the process can be made faster, more consistent, and more data-driven.

---

## Tech Stack

- **Python**
- **Jupyter Notebook**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Joblib**
- **Streamlit**

---

## Repository Structure

```bash
Credit-Wise-Loan-System/
│
├── Creditwise_loanSystem-Copy1.ipynb   # Main notebook with full ML workflow
├── app.py                              # Streamlit app
├── loan_approval_data.csv              # Dataset
├── loan_model.pkl                      # Trained model
├── num_imputer.pkl                     # Numerical preprocessing artifact
├── cat_imputer.pkl                     # Categorical preprocessing artifact
├── feature_columns.pkl                 # Saved feature schema
├── requirements.txt                    # Project dependencies
└── README.md                           # Project documentation
