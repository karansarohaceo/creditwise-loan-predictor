import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="CreditWise Loan Predictor")

st.title("🏦 CreditWise Loan Approval Predictor")

st.write("Fill the applicant details below")

# -------------------------
# Numeric Inputs
# -------------------------

applicant_income = st.number_input("Applicant Income", min_value=0.0)

coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0)

age = st.number_input("Age", min_value=18, max_value=100)

dependents = st.number_input("Dependents", min_value=0)

existing_loans = st.number_input("Existing Loans", min_value=0)

savings = st.number_input("Savings", min_value=0.0)

collateral_value = st.number_input("Collateral Value", min_value=0.0)

loan_amount = st.number_input("Loan Amount", min_value=0.0)

loan_term = st.number_input("Loan Term (Months)", min_value=1)

education_level = st.number_input("Education Level", min_value=0)

dti_ratio = st.number_input("DTI Ratio", min_value=0.0)

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900
)

# -------------------------
# Categorical Inputs
# -------------------------

employment_status = st.selectbox(
    "Employment Status",
    ["Salaried", "Self-employed", "Unemployed"]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Married", "Single"]
)

loan_purpose = st.selectbox(
    "Loan Purpose",
    ["Business", "Car", "Education", "Home", "Personal"]
)

property_area = st.selectbox(
    "Property Area",
    ["Rural", "Semiurban", "Urban"]
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

employer_category = st.selectbox(
    "Employer Category",
    ["Government", "MNC", "Private", "Unemployed"]
)

# -------------------------
# One Hot Encoding
# -------------------------

employment_salaried = 1 if employment_status == "Salaried" else 0
employment_self = 1 if employment_status == "Self-employed" else 0
employment_unemp = 1 if employment_status == "Unemployed" else 0

marital_single = 1 if marital_status == "Single" else 0

loan_car = 1 if loan_purpose == "Car" else 0
loan_education = 1 if loan_purpose == "Education" else 0
loan_home = 1 if loan_purpose == "Home" else 0
loan_personal = 1 if loan_purpose == "Personal" else 0

property_semiurban = 1 if property_area == "Semiurban" else 0
property_urban = 1 if property_area == "Urban" else 0

gender_male = 1 if gender == "Male" else 0

emp_gov = 1 if employer_category == "Government" else 0
emp_mnc = 1 if employer_category == "MNC" else 0
emp_private = 1 if employer_category == "Private" else 0
emp_unemployed = 1 if employer_category == "Unemployed" else 0

# Feature Engineering
dti_ratio_sq = dti_ratio ** 2
credit_score_sq = credit_score ** 2

# -------------------------
# Prediction
# -------------------------

if st.button("Predict Loan Approval"):

    features = np.array([[
        applicant_income,
        coapplicant_income,
        age,
        dependents,
        existing_loans,
        savings,
        collateral_value,
        loan_amount,
        loan_term,
        education_level,
        employment_salaried,
        employment_self,
        employment_unemp,
        marital_single,
        loan_car,
        loan_education,
        loan_home,
        loan_personal,
        property_semiurban,
        property_urban,
        gender_male,
        emp_gov,
        emp_mnc,
        emp_private,
        emp_unemployed,
        dti_ratio_sq,
        credit_score_sq
    ]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")