import streamlit as st
import requests

st.header("TELCO-CUSTOMER-CHURN")
payload = {}

with st.form(key = "user_data"):
    payload["tenure"] = st.number_input(label = "Tenure", min_value = 0)
    payload["MonthlyCharges"] = st.number_input(label = "Monthly Charges", min_value = 0)
    payload["TotalCharges"] = st.number_input(label = "Total Charges", min_value = 0)
    payload["gender"] = st.selectbox(label = "Gender", options = ["Male", "Female"])
    payload["SeniorCitizen"] = "1" if st.selectbox(label = "Are you a seniour citizen?", options = ["Yes", "No"]) == "Yes" else "0"
    payload["Partner"] = st.selectbox(label = "Are you a partner?", options = ["Yes", "No"])
    payload["Dependents"] = st.selectbox(label = "Are you a dependent?", options = ["Yes", "No"])
    payload["InternetService"] = st.selectbox(label = "Internet Service", options = ["Fiber optic", "DSL", "No"])
    payload["Contract"] = st.selectbox(label = "Your Contract", options = ["Month-to-month", "Two year", "One year"])
    payload["PaymentMethod"] = st.selectbox(label = "Payment Method", options = ["Electronic check", "Electronic check", "Bank transfer (automatic)", "Credit card (automatic)"])

    submit = st.form_submit_button(label = "Submit")

if submit:
    url = url = "http://127.0.0.1:8000/predict"
    response = requests.post(url, json = payload)
    st.success(response.json())