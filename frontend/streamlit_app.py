import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title = "Telco Customer Churn Predictor",
    layout     = "centered"
)

st.title("Telco Customer Churn Predictor")
st.caption("Fill in the customer details to predict whether they will churn.")

st.divider()

# Section 1: Usage Details 
st.subheader("Usage Details")
col1, col2, col3 = st.columns(3)

tenure = col1.number_input(
    "Tenure (months)",
    min_value = 0,
    step      = 1
)
monthly_charges = col2.number_input(
    "Monthly Charges ($)",
    min_value = 0.0,
    step      = 0.01,
    format    = "%.2f"
)
total_charges = col3.number_input(
    "Total Charges ($)",
    min_value = 0.0,
    step      = 0.01,
    format    = "%.2f"
)

st.divider()

# Section 2: Customer Profile 
st.subheader("Customer Profile")
col1, col2, col3, col4 = st.columns(4)

gender = col1.selectbox(
    "Gender",
    ["Male", "Female"]
)
senior_raw = col2.selectbox(
    "Senior Citizen",
    ["No", "Yes"]
)
partner = col3.selectbox(
    "Partner",
    ["Yes", "No"]
)
dependents = col4.selectbox(
    "Dependents",
    ["Yes", "No"]
)

st.divider()

# Section 3: Service Details 
st.subheader("Service Details")
col1, col2, col3 = st.columns(3)

internet_service = col1.selectbox(
    "Internet Service",
    ["Fiber optic", "DSL", "No"]
)
contract = col2.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)
payment_method = col3.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

st.divider()

# Predict Button 
predict_btn = st.button("Predict Churn", use_container_width=True)

# Result 
if predict_btn:
    payload = {
        "tenure"         : int(tenure),
        "MonthlyCharges" : float(monthly_charges),
        "TotalCharges"   : float(total_charges),
        "gender"         : gender,
        "SeniorCitizen"  : "1" if senior_raw == "Yes" else "0",
        "Partner"        : partner,
        "Dependents"     : dependents,
        "InternetService": internet_service,
        "Contract"       : contract,
        "PaymentMethod"  : payment_method
    }

    with st.spinner("Predicting..."):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json = payload
            )

            if response.status_code == 200:
                result           = response.json()
                prediction       = result.get("Predicted Value",  "Unknown")
                confidence_score = result.get("Confidence Score", 0.0)

                st.divider()
                st.subheader("Prediction Result")

                col1, col2, col3 = st.columns(3)
                col1.metric(label="Churn Prediction", value=prediction)
                col2.metric(label="Confidence Score", value=f"{confidence_score}%")
                col3.metric(
                    label="Risk Level",
                    value="High" if confidence_score >= 70
                        else "Medium" if confidence_score >= 40
                        else "Low"
                )

                st.progress(min(max(int(confidence_score), 0), 100))

                if prediction == "YES":
                    st.error(
                        "This customer is likely to churn. "
                        "Consider offering a retention plan."
                    )
                else:
                    st.success(
                        "This customer is unlikely to churn."
                    )

        except Exception as e:
            st.error(f"Connection error: {e}")