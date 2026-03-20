import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title = "Telco Customer Churn",
    page_icon  = "T",
    layout     = "centered"
)

# Custom CSS 
st.markdown("""
<style>
    .block-container { padding-top: 2rem; max-width: 700px; }
    .section-label {
        font-size: 11px;
        font-weight: 500;
        color: gray;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .stButton > button {
        width: 100%;
        padding: 10px;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Header 
st.markdown("## Telco Customer Churn")
st.markdown(
    "<p style='color:gray; font-size:14px; margin-top:-10px;'>"
    "Fill in the customer details below to predict churn likelihood."
    "</p>",
    unsafe_allow_html=True
)

st.divider()

payload = {}

with st.form(key="user_data"):

    # Usage Details 
    with st.container(border=True):
        st.markdown(
            "<p class='section-label'>Usage details</p>",
            unsafe_allow_html=True
        )
        c1, c2, c3 = st.columns(3)
        payload["tenure"] = c1.number_input(
            "Tenure (months)", min_value=0
        )
        payload["MonthlyCharges"] = c2.number_input(
            "Monthly charges ($)", min_value=0
        )
        payload["TotalCharges"] = c3.number_input(
            "Total charges ($)", min_value=0
        )

    # Customer Profile 
    with st.container(border=True):
        st.markdown(
            "<p class='section-label'>Customer profile</p>",
            unsafe_allow_html=True
        )
        c1, c2 = st.columns(2)
        payload["gender"] = c1.selectbox(
            "Gender", ["Male", "Female"]
        )
        payload["SeniorCitizen"] = "1" if c2.selectbox(
            "Senior citizen", ["No", "Yes"]
        ) == "Yes" else "0"

        c3, c4 = st.columns(2)
        payload["Partner"] = c3.selectbox(
            "Partner", ["Yes", "No"]
        )
        payload["Dependents"] = c4.selectbox(
            "Dependents", ["Yes", "No"]
        )

    # Service Details 
    with st.container(border=True):
        st.markdown(
            "<p class='section-label'>Service details</p>",
            unsafe_allow_html=True
        )
        c1, c2, c3 = st.columns(3)
        payload["InternetService"] = c1.selectbox(
            "Internet service",
            ["Fiber optic", "DSL", "No"]
        )
        payload["Contract"] = c2.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        payload["PaymentMethod"] = c3.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    submit = st.form_submit_button("Predict churn")

# Result
if submit:
    with st.spinner("Analyzing..."):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json = payload
            )

            if response.status_code == 200:
                result = response.json()

                st.divider()
                st.markdown(
                    "<p class='section-label'>Result</p>",
                    unsafe_allow_html=True
                )

                prediction = result.get("Prediction", "Unknown")
                confidence = result.get("Confidence_Score", 0.0)

                # Metric cards
                c1, c2 = st.columns(2)
                c1.metric("Prediction", prediction)
                c2.metric("Confidence", f"{confidence}%")

                # Confidence bar + message
                with st.container(border=True):
                    st.progress(min(max(int(confidence), 0), 100))

                    if prediction == "Yes":
                        st.error(
                            "This customer is likely to churn. "
                            "Consider offering a retention plan."
                        )
                    else:
                        st.success(
                            "This customer is unlikely to churn."
                        )

            else:
                st.error(
                    f"API Error {response.status_code}: {response.text}"
                )

        except Exception as e:
            st.error(f"Connection error: {e}")