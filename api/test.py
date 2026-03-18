import requests

url = "http://127.0.0.1:8000/predict"
payload = {
    "tenure": 12,
    "MonthlyCharges": 70.5,
    "TotalCharges": 845.0,
    "gender": "Male",
    "SeniorCitizen": "0",
    "Partner": "Yes",
    "Dependents": "No",
    "InternetService": "Fiber optic",
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check"
}

response = requests.post(url, json=payload)
print(response.json())
