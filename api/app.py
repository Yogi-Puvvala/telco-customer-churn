import pandas as pd
from src.predict import predict_value
from fastapi import FastAPI
from pydantic import BaseModel

class Data(BaseModel):
    tenure         : int
    MonthlyCharges : float
    TotalCharges   : float
    gender         : str
    SeniorCitizen  : str
    Partner        : str
    Dependents     : str
    InternetService: str
    Contract       : str
    PaymentMethod  : str

app = FastAPI()

@app.get("/")
def sayHie():
    return {"message": "Hie Yogi"}

@app.post("/predict")
def predict(user_data: Data):
    df = pd.DataFrame([user_data.model_dump()])

    predicted_class, confidence_score = predict_value(df)

    if predicted_class is None:
        return {"error": "Prediction failed"}

    res = "YES" if predicted_class == 1 else "NO"

    return {
        "Predicted Value" : res,
        "Confidence Score": confidence_score
    }