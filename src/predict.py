import pickle

def load_model():
    with open("models/model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

def predict_value(data):
    try:
        prediction = model.predict(data)
        return prediction
    except Exception as e:
        print("Prediction error:", e)
        return None