import pickle

def load_model():
    with open("models/model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

def predict_value(data):
    try:
        prediction       = model.predict(data)
        probabilities    = model.predict_proba(data)

        predicted_class  = int(prediction[0])
        confidence_score = round(float(probabilities[0][predicted_class]) * 100, 2)

        return predicted_class, confidence_score

    except Exception as e:
        print("Prediction error:", e)
        return None, None