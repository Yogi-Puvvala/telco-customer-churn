import pandas as pd
import pickle

# Loading data
df = pd.read_csv("data/raw/telo_churn.csv")

# Dropping un-necessary columns
df = df.drop([
    "customerID", "PaperlessBilling", "StreamingMovies",
    "StreamingTV", "TechSupport", "DeviceProtection",
    "OnlineBackup", "OnlineSecurity", "MultipleLines",
    "PhoneService"
], axis=1)

# Type Conversion
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df["SeniorCitizen"] = df["SeniorCitizen"].astype("object")

# Splitting features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Convert target for XGBoost & RF
y = y.str.strip().str.lower()
y = y.map({"no": 0, "yes": 1})

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=44
)

numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object"]).columns

with open("data/processed/X_train.pkl", "wb") as f:
    pickle.dump(X_train, f)

with open("data/processed/X_test.pkl", "wb") as f:
    pickle.dump(X_test, f)

with open("data/processed/y_train.pkl", "wb") as f:
    pickle.dump(y_train, f)

with open("data/processed/y_test.pkl", "wb") as f:
    pickle.dump(y_test, f)

with open("data/processed/numerical_cols.pkl", "wb") as f:
    pickle.dump(numerical_cols, f)

with open("data/processed/categorical_cols.pkl", "wb") as f:
    pickle.dump(categorical_cols, f)