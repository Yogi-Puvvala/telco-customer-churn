# Importing the data
import pickle

with open("data/processed/X_train.pkl", "rb") as f:
    X_train = pickle.load(f)

with open("data/processed/X_test.pkl", "rb") as f:
    X_test = pickle.load(f)

with open("data/processed/y_train.pkl", "rb") as f:
    y_train = pickle.load(f)

with open("data/processed/y_test.pkl", "rb") as f:
    y_test = pickle.load(f)

with open("data/processed/numerical_cols.pkl", "rb") as f:
    numerical_cols = pickle.load(f)

with open("data/processed/categorical_cols.pkl", "rb") as f:
    categorical_cols = pickle.load(f)

# =====================
# Logistic Regression
# =====================

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score

num_pipeline_lr = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline_lr = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

lr_preprocessor = ColumnTransformer([
    ("num", num_pipeline_lr, numerical_cols),
    ("cat", cat_pipeline_lr, categorical_cols)
])

lr = Pipeline([
    ("preprocessing", lr_preprocessor),
    ("smote", SMOTE(random_state = 44)),
    ("model", LogisticRegression(max_iter=1000))
])

lr.fit(X_train, y_train)

print("Training Score(LR):", lr.score(X_train, y_train))
print("Testing Score(LR):", lr.score(X_test, y_test))


# =====================
# Decision Tree
# =====================

num_pipeline_tree = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

cat_pipeline_tree = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

dt_preprocessor = ColumnTransformer([
    ("num", num_pipeline_tree, numerical_cols),
    ("cat", cat_pipeline_tree, categorical_cols)
])

dt = Pipeline([
    ("preprocessing", dt_preprocessor), 
    ("smote", SMOTE(random_state = 44)),
    ("model", DecisionTreeClassifier(max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=44))
])

dt.fit(X_train, y_train)

print("Training Score(DT):", dt.score(X_train, y_train))
print("Testing Score(DT):", dt.score(X_test, y_test))


# =====================
# XGBoost
# =====================

xgb = Pipeline([
    ("preprocessing", dt_preprocessor),
    ("smote", SMOTE(random_state = 44)),
    ("model", XGBClassifier(n_estimators=200, 
                            max_depth=4, 
                            learning_rate=0.05, 
                            eval_metric="logloss"))
])

xgb.fit(X_train, y_train)

print("Training Score(XGB):", xgb.score(X_train, y_train))
print("Testing Score(XGB):", xgb.score(X_test, y_test))


# ==============================
# Experiment Tracking
# ==============================

import mlflow
import os

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
)
mlflow.set_experiment("TELCOM-Customer-Churn")

with mlflow.start_run(run_name = "Logistic-Regression"):

    mlflow.log_params({"max_iter": 100})

    lr_pred = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_rec = recall_score(y_test, lr_pred)
    mlflow.log_metrics({
        "accuracy": lr_acc,
        "recall": lr_rec
    })

    mlflow.sklearn.log_model(lr, name = "logistic_pipeline")

with mlflow.start_run(run_name = "Decision-Trees"):

    mlflow.log_params({"max_depth" : 5,
                        "min_samples_split" : 10,
                        "min_samples_leaf" : 5,
                        "max_features" : "sqrt",
                        "random_state" : 44})
    
    dt_pred = dt.predict(X_test)
    dt_acc = accuracy_score(y_test, dt_pred)
    dt_rec = recall_score(y_test, dt_pred)
    mlflow.log_metrics({"accuracy": dt_acc,
                        "recall": dt_rec})
    
    mlflow.sklearn.log_model(dt, "decision_tree_pipeline")

with mlflow.start_run(run_name = "XGBoost"):

    mlflow.log_params({"n_estimators" : 200, 
                        "max_depth" : 4, 
                        "learning_rate" : 0.05, 
                        "eval_metric" : "logloss"})
    
    xgb_pred = xgb.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_rec = recall_score(y_test, xgb_pred)
    mlflow.log_metrics({"accuracy": xgb_acc,
                        "recall": xgb_rec})
    
    mlflow.sklearn.log_model(xgb, "XGBoost")
