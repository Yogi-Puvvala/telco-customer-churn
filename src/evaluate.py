import mlflow
import pickle

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("TELCOM-Customer-Churn")

experiment = mlflow.get_experiment_by_name("TELCOM-Customer-Churn")
experiment_id = experiment.experiment_id

runs = mlflow.search_runs(experiment_ids=[experiment_id], max_results = 1, order_by = ["metrics.recall DESC"])

if runs.empty:
    raise Exception("No runs found! Run training pipeline first.")

best_run = runs.iloc[0]
run_id = best_run.run_id
run_name = best_run["tags.mlflow.runName"]

print("Run ID:", run_id)
print("Run Name:", run_name)

mapping = {
    "Logistic-Regression": "logistic_pipeline",
    "Decision-Trees": "decision_tree_pipeline",
    "XGBoost": "xgboost_pipeline"
}

model_path = mapping[run_name]
best_model = mlflow.sklearn.load_model(f"runs:/{run_id}/{model_path}")

with open("data/processed/y_test.pkl", "rb") as f:
    y_test = pickle.load(f)
with open("data/processed/X_test.pkl", "rb") as f:
    X_test = pickle.load(f)

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]  # for ROC curve

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Confusion Matrix")
plt.savefig("plots/confusion_matrix.png")
plt.close()

from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color="blue", label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="red", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.savefig("plots/roc_curve.png")
plt.close()



with open("models/model.pkl", "wb") as f:
    pickle.dump(best_model, f)