# Telco Customer Churn Prediction

A production-ready machine learning project that predicts customer churn for a telecom company. The project covers the full ML lifecycle — from raw data to a deployed web application — using a proper MLOps stack with experiment tracking, data versioning, and containerized deployment.

**Live App:** https://churn-frontend-ejth.onrender.com

---

## What This Project Does

Customer churn is one of the more expensive problems for any subscription-based business. This project builds a binary classification model that predicts whether a customer is likely to churn, based on their account and usage data. The goal was not just to train a model, but to build a pipeline that mirrors how this kind of work is actually done in production.

---

## Project Structure

```
telco-customer-churn/
├── api/                    # FastAPI application (prediction endpoint)
├── frontend/               # Streamlit UI
├── src/                    # ML pipeline scripts
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── models/                 # Saved model artifacts
├── Dockerfile.api
├── Dockerfile.frontend
├── docker-compose.yaml
├── dvc.yaml                # DVC pipeline definition
├── dvc.lock
└── requirements.txt
```

---

## ML Pipeline

The pipeline is defined in `dvc.yaml` and has three stages:

**Preprocess** — reads the raw CSV, handles missing values, encodes categorical features, and splits the data into train/test sets. Outputs are saved as pickle files for the downstream stages.

**Train** — loads the processed training data and trains the model. Experiment runs are tracked using MLflow.

**Evaluate** — loads the test set and the trained model, generates metrics, and saves a confusion matrix and ROC curve to the `plots/` directory.

To reproduce the full pipeline:

```bash
dvc repro
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| Data versioning | DVC |
| Experiment tracking | MLflow |
| Modeling | Scikit-learn, XGBoost, imbalanced-learn |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Containerization | Docker, Docker Compose |
| Deployment | Render |

---

## Running Locally

**Prerequisites:** Docker and Docker Compose installed.

Clone the repo and start all services:

```bash
git clone https://github.com/Yogi-Puvvala/telco-customer-churn.git
cd telco-customer-churn
docker-compose up --build
```

This spins up three services:

- MLflow tracking server at `http://localhost:5000`
- FastAPI backend at `http://localhost:8000`
- Streamlit frontend at `http://localhost:8501`

The frontend communicates with the API, which loads the trained model and returns a churn prediction with a probability score.

---

## Running the ML Pipeline Manually

If you want to run the pipeline outside of Docker:

```bash
pip install -r requirements.txt
dvc repro
```

Individual stages can also be run directly:

```bash
python src/preprocess.py
python src/train.py
python src/evaluate.py
```

---

## Deployment

The API and frontend are deployed as separate services on Render. Each has its own Dockerfile (`Dockerfile.api` and `Dockerfile.frontend`). The frontend is configured to point to the live API URL via the `API_URL` environment variable.

---

## Notes

- The raw data file is `data/raw/telo_churn.csv` and is tracked by DVC, not committed to Git directly.
- MLflow experiment logs are stored locally under `mlruns/` during development.
- Class imbalance in the dataset is handled using `imbalanced-learn`.