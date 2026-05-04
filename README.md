# Internals_Basics
MLOPs Lab CIE Submission

# SteelForge Production Time Prediction (MLOps Project)

## Overview

This project implements a complete **end-to-end MLOps pipeline** for predicting steel batch production time. It includes model training, hyperparameter tuning, API deployment, logging, and retraining based on new data.

---

## Problem Statement

SteelForge needs to predict **production hours** to optimize furnace scheduling and shift planning.

---

## Dataset

Features:

* `batch_size_tons` (1–50)
* `machine_count` (1–10)
* `alloy_grade` (1–5)
* `shift_type` (1–3)

Target:

* `production_hours`

Datasets used:

* `training_data.csv`
* `new_data.csv` (for retraining)

---

## Pipeline Overview

1. Model Training (Ridge & RandomForest)
2. MLflow Experiment Tracking
3. Hyperparameter Tuning (RandomForest)
4. FastAPI Deployment
5. Prediction Logging
6. Retraining & Model Promotion

---

## Tasks Implemented

### Task 1: Model Training & MLflow

* Trained Ridge and RandomForest
* Logged metrics: MAE, RMSE, R², MAPE
* Selected best model based on MAE

---

### Task 2: Hyperparameter Tuning

* Used RandomizedSearchCV
* 3-fold cross-validation
* Tuned RandomForest
* Logged nested MLflow runs

---

### Task 3: FastAPI Deployment

Endpoints:

* `GET /heartbeat` → health check
* `POST /score` → prediction

Includes:

* Input validation (Pydantic)
* JSONL logging
* Automatic result generation

---

### Task 4: Retraining Pipeline

* Combined old + new data
* Compared models using RMSE
* Promotion rule applied:

  * Promote if improvement ≥ 0.3

---

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* MLflow
* FastAPI
* Uvicorn
* Joblib

---

## 📁 Project Structure

```
Internals_Basics/
└── MLOPs_Lab_CIE/
    ├── data/
    ├── logs/
    ├── models/
    ├── results/
    ├── src/
    │   ├── train.py
    │   ├── tune.py
    │   ├── api.py
    │   ├── retrain.py
    ├── requirements.txt
    └── README.md
```

---

## How to Run

### 1️⃣ Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

### 2️⃣ Train Model

```bash
cd src
python train.py
```

---

### 3️⃣ Tune Model

```bash
python tune.py
```

---

### 4️⃣ Run API

```bash
uvicorn api:app --host 127.0.0.1 --port 9000
```

Open:

```
http://127.0.0.1:9000/docs
```

---

### 5️⃣ Retrain Model

```bash
python retrain.py
```

---

## 🌐 API Example

### POST `/score`

```json
{
  "batch_size_tons": 17.2,
  "machine_count": 4,
  "alloy_grade": 2,
  "shift_type": 2
}
```

Response:

```json
{
  "prediction": 9.56
}
```

---

## Results

* Model comparison completed
* Hyperparameter tuning improved performance
* Retraining decision: **promoted**
* All outputs saved in `results/` as JSON

---

## Key Highlights

* End-to-end MLOps pipeline
* Experiment tracking with MLflow
* API deployment with validation
* Real-time prediction logging
* Automated retraining decision system

---

## Conclusion

This project demonstrates a production-ready MLOps workflow integrating training, deployment, monitoring, and retraining — ensuring scalable and reliable machine learning systems.
