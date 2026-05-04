import pandas as pd
import numpy as np
import json
import os
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ----------------------------
# CONFIG
# ----------------------------
DATA_PATH = "../data/training_data.csv"
RESULT_PATH = "../results/step2_s2.json"
EXPERIMENT_NAME = "steelforge-production-hours"

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv(DATA_PATH)

X = df.drop("production_hours", axis=1)
y = df["production_hours"]

# ----------------------------
# SPLIT (MANDATORY)
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# PARAM GRID (EXACT FROM QUESTION)
# ----------------------------
param_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 7, 15],
    "min_samples_split": [2, 4]
}

# ----------------------------
# MODEL
# ----------------------------
model = RandomForestRegressor(random_state=42)

# ----------------------------
# RANDOM SEARCH (MANDATORY)
# ----------------------------
search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=10,
    scoring="neg_mean_absolute_error",
    cv=3,
    random_state=42,
    n_jobs=-1
)

# ----------------------------
# MLflow
# ----------------------------
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="tuning-steelforge"):

    # Fit search
    search.fit(X_train, y_train)

    # Log nested runs (each trial)
    for i, params in enumerate(search.cv_results_["params"]):
        with mlflow.start_run(run_name=f"trial_{i}", nested=True):
            mlflow.log_params(params)
            mae = -search.cv_results_["mean_test_score"][i]
            mlflow.log_metric("mae", mae)

    # Best results
    best_params = search.best_params_
    best_cv_mae = -search.best_score_

    # Evaluate on test set
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    best_mae = mean_absolute_error(y_test, y_pred)

    # Save tuned model
    os.makedirs("../models", exist_ok=True)
    joblib.dump(best_model, "../models/best_tuned_model.pkl")

# ----------------------------
# SAVE JSON
# ----------------------------
output = {
    "search_type": "random",
    "n_folds": 3,
    "total_trials": len(search.cv_results_["params"]),
    "best_params": best_params,
    "best_mae": best_mae,
    "best_cv_mae": best_cv_mae,
    "parent_run_name": "tuning-steelforge"
}

os.makedirs("../results", exist_ok=True)

with open(RESULT_PATH, "w") as f:
    json.dump(output, f, indent=4)

print("Task 2 completed. Results saved to:", RESULT_PATH)