import pandas as pd
import numpy as np
import os
import json
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------
# CONFIG
# ----------------------------
DATA_PATH = "../data/training_data.csv"
RESULT_PATH = "../results/step1_s1.json"
EXPERIMENT_NAME = "steelforge-production-hours"

# ----------------------------
# METRICS FUNCTION
# ----------------------------
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, r2, mape

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv(DATA_PATH)

X = df.drop("production_hours", axis=1)
y = df["production_hours"]

# ----------------------------
# SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# MLflow setup
# ----------------------------
mlflow.set_experiment(EXPERIMENT_NAME)

results = []

# ----------------------------
# MODELS
# ----------------------------
models = {
    "Ridge": Ridge(),
    "RandomForest": RandomForestRegressor(random_state=42)
}

best_model_name = None
best_mae = float("inf")

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        
        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        mae, rmse, r2, mape = compute_metrics(y_test, y_pred)

        # Log params
        for param, value in model.get_params().items():
            mlflow.log_param(param, value)

        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mape", mape)

        # Tag
        mlflow.set_tag("team", "ml_engineering")

        # Save model
        os.makedirs("../models", exist_ok=True)
        model_path = f"../models/{name}.pkl"
        import joblib
        joblib.dump(model, model_path)

        # Track results
        results.append({
            "name": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape
        })

        # Best model selection (by MAE)
        if mae < best_mae:
            best_mae = mae
            best_model_name = name

# ----------------------------
# SAVE JSON OUTPUT
# ----------------------------
output = {
    "experiment_name": EXPERIMENT_NAME,
    "models": results,
    "best_model": best_model_name,
    "best_metric_name": "mae",
    "best_metric_value": best_mae
}

os.makedirs("../results", exist_ok=True)

with open(RESULT_PATH, "w") as f:
    json.dump(output, f, indent=4)

print("Task 1 completed. Results saved to:", RESULT_PATH)