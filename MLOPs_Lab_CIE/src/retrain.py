import pandas as pd
import numpy as np
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ----------------------------
# PATHS
# ----------------------------
TRAIN_PATH = "../data/training_data.csv"
NEW_PATH = "../data/new_data.csv"
RESULT_PATH = "../results/step4_s8.json"

# ----------------------------
# LOAD DATA
# ----------------------------
df_train = pd.read_csv(TRAIN_PATH)
df_new = pd.read_csv(NEW_PATH)

# Combine dataset
df_combined = pd.concat([df_train, df_new], ignore_index=True)

# ----------------------------
# SPLIT (ONLY ON ORIGINAL DATA)
# ----------------------------
X = df_train.drop("production_hours", axis=1)
y = df_train["production_hours"]

X_train_old, X_test, y_train_old, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# CHAMPION MODEL (OLD DATA)
# ----------------------------
champion_model = RandomForestRegressor(random_state=42)
champion_model.fit(X_train_old, y_train_old)

y_pred_old = champion_model.predict(X_test)
champion_rmse = np.sqrt(mean_squared_error(y_test, y_pred_old))

# ----------------------------
# RETRAINED MODEL (COMBINED DATA)
# ----------------------------
X_combined = df_combined.drop("production_hours", axis=1)
y_combined = df_combined["production_hours"]

retrained_model = RandomForestRegressor(random_state=42)
retrained_model.fit(X_combined, y_combined)

# IMPORTANT → evaluate on SAME test set
y_pred_new = retrained_model.predict(X_test)
retrained_rmse = np.sqrt(mean_squared_error(y_test, y_pred_new))

# ----------------------------
# IMPROVEMENT
# ----------------------------
improvement = champion_rmse - retrained_rmse
threshold = 0.3

if improvement >= threshold:
    action = "promoted"
else:
    action = "kept_champion"

# ----------------------------
# SAVE RESULT
# ----------------------------
output = {
    "original_data_rows": len(df_train),
    "new_data_rows": len(df_new),
    "combined_data_rows": len(df_combined),
    "champion_rmse": champion_rmse,
    "retrained_rmse": retrained_rmse,
    "improvement": improvement,
    "min_improvement_threshold": threshold,
    "action": action,
    "comparison_metric": "rmse"
}

os.makedirs("../results", exist_ok=True)

with open(RESULT_PATH, "w") as f:
    json.dump(output, f, indent=4)

print("Task 4 completed (correct test set usage).")