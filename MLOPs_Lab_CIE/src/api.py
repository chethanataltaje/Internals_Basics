import joblib
import json
import os
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ----------------------------
# PATHS
# ----------------------------
MODEL_PATH = "../models/best_tuned_model.pkl"
LOG_PATH = "../logs/predictions.jsonl"
RESULT_PATH = "../results/step3_s4.json"

# ----------------------------
# LOAD MODEL
# ----------------------------
try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
except Exception as e:
    model = None
    model_loaded = False
    print("Model load error:", e)

# ----------------------------
# FASTAPI INIT
# ----------------------------
app = FastAPI()

# ----------------------------
# INPUT VALIDATION
# ----------------------------
class InputData(BaseModel):
    batch_size_tons: float = Field(..., ge=1, le=50)
    machine_count: int = Field(..., ge=1, le=10)
    alloy_grade: int = Field(..., ge=1, le=5)
    shift_type: int = Field(..., ge=1, le=3)

# Ensure folders exist
os.makedirs("../logs", exist_ok=True)
os.makedirs("../results", exist_ok=True)

# ----------------------------
# HEALTH ENDPOINT
# ----------------------------
@app.get("/heartbeat")
def heartbeat():
    return {
        "status": "healthy",
        "model_loaded": model_loaded
    }

# ----------------------------
# PREDICT ENDPOINT
# ----------------------------
@app.post("/score")
def predict(data: InputData):

    if not model_loaded:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Prepare input
    features = np.array([[
        data.batch_size_tons,
        data.machine_count,
        data.alloy_grade,
        data.shift_type
    ]])

    # Prediction
    prediction = float(model.predict(features)[0])

    # ----------------------------
    # LOGGING (JSONL)
    # ----------------------------
    log_entry = {
        "timestamp": str(datetime.now()),
        "input": data.dict(),
        "prediction": prediction,
        "endpoint": "/score"
    }

    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    # ----------------------------
    # SAVE STEP 3 RESULT JSON
    # ----------------------------
    result = {
        "health_endpoint": "/heartbeat",
        "predict_endpoint": "/score",
        "port": 9000,
        "health_response": {
            "status": "healthy",
            "model_loaded": model_loaded
        },
        "test_input": data.dict(),
        "prediction": prediction
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=4)

    return {
        "prediction": prediction
    }