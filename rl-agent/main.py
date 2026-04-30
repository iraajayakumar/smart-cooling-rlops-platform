import os
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Smart Cooling RL Agent Placeholder")

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/model.pkl")
INFERENCE_MODE = os.getenv("INFERENCE_MODE", "api")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


class StateInput(BaseModel):
    temperature: float
    workload: float


@app.get("/")
def root():
    return {
        "message": "RL agent placeholder is running",
        "inference_mode": INFERENCE_MODE,
        "model_path": MODEL_PATH
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics_placeholder():
    return {"message": "Metrics endpoint placeholder"}


@app.post("/predict")
def predict(data: StateInput):
    cooling_action = 1 if data.temperature > 30 or data.workload > 0.75 else 0
    return {
        "temperature": data.temperature,
        "workload": data.workload,
        "cooling": cooling_action,
        "note": "placeholder inference response"
    }