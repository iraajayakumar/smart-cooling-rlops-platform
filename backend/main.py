from fastapi import FastAPI
from simulator import Simulator
from metrics import Metrics
from schemas import State
from rl_stub import predict_action
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()

sim = Simulator()
metrics = Metrics()


@app.get("/")
def home():
    return {"message": "Backend is running!"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/state", response_model=State)
def get_state():
    return sim.get_state()


@app.post("/optimize", response_model=State)
def optimize():
    state = sim.get_state()
    action = predict_action(state.model_dump())
    logging.info(f"Predicted action: {action}")
    new_state = sim.step(action)
    metrics.record(new_state.temperature, new_state.workload, new_state.cooling)
    return new_state


@app.get("/metrics")
def get_metrics():
    return metrics.latest()