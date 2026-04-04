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


@app.get("/state", response_model=State)
def get_state():
    return sim.get_state()


@app.post("/optimize")
def optimize():
    state = sim.get_state()

    # TEMP: dummy action (later RL will replace)
    action = predict_action(state) 

    new_state = sim.step(action)

    metrics.log(new_state, sim.energy)
    logging.info(f"State: {state}, Action: {action}")

    return new_state


@app.get("/metrics")
def get_metrics():
    return metrics.get_metrics()
