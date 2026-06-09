from fastapi import FastAPI
from backend.simulator import Simulator
from backend.metrics import Metrics
from backend.schemas import State
from backend.rl_client import predict_action
from rl_engine.reward import compute_reward, reward_breakdown
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
    temperature_before = state["temperature"]
    action = predict_action(state)
    new_state = sim.step(action)
    reward = compute_reward(
        temperature=new_state["temperature"],
        energy=sim.energy,
        cooling=new_state["cooling"],
    )
    breakdown = reward_breakdown(
        temperature=new_state["temperature"],
        energy=sim.energy,
        cooling=new_state["cooling"],
    )
    metrics.log(
        new_state,
        sim.energy,
        action=action,
        reward=reward,
        done=False,
        temperature_before=temperature_before,
        reward_breakdown=breakdown,
    )
    logging.info(f"State: {state}, Action: {action}")
    return new_state


@app.get("/metrics")
def get_metrics():
    return metrics.get_metrics()
