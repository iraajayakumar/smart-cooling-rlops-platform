from fastapi import FastAPI
from simulator import DataCenterSimulator
from metrics import Metrics

app = FastAPI()

sim = DataCenterSimulator()
metrics = Metrics()

@app.get("/")
def home():
    return {"message": "Backend is running!"}


@app.get("/state")
def get_state():
    return sim.get_state()


@app.post("/optimize")
def optimize():
    state = sim.get_state()

    # TEMP: dummy action (later RL will replace)
    action = 1  

    new_state = sim.step(action)

    metrics.log(new_state, sim.energy)

    return new_state


@app.get("/metrics")
def get_metrics():
    return metrics.get_metrics()
