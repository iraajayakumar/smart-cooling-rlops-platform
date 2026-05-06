from fastapi import FastAPI, Response
from pydantic import BaseModel
from prometheus_client import (
    generate_latest,
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
)

from inference import CoolingAgent

app = FastAPI(title="Smart Cooling RL Agent Demo API")
agent = CoolingAgent()

PREDICTION_REQUESTS = Counter(
    "cooling_prediction_requests_total",
    "Total number of prediction requests"
)

CURRENT_TEMPERATURE = Gauge(
    "cooling_temperature_celsius",
    "Latest requested temperature in celsius"
)

CURRENT_WORKLOAD = Gauge(
    "cooling_workload_level",
    "Latest requested workload level"
)

CURRENT_COOLING = Gauge(
    "cooling_current_level",
    "Latest input cooling level"
)

PREDICTED_ACTION = Gauge(
    "cooling_predicted_action",
    "Predicted cooling action (0 low, 1 medium, 2 high)"
)

REQUEST_LATENCY = Histogram(
    "cooling_prediction_latency_seconds",
    "Latency of prediction requests"
)


class CoolingStateRequest(BaseModel):
    temperature: float
    workload: float
    cooling: int


@app.get("/")
def root():
    return {"message": "RL agent demo service is running", "mode": "demo"}


@app.get("/health")
def health():
    return {"status": "ok", "mode": "demo"}


@app.post("/predict")
def predict(state: CoolingStateRequest):
    import time
    start = time.perf_counter()

    payload = state.model_dump()
    result = agent.predict_with_confidence(payload)

    PREDICTION_REQUESTS.inc()
    CURRENT_TEMPERATURE.set(payload["temperature"])
    CURRENT_WORKLOAD.set(payload["workload"])
    CURRENT_COOLING.set(payload["cooling"])
    PREDICTED_ACTION.set(result["action"])
    REQUEST_LATENCY.observe(time.perf_counter() - start)

    return result


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)