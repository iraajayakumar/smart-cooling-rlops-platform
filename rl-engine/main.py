from fastapi import FastAPI
from pydantic import BaseModel
from inference import CoolingAgent

app = FastAPI()

agent = CoolingAgent()


class CoolingStateRequest(BaseModel):
    temperature: float
    workload: float
    cooling: int


@app.get("/")
def root():
    return {"message": "RL agent service is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(state: CoolingStateRequest):
    result = agent.predict_with_confidence(state.model_dump())
    return result


@app.get("/metrics")
def metrics():
    return {
        "service": "rl-agent",
        "status": "up"
    }