from pydantic import BaseModel

class State(BaseModel):
    temperature: float
    workload: float
    cooling: int


class Action(BaseModel):
    action: int


class MetricsResponse(BaseModel):
    temperature: float
    cooling: int
    energy: float
