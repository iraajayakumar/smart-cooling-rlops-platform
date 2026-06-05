from pydantic import BaseModel, Field


class State(BaseModel):
    temperature: float = Field(..., ge=15.0, le=90.0)
    workload: float = Field(..., ge=0.0, le=10.0)
    cooling: int = Field(..., ge=0, le=2)


class Action(BaseModel):
    action: int = Field(..., ge=0, le=2)


class MetricsResponse(BaseModel):
    temperature: float
    cooling: int = Field(..., ge=0, le=2)
    energy: float