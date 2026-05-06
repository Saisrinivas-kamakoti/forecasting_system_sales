from pydantic import BaseModel, Field


class ForecastPoint(BaseModel):
    state: str
    date: str
    forecast_sales: float
    model: str


class ForecastResponse(BaseModel):
    state: str
    horizon_weeks: int
    predictions: list[ForecastPoint]


class HealthResponse(BaseModel):
    status: str = "ok"
    model_artifacts_loaded: bool
    trained_states: int = Field(ge=0)

