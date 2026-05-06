import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from app.config import ARTIFACT_DIR, FORECAST_HORIZON_WEEKS
from app.schemas import ForecastPoint, ForecastResponse, HealthResponse


app = FastAPI(
    title="State Sales Forecasting API",
    description="Backend service for 8-week state-level sales forecasts.",
    version="1.0.0",
)


def _forecast_path() -> Path:
    return ARTIFACT_DIR / "forecast_8_weeks.csv"


def _summary_path() -> Path:
    return ARTIFACT_DIR / "training_summary.json"


def load_forecasts() -> pd.DataFrame:
    path = _forecast_path()
    if not path.exists():
        raise HTTPException(status_code=503, detail="Forecast artifacts not found. Run `python -m app.train` first.")
    return pd.read_csv(path)


@app.get("/health", response_model=HealthResponse)
def health():
    summary_exists = _summary_path().exists()
    trained_states = 0
    if summary_exists:
        with _summary_path().open("r", encoding="utf-8") as f:
            trained_states = len(json.load(f).get("states", []))
    return HealthResponse(model_artifacts_loaded=summary_exists and _forecast_path().exists(), trained_states=trained_states)


@app.get("/states")
def states():
    forecasts = load_forecasts()
    return {"states": sorted(forecasts["state"].unique().tolist())}


@app.get("/forecast/{state}", response_model=ForecastResponse)
def forecast_state(state: str, horizon_weeks: int = Query(default=FORECAST_HORIZON_WEEKS, ge=1, le=8)):
    forecasts = load_forecasts()
    state_rows = forecasts[forecasts["state"].str.lower() == state.lower()].head(horizon_weeks)
    if state_rows.empty:
        raise HTTPException(status_code=404, detail=f"No forecast found for state '{state}'.")
    predictions = [ForecastPoint(**row) for row in state_rows.to_dict(orient="records")]
    return ForecastResponse(state=state_rows.iloc[0]["state"], horizon_weeks=horizon_weeks, predictions=predictions)


@app.get("/forecast")
def forecast_all(horizon_weeks: int = Query(default=FORECAST_HORIZON_WEEKS, ge=1, le=8)):
    forecasts = load_forecasts()
    return {
        "horizon_weeks": horizon_weeks,
        "predictions": forecasts.groupby("state", group_keys=False).head(horizon_weeks).to_dict(orient="records"),
    }

