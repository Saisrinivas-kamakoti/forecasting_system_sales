# Forecasting System Sales

End-to-end time series forecasting backend for the assignment: forecast the next 8 weeks of sales for each state from the provided case-study dataset.

## Repository Structure

```text
forecasting_system_sales/
├── app/                # Training and API code
│   ├── train.py        # Model training, validation, selection, and forecast export
│   ├── api.py          # FastAPI app entrypoint
│   ├── main.py         # API routes
│   ├── utils.py        # Shared utility exports
│   ├── data.py         # Excel loading, cleaning, weekly aggregation
│   ├── features.py     # Lag, rolling, calendar, and holiday features
│   └── models/         # SARIMA, Prophet, XGBoost, and LSTM wrappers
├── data/               # Sample dataset
├── notebooks/          # Exploratory analysis and feature engineering notes
├── outputs/            # Prediction CSVs, summaries, and plots
│   └── plots/          # Forecast vs. actual preview plots
├── docs/               # Report, architecture, and video script
├── scripts/            # Helper scripts for preview outputs
├── requirements.txt    # Dependencies
├── Dockerfile          # Optional container deployment
└── README.md           # Documentation
```

## Objective

Forecast the next 8 weeks of sales for each state using historical sales data. The system handles missing dates and missing values, captures trend and seasonality, compares multiple models, selects the best model automatically, and serves predictions through a REST API.

## Dataset

Source file:

```text
data/Forecasting Case- Study.xlsx
```

Observed columns:

| Column | Description |
| --- | --- |
| `State` | Forecast group |
| `Date` | Historical date |
| `Total` | Sales amount |
| `Category` | Product category |

The dataset is aggregated to weekly state-level sales. Missing weekly dates are created per state, and missing values are filled with time interpolation plus forward/backward fill.

## Models Implemented

The training pipeline compares these mandatory models for every state:

- ARIMA/SARIMA using `statsmodels`
- Facebook Prophet
- XGBoost with lag features
- LSTM deep learning model using TensorFlow/Keras

The best model is selected per state using validation RMSE.

## Feature Engineering

The machine-learning feature set includes:

- Lag features: `t-1`, `t-7`, `t-30`
- Rolling mean and rolling standard deviation
- Day of week
- Month
- Week of year
- Quarter
- Holiday flag

All lag and rolling features are shifted so the model only learns from past values.

## Training Pipeline

The pipeline uses time-series validation:

1. Load Excel data.
2. Aggregate sales weekly by state.
3. Fill missing dates and values.
4. Split chronologically: final 8 weeks as validation.
5. Train SARIMA, Prophet, XGBoost, and LSTM.
6. Score models with MAE, RMSE, and MAPE.
7. Select the lowest-RMSE model per state.
8. Retrain the selected model on full history.
9. Export 8-week forecasts to `outputs/`.

## Installation

```bash
python3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Windows, if `python3.11` is unavailable, use:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Training

```bash
python -m app.train --data-path ./data/Forecasting Case- Study.xlsx
```

Training writes:

- `outputs/metrics.csv`
- `outputs/forecast_8_weeks.csv`
- `outputs/training_summary.json`
- `outputs/models/`

## Run API

```bash
uvicorn app.api:app --reload
```

Alternative:

```bash
python run_api.py
```

Swagger UI:

```text
http://localhost:8000/docs
```

## API Usage

Health check:

```text
GET /health
```

List available states:

```text
GET /states
```

Assignment-style prediction endpoint:

```text
GET /predict?state=Alabama&weeks=8
```

Path-style prediction endpoint:

```text
GET /forecast/California?horizon_weeks=8
```

All state forecasts:

```text
GET /forecast?weeks=8
```

## Example Response

```json
{
  "state": "Alabama",
  "horizon_weeks": 8,
  "predictions": [
    {
      "state": "Alabama",
      "date": "2023-12-10",
      "forecast_sales": 246273728.73,
      "model": "Sample seasonal-naive preview"
    }
  ]
}
```

## Outputs Included

The repository includes preview outputs so reviewers can inspect file formats without running the full training job:

- `outputs/forecast_8_weeks.csv`
- `outputs/sample_predictions.csv`
- `outputs/training_summary.json`
- `outputs/plots/alabama_forecast_preview.png`
- `outputs/plots/california_forecast_preview.png`
- `outputs/plots/florida_forecast_preview.png`
- `outputs/plots/new_york_forecast_preview.png`
- `outputs/plots/texas_forecast_preview.png`

These preview outputs are clearly marked as sample seasonal-naive previews. Running `python -m app.train` replaces them with outputs from the automatic SARIMA/Prophet/XGBoost/LSTM model-selection pipeline.

## Demo Video

The script for recording the required demo is available at:

```text
docs/VIDEO_SCRIPT.md
```

Recommended demo flow:

1. Show the repository structure.
2. Run `python -m app.train`.
3. Run `uvicorn app.api:app --reload`.
4. Open `http://localhost:8000/docs`.
5. Query `/predict?state=Alabama&weeks=8`.

Add the Google Drive or YouTube demo video link here after recording:

```text
Demo video: <paste link here>
```

## Optional Deployment

The included `Dockerfile` can be used to deploy the API to Render, Railway, Azure App Service, or another container-friendly platform.
