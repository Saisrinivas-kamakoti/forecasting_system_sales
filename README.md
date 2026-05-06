# End-to-End Time Series Forecasting System with API

This project forecasts the next 8 weeks of sales for each state from the Excel case-study dataset. It is structured like a backend service: data ingestion, feature engineering, model training, model selection, persisted artifacts, and REST endpoints for predictions.

## Dataset

Source file: `data/Forecasting Case- Study.xlsx`

Observed columns:

| Column | Purpose |
| --- | --- |
| `State` | Forecast entity |
| `Date` | Historical sales date |
| `Total` | Sales value to forecast |
| `Category` | Product category, currently `Beverages` |

The pipeline aggregates the data to weekly state-level sales, fills missing weekly dates per state, and interpolates missing values where required.

## Models Implemented

The training job compares the following models for every state:

1. SARIMA using `statsmodels`
2. Facebook Prophet
3. XGBoost with lag and rolling-window features
4. LSTM neural network using TensorFlow/Keras

The best model is selected independently for each state using validation RMSE from a time-series split. The final chosen model is retrained on all available history and used to produce the 8-week forecast.

## Feature Engineering

The XGBoost model uses:

- Lag features: `t-1`, `t-7`, `t-30`
- Rolling mean and standard deviation: 4, 8, and 12 weeks
- Calendar features: day of week, month, week of year, quarter
- Holiday flag: built-in US holiday indicator

The split is chronological: the last 8 weeks are used for validation, so future data never leaks into training.

## Setup

```bash
cd "C:\Users\bhara\Downloads\forecasting_time_series_api"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy the Excel file into the project data folder:

```bash
copy "C:\Users\bhara\Downloads\Forecasting Case- Study.xlsx" "data\Forecasting Case- Study.xlsx"
```

## Train Models

```bash
python -m app.train
```

Outputs:

- `artifacts/metrics.csv`: model comparison by state
- `artifacts/forecast_8_weeks.csv`: final 8-week predictions
- `artifacts/training_summary.json`: selected model per state
- `artifacts/models/`: pickled best model per state

## Run API

```bash
python run_api.py
```

Open:

- Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- List states: `http://localhost:8000/states`
- One state forecast: `http://localhost:8000/forecast/California`
- All forecasts: `http://localhost:8000/forecast`

## Example API Response

```json
{
  "state": "California",
  "horizon_weeks": 8,
  "predictions": [
    {
      "state": "California",
      "date": "2023-12-10",
      "forecast_sales": 123456789.0,
      "model": "XGBoost"
    }
  ]
}
```

## Production Design Notes

- Separate modules for data preparation, features, metrics, model wrappers, training, and API.
- Model selection is automated by validation RMSE.
- Each state receives its own best model because state-level sales behavior can differ.
- REST API serves persisted forecast artifacts instead of retraining during requests.
- Dockerfile is included for deployment.

