# Assignment Report

## Objective

Build a production-ready forecasting system that forecasts the next 8 weeks of sales for each state using historical sales data from the Excel case-study file.

## Data Understanding

The workbook contains one sheet with 8,084 rows and four fields:

| Field | Description |
| --- | --- |
| `State` | State-level forecasting group |
| `Date` | Historical observation date |
| `Total` | Sales amount |
| `Category` | Product category |

Initial profiling found:

- 43 states
- Date range: 2019-01-12 to 2023-12-03
- Category: Beverages
- No missing values in the source file
- Dates are not perfectly spaced, so the service resamples to weekly state-level data

## Data Cleaning

The ingestion layer:

1. Validates that `State`, `Date`, and `Total` exist.
2. Converts `Date` to datetime.
3. Converts `Total` to numeric.
4. Aggregates sales by state and weekly date.
5. Creates a complete weekly timeline per state.
6. Fills missing weekly values using time interpolation, forward fill, and backward fill.

## Feature Engineering

The XGBoost model uses explicit supervised learning features:

| Feature Type | Features |
| --- | --- |
| Lag | `lag_1`, `lag_7`, `lag_30` |
| Rolling mean | 4, 8, and 12 week rolling means |
| Rolling std | 4, 8, and 12 week rolling standard deviations |
| Calendar | day of week, month, week of year, quarter |
| Holiday | US holiday flag |

All lag and rolling features are shifted so they only use past values.

## Train and Validation Split

The split is chronological:

- Training: all observations except the final 8 weeks
- Validation: final 8 weeks

This avoids look-ahead bias and simulates a real forecasting task.

## Model Training

The system trains four mandatory models for every state:

1. SARIMA
2. Facebook Prophet
3. XGBoost with lag features
4. LSTM neural network

Each model predicts the validation period. The service computes MAE, RMSE, and MAPE for comparison.

## Model Selection

Model selection is automatic and state-specific:

1. Train all candidate models.
2. Score each model on the same validation window.
3. Select the model with the lowest RMSE.
4. Retrain the selected model on the full historical series.
5. Generate 8 future weekly forecasts.

This allows different states to use different best-performing algorithms.

## API Design

The API is built with FastAPI and serves saved forecast artifacts.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Confirms artifact availability |
| `GET /states` | Lists all forecastable states |
| `GET /forecast/{state}` | Returns the next 8 weeks for one state |
| `GET /forecast` | Returns forecasts for all states |

Training and serving are separated so API requests remain fast and stable.

## Deliverables

- Source code
- Model training pipeline
- REST API
- Dockerfile
- README
- Architecture document
- Video walkthrough script

