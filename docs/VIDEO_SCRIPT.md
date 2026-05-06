# Short Video Script

Use this as a 3 to 5 minute walkthrough.

## 1. Introduction

Hello, this is my end-to-end data science assignment for building a production-ready time series forecasting system with an API. The objective is to forecast the next 8 weeks of sales for each state using the Excel case-study dataset.

## 2. Dataset Overview

The dataset contains state-level sales history with four columns: State, Date, Total, and Category. The target variable is Total sales. The system aggregates the data to weekly state-level sales, handles missing weeks, and fills missing values using time-aware interpolation.

## 3. Feature Engineering

For machine learning, I created lag features for t-1, t-7, and t-30, rolling mean and standard deviation features, and calendar features such as day of week, month, week of year, quarter, and a holiday flag. The training and validation split is chronological, using the last 8 weeks as validation to avoid leakage.

## 4. Models

The pipeline trains and compares four models: SARIMA, Facebook Prophet, XGBoost with engineered lag features, and an LSTM deep learning model. Each model is evaluated using MAE, RMSE, and MAPE.

## 5. Model Selection

For every state, the model with the lowest validation RMSE is selected automatically. The best model is retrained on the full historical data and saved as a model artifact. The final 8-week forecast is saved to a CSV artifact.

## 6. Backend API

The project exposes forecasts through a FastAPI service. The API has endpoints for health checking, listing states, getting one state's forecast, and getting all state forecasts. This design keeps training separate from serving, which is closer to a real backend architecture.

## 7. Demo

Show the commands:

```bash
python -m app.train
python run_api.py
```

Then open Swagger at:

```text
http://localhost:8000/docs
```

Call:

```text
/forecast/California
```

Explain that the response contains 8 weekly predictions, forecast dates, forecast values, and the selected model.

## 8. Closing

This solution covers data cleaning, feature engineering, multiple forecasting algorithms, automatic model comparison, saved artifacts, and REST API serving, making it an end-to-end forecasting backend system.

