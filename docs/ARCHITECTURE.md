# Architecture

```mermaid
flowchart LR
    A["Excel Dataset"] --> B["Data Loader"]
    B --> C["Weekly State Aggregation"]
    C --> D["Missing Week and Value Handling"]
    D --> E["Time Series Split"]
    E --> F1["SARIMA"]
    E --> F2["Prophet"]
    E --> F3["XGBoost + Lag Features"]
    E --> F4["LSTM"]
    F1 --> G["Metric Comparison"]
    F2 --> G
    F3 --> G
    F4 --> G
    G --> H["Best Model Per State"]
    H --> I["8 Week Forecast Artifacts"]
    I --> J["FastAPI Prediction Service"]
```

## Service Boundary

Training is an offline batch job:

```bash
python -m app.train
```

Serving is a lightweight API that reads saved artifacts:

```bash
python run_api.py
```

This avoids slow model training during user-facing API requests.

