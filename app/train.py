import argparse
import json
from datetime import datetime, timezone

import pandas as pd
import joblib

from app.config import ARTIFACT_DIR, DATA_PATH, DATE_COL, FORECAST_HORIZON_WEEKS, MODEL_DIR, STATE_COL, TARGET_COL, VALIDATION_WEEKS
from app.data import load_raw_data, prepare_weekly_state_sales, time_series_split
from app.metrics import mae, mape, rmse
from app.models.lstm import LstmForecaster
from app.models.ml import XGBoostForecaster
from app.models.statistical import ProphetForecaster, SarimaForecaster


def candidate_models():
    return [
        SarimaForecaster(),
        ProphetForecaster(),
        XGBoostForecaster(),
        LstmForecaster(),
    ]


def evaluate_model(model, train_df: pd.DataFrame, validation_df: pd.DataFrame) -> dict:
    model.fit(train_df)
    pred = model.predict(len(validation_df), train_df)
    y_true = validation_df[TARGET_COL].to_numpy()
    y_pred = pred.to_numpy()
    return {
        "model": model.name,
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "status": "ok",
        "error": "",
        "fitted_model": model,
    }


def train_all(data_path=DATA_PATH):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw_data(data_path)
    weekly = prepare_weekly_state_sales(raw)
    metrics_rows = []
    forecasts = []
    summary = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "horizon_weeks": FORECAST_HORIZON_WEEKS,
        "validation_weeks": VALIDATION_WEEKS,
        "states": [],
        "best_models": {},
    }

    for state, state_df in weekly.groupby(STATE_COL):
        state_df = state_df.sort_values(DATE_COL).reset_index(drop=True)
        train_df, validation_df = time_series_split(state_df, VALIDATION_WEEKS)
        state_results = []

        for model in candidate_models():
            try:
                result = evaluate_model(model, train_df, validation_df)
            except Exception as exc:
                result = {
                    "model": model.name,
                    "mae": None,
                    "rmse": None,
                    "mape": None,
                    "status": "failed",
                    "error": str(exc),
                    "fitted_model": None,
                }
            metrics_rows.append({k: v for k, v in result.items() if k != "fitted_model"} | {"state": state})
            if result["status"] == "ok":
                state_results.append(result)

        if not state_results:
            raise RuntimeError(f"No model could be trained for state={state}. Check dependencies and data.")

        best = sorted(state_results, key=lambda r: r["rmse"])[0]
        best_model = best["fitted_model"]
        best_model.fit(state_df)
        model_path = MODEL_DIR / f"{state.replace(' ', '_').lower()}__{best_model.name.lower()}.joblib"
        try:
            joblib.dump(best_model, model_path)
            persisted_model_path = str(model_path)
        except Exception as exc:
            persisted_model_path = f"Model persistence skipped: {exc}"

        future = best_model.predict(FORECAST_HORIZON_WEEKS, state_df)
        for date, value in future.items():
            forecasts.append(
                {
                    "state": state,
                    "date": pd.Timestamp(date).date().isoformat(),
                    "forecast_sales": float(value),
                    "model": best_model.name,
                }
            )

        summary["states"].append(state)
        summary["best_models"][state] = {
            "model": best_model.name,
            "rmse": best["rmse"],
            "mae": best["mae"],
            "mape": best["mape"],
            "model_path": persisted_model_path,
        }

    pd.DataFrame(metrics_rows).to_csv(ARTIFACT_DIR / "metrics.csv", index=False)
    pd.DataFrame(forecasts).to_csv(ARTIFACT_DIR / "forecast_8_weeks.csv", index=False)
    with (ARTIFACT_DIR / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=str(DATA_PATH), help="Path to the Excel dataset.")
    args = parser.parse_args()
    summary = train_all(data_path=args.data_path)
    print(json.dumps(summary["best_models"], indent=2))


if __name__ == "__main__":
    main()
