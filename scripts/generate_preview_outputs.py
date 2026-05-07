import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.config import DATA_PATH, DATE_COL, STATE_COL, TARGET_COL
from app.data import load_raw_data, prepare_weekly_state_sales


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _draw_plot(state: str, history: pd.DataFrame, forecast: pd.DataFrame, path: Path) -> None:
    values = list(history[TARGET_COL].astype(float)) + list(forecast["forecast_sales"].astype(float))
    width, height, margin = 1100, 620, 80
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(28)
    body_font = _font(18)

    draw.text((margin, 25), f"{state}: historical sales and 8-week forecast preview", fill=(20, 37, 63), font=title_font)
    draw.text(
        (margin, 58),
        "Preview output. Run python -m app.train to replace with selected SARIMA/Prophet/XGBoost/LSTM outputs.",
        fill=(80, 80, 80),
        font=body_font,
    )
    draw.line((margin, height - margin, width - margin, height - margin), fill=(80, 80, 80), width=2)
    draw.line((margin, margin, margin, height - margin), fill=(80, 80, 80), width=2)

    ymin, ymax = min(values), max(values)
    span = max(ymax - ymin, 1)
    total = len(values)

    def xy(index: int, value: float):
        x = margin + index * (width - 2 * margin) / (total - 1)
        y = (height - margin) - (value - ymin) * (height - 2 * margin - 45) / span
        return x, y

    historical_points = [xy(i, value) for i, value in enumerate(history[TARGET_COL].astype(float))]
    forecast_values = [history[TARGET_COL].iloc[-1]] + list(forecast["forecast_sales"].astype(float))
    forecast_points = [xy(len(historical_points) - 1 + i, value) for i, value in enumerate(forecast_values)]

    draw.line(historical_points, fill=(31, 97, 141), width=4)
    draw.line(forecast_points, fill=(218, 95, 36), width=4)
    draw.line((historical_points[-1][0], margin, historical_points[-1][0], height - margin), fill=(120, 120, 120), width=2)
    draw.text((margin, height - 55), "Last 52 historical weeks", fill=(31, 97, 141), font=body_font)
    draw.text((width - margin - 290, height - 55), "Next 8 forecast weeks", fill=(218, 95, 36), font=body_font)
    image.save(path)


def main() -> None:
    output_dir = Path("outputs")
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    weekly = prepare_weekly_state_sales(load_raw_data(DATA_PATH))
    rows = []
    plot_states = {"Alabama", "California", "Florida", "New York", "Texas"}

    for state, group in weekly.groupby(STATE_COL):
        group = group.sort_values(DATE_COL).reset_index(drop=True)
        recent = group[TARGET_COL].tail(8).astype(float)
        previous = group[TARGET_COL].iloc[-16:-8].astype(float) if len(group) >= 16 else recent
        growth = (recent.mean() - previous.mean()) / max(abs(previous.mean()), 1e-9)
        dates = pd.date_range(group[DATE_COL].max() + pd.offsets.Week(weekday=6), periods=8, freq="W-SUN")

        state_rows = []
        for index, date in enumerate(dates):
            forecast_sales = max(float(recent.iloc[index % len(recent)] * (1 + growth * 0.5)), 0.0)
            row = {
                "state": state,
                "date": date.date().isoformat(),
                "forecast_sales": round(forecast_sales, 2),
                "model": "Sample seasonal-naive preview",
            }
            rows.append(row)
            state_rows.append(row)

        if state in plot_states:
            safe_state = state.lower().replace(" ", "_")
            _draw_plot(
                state,
                group.tail(52)[[DATE_COL, TARGET_COL]],
                pd.DataFrame(state_rows),
                plots_dir / f"{safe_state}_forecast_preview.png",
            )

    forecast = pd.DataFrame(rows)
    forecast.to_csv(output_dir / "forecast_8_weeks.csv", index=False)
    forecast.head(40).to_csv(output_dir / "sample_predictions.csv", index=False)

    summary = {
        "note": "Preview outputs generated for repository review. Running python -m app.train replaces these with selected model outputs.",
        "states": sorted(weekly[STATE_COL].unique().tolist()),
        "horizon_weeks": 8,
        "preview_model": "Sample seasonal-naive preview",
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {len(forecast)} forecast rows and {len(list(plots_dir.glob('*.png')))} plots.")


if __name__ == "__main__":
    main()
