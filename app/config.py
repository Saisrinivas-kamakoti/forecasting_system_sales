from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "Forecasting Case- Study.xlsx"
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"

DATE_COL = "Date"
STATE_COL = "State"
TARGET_COL = "Total"
CATEGORY_COL = "Category"

FORECAST_HORIZON_WEEKS = 8
VALIDATION_WEEKS = 8
WEEKLY_FREQ = "W-SUN"

