import math
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from typing import Optional
import numpy as np
import pandas as pd
from model import PrecipitationModel, DataProcessor, train_precipitation_model
import config
from predict import predict_precip_for_month

DATA_FILE = config.NCEI_DATA_FILE

FEATURE_COLUMNS = [
    "precipitation",
    "month",
    "season",
    "precipitation_lag1",
    "precipitation_lag2",
    "precipitation_lag3",
]


app = FastAPI()

processor: DataProcessor
model: PrecipitationModel
data: pd.DataFrame
X, y, scalers = None, None, None

@app.on_event("startup")
def load_resources():
    global processor, model, data, X, y, scalers

    processor = DataProcessor()
    model      = PrecipitationModel(sequence_length=12)
    model.load_model()

    raw = pd.read_csv(DATA_FILE, parse_dates=["date"]).sort_values("date")

    raw["month"]  = raw["date"].dt.month
    raw["season"] = raw["month"].map(lambda m:
                        1 if m in [12, 1, 2] else
                        2 if m in [3, 4, 5] else
                        3 if m in [6, 7, 8] else 4)

    for lag in (1, 2, 3):
        raw[f"precipitation_lag{lag}"] = raw["precipitation"].shift(lag)

    monthly = raw.dropna().reset_index(drop=True)
    monthly_selected = monthly[["date"] + FEATURE_COLUMNS]

    X, y, scalers = processor.create_sequences(
        monthly_selected,
        sequence_length=12,
        target_column="precipitation"
    )

    model.set_feature_columns(FEATURE_COLUMNS)
    model.set_scalers(scalers)
    data = raw 

@app.get("/")
def redirect_to_docs():
    return RedirectResponse(url="/docs")

@app.post("/train-model")
def train_model():
    global model, X, y, scalers, data

    model = train_precipitation_model(config.NCEI_DATA_FILE)

    # Reload the same monthly file for fresh scalers
    raw = pd.read_csv(DATA_FILE, parse_dates=["date"]).sort_values("date")

    raw["month"]  = raw["date"].dt.month
    raw["season"] = raw["month"].map(lambda m:
                    1 if m in [12,1,2] else
                    2 if m in [3,4,5] else
                    3 if m in [6,7,8] else 4)
    for lag in (1, 2, 3):
        raw[f"precipitation_lag{lag}"] = raw["precipitation"].shift(lag)

    raw = raw.dropna().reset_index(drop=True)
    monthly_selected = raw[["date"] + FEATURE_COLUMNS]

    X, y, scalers = processor.create_sequences(
        monthly_selected,
        sequence_length=12,
        target_column="precipitation"
    )
    model.set_feature_columns(FEATURE_COLUMNS)
    model.set_scalers(scalers)

    return {"message": "Model retrained and reloaded with NCEI data."}


@app.get("/predict")
def predict(month: Optional[str] = None):
    if not month:
        return {"error": "Please provide a month like '2025-06'"}

    try:
        rain = predict_precip_for_month(month)
        if not math.isfinite(rain):
            return {"error": "Prediction was NaN/Inf – check training data"}
    except Exception as e:
        return {"error": str(e)}

    return {
        "month": month,
        "predicted_monthly_precipitation_inches": float(rain)  # ensure JSON‑safe
    }




