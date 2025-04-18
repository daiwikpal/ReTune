from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from typing import Optional
import numpy as np
import pandas as pd
from model import PrecipitationModel, DataProcessor, train_precipitation_model
import config
from predict import predict_precip_for_month


app = FastAPI()

processor: DataProcessor
model: PrecipitationModel
data: pd.DataFrame
X, y, scalers = None, None, None

@app.on_event("startup")
def load_resources():
    global processor, model, data, X, y, scalers
    processor = DataProcessor()
    model = PrecipitationModel(sequence_length=12)
    model.load_model()

    # api.py → load_resources()
    raw = pd.read_csv(config.OUTPUT_FILE, parse_dates=["date"])
    raw = raw.set_index("date")

    # 1) aggregate to month:
    monthly = raw.resample("M").agg({
        "precipitation": "sum",
        "temperature_max": "mean",
        "temperature_min": "mean",
        "humidity": "mean",
        "wind_speed": "mean",
        "pressure": "mean",
    }).reset_index()

    # 2) add month/season + lag features (exactly like training!)
    monthly["month"]  = monthly["date"].dt.month
    monthly["season"] = monthly["month"].map(lambda m:
                        1 if m in [12,1,2] else
                        2 if m in [3,4,5]  else
                        3 if m in [6,7,8]  else
                        4)
    for lag in (1,2,3):
        monthly[f"precipitation_lag{lag}"] = monthly["precipitation"].shift(lag)
    monthly = monthly.dropna().reset_index(drop=True)

    # 3) build your sequences exactly as you did at training time:
    X, y, scalers = processor.create_sequences(
        monthly,
        sequence_length=12,      # must be the same SEQUENCE_LENGTH you trained with
        target_column="precipitation"
    )
    model.set_scalers(scalers)
    data = raw

@app.get("/")
def redirect_to_docs():
    return RedirectResponse(url="/docs")

@app.post("/train-model")
def train_model():
    global model, X, y, scalers, data

    model = train_precipitation_model()
    data = pd.read_csv(config.OUTPUT_FILE)
    data["date"] = pd.to_datetime(data["date"])
    X, y, scalers = processor.create_sequences(data, sequence_length=30)
    model.set_scalers(scalers)

    return {"message": "Model retrained and reloaded."}

@app.get("/predict")
def predict(month: Optional[str] = None):
    if not month:
        return {"error": "Please provide a month like '2025-06'"}

    try:
        rain = predict_precip_for_month(month)
    except ValueError as e:
        return {"error": str(e)}
    except Exception:
        return {"error": "Something went wrong while predicting"}

    return {
        "month": month,
        "predicted_monthly_precipitation_inches": rain
    }




