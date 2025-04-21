from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, conlist
from datetime import date
import pandas as pd

import config
from model import PrecipitationModel, train_precipitation_model, FEATURE_COLUMNS

app = FastAPI()

# single global model instance
model = PrecipitationModel(sequence_length=12)

@app.on_event("startup")
def startup():
    # try loading an existing model+scalers at startup
    try:
        model.load()
    except Exception:
        pass

class Record(BaseModel):
    date: date
    precipitation: float
    temperature_max: float
    temperature_min: float
    humidity: float
    wind_speed: float
    pressure: float
    temperature_range: float
    month: int
    season: int
    month_cos: float
    month_sin: float
    precipitation_lag1: float
    precipitation_lag2: float
    precipitation_lag3: float
    precipitation_lag7: float
    precipitation_rolling_mean_7d: float
    precipitation_rolling_max_7d: float

class ForecastInput(BaseModel):
    # exactly 12 records required
    recent: conlist(Record, min_items=12, max_items=12)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.post("/train-model", summary="Train or retrain the LSTM on the NCEI data")
def train_model_endpoint():
    # retrain from scratch
    train_precipitation_model(config.NCEI_DATA_FILE)
    # reload into our global model
    try:
        model.load()
    except Exception:
        pass
    return {"message": "Model trained and loaded."}

@app.post("/forecast", summary="Forecast next month from 12 months of input data")
def forecast(input: ForecastInput):
    # build a DataFrame from the incoming records
    recent_df = pd.DataFrame([r.dict() for r in input.recent])
    try:
        # pick out exactly the columns your model was trained on
        df_in = recent_df[["date"] + FEATURE_COLUMNS]
        pred = model.forecast_next(df_in)
    except Exception as e:
        # return any errors as HTTP 400
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "predicted_monthly_precipitation_inches": round(pred, 4)
    }








