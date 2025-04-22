
import os
import time
import warnings
import logging

# suppress sklearn warning about feature names
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but MinMaxScaler was fitted with feature names"
)
# suppress TensorFlow INFO & retracing warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)
# suppress Keras input_shape warning
warnings.filterwarnings(
    "ignore",
    message="Do not pass an `input_shape`/`input_dim` argument to a layer"
)

import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timezone
from pandas.tseries.offsets import MonthEnd

import config
from data_processor import MarketDataProcessor
from market_model_full import train_market_model, MODEL_PATH, SCALER_PATH

# Load model & scaler once at startup
_model = tf.keras.models.load_model(MODEL_PATH)
_scaler = joblib.load(SCALER_PATH)
# Prime the graph
_dummy = np.zeros((1, config.MARKET_SEQUENCE_LENGTH, 10))
_model.predict(_dummy, verbose=0)

app = FastAPI(
    title="Rainmaker Market‑Trend Model API",
    description="Train and predict market trends for the Rain in NYC this month contract",
    version="1.0",
)

class TrainResponse(BaseModel):
    message: str

class PredictLiveResponse(BaseModel):
    current_price: float
    forecast_price: float
    sentiment: str
    suggested_action: str

class PredictEOMResponse(BaseModel):
    current_price: float
    forecast_price: float
    target_date: datetime
    sentiment: str
    suggested_action: str

def choose_price_from_row(row: pd.Series) -> float:
    for col in ("price.close", "yes_ask.close", "yes_bid.close"):
        if col in row.index and pd.notna(row[col]):
            return row[col]
    raise KeyError("No valid price field in row")

@app.post("/train-model", response_model=TrainResponse)
def train_model():
    csv_path = os.path.join(config.DATA_DIR, "KXRAINNYCM_4inch_daily.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"CSV not found at {csv_path}")
    try:
        train_market_model(csv_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"message": "Model trained successfully."}

@app.get("/predict-live", response_model=PredictLiveResponse)
def predict_live():
    mdp = MarketDataProcessor()
    now_dt = datetime.now(timezone.utc)
    month_start = datetime(now_dt.year, now_dt.month, 1, tzinfo=timezone.utc)

    # 1) Fetch daily bars since month start
    daily = mdp.candlesticks(
        config.SERIES_TICKER,
        config.CURRENT_RAIN_MARKET_TICKER,
        int(month_start.timestamp()),
        int(now_dt.timestamp()),
        period_interval=config.DAILY_INTERVAL,
    )
    if daily.empty:
        raise HTTPException(status_code=404, detail="No daily data for this month.")

    # 2) Build features & drop NaNs
    df = daily.copy()
    df["market_price"]   = df.apply(lambda r: choose_price_from_row(r) / 100.0, axis=1)
    df["open_interest"]  = df["open_interest"]
    df["trading_volume"] = df["volume"]

    for lag in (1, 2):
        df[f"market_price_lag{lag}"]   = df["market_price"].shift(lag)
        df[f"open_interest_lag{lag}"]  = df["open_interest"].shift(lag)
        df[f"trading_volume_lag{lag}"] = df["trading_volume"].shift(lag)
    df["trading_volume_roll7"] = df["trading_volume"].rolling(7).mean()

    feature_cols = [
        "market_price",
        "market_price_lag1","market_price_lag2",
        "open_interest","open_interest_lag1","open_interest_lag2",
        "trading_volume","trading_volume_lag1","trading_volume_lag2",
        "trading_volume_roll7",
    ]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    if len(df) < config.MARKET_SEQUENCE_LENGTH:
        raise HTTPException(status_code=422, detail="Not enough data to predict")

    # 3) Next‑day forecast
    window = df[feature_cols].values[-config.MARKET_SEQUENCE_LENGTH:]
    X = _scaler.transform(window).reshape((1, config.MARKET_SEQUENCE_LENGTH, len(feature_cols)))
    forecast_price = float(_model.predict(X, verbose=0)[0,0])

    # 4) Live current price
    end_ts   = int(time.time())
    start_ts = end_ts - 12 * 3600
    minute = mdp.candlesticks(
        config.SERIES_TICKER,
        config.CURRENT_RAIN_MARKET_TICKER,
        start_ts,
        end_ts,
        period_interval=1,
    )
    current_price = None
    if not minute.empty:
        for _, row in minute[::-1].iterrows():
            try:
                current_price = choose_price_from_row(row) / 100.0
                break
            except KeyError:
                continue
    if current_price is None:
        info = mdp.get_market_info(config.CURRENT_RAIN_MARKET_TICKER)
        current_price = info.get("last_price", 0) / 100.0

    # 5) Sentiment & action
    sentiment = "undervalued" if forecast_price > current_price else "overvalued"
    action    = "go long" if sentiment == "undervalued" else "go short"

    return PredictLiveResponse(
        current_price=current_price,
        forecast_price=forecast_price,
        sentiment=sentiment,
        suggested_action=action,
    )

@app.get("/predict-eom", response_model=PredictEOMResponse)
def predict_end_of_month():
    mdp = MarketDataProcessor()
    now_dt = datetime.now(timezone.utc)
    month_start = datetime(now_dt.year, now_dt.month, 1, tzinfo=timezone.utc)

    # 1) Fetch daily history
    daily = mdp.candlesticks(
        config.SERIES_TICKER,
        config.CURRENT_RAIN_MARKET_TICKER,
        int(month_start.timestamp()),
        int(now_dt.timestamp()),
        period_interval=config.DAILY_INTERVAL,
    )
    if daily.empty:
        raise HTTPException(status_code=404, detail="No daily data for this month.")

    # 2) Build history & features
    df = daily.copy()
    df["market_price"]   = df.apply(lambda r: choose_price_from_row(r) / 100.0, axis=1)
    df["open_interest"]  = df["open_interest"]
    df["trading_volume"] = df["volume"]

    for lag in (1, 2):
        df[f"market_price_lag{lag}"]   = df["market_price"].shift(lag)
        df[f"open_interest_lag{lag}"]  = df["open_interest"].shift(lag)
        df[f"trading_volume_lag{lag}"] = df["trading_volume"].shift(lag)
    df["trading_volume_roll7"] = df["trading_volume"].rolling(7).mean()

    feature_cols = [
        "market_price",
        "market_price_lag1","market_price_lag2",
        "open_interest","open_interest_lag1","open_interest_lag2",
        "trading_volume","trading_volume_lag1","trading_volume_lag2",
        "trading_volume_roll7",
    ]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    # 3) Roll forward to month‑end
    last_ts   = df["timestamp"].iloc[-1]
    month_end = last_ts + MonthEnd(0)
    days_left = (month_end - last_ts).days
    price_eom = None

    for _ in range(days_left):
        window = df[feature_cols].values[-config.MARKET_SEQUENCE_LENGTH:]
        X = _scaler.transform(window).reshape((1, config.MARKET_SEQUENCE_LENGTH, len(feature_cols)))
        price_eom = float(_model.predict(X, verbose=0)[0,0])

        prev  = df.iloc[-1]
        prev2 = df.iloc[-2]
        last7 = df["trading_volume"].iloc[-6:].tolist() + [prev["trading_volume"]]
        new = {
            "timestamp":            prev["timestamp"] + pd.Timedelta(days=1),
            "market_price":         price_eom,
            "open_interest":        prev["open_interest"],
            "trading_volume":       prev["trading_volume"],
            "market_price_lag1":    price_eom,
            "market_price_lag2":    prev["market_price"],
            "open_interest_lag1":   prev["open_interest"],
            "open_interest_lag2":   prev2["open_interest"],
            "trading_volume_lag1":  prev["trading_volume"],
            "trading_volume_lag2":  prev2["trading_volume"],
            "trading_volume_roll7": sum(last7) / len(last7),
        }
        df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)

    # 4) Live current price
    end_ts   = int(time.time())
    start_ts = end_ts - 12 * 3600
    minute = mdp.candlesticks(
        config.SERIES_TICKER,
        config.CURRENT_RAIN_MARKET_TICKER,
        start_ts,
        end_ts,
        period_interval=1,
    )
    current_price = None
    if not minute.empty:
        for _, row in minute[::-1].iterrows():
            try:
                current_price = choose_price_from_row(row) / 100.0
                break
            except KeyError:
                continue
    if current_price is None:
        info = mdp.get_market_info(config.CURRENT_RAIN_MARKET_TICKER)
        current_price = info.get("last_price", 0) / 100.0

    # 5) Sentiment & action
    sentiment = "undervalued" if price_eom > current_price else "overvalued"
    action    = "go long" if sentiment == "undervalued" else "go short"

    return PredictEOMResponse(
        current_price=current_price,
        forecast_price=price_eom,
        target_date=month_end,
        sentiment=sentiment,
        suggested_action=action,
    )

@app.post("/update-model", response_model=TrainResponse)
def update_model():
    csv_path = os.path.join(config.DATA_DIR, "KXRAINNYCM_4inch_daily.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"CSV not found at {csv_path}")
    try:
        train_market_model(csv_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"message": "Model updated successfully."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002) 
