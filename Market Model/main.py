import os
import time
import warnings
import logging

# suppress sklearn feature‑name warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but MinMaxScaler was fitted with feature names"
)

# suppress TensorFlow logs
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from datetime import datetime, timezone

import config
from data_processor import MarketDataProcessor
from market_model_full import train_market_model, MODEL_PATH, SCALER_PATH


def choose_price_from_row(row: pd.Series) -> float:
    """Return first available of yes_ask.close, yes_bid.close, or price.close (in cents)."""
    for col in ("yes_ask.close", "yes_bid.close", "price.close"):
        if col in row.index and pd.notna(row[col]):
            return row[col]
    raise KeyError("no price column found")


def run_rain_pipeline() -> None:
    # retrain LSTM on daily CSV
    os.makedirs(config.DATA_DIR, exist_ok=True)
    csv_path = os.path.join(config.DATA_DIR, "KXRAINNYCM_4inch_daily.csv")
    train_market_model(csv_path)

    # fetch this month's daily bars
    mdp = MarketDataProcessor()
    now = datetime.now(timezone.utc)
    month_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
    daily = mdp.candlesticks(
        config.SERIES_TICKER,
        config.CURRENT_RAIN_MARKET_TICKER,
        int(month_start.timestamp()),
        int(now.timestamp()),
        period_interval=config.DAILY_INTERVAL,
    )
    if daily.empty:
        return

    # build feature dataframe
    df = daily.copy()
    df["market_price"] = df.apply(lambda r: choose_price_from_row(r) / 100.0, axis=1)
    df["open_interest"] = df["open_interest"]
    df["trading_volume"] = df["volume"]
    df = (
        df[["time", "market_price", "open_interest", "trading_volume"]]
        .rename(columns={"time": "timestamp"})
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # add lags and rolling mean
    for lag in (1, 2):
        df[f"market_price_lag{lag}"] = df["market_price"].shift(lag)
        df[f"open_interest_lag{lag}"] = df["open_interest"].shift(lag)
        df[f"trading_volume_lag{lag}"] = df["trading_volume"].shift(lag)
    df["trading_volume_roll7"] = df["trading_volume"].rolling(7).mean()

    features = [
        "market_price",
        "market_price_lag1", "market_price_lag2",
        "open_interest", "open_interest_lag1", "open_interest_lag2",
        "trading_volume", "trading_volume_lag1", "trading_volume_lag2",
        "trading_volume_roll7",
    ]
    df = df.dropna(subset=features).reset_index(drop=True)
    if len(df) < config.MARKET_SEQUENCE_LENGTH:
        return

    # load model and scaler
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # predict next value
    window = df[features].values[-config.MARKET_SEQUENCE_LENGTH:]
    X = scaler.transform(window).reshape((1, config.MARKET_SEQUENCE_LENGTH, len(features)))
    forecast_price = float(model.predict(X, verbose=0)[0, 0])

    # fetch most recent minute bar for live price
    end_ts = int(time.time())
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

    # determine signal
    sentiment = "undervalued" if forecast_price > current_price else "overvalued"
    action = "go long" if sentiment == "undervalued" else "go short"

    # print results
    print(config.CURRENT_RAIN_MARKET_TICKER)
    print(f"Current price: {current_price:.4f}")
    print(f"Forecast:      {forecast_price:.4f}")
    print(f"Sentiment:     {sentiment} → {action}")


if __name__ == "__main__":
    run_rain_pipeline()
