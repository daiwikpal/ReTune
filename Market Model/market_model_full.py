import os
import hashlib
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import config

MODEL_PATH = os.path.join(config.DATA_DIR, "market_lstm_model.keras")
SCALER_PATH = os.path.join(config.DATA_DIR, "market_scaler.save")
VERSION_PATH = os.path.join(config.DATA_DIR, "market_model.version")


def preprocess_data(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Sort by time and drop rows with missing features."""
    return df.sort_values("timestamp").dropna(subset=features).reset_index(drop=True)


def create_sequences(array: np.ndarray, seq_len: int, target_idx: int):
    """Convert array into sequences for LSTM."""
    X, y = [], []
    for i in range(len(array) - seq_len):
        X.append(array[i : i + seq_len])
        y.append(array[i + seq_len, target_idx])
    return np.array(X), np.array(y)


def train_market_model(csv_path: str) -> float:
    """Train LSTM on market data and return validation RMSE."""
    df = pd.read_csv(csv_path)

    # rename columns for consistency
    if "date" in df and "timestamp" not in df:
        df = df.rename(columns={"date": "timestamp"})
    if "close" in df and "market_price" not in df:
        df = df.rename(columns={"close": "market_price", "volume": "trading_volume"})

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # add lag and rolling features
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
    seq_len = config.MARKET_SEQUENCE_LENGTH

    df = preprocess_data(df, features)

    # scale features
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])
    joblib.dump(scaler, SCALER_PATH)

    # create sequences
    X, y = create_sequences(scaled, seq_len, features.index("market_price"))
    split = int(len(X) * config.MARKET_TRAIN_SPLIT)

    # build and train model
    model = Sequential([
        LSTM(64, input_shape=(seq_len, len(features))),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    history = model.fit(
        X[:split], y[:split],
        epochs=50,
        batch_size=16,
        validation_split=0.2,
        callbacks=[EarlyStopping("val_loss", patience=5, restore_best_weights=True)],
        verbose=0,
    )
    model.save(MODEL_PATH)

    # record version as SHA-1 of CSV
    with open(csv_path, "rb") as f:
        sha1 = hashlib.sha1(f.read()).hexdigest()
    with open(VERSION_PATH, "w") as f:
        f.write(sha1 + "\n")

    val_rmse = float(history.history["val_loss"][-1] ** 0.5)
    print(f"Model saved. val_RMSE={val_rmse:.4f}, version={sha1[:7]}")
    return val_rmse


def predict_from_sequence(df_seq: pd.DataFrame) -> float:
    """Load model and scaler, then predict next value from the most recent sequence."""
    features = [
        "market_price",
        "market_price_lag1", "market_price_lag2",
        "open_interest", "open_interest_lag1", "open_interest_lag2",
        "trading_volume", "trading_volume_lag1", "trading_volume_lag2",
        "trading_volume_roll7",
    ]
    seq_len = config.MARKET_SEQUENCE_LENGTH

    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    window = df_seq[features].values[-seq_len:]
    X = np.expand_dims(scaler.transform(window), axis=0)
    return round(float(model.predict(X, verbose=0)[0, 0]), 4)
