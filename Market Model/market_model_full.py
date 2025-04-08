import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import config
from datetime import datetime
from data_processor import MarketDataProcessor

MODEL_PATH = os.path.join(config.DATA_DIR, "market_lstm_model.keras")
SCALER_PATH = os.path.join(config.DATA_DIR, "market_scaler.save")

def preprocess_market_data(df: pd.DataFrame, features: list) -> pd.DataFrame:
    df = df.sort_values("date").dropna(subset=features)
    df = df[features + ["date"]].reset_index(drop=True)
    return df

def create_sequences(data: np.ndarray, sequence_length: int, target_index: int):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length, target_index])
    return np.array(X), np.array(y)

def train_market_model(csv_file_path: str):
    df = pd.read_csv(csv_file_path)
    features = ["market_price", "open_interest", "trading_volume"]
    sequence_length = config.MARKET_SEQUENCE_LENGTH
    target_column = "market_price"

    df = preprocess_market_data(df, features)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    joblib.dump(scaler, SCALER_PATH)

    target_index = features.index(target_column)
    X, y = create_sequences(scaled_data, sequence_length, target_index)

    split = int(len(X) * config.MARKET_TRAIN_SPLIT)
    X_train, y_train = X[:split], y[:split]

    model = Sequential([
        LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        validation_split=0.2,  # <-- this creates a validation set from training data
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
        verbose=1
)
    model.save(MODEL_PATH)
    return model

def predict_market(input_features: list) -> dict:
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise RuntimeError("Trained model or scaler not found. Train the model first.")

    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    sequence_length = config.MARKET_SEQUENCE_LENGTH
    sequence = np.array([input_features] * sequence_length)
    scaled_seq = scaler.transform(sequence)
    input_seq = np.expand_dims(scaled_seq, axis=0)

    predicted_price = model.predict(input_seq, verbose=0)[0][0]
    current_price = input_features[0]
    signal = "undervalued" if predicted_price > current_price else "overvalued"

    return {
        "predicted_market_price": round(float(predicted_price), 4),
        "current_market_price": current_price,
        "suggested_action": f"go long" if signal == "undervalued" else "go short",
        "market_sentiment": signal
    }

def collect_and_prepare_market_data_only():
    mdp = MarketDataProcessor()
    market_df = mdp.collect_market_data()

    # Simulate date (since Kalshi data has no timestamps)
    market_df["date"] = datetime.utcnow().date()

    try:
        market_df = market_df[["date", "last_price", "open_interest", "volume"]]
    except KeyError:
        raise RuntimeError("Expected market data fields ('last_price', 'open_interest', 'volume') not found.")

    market_df.rename(columns={
        "last_price": "market_price",
        "volume": "trading_volume"
    }, inplace=True)

    output_path = config.MARKET_OUTPUT_FILE
    os.makedirs(config.DATA_DIR, exist_ok=True)
    market_df.to_csv(output_path, index=False)
    return output_path
