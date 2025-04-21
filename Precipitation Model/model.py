"""
LSTM model for predicting NYC precipitation using the past 12 months of data.
"""
import os
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import config
from weather_data.data_processor import DataProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# these are the only columns we ever scale
FEATURE_COLUMNS = [
    "precipitation",
    "month",
    "season",
    "precipitation_lag1",
    "precipitation_lag2",
    "precipitation_lag3",
]

class PrecipitationModel:
    """
    LSTM model that, given 12 months of features, predicts the next month's precipitation.
    """
    def __init__(self, sequence_length: int = 12):
        self.sequence_length = sequence_length
        self.model: tf.keras.Model | None = None
        self.scalers: Dict[str, any] = {}
        os.makedirs(os.path.join(config.DATA_DIR, "models"), exist_ok=True)

    def build_model(self, input_shape: Tuple[int,int]) -> None:
        """Builds a two‑layer LSTM."""
        m = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, name="precipitation_output")
        ])
        m.compile(optimizer="adam", loss="mse", metrics=["mae"])
        self.model = m
        logger.info(f"Built LSTM with input shape {input_shape}")
        m.summary()

    def train(self,
              X_train: np.ndarray, y_train: np.ndarray,
              X_val:   np.ndarray, y_val:   np.ndarray,
              epochs: int = 100, batch_size: int = 32) -> None:
        """Trains with early stopping + best‑model checkpointing."""
        if self.model is None:
            self.build_model((X_train.shape[1], X_train.shape[2]))

        ckpt = os.path.join(config.DATA_DIR, "models", "precip_model.h5")
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                ckpt, monitor="val_loss", save_best_only=True
            )
        ]
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs, batch_size=batch_size,
            callbacks=callbacks, verbose=1
        )
        logger.info("Training complete, best model saved to %s", ckpt)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Returns MSE, RMSE, MAE & R² on held‑out test set."""
        preds = self.model.predict(X_test).flatten()
        mse = mean_squared_error(y_test, preds)
        return {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": mean_absolute_error(y_test, preds),
            "r2":   r2_score(y_test, preds)
        }

    def forecast_next(self, recent_df: pd.DataFrame) -> float:
        """
        Given the most recent `sequence_length` rows (with 'date' + FEATURE_COLUMNS),
        returns next‑month precipitation in original units.
        """
        # 1) Sort
        df = recent_df.copy().sort_values("date")
        # 2) Pull out only the columns we have scalers for:
        feats = df[FEATURE_COLUMNS]
        # 3) Normalize each column with its scaler
        arr = np.zeros(feats.shape)
        for i, col in enumerate(FEATURE_COLUMNS):
            arr[:, i] = self.scalers[col] \
                         .transform(feats[col].values.reshape(-1,1)) \
                         .flatten()
        # 4) Reshape to (1, seq_len, n_features)
        X_in = arr.reshape((1, self.sequence_length, feats.shape[1]))
        # 5) Predict then invert the precipitation scaler
        p_norm = self.model.predict(X_in)
        return float(
            self.scalers["precipitation"]
                .inverse_transform(p_norm)[0,0]
        )

    def save(self, path: str = None) -> None:
        """Save model & scalers."""
        if path is None:
            path = os.path.join(config.DATA_DIR, "models", "precip_model.h5")
        self.model.save(path)
        np.save(
            os.path.join(config.DATA_DIR, "models", "scalers.npy"),
            self.scalers,
        )
        logger.info("Model and scalers saved.")

    def load(self, path: str = None) -> None:
        """Load model & scalers."""
        if path is None:
            path = os.path.join(config.DATA_DIR, "models", "precip_model.h5")
        self.model = tf.keras.models.load_model(path)
        self.scalers = np.load(
            os.path.join(config.DATA_DIR, "models", "scalers.npy"),
            allow_pickle=True
        ).item()
        logger.info("Model and scalers loaded.")


def train_precipitation_model(data_path: str = None) -> PrecipitationModel:
    """
    Reads CSV, builds 12‑step sequences over exactly FEATURE_COLUMNS,
    trains & returns the model.
    """
    if data_path is None:
        data_path = config.NCEI_DATA_FILE

    # 1) load & engineer
    df = pd.read_csv(data_path, parse_dates=["date"]).sort_values("date")
    df["month"]  = df["date"].dt.month
    df["season"] = df["month"].map(lambda m:
        1 if m in [12,1,2] else
        2 if m in [3,4,5]  else
        3 if m in [6,7,8]  else 4
    )
    for lag in (1,2,3):
        df[f"precipitation_lag{lag}"] = df["precipitation"].shift(lag)

    # drop any rows missing exactly those FEATURE_COLUMNS:
    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    logger.info("After dropna, %d rows remain", len(df))

    # hand off to DataProcessor
    monthly = df[["date"] + FEATURE_COLUMNS]
    proc = DataProcessor()
    X, y, scalers = proc.create_sequences(
        monthly,
        sequence_length=12,
        target_column="precipitation",
    )

    # splits
    total     = len(X)
    train_end = int(total * config.TRAIN_SPLIT)
    val_end   = train_end + int((total - train_end)*0.2)

    X_train, X_val, X_test = (
        X[:train_end],
        X[train_end:val_end],
        X[val_end:]
    )
    y_train, y_val, y_test = (
        y[:train_end],
        y[train_end:val_end],
        y[val_end:]
    )

    # build + train + save
    model = PrecipitationModel(sequence_length=12)
    model.scalers = scalers
    model.build_model((12, X.shape[2]))
    model.train(X_train, y_train, X_val, y_val)
    print("Test metrics:", model.evaluate(X_test, y_test))
    model.save()
    return model


if __name__ == "__main__":
    m = train_precipitation_model()
    # just the last 12 months of features
    df_all = pd.read_csv(config.NCEI_DATA_FILE, parse_dates=["date"]).sort_values("date")
    df_all["month"]  = df_all["date"].dt.month
    df_all["season"] = df_all["month"].map(lambda m:
        1 if m in [12,1,2] else
        2 if m in [3,4,5] else
        3 if m in [6,7,8] else 4)
    for lag in (1,2,3):
        df_all[f"precipitation_lag{lag}"] = df_all["precipitation"].shift(lag)
    df_all = df_all.dropna().reset_index(drop=True)

    last_12 = df_all[["date"] + FEATURE_COLUMNS].iloc[-12:]
    print("Next‐month:", m.forecast_next(last_12))



