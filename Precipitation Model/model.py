"""
LSTM model for predicting NYC precipitation using the past 12 months of data.
"""
import os
import logging
from typing import Dict, Tuple, List

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

# Define the target column
TARGET_COLUMN = "PRECIPITATION"

# Feature columns will be updated dynamically to include all columns except date and target
FEATURE_COLUMNS = [
    "wind_speed_mean", "wind_speed_min", "wind_speed_max",
    "temperature_max_mean", "temperature_max_min", "temperature_max_max",
    "temperature_min_mean", "temperature_min_min", "temperature_min_max",
    "month_cos", "month_sin"
]

class PrecipitationModel:
    """
    LSTM model that, given 12 months of features, predicts the next month's precipitation.
    """
    def __init__(self, sequence_length: int = 12):
        self.sequence_length = sequence_length
        self.model: tf.keras.Model | None = None
        self.scalers: Dict[str, any] = {}
        self.num_features: int = 0
        os.makedirs(os.path.join(config.DATA_DIR, "models"), exist_ok=True)

    def build_model(self, input_shape: Tuple[int,int]) -> None:
        """Builds a two‑layer LSTM with input shape (sequence_length, n_features)."""
        self.num_features = input_shape[1]  # Store the number of features used during training
        logger.info(f"Building model with {self.num_features} features")
        
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
        
        # Verify dimensions
        logger.info(f"Training with X shape: {X_train.shape}, y shape: {y_train.shape}")
        logger.info(f"Validation with X shape: {X_val.shape}, y shape: {y_val.shape}")

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
        Given the most recent 12 months of data with exactly the same feature columns
        used during training, returns next-month precipitation in original units.
        
        Args:
            recent_df: DataFrame with exactly 12 rows and FEATURE_COLUMNS (no target column)
        
        Returns:
            float: Predicted precipitation for the next month
        """
        # Ensure we have exactly 12 months of data
        if len(recent_df) != self.sequence_length:
            raise ValueError(f"Expected {self.sequence_length} months of data, got {len(recent_df)}")
            
        # Sort by date to ensure chronological order
        df = recent_df.copy().sort_values("date")
        
        # Verify we have the exact feature columns needed
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"Missing required feature column: {col}")
        
        # No need to check for target column - it should be excluded for prediction
        
        # Extract just the feature columns in the correct order
        feats = df[FEATURE_COLUMNS]
        
        # Normalize each column with its scaler
        arr = np.zeros((len(feats), len(FEATURE_COLUMNS)))
        for i, col in enumerate(FEATURE_COLUMNS):
            arr[:, i] = self.scalers[col].transform(feats[col].values.reshape(-1,1)).flatten()
        
        # Reshape to (1, sequence_length, n_features)
        X_in = arr.reshape((1, self.sequence_length, len(FEATURE_COLUMNS)))
        
        # Make prediction and denormalize
        p_norm = self.model.predict(X_in, verbose=0)
        result = float(self.scalers[TARGET_COLUMN].inverse_transform(p_norm)[0,0])
        
        # Precipitation can't be negative
        return max(0.0, result)

    def save(self, path: str = None) -> None:
        """Save model & scalers."""
        if path is None:
            path = os.path.join(config.DATA_DIR, "models", "precip_model.h5")
        self.model.save(path)
        
        # Save scalers and metadata
        metadata = {
            'scalers': self.scalers,
            'num_features': self.num_features,
            'feature_columns': FEATURE_COLUMNS
        }
        
        np.save(
            os.path.join(config.DATA_DIR, "models", "model_metadata.npy"),
            metadata,
        )
        logger.info("Model, scalers, and metadata saved.")

    def load(self, path: str = None) -> None:
        """Load model & scalers."""
        if path is None:
            path = os.path.join(config.DATA_DIR, "models", "precip_model.h5")
        self.model = tf.keras.models.load_model(path)
        
        # Load metadata
        metadata_path = os.path.join(config.DATA_DIR, "models", "model_metadata.npy")
        if os.path.exists(metadata_path):
            metadata = np.load(
                metadata_path,
                allow_pickle=True
            ).item()
            
            self.scalers = metadata['scalers']
            self.num_features = metadata['num_features']
            global FEATURE_COLUMNS
            FEATURE_COLUMNS = metadata['feature_columns']
            logger.info(f"Model, scalers, and metadata loaded. Model has {self.num_features} features.")
        else:
            # Backward compatibility
            self.scalers = np.load(
                os.path.join(config.DATA_DIR, "models", "scalers.npy"),
                allow_pickle=True
            ).item()
            logger.info("Model and scalers loaded (legacy format).")


def train_precipitation_model(data_path: str = None) -> PrecipitationModel:
    """
    Reads CSV, builds 12‑step sequences using all features except date and precipitation,
    trains & returns the model.
    """
    if data_path is None:
        data_path = config.NCEI_DATA_FILE
    
    # 1) load & engineer
    df = pd.read_csv(data_path, parse_dates=["date"]).sort_values("date")
    
    # Get all columns except date and precipitation_sum as features
    # Also exclude non-numeric columns like 'TimeStamp' that can't be normalized
    global FEATURE_COLUMNS
    non_feature_columns = ['date', TARGET_COLUMN, 'TimeStamp']
    FEATURE_COLUMNS = [col for col in df.columns if col not in non_feature_columns]
    logger.info(f"Using {len(FEATURE_COLUMNS)} features for training: {FEATURE_COLUMNS}")
    
    # Drop rows with NaN values in any column we'll use
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)
    logger.info("After dropna, %d rows remain", len(df))
    
    # hand off to DataProcessor - IMPORTANT: we'll modify this to exclude the target column from sequences
    proc = DataProcessor()
    
    # Make a copy of the dataframe that only includes the features we want to use
    # This ensures that target column is NOT included in input features
    training_df = df[['date'] + FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    
    X, y, scalers = proc.create_sequences(
        training_df,
        sequence_length=12,
        target_column=TARGET_COLUMN,
        exclude_target_from_features=True,  # Add this parameter to ensure target is excluded
    )
    
    # Let's check the actual feature dimensions
    logger.info(f"X shape after sequence creation: {X.shape}")
    logger.info(f"Actual number of features in training data: {X.shape[2]}")
    if X.shape[2] != len(FEATURE_COLUMNS):
        logger.warning(f"Feature count mismatch: expected {len(FEATURE_COLUMNS)} but got {X.shape[2]}. Check DataProcessor.create_sequences implementation.")
    
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
    # Train the model - this now excludes the target column from features
    m = train_precipitation_model()
    
    # Get the full dataset
    df_all = pd.read_csv(config.NCEI_DATA_FILE, parse_dates=["date"]).sort_values("date")
    
    # Get the most recent 12 months of data
    last_12 = df_all.iloc[-12:].reset_index(drop=True)
    
    # Ensure we have exactly 12 months of data
    if len(last_12) < 12:
        logger.error(f"Insufficient data for forecasting, need 12 months but only have {len(last_12)}")
    else:
        try:
            # Verify we have all required feature columns
            missing_cols = set(FEATURE_COLUMNS) - set(last_12.columns)
            if missing_cols:
                logger.error(f"Missing required feature columns for forecasting: {missing_cols}")
            else:
                # Make sure data has only the required feature columns plus date (no target column)
                forecast_data = last_12[['date'] + FEATURE_COLUMNS].copy()
                
                # Generate forecast
                forecast = m.forecast_next(forecast_data)
                print(f"Next-month precipitation forecast: {forecast:.6f}")
                
                # Optionally, print the date we're forecasting for
                next_month = pd.to_datetime(forecast_data['date'].iloc[-1]) + pd.DateOffset(months=1)
                print(f"Forecast for: {next_month.strftime('%Y-%m')}")
        except Exception as e:
            logger.error(f"Error during forecasting: {str(e)}")
            import traceback
            traceback.print_exc()
