"""
LSTM model for predicting NYC precipitation.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import config
from weather_data.data_processor import DataProcessor

FEATURE_COLUMNS = [
    "precipitation",
    "month",
    "season",
    "precipitation_lag1",
    "precipitation_lag2",
    "precipitation_lag3",
]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PrecipitationModel:
    """
    LSTM model for predicting precipitation in NYC.
    """
    
    def __init__(self, sequence_length: int = None):
        """
        Initialize the precipitation prediction model.
        
        Args:
            sequence_length: Number of time steps in each sequence
        """
        self.sequence_length = sequence_length or config.SEQUENCE_LENGTH
        self.model = None
        self.scalers = None
        self.feature_columns = None
        
        # Create model directory if it doesn't exist
        os.makedirs(os.path.join(config.DATA_DIR, "models"), exist_ok=True)
    
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build the LSTM model architecture.
        
        Args:
            input_shape: Shape of input data (sequence_length, num_features)
        """
        model = Sequential()
        
        # LSTM layers
        model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        
        model.add(LSTM(32, return_sequences=False))
        model.add(Dropout(0.2))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile the model
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        self.model = model
        logger.info(f"Model built with input shape: {input_shape}")
        
        # Print model summary
        model.summary()
    
    def train(self, 
             X_train: np.ndarray, 
             y_train: np.ndarray, 
             X_val: np.ndarray = None, 
             y_val: np.ndarray = None,
             epochs: int = 100,
             batch_size: int = 32) -> Dict[str, List[float]]:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary with training history
        """
        if self.model is None:
            self.build_model((X_train.shape[1], X_train.shape[2]))
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(
                filepath=os.path.join(config.DATA_DIR, "models", "precipitation_model.h5"),
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Train the model
        if X_val is not None and y_val is not None:
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
        else:
            # Use a portion of training data for validation
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1
            )
        
        logger.info("Model training completed")
        return history.history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            logger.error("Model not trained yet")
            return {}
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
        
        logger.info(f"Model evaluation metrics: {metrics}")
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the trained model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted values
        """
        if self.model is None:
            logger.error("Model not trained yet")
            return np.array([])
        
        # Make predictions
        predictions = self.model.predict(X)
        
        return predictions
    
    def save_model(self, filepath: str = None) -> None:
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            logger.error("No model to save")
            return
        
        if filepath is None:
            filepath = os.path.join(config.DATA_DIR, "models", "precipitation_model.h5")
        
        # Save the model
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str = None) -> None:
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
        """
        if filepath is None:
            filepath = os.path.join(config.DATA_DIR, "models", "precipitation_model.h5")
        
        # Check if model file exists
        if not os.path.exists(filepath):
            logger.error(f"Model file not found: {filepath}")
            return
        
        # Load the model
        self.model = tf.keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
        
    
    def set_scalers(self, scalers: Dict) -> None:
        """
        Set the scalers for denormalizing predictions.
        
        Args:
            scalers: Dictionary of scalers for each feature
        """
        self.scalers = scalers
    
    def set_feature_columns(self, feature_columns: List[str]) -> None:
        """
        Set the feature columns used in the model.
        
        Args:
            feature_columns: List of feature column names
        """
        self.feature_columns = feature_columns


def train_precipitation_model(data_path: str = None, sequence_length: int = None) -> PrecipitationModel:
    """
    Train a precipitation prediction model using NCEI-only monthly data.
    """
    if data_path is None:
        data_path = config.NCEI_DATA_FILE

    if sequence_length is None:
        sequence_length = config.SEQUENCE_LENGTH

    ### NOTE: Read monthly NCEI data directly
    data = pd.read_csv(data_path)
    data["date"] = pd.to_datetime(data["date"])
    monthly = data.sort_values("date").copy()

    # Add temporal features
    monthly["month"] = monthly["date"].dt.month
    monthly["season"] = monthly["month"].map(lambda m: 
        1 if m in [12,1,2] else
        2 if m in [3,4,5]  else
        3 if m in [6,7,8]  else 4
    )

    # Add lag features
    for lag in (1, 2, 3):
        monthly[f"precipitation_lag{lag}"] = monthly["precipitation"].shift(lag)

    monthly = monthly.dropna().reset_index(drop=True)


    monthly_selected = monthly[["date"] + FEATURE_COLUMNS]

    processor = DataProcessor()
    X, y, scalers = processor.create_sequences(
        monthly_selected,              # <‑‑ use the restricted frame
        sequence_length=sequence_length,
        target_column="precipitation"
    )

    train_size = int(len(X) * config.TRAIN_SPLIT)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    val_size = int(len(X_train) * 0.2)
    X_train, X_val = X_train[:-val_size], X_train[-val_size:]
    y_train, y_val = y_train[:-val_size], y_train[-val_size:]

    model = PrecipitationModel(sequence_length)
    model.build_model((X_train.shape[1], X_train.shape[2]))
    history = model.train(X_train, y_train, X_val, y_val)
    metrics = model.evaluate(X_test, y_test)
    model.save_model()
    model.set_scalers(scalers)
    model.set_feature_columns(FEATURE_COLUMNS)

    # model.plot_training_history(history, os.path.join(config.DATA_DIR, "training_history.png"))

    y_pred = model.predict(X_test)
    test_dates = monthly["date"].iloc[sequence_length + train_size : sequence_length + train_size + len(y_test)]
    y_test_denorm = scalers["precipitation"].inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_denorm = scalers["precipitation"].inverse_transform(y_pred).flatten()

    # model.plot_predictions(test_dates, y_test_denorm, y_pred_denorm, 
    #                        os.path.join(config.DATA_DIR, "predictions.png"))
    
    return model


if __name__ == "__main__":
    # Train the model
    model = train_precipitation_model()
