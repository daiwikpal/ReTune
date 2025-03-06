"""
LSTM model for predicting NYC precipitation.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import config
from weather_data.data_processor import DataProcessor

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
    
    def plot_training_history(self, history: Dict[str, List[float]], save_path: str = None) -> None:
        """
        Plot the training history.
        
        Args:
            history: Training history
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot MAE
        plt.subplot(1, 2, 2)
        plt.plot(history['mae'], label='Training MAE')
        plt.plot(history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        
        # Save the plot if a path is provided
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_predictions(self, 
                       dates: pd.Series, 
                       y_true: np.ndarray, 
                       y_pred: np.ndarray,
                       save_path: str = None) -> None:
        """
        Plot the actual vs. predicted precipitation.
        
        Args:
            dates: Dates corresponding to the data points
            y_true: Actual precipitation values
            y_pred: Predicted precipitation values
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        plt.plot(dates, y_true, label='Actual Precipitation', marker='o')
        plt.plot(dates, y_pred, label='Predicted Precipitation', marker='x')
        
        plt.title('Actual vs. Predicted Precipitation')
        plt.xlabel('Date')
        plt.ylabel('Precipitation (inches)')
        plt.legend()
        plt.grid(True)
        
        # Rotate date labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot if a path is provided
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Predictions plot saved to {save_path}")
        
        plt.show()
    
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
    Train a precipitation prediction model using data from the specified path.
    
    Args:
        data_path: Path to the processed data CSV
        sequence_length: Number of time steps in each sequence
        
    Returns:
        Trained PrecipitationModel
    """
    if data_path is None:
        data_path = config.OUTPUT_FILE
    
    if sequence_length is None:
        sequence_length = config.SEQUENCE_LENGTH
    
    # Load the data
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        # Collect data if file doesn't exist
        processor = DataProcessor()
        historical_data = processor.collect_historical_data()
        forecast_data = processor.collect_forecast_data()
        prepared_data = processor.prepare_data_for_model(historical_data, forecast_data)
        processor.save_data(prepared_data, data_path)
    
    data = pd.read_csv(data_path)
    data["date"] = pd.to_datetime(data["date"])
    
    # Create sequences
    processor = DataProcessor()
    X, y, scalers = processor.create_sequences(data, sequence_length, "precipitation")
    
    # Split into train, validation, and test sets
    train_size = int(len(X) * config.TRAIN_SPLIT)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Further split training data into train and validation
    val_size = int(len(X_train) * 0.2)
    X_train, X_val = X_train[:-val_size], X_train[-val_size:]
    y_train, y_val = y_train[:-val_size], y_train[-val_size:]
    
    # Create and train the model
    model = PrecipitationModel(sequence_length)
    model.build_model((X_train.shape[1], X_train.shape[2]))
    
    # Train the model
    history = model.train(X_train, y_train, X_val, y_val)
    
    # Evaluate the model
    metrics = model.evaluate(X_test, y_test)
    
    # Save the model
    model.save_model()
    
    # Set scalers and feature columns for later use
    model.set_scalers(scalers)
    model.set_feature_columns(data.drop(columns=["date"]).columns.tolist())
    
    # Plot training history
    model.plot_training_history(history, os.path.join(config.DATA_DIR, "training_history.png"))
    
    # Make predictions on test data
    y_pred = model.predict(X_test)
    
    # Get dates for test data
    test_dates = data["date"].iloc[train_size + sequence_length:train_size + sequence_length + len(y_test)]
    
    # Denormalize predictions and actual values
    y_test_denorm = scalers["precipitation"].inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_denorm = scalers["precipitation"].inverse_transform(y_pred).flatten()
    
    # Plot predictions
    model.plot_predictions(test_dates, y_test_denorm, y_pred_denorm, 
                         os.path.join(config.DATA_DIR, "predictions.png"))
    
    return model


if __name__ == "__main__":
    # Train the model
    model = train_precipitation_model()
