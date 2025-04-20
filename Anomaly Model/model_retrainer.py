import os
import sys
import pandas as pd
import numpy as np
import pickle
from typing import Optional, Dict, Any, Tuple
import torch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the Model directory
from Model.lstm_model import clean_and_split_data, create_dataloaders, Regressor
from Model.lstm_model import train, evaluate, preprocess_data, make_windows

from alert_data_processor import AlertDataProcessor


class ModelRetrainer:
    """
    Class for retraining the LSTM model with new weather data.
    """
    
    def __init__(self, 
                 model_path: str, 
                 data_path: str,
                 window_size: int = 12,
                 stride: int = 1,
                 test_size: float = 0.1,
                 val_size: float = 0.1,
                 hidden_size: int = 16,
                 num_layers: int = 1,
                 learning_rate: float = 1e-4,
                 batch_size: int = 16,
                 num_epochs: int = 200,
                 patience: int = 10,
                 dropout: float = 0.1):
        """
        Initialize the model retrainer.
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model
        data_path : str
            Path to the processed weather data
        window_size : int
            Size of the sequence window for training
        stride : int
            Stride for creating windows
        test_size : float
            Proportion of data to use for testing
        val_size : float
            Proportion of data to use for validation
        hidden_size : int
            Hidden size of the LSTM model
        num_layers : int
            Number of LSTM layers
        learning_rate : float
            Learning rate for optimizer
        batch_size : int
            Batch size for training
        num_epochs : int
            Maximum number of training epochs
        patience : int
            Number of epochs to wait for improvement in validation loss
        dropout : float
            Dropout rate for the model
        """
        self.model_path = model_path
        self.data_path = data_path
        self.window_size = window_size
        self.stride = stride
        self.test_size = test_size
        self.val_size = val_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.dropout = dropout
        
        # Load the model if it exists
        self.model = None
        self.scaler = None
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    saved_dict = pickle.load(f)
                self.model = saved_dict.get('model')
                self.scaler = saved_dict.get('scaler')
            except Exception as e:
                print(f"Error loading model: {e}")
    
    def retrain_model(self, 
                      save_path: Optional[str] = None, 
                      use_existing_model: bool = True) -> Dict[str, Any]:
        """
        Retrain the model with new data.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the retrained model
        use_existing_model : bool
            Whether to start from the existing model or train from scratch
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing the trained model, scaler, and evaluation metrics
        """
        # Load the data
        data = pd.read_csv(self.data_path)
        
        # Clean and split the data
        # The clean_and_split_data function uses CONFIG values for splits
        train_data, val_data, test_data, feature_cols = clean_and_split_data(data)
        
        # Create windows
        X_train, y_train = make_windows(
            train_data, 
            window=self.window_size, 
            stride=self.stride,
            feature_cols=feature_cols
        )
        X_val, y_val = make_windows(
            val_data, 
            window=self.window_size, 
            stride=self.stride,
            feature_cols=feature_cols
        )
        X_test, y_test = make_windows(
            test_data, 
            window=self.window_size, 
            stride=self.stride,
            feature_cols=feature_cols
        )
        
        # Create dataloaders
        train_loader, val_loader, test_loader, input_size, self.scaler = create_dataloaders(
            train_data, val_data, test_data, feature_cols, self.stride
        )
        
        # Initialize or reuse model
        if use_existing_model and self.model is not None:
            model = self.model
        else:
            model = Regressor(
                "lstm",  # Use LSTM by default
                input_size
            )
        
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Train the model
        val_mae = train(
            train_loader,
            val_loader,
            model,
            device,
            max_ep=self.num_epochs,
            patience=self.patience
        )
        
        # Evaluate the model
        test_mae = evaluate(
            model, 
            test_loader, 
            device
        )
        print(f"Test MAE: {test_mae:.4f}")
        
        # Save the model
        if save_path is None:
            save_path = self.model_path
            
        save_dict = {
            'model': model,
            'scaler': self.scaler,
            'config': {
                'window_size': self.window_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate
            },
            'metrics': {
                'test_mae': test_mae
            }
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"Model saved to {save_path}")
        
        return save_dict
    
    def update_and_retrain(self, 
                          begints: str, 
                          endts: str, 
                          wfos: list,
                          save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Update the dataset with new alert data and retrain the model.
        
        Parameters:
        -----------
        begints : str
            Start date in YYYY-MM-DD format
        endts : str
            End date in YYYY-MM-DD format
        wfos : list
            List of Weather Forecast Office codes
        save_path : str, optional
            Path to save the retrained model
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing the trained model, scaler, and evaluation metrics
        """
        # Process new alert data
        processor = AlertDataProcessor(self.data_path)
        
        # Fetch and process alerts, merge with existing data
        updated_data = processor.process_and_merge_alerts(
            begints=begints,
            endts=endts,
            wfos=wfos,
            output_path=self.data_path
        )
        
        # Retrain the model
        return self.retrain_model(save_path=save_path)


def main():
    """Main function to demonstrate usage."""
    model_path = "Anomaly Model/saved_models/lstm_stride1_valMAE1.5575.pkl"
    data_path = "Anomaly Model/processed_weather_data.csv"
    
    retrainer = ModelRetrainer(
        model_path=model_path,
        data_path=data_path
    )
    
    # Example usage: update and retrain
    retrainer.update_and_retrain(
        begints="2025-01-01",
        endts="2025-04-01",
        wfos=["OKX", "PHI", "CTP"],  # New York City
        save_path="Anomaly Model/saved_models/lstm_retrained.pkl"
    )
    
    
if __name__ == "__main__":
    main() 