import requests
import json
import numpy as np
from typing import Dict, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("integration_logs.log")
    ]
)
logger = logging.getLogger(__name__)

class ModelIntegrator:
    """Integration layer for anomaly and precipitation models"""
    
    def __init__(self, anomaly_model_url: str = "http://localhost:8000", precipitation_model_url: str = "http://localhost:8001"):
        """
        Initialize the model integrator
        
        Args:
            anomaly_model_url: Base URL for the anomaly model API
        """
        self.anomaly_model_url = anomaly_model_url
        self.precipitation_model_url = precipitation_model_url
        logger.info(f"Initialized ModelIntegrator with anomaly model at {anomaly_model_url}")
    
    def get_anomaly_prediction(self, target_month: str) -> Optional[float]:
        """
        Get prediction from the anomaly model
        
        Args:
            target_month: Target month in YYYY-MM format
        
        Returns:
            Prediction value or None if the request failed
        """
        try:
            endpoint = f"{self.anomaly_model_url}/predict_simple"
            payload = {"target_month": target_month}
            
            logger.info(f"Requesting anomaly prediction for {target_month}")
            response = requests.post(endpoint, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                prediction = result.get("prediction")
                logger.info(f"Received anomaly prediction: {prediction}")
                return prediction
            else:
                logger.error(f"Anomaly model request failed with status {response.status_code}: {response.text}")
                return None
        except Exception as e:
            logger.exception(f"Error getting anomaly prediction: {str(e)}")
            return None
    
    def get_precipitation_prediction(self, target_month: str) -> Optional[float]:
        """
        Get prediction from the precipitation model
        """

        try:
            endpoint = f"{self.precipitation_model_url}/predict"
            payload = {
                "year": int(target_month[:4]),
                "month": int(target_month[5:])
            }
            
            logger.info(f"Requesting precipitation prediction for {target_month}")
            response = requests.post(endpoint, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                prediction = result.get("predicted_monthly_precipitation_inches")
                logger.info(f"Received precipitation prediction: {prediction}")
                return prediction
            else:
                logger.error(f"precipitation model request failed with status {response.status_code}: {response.text}")
                return None
        except Exception as e:
            logger.exception(f"Error getting precipitation prediction: {str(e)}")
            return None
        
    
    def bayesian_model_averaging(self, anomaly_prediction: float, precipitation_prediction: float, 
                                 anomaly_weight: float = 0.5) -> float:
        """
        Perform Bayesian model averaging between anomaly and precipitation predictions
        
        Args:
            anomaly_prediction: Prediction from anomaly model
            precipitation_prediction: Prediction from precipitation model
            anomaly_weight: Weight for anomaly model (between 0 and 1)
        
        Returns:
            Combined prediction value
        """
        # Simple weighted average for now - can be extended with proper Bayesian averaging
        combined_prediction = (anomaly_weight * anomaly_prediction + 
                              (1 - anomaly_weight) * precipitation_prediction)
        
        logger.info(f"Combined prediction: {combined_prediction} (weights: anomaly={anomaly_weight}, "
                   f"precipitation={1-anomaly_weight})")
        
        return combined_prediction
    
    def get_integrated_prediction(self, target_month: str, anomaly_weight: float = 0.5) -> Dict:
        """
        Get integrated prediction combining anomaly and precipitation models
        
        Args:
            target_month: Target month in YYYY-MM format
            anomaly_weight: Weight for anomaly model (between 0 and 1)
        
        Returns:
            Dictionary with prediction results from both models and their integration
        """
        # Get anomaly prediction
        anomaly_prediction = self.get_anomaly_prediction(target_month)
        
        if anomaly_prediction is None:
            return {
                "success": False,
                "error": "Failed to get anomaly prediction",
                "timestamp": datetime.now().isoformat()
            }
        
        # Get precipitation prediction
        precipitation_prediction = self.get_precipitation_prediction(target_month)
        
        if precipitation_prediction is None:
            return {
                "success": False,
                "error": "Failed to get precipitation prediction",
                "timestamp": datetime.now().isoformat()
            }
        
        # Calculate integrated prediction using Bayesian model averaging
        integrated_prediction = self.bayesian_model_averaging(
            anomaly_prediction, 
            precipitation_prediction,
            anomaly_weight
        )
        
        return {
            "success": True,
            "anomaly_prediction": anomaly_prediction,
            "precipitation_prediction": precipitation_prediction,
            "integrated_prediction": integrated_prediction,
            "weights": {
                "anomaly": anomaly_weight,
                "precipitation": 1 - anomaly_weight
            },
            "timestamp": datetime.now().isoformat()
        }


# Example usage
if __name__ == "__main__":
    # Example of how to use the integrator
    integrator = ModelIntegrator(anomaly_model_url="http://localhost:8000", precipitation_model_url="http://localhost:8001")
    
    # Get integrated prediction with automatic fetching from both models
    result = integrator.get_integrated_prediction(target_month="2023-05")
    print(json.dumps(result, indent=2))
    
    # Example with custom weights
    result_with_custom_weights = integrator.get_integrated_prediction(
        target_month="2023-05",
        anomaly_weight=0.6  # Give more weight to anomaly model
    )
    print(json.dumps(result_with_custom_weights, indent=2)) 