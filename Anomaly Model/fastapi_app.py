import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from Model.lstm_model import Regressor


# Path to the saved model
MODEL_PATH = Path(__file__).parent / "saved_models" / "lstm_stride1_valMAE1.5575.pkl"

# Load the model
def load_model():
    try:
        with open(MODEL_PATH, 'rb') as f:
            saved_model = pickle.load(f)
        
        # Get model information
        model_info = saved_model['model_info']
        
        # Create model instance using our embedded Regressor class
        model = Regressor(model_info['type'], model_info['in_size'])
        model.load_state_dict(saved_model['model_state'])
        model.eval()  # Set model to evaluation mode
        
        return model, model_info, saved_model['scaler']
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

# Load the model
model, model_info, scaler = load_model()

if model is None:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}")

# Create FastAPI app
app = FastAPI(
    title="Weather Anomaly Prediction API",
    description="API for predicting precipitation anomalies based on weather data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input data model based on processed_weather_data.csv
class WeatherData(BaseModel):
    """Weather data for prediction (12-month window)"""
    data: List[Dict[str, float]]

# Response model
class PredictionResponse(BaseModel):
    prediction: float

@app.get("/")
async def root():
    """Root endpoint returning model info"""
    return {
        "message": "Weather Anomaly Prediction API",
        "model_type": model_info['type'],
        "stride": model_info['stride'],
        "window_size": model_info['config']['window_size'],
        "validation_mae": model_info['val_mae'],
        "test_mae": model_info['test_mae'],
        "input_features": model_info['in_size']
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: WeatherData):
    """
    Predict precipitation anomalies based on 12 months of weather data
    
    Expects 12 months of data with required features based on processed_weather_data.csv
    Returns prediction and anomaly status
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Check if we have correct number of months (window_size)
    window_size = model_info['config']['window_size']
    if len(data.data) != window_size:
        raise HTTPException(
            status_code=400, 
            detail=f"Input data must contain exactly {window_size} months of data"
        )
    
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame(data.data)
        
        # Check for required features - Note: PRECIPITATION is excluded as it's what we're predicting
        required_features = [
            'AVALANCHE', 'BLIZZARD', 'COASTAL_FLOOD', 'COLD/WIND_CHILL', 
            'DEBRIS_FLOW', 'DENSE_FOG', 'DENSE_SMOKE', 'DROUGHT', 
            'DUST_DEVIL', 'DUST_STORM', 'EXCESSIVE_HEAT', 'ASTRONOMICAL_LOW_TIDE', 
            'EXTREME_COLD/WIND_CHILL', 'FLASH_FLOOD', 'FLOOD', 'FREEZING_FOG', 
            'FROST/FREEZE', 'FUNNEL_CLOUD', 'HAIL', 'HEAT', 'HEAVY_RAIN', 
            'HEAVY_SNOW', 'HIGH_SURF', 'HIGH_WIND', 'HURRICANE_(TYPHOON)', 
            'ICE_STORM', 'LAKE-EFFECT_SNOW', 'LAKESHORE_FLOOD', 'LIGHTNING', 
            'MARINE_HAIL', 'MARINE_HIGH_WIND', 'MARINE_STRONG_WIND', 
            'MARINE_THUNDERSTORM_WIND', 'RIP_CURRENT', 'SEICHE', 'SLEET', 
            'SNEAKERWAVE', 'STORM_SURGE/TIDE', 'STRONG_WIND', 'THUNDERSTORM_WIND', 
            'TORNADO', 'TROPICAL_DEPRESSION', 'TROPICAL_STORM', 'TSUNAMI', 
            'VOLCANIC_ASH', 'WATERSPOUT', 'WILDFIRE', 'WINTER_STORM', 
            'WINTER_WEATHER', 'precip_12m_mean', 'precip_12m_std', 'precip_12m_z',
            'season_sin', 'season_cos'
        ]
        
        # Validate required features
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features: {', '.join(missing_features)}"
            )
        
        # Ensure features are in the correct order
        df = df[required_features]
        
        # Apply scaler to normalize input data
        input_data = df.values
        input_data = scaler.transform(input_data)
        
        # Convert to PyTorch tensor
        input_tensor = torch.tensor(input_data).float().unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor)
            prediction_value = prediction.item()
        
        return {
            "prediction": prediction_value
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=True)
