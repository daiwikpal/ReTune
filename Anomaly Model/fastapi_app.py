import os
import pickle
import sys
import datetime
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.lstm_model import Regressor
from model_retrainer import ModelRetrainer
from alert_data_processor import AlertDataProcessor

# Path settings
MODEL_DIR = Path(__file__).parent / "saved_models"
DEFAULT_MODEL_PATH = MODEL_DIR / "lstm_stride1_valMAE1.5575.pkl"
DATA_PATH = Path(__file__).parent / "processed_weather_data.csv"
DEFAULT_WFOS = ["OKX", "PHI", "CTP"]  # Default WFOs to use for retraining

# Ensure model directory exists
MODEL_DIR.mkdir(exist_ok=True, parents=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent / "prediction_logs.log")
    ]
)
logger = logging.getLogger(__name__)

# Function to find the most recent model file
def find_latest_model():
    model_files = list(MODEL_DIR.glob("*.pkl"))
    if not model_files:
        return DEFAULT_MODEL_PATH
    
    # Sort by modification time, newest first
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    return latest_model

# Set the model path to the latest model
MODEL_PATH = find_latest_model()

# Load the model
def load_model(model_path=None):
    if model_path is None:
        model_path = MODEL_PATH
        
    try:
        with open(model_path, 'rb') as f:
            saved_model = pickle.load(f)
        
        # Check if this is in the format from the ModelRetrainer
        if 'model' in saved_model and isinstance(saved_model['model'], nn.Module):
            # Convert to the format expected by the FastAPI app
            model = saved_model['model']
            
            # Need to determine the input size - try to infer from model structure
            # For LSTM the first parameter of rnn module is the input size
            input_size = None
            for name, param in model.named_parameters():
                if 'rnn.weight_ih_l0' in name:
                    input_size = param.shape[1]
                    break
            
            if input_size is None:
                input_size = saved_model['config'].get('window_size', 12)
            
            model_info = {
                'type': 'lstm',  # Assume LSTM for now
                'in_size': input_size,
                'stride': 1,
                'config': saved_model['config'],
                'val_mae': 0.0,
                'test_mae': saved_model['metrics'].get('test_mae', 0.0)
            }
            
            return model, model_info, saved_model['scaler']
            
        # Else assume it's in the standard format
        model_info = saved_model.get('model_info', {})
        
        # Create model instance using our embedded Regressor class
        model = Regressor(model_info.get('type', 'lstm'), model_info.get('in_size', 12))
        
        # Check if the model state is stored directly or in 'model_state'
        if 'model_state' in saved_model:
            model.load_state_dict(saved_model['model_state'])
        else:
            # Try to load state from the model itself
            model.load_state_dict(saved_model.get('model', model).state_dict())
            
        model.eval()  # Set model to evaluation mode
        
        return model, model_info, saved_model.get('scaler')
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

# Model for retraining requests
class RetrainingRequest(BaseModel):
    """Request model for retraining the model with new data"""
    begints: str  # Start date in YYYY-MM-DD format
    endts: str    # End date in YYYY-MM-DD format
    
# Response model for retraining
class RetrainingResponse(BaseModel):
    """Response model for retraining results"""
    success: bool
    model_path: str
    test_mae: float
    timestamp: str

# Add a new model for simplified prediction request
class SimplePredictionRequest(BaseModel):
    """Request model for simplified prediction that only requires target month"""
    target_month: str = Field(..., description="Target month for prediction in YYYY-MM format")
    
# Function to reload the model
def reload_model():
    global model, model_info, scaler, MODEL_PATH
    MODEL_PATH = find_latest_model()
    model, model_info, scaler = load_model(MODEL_PATH)
    return model is not None

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
        "input_features": model_info['in_size'],
        "current_model": str(MODEL_PATH)
    }

@app.post("/retrain", response_model=RetrainingResponse)
async def retrain(request: RetrainingRequest, background_tasks: BackgroundTasks):
    """
    Retrain the model with new data
    
    Requires begin timestamp and end timestamp in YYYY-MM-DD format
    Uses default WFOs and saves model to a timestamped location
    """
    try:
        # Generate timestamp for the new model
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = MODEL_DIR / f"lstm_retrained_{timestamp}.pkl"
        
        # Initialize model retrainer
        retrainer = ModelRetrainer(
            model_path=str(MODEL_PATH),
            data_path=str(DATA_PATH)
        )
        
        # Update and retrain the model
        result = retrainer.update_and_retrain(
            begints=request.begints,
            endts=request.endts,
            wfos=DEFAULT_WFOS,
            save_path=str(save_path)
        )
        
        # The model is now saved, but potentially in an incompatible format
        # We'll reload it immediately to ensure it works with our system
        success = reload_model()
        
        if not success:
            raise HTTPException(
                status_code=500, 
                detail="Model retraining was successful but failed to load the new model"
            )
        
        return {
            "success": True,
            "model_path": str(save_path),
            "test_mae": result['metrics']['test_mae'],
            "timestamp": timestamp
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

@app.post("/predict_simple", response_model=PredictionResponse)
async def predict_simple(request: SimplePredictionRequest):
    """
    Predict precipitation anomalies based on target month
    
    Only requires the target month in YYYY-MM format
    Automatically fetches the required alert data for model input
    Returns prediction
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        logger.info(f"Received prediction request for target month: {request.target_month}")
        
        # Convert target month to date format
        target_date = datetime.datetime.strptime(request.target_month, "%Y-%m")
        
        # Calculate the start date (12 months before target)
        window_size = model_info['config']['window_size']
        start_date = target_date.replace(day=1)
        for _ in range(window_size):
            start_date = (start_date.replace(day=1) - datetime.timedelta(days=1)).replace(day=1)
        
        # Format dates for API calls
        begints = start_date.strftime("%Y-%m-%d")
        endts = (target_date.replace(day=28) + datetime.timedelta(days=4)).strftime("%Y-%m-%d")
        
        logger.info(f"Calculated date range: {begints} to {endts} for window size: {window_size}")
        
        # Use AlertDataProcessor to get the alert data
        logger.info("Initializing AlertDataProcessor...")
        processor = AlertDataProcessor(str(DATA_PATH))
        
        # Add monkey patching to log the NCEI client calls
        original_fetch_precipitation = processor.ncei_client.get_monthly_precipitation
        
        def logged_fetch_precipitation(begints, endts):
            logger.info(f"Calling NCEI API to fetch precipitation data from {begints} to {endts}")
            result = original_fetch_precipitation(begints, endts)
            logger.info(f"NCEI API returned {len(result)} records")
            return result
        
        processor.ncei_client.get_monthly_precipitation = logged_fetch_precipitation
        
        # Add monkey patching for VTEC client too
        original_get_alerts = processor.vtec_client.get_alerts
        
        def logged_get_alerts(begints, endts, wfos):
            logger.info(f"Calling VTEC API to fetch alerts from {begints} to {endts} for WFOs: {wfos}")
            result = original_get_alerts(begints=begints, endts=endts, wfos=wfos)
            alert_count = len(result.get('data', [])) if result and 'data' in result else 0
            logger.info(f"VTEC API returned {alert_count} alerts")
            return result
        
        processor.vtec_client.get_alerts = logged_get_alerts
        
        # Process alerts for the date range
        logger.info(f"Processing alerts for date range with WFOs: {DEFAULT_WFOS}")
        alerts_data = processor.process_and_merge_alerts(
            begints=begints,
            endts=endts,
            wfos=DEFAULT_WFOS
        )
        
        logger.info(f"Processed data has {len(alerts_data)} total records")
        
        # Convert TimeStamp to datetime for filtering
        alerts_data['date'] = pd.to_datetime(alerts_data['TimeStamp'])
        
        # Get the window of data for prediction
        window_data = alerts_data[
            (alerts_data['date'] >= pd.to_datetime(begints)) & 
            (alerts_data['date'] <= pd.to_datetime(endts))
        ]
        
        # Sort by date
        window_data = window_data.sort_values('date')
        
        logger.info(f"After filtering, window has {len(window_data)} records")
        
        # Take the last window_size months
        if len(window_data) > window_size:
            window_data = window_data.iloc[-window_size:]
            logger.info(f"Taking last {window_size} months, now have {len(window_data)} records")
        
        # Check if we have sufficient data
        if len(window_data) < window_size:
            logger.error(f"Not enough data: need {window_size} months, but only found {len(window_data)}")
            raise HTTPException(
                status_code=400,
                detail=f"Not enough historical data available. Need {window_size} months, but only found {len(window_data)}."
            )
        
        # Extract required features (same as in regular predict endpoint)
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
        
        # Validate all required features are present
        missing_features = [f for f in required_features if f not in window_data.columns]
        if missing_features:
            logger.error(f"Missing features: {missing_features}")
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features: {', '.join(missing_features)}"
            )
        
        # Prepare input data
        df = window_data[required_features]
        
        # Log summary statistics of the input data
        logger.info("Input data summary statistics:")
        for col in df.columns:
            non_zero = (df[col] != 0).sum()
            if non_zero > 0:
                logger.info(f"  {col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.4f}, non-zero={non_zero}")
        
        # Apply scaler to normalize input data
        input_data = df.values
        logger.info(f"Input data shape before scaling: {input_data.shape}")
        input_data = scaler.transform(input_data)
        logger.info(f"Input data shape after scaling: {input_data.shape}")
        
        # Convert to PyTorch tensor
        input_tensor = torch.tensor(input_data).float().unsqueeze(0)  # Add batch dimension
        logger.info(f"Input tensor shape: {input_tensor.shape}")
        
        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor)
            prediction_value = prediction.item()
        
        logger.info(f"Model prediction for {request.target_month}: {prediction_value}")
        
        return {
            "prediction": prediction_value
        }
    
    except Exception as e:
        logger.exception(f"Error in predict_simple: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model_status")
async def model_status():
    """Get information about the currently loaded model"""
    return {
        "current_model": str(MODEL_PATH),
        "last_modified": datetime.datetime.fromtimestamp(MODEL_PATH.stat().st_mtime).isoformat(),
        "available_models": [str(p) for p in MODEL_DIR.glob("*.pkl")],
        "model_info": {
            "type": model_info.get('type', 'unknown'),
            "stride": model_info.get('stride', 'unknown'),
            "window_size": model_info.get('config', {}).get('window_size', 'unknown'),
            "test_mae": model_info.get('test_mae', 'unknown')
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=True)
