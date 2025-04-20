import os
import sys
import pickle
import json
import numpy as np
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import pandas as pd
import uvicorn
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_retrainer import ModelRetrainer

app = FastAPI(title="Weather Anomaly API", 
              description="API for retraining weather anomaly models and making predictions")

# Model configurations
MODEL_PATH = "Anomaly Model/saved_models/lstm_stride1_valMAE1.5575.pkl"
DATA_PATH = "Anomaly Model/processed_weather_data.csv"


class RetrainRequest(BaseModel):
    """Request model for retraining the model with new data."""
    begints: str = Field(..., description="Start date in YYYY-MM-DD format")
    endts: str = Field(..., description="End date in YYYY-MM-DD format")
    wfos: List[str] = Field(..., description="List of Weather Forecast Office codes")
    save_model_name: Optional[str] = Field(None, description="Name for the saved model (without path or extension)")
    use_existing_model: bool = Field(True, description="Whether to use existing model as starting point")


class PredictionRequest(BaseModel):
    """Request model for making predictions."""
    date: str = Field(..., description="Date for prediction in YYYY-MM-DD format")


class RetrainResponse(BaseModel):
    """Response model for retraining requests."""
    status: str
    message: str
    model_path: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None


class PredictionResponse(BaseModel):
    """Response model for prediction requests."""
    date: str
    prediction: float
    is_anomaly: bool
    anomaly_threshold: float
    model_used: str


def format_model_name(base_name: Optional[str] = None) -> str:
    """Format the model name with timestamp if not provided."""
    if base_name:
        return f"Anomaly Model/saved_models/{base_name}.pkl"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"Anomaly Model/saved_models/lstm_retrained_{timestamp}.pkl"


@app.post("/retrain", response_model=RetrainResponse)
async def retrain_model(request: RetrainRequest, background_tasks: BackgroundTasks):
    """
    Endpoint to retrain the model with new weather alert data.
    
    The retraining happens in the background to avoid blocking the API.
    """
    try:
        # Check date formats
        try:
            start_date = datetime.strptime(request.begints, "%Y-%m-%d")
            end_date = datetime.strptime(request.endts, "%Y-%m-%d")
            
            if start_date > end_date:
                raise HTTPException(status_code=400, detail="Start date must be before end date")
                
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        # Format save path
        save_path = format_model_name(request.save_model_name)
        
        # Start retraining in the background
        background_tasks.add_task(
            perform_retraining,
            request.begints,
            request.endts,
            request.wfos,
            save_path,
            request.use_existing_model
        )
        
        return RetrainResponse(
            status="processing",
            message="Model retraining has been initiated in the background",
            model_path=save_path
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initiating retraining: {str(e)}")


def perform_retraining(begints: str, endts: str, wfos: List[str], 
                       save_path: str, use_existing_model: bool) -> None:
    """
    Perform the actual model retraining in the background.
    
    Parameters:
    -----------
    begints : str
        Start date in YYYY-MM-DD format
    endts : str
        End date in YYYY-MM-DD format
    wfos : List[str]
        List of Weather Forecast Office codes
    save_path : str
        Path to save the retrained model
    use_existing_model : bool
        Whether to use the existing model as a starting point
    """
    try:
        # Initialize the model retrainer
        retrainer = ModelRetrainer(
            model_path=MODEL_PATH,
            data_path=DATA_PATH
        )
        
        # Update data and retrain the model
        if begints and endts and wfos:
            # Use the combined function to update data and retrain
            result = retrainer.update_and_retrain(
                begints=begints,
                endts=endts,
                wfos=wfos,
                save_path=save_path
            )
        else:
            # Just retrain without updating data
            result = retrainer.retrain_model(
                save_path=save_path,
                use_existing_model=use_existing_model
            )
            
        print(f"Retraining complete. Model saved to {save_path}")
        print(f"Test MAE: {result['metrics']['test_mae']:.4f}")
        
    except Exception as e:
        print(f"Error during retraining: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a prediction for the given date.
    
    The endpoint loads the most recent model and returns the prediction
    along with information about whether it's an anomaly.
    """
    try:
        # Check date format
        try:
            prediction_date = datetime.strptime(request.date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        # Load the model
        try:
            with open(MODEL_PATH, 'rb') as f:
                model_dict = pickle.load(f)
                
            model = model_dict['model']
            scaler = model_dict['scaler']
            window_size = model_dict['config']['window_size']
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")
        
        # Load the data
        try:
            data = pd.read_csv(DATA_PATH)
            
            # Check if the date exists in the data
            data['date'] = pd.to_datetime(data['date'])
            
            # If the date is not in the data, return an error
            if prediction_date not in data['date'].values:
                # Find the closest date in the future
                closest_future_date = data[data['date'] > prediction_date]['date'].min()
                
                if pd.isna(closest_future_date):
                    raise HTTPException(
                        status_code=400, 
                        detail=f"The date {request.date} is not in the data and no future dates are available"
                    )
                else:
                    days_diff = (closest_future_date - prediction_date).days
                    raise HTTPException(
                        status_code=400, 
                        detail=f"The date {request.date} is not in the data. The closest future date is {closest_future_date.strftime('%Y-%m-%d')} ({days_diff} days ahead)"
                    )
            
            # Get the row for the date
            date_row = data[data['date'] == prediction_date]
            
            # Get the actual consumption
            actual_consumption = date_row['consumption'].values[0]
            
            # Get the window of data preceding this date
            window_start_idx = data[data['date'] <= prediction_date].index[-1] - window_size + 1
            
            if window_start_idx < 0:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Not enough historical data before {request.date} for prediction"
                )
                
            window_data = data.iloc[window_start_idx:window_start_idx + window_size]
            
            # Prepare the data (similar to what's done in make_windows)
            # This is simplified and might need adjustment depending on the model details
            X = window_data.drop(['date', 'consumption'], axis=1).values
            X = np.expand_dims(X, axis=0)  # Add batch dimension
            
            # Make prediction
            import torch
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32)
                prediction = model(X_tensor).item()
            
            # Determine if it's an anomaly (example threshold - adjust as needed)
            anomaly_threshold = 1.0  # Define your threshold
            is_anomaly = abs(prediction - actual_consumption) > anomaly_threshold
            
            return PredictionResponse(
                date=request.date,
                prediction=float(prediction),
                is_anomaly=is_anomaly,
                anomaly_threshold=anomaly_threshold,
                model_used=os.path.basename(MODEL_PATH)
            )
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs("Anomaly Model/saved_models", exist_ok=True)
    
    # Run the API
    uvicorn.run(app, host="0.0.0.0", port=8000) 