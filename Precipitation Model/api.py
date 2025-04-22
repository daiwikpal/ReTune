from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, conlist
from datetime import date, datetime, timedelta
import pandas as pd
import os
import logging
import numpy as np

import config
from model import PrecipitationModel, train_precipitation_model, FEATURE_COLUMNS
from weather_data.data_processor import DataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# single global model instance
model = PrecipitationModel(sequence_length=12)
data_processor = DataProcessor()

@app.on_event("startup")
def startup():
    # try loading an existing model+scalers at startup
    try:
        model.load()
    except Exception as e:
        logger.warning(f"Failed to load model at startup: {str(e)}")
        pass

class Record(BaseModel):
    date: date
    precipitation: float
    temperature_max: float
    temperature_min: float
    humidity: float
    wind_speed: float
    pressure: float
    temperature_range: float
    month: int
    season: int
    month_cos: float
    month_sin: float
    precipitation_lag1: float
    precipitation_lag2: float
    precipitation_lag3: float
    precipitation_lag7: float
    precipitation_rolling_mean_7d: float
    precipitation_rolling_max_7d: float

class ForecastInput(BaseModel):
    # exactly 12 records required
    recent: conlist(Record, min_items=12, max_items=12)

class PredictInput(BaseModel):
    year: int
    month: int

class RetrainInput(BaseModel):
    year: int
    month: int

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.post("/train-model", summary="Train or retrain the LSTM on the NCEI data")
def train_model_endpoint():
    # retrain from scratch
    train_precipitation_model(config.NCEI_DATA_FILE)
    # reload into our global model
    try:
        model.load()
    except Exception:
        pass
    return {"message": "Model trained and loaded."}

def retrain_model_task(year: int, month: int):
    """Background task to collect new data and retrain the model"""

    # check "data/monthly_weather_data.csv" if month and year are alrady in the file then dont do anything an log 

    if os.path.exists(config.DATA_DIR + "/monthly_weather_data.csv"):
        df = pd.read_csv(config.DATA_DIR + "/monthly_weather_data.csv")
        if (year, month) in df.values:
            logger.info(f"Data for {year}-{month} already exists in monthly_weather_data.csv")
            return
        
        # use the data processor to collect the data for that month and year

        data_processor.collect_ncei_daily_data(f"{year}-{month}-01", f"{year}-{month}-30", "data/temp_monthly_weather_data.csv")

        # append the data to the existing monthly_weather_data.csv file

        df = pd.read_csv("data/temp_monthly_weather_data.csv")
        df.to_csv(config.DATA_DIR + "/monthly_weather_data.csv", mode="a", header=False, index=False)   

        # remove the temp file
        os.remove("data/temp_monthly_weather_data.csv")

        # retrain the model
        train_precipitation_model(config.NCEI_DATA_FILE)
        model.load()

        # save the model
        model.save()


@app.post("/retrain", summary="Retrain the model with data for a specific month")
def retrain_endpoint(background_tasks: BackgroundTasks, input: RetrainInput):
    """
    Retrain the model with data for a specific month.
    
    This endpoint:
    1. Takes a year and month as input
    2. Collects data for that month and the month before
    3. Appends this data to the existing monthly_weather_data.csv file
    4. Retrains the model on the combined dataset
    5. Ensures the predict endpoint uses the newly retrained model
    """
    # Validate input
    if not 1 <= input.month <= 12:
        raise HTTPException(status_code=400, detail="Month must be between 1 and 12")
    
    current_year = datetime.now().year
    if input.year < 1970 or input.year > current_year:
        raise HTTPException(status_code=400, detail=f"Year must be between 1970 and {current_year}")
    
    # Start retraining as a background task
    background_tasks.add_task(retrain_model_task, input.year, input.month)
    
    return {
        "message": f"Retraining started in the background for {input.year}-{input.month}."
    }

@app.post("/forecast", summary="Forecast next month from 12 months of input data")
def forecast(input: ForecastInput):
    # build a DataFrame from the incoming records
    recent_df = pd.DataFrame([r.dict() for r in input.recent])
    try:
        # pick out exactly the columns your model was trained on
        df_in = recent_df[["date"] + FEATURE_COLUMNS]
        pred = model.forecast_next(df_in)
    except Exception as e:
        # return any errors as HTTP 400
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "predicted_monthly_precipitation_inches": round(pred, 4)
    }

@app.post("/predict", summary="Predict precipitation using automatically collected data for the past 12 months")
def predict(input: PredictInput):
    """
    Predict precipitation for the specified year and month by automatically collecting
    the past 12 months of data using NCEI's data service.
    """
    # Validate input
    if not 1 <= input.month <= 12:
        raise HTTPException(status_code=400, detail="Month must be between 1 and 12")
    
    try:
        # Create target date from input
        target_date = datetime(input.year, input.month, 1)
        
        # Calculate date range for data collection
        # We need 12 months prior to the target date
        end_date = target_date - timedelta(days=1)  # Last day of previous month
        start_date = target_date - timedelta(days=365)  # Roughly 1 year before
        
        # Format dates for API
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Check if we already have a monthly data file we can use
        monthly_data_path = os.path.join(config.DATA_DIR, "monthly_weather_data.csv")
        use_existing_data = False
        
        if os.path.exists(monthly_data_path):
            try:
                # Try to use existing data if it covers our date range
                existing_data = pd.read_csv(monthly_data_path, parse_dates=["date"])
                
                # Check if the date range is covered
                if (not existing_data.empty and 
                   min(existing_data["date"]) <= start_date and 
                   max(existing_data["date"]) >= end_date):
                    logger.info("Using existing monthly data file for prediction")
                    
                    # Filter to just our date range
                    monthly_data = existing_data[
                        (existing_data["date"] >= start_date) & 
                        (existing_data["date"] <= end_date)
                    ]
                    use_existing_data = True
            except Exception as e:
                logger.warning(f"Could not use existing data file: {str(e)}")
                use_existing_data = False
        
        if not use_existing_data:
            # Create temporary file path
            temp_daily_csv = os.path.join(config.DATA_DIR, "temp_predict_daily.csv")
            
            # Collect NCEI daily data and save to temp file
            logger.info(f"Collecting data from {start_str} to {end_str}")
            data_processor.collect_ncei_daily_data(start_str, end_str, save_path=temp_daily_csv)
            
            # Check if file was created and has data
            if not os.path.exists(temp_daily_csv) or os.path.getsize(temp_daily_csv) < 100:
                raise HTTPException(status_code=404, detail="Failed to collect weather data for the specified period")
            
            # Load the saved data
            daily_data = pd.read_csv(temp_daily_csv, parse_dates=["date"])
            
            if daily_data.empty:
                raise HTTPException(status_code=404, detail="Collected data is empty")
            
            # Aggregate to monthly data
            monthly_data = data_processor.convert_to_monthly(daily_data)
            
            # Clean up temporary file
            try:
                os.remove(temp_daily_csv)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file: {str(e)}")
        
        # Ensure we have at least 12 months of data
        if len(monthly_data) < 12:
            raise HTTPException(
                status_code=400, 
                detail=f"Insufficient data: only {len(monthly_data)} months available, need 12 months"
            )
        
        # Get the most recent 12 months
        forecast_data = monthly_data.sort_values("date").tail(12).reset_index(drop=True)
        
        # Check that we have all required feature columns
        missing_cols = set(FEATURE_COLUMNS) - set(forecast_data.columns)
        if missing_cols:
            raise HTTPException(
                status_code=500,
                detail=f"Missing required feature columns: {missing_cols}"
            )
        
        # Make sure data has only the required feature columns plus date (no target column)
        df_in = forecast_data[["date"] + FEATURE_COLUMNS].copy()
        
        # Generate forecast
        pred = model.forecast_next(df_in)
        
        # Calculate the date we're predicting for
        prediction_date = target_date.strftime('%Y-%m')
        
        return {
            "prediction_for": prediction_date,
            "predicted_monthly_precipitation_inches": round(pred, 4),
            "data_range_used": f"{forecast_data['date'].min().strftime('%Y-%m')} to {forecast_data['date'].max().strftime('%Y-%m')}"
        }
        
    except Exception as e:
        # Log the error for debugging
        logger.error(f"Prediction error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Return a useful error message
        raise HTTPException(status_code=500, detail=str(e))

# Run the API server on port 8001 when the script is run directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)