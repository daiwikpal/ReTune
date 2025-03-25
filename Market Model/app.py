"""
FastAPI application to host the Kalshi Market Prediction Model.
Exposes endpoints to train, predict, and update the model.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging
from market_data.data_processor import MarketDataProcessor
from noaa_data import collect_noaa_data, combine_market_and_noaa_data
from market_model import train_market_model  # Ensure these functions exist
import config

logger = logging.getLogger(__name__)
app = FastAPI(title="Kalshi Market Prediction API")

@app.post("/train-model")
def train_model():
    """
    Trigger training of the LSTM-based market trend model.
    This endpoint:
      - Collects market data.
      - Collects NOAA data.
      - Combines both datasets.
      - Saves the combined data.
      - Trains the model using the combined dataset.
    """
    try:
        # Collect market data
        mdp = MarketDataProcessor()
        market_df = mdp.collect_market_data()
        if market_df.empty:
            raise HTTPException(status_code=500, detail="Failed to collect market data.")
        
        # Collect NOAA data
        noaa_df = collect_noaa_data(config.HISTORICAL_START_DATE, config.HISTORICAL_END_DATE)
        if noaa_df.empty:
            raise HTTPException(status_code=500, detail="Failed to collect NOAA data.")
        
        # Combine datasets
        combined_df = combine_market_and_noaa_data(market_df, noaa_df)
        combined_file = config.MARKET_OUTPUT_FILE
        mdp.save_data(combined_df, combined_file)
        
        # Train the model (train_market_model should integrate these features)
        model = train_market_model(combined_file)
        return {"message": "Model trained successfully", "metrics": "dummy_metrics"}
    except Exception as e:
        logger.error(f"Error in /train-model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class MarketSnapshot(BaseModel):
    """
    Input model for market snapshot data used in prediction.
    Extend with additional features as needed.
    """
    market_price: float
    open_interest: float
    trading_volume: float

@app.post("/predict")
def predict(snapshot: MarketSnapshot):
    """
    Accept market snapshot data as input and return a prediction.
    The prediction includes a probability distribution and a suggested trading signal.
    """
    try:
        input_features = [snapshot.market_price, snapshot.open_interest, snapshot.trading_volume]
        # predict_market should process these features and return prediction results.
        prediction = predict_market(input_features)
        return {"prediction": prediction}
    except Exception as e:
        logger.error(f"Error in /predict: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update-model")
def update_model():
    """
    Endpoint to update the deployed model if retraining improves performance.
    This is a placeholder implementation.
    """
    try:
        # In a full implementation, compare current vs. retrained model metrics.
        updated = True  # Placeholder logic
        if updated:
            return {"message": "Model updated successfully."}
        else:
            return {"message": "Model update not required."}
    except Exception as e:
        logger.error(f"Error in /update-model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
