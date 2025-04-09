from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import pandas as pd

# Load the model and scaler
model = load_model('Anomaly Model/lstm_model.h5')
scaler = joblib.load('Anomaly Model/scaler.pkl')

# Initialize FastAPI app
app = FastAPI()

# Define request model
class PredictionRequest(BaseModel):
    data: list

# Define /predict endpoint
@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # Convert input data to numpy array
        input_data = np.array(request.data)
        
        # Check input shape
        if input_data.shape[1] != 12:
            raise HTTPException(status_code=400, detail="Input data must have 12 months of data.")
        
        # Scale input data
        input_data_scaled = scaler.transform(input_data)
        input_data_scaled = np.expand_dims(input_data_scaled, axis=0)
        
        # Make prediction
        prediction_scaled = model.predict(input_data_scaled)
        
        # Inverse scale the prediction
        prediction = scaler.inverse_transform(prediction_scaled)
        
        # Convert prediction to list
        prediction_list = prediction.tolist()[0]
        
        return {"prediction": prediction_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Example usage
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 