from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from typing import Optional
import pandas as pd
from model import PrecipitationModel, DataProcessor

app = FastAPI()

# Load processor and model at startup
processor = DataProcessor()
model = PrecipitationModel()
model.load_model()  # Loads precipitation_model.h5

# Load the same processed data used for training
data = pd.read_csv("data/nyc_weather_data.csv")
data["date"] = pd.to_datetime(data["date"])

# Recreate sequences and scalers
X, y, scalers = processor.create_sequences(data, sequence_length=30)
model.set_scalers(scalers)

@app.get("/")
def redirect_to_docs():
    return RedirectResponse(url="/docs")

@app.post("/train-model")
def train_model():
    model = train_precipitation_model()
    return {"message": "Model trained and saved successfully."}

@app.get("/predict")
def predict(date_range: Optional[str] = None, location: str = "NYC"):
    if not date_range:
        return {"error": "Please provide a date_range like '2024-12-01,2024-12-07'"}
    
    try:
        start_str, end_str = date_range.split(",")
        start_date = pd.to_datetime(start_str)
        end_date = pd.to_datetime(end_str)
    except Exception as e:
        return {"error": "Invalid date_range format. Use 'YYYY-MM-DD,YYYY-MM-DD'"}

    # Slice the appropriate portion from the original data
    mask = (data["date"] >= start_date) & (data["date"] <= end_date)
    dates_to_predict = data.loc[mask]

    if dates_to_predict.empty:
        return {"error": "No data available for the given date range"}

    # Get the last valid sequence that ends before this range starts
    sequence_start_index = data.index[data["date"] == start_date].tolist()
    if not sequence_start_index:
        return {"error": "Start date not found in data"}
    
    seq_idx = sequence_start_index[0] - 30
    if seq_idx < 0:
        return {"error": "Not enough data before start date to form a sequence"}

    X_input = X[seq_idx]

    # Predict (note: we only predict the first day for simplicity)
    y_pred = model.predict(X_input.reshape(1, *X_input.shape))
    pred = scalers["precipitation"].inverse_transform(y_pred).flatten()[0]

    return {
        "start_date": start_str,
        "end_date": end_str,
        "predicted_precipitation_first_day": round(float(pred), 4)
    }

