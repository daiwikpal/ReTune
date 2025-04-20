# Weather Anomaly Prediction API

A FastAPI service for predicting precipitation anomalies based on weather data using an LSTM model.

## Overview

This API serves a pre-trained LSTM model that takes a 12-month window of weather data and predicts whether the next month's precipitation will be anomalous (over 1 inch).

## Installation

1. Install the required dependencies:

```bash
pip install fastapi uvicorn pandas numpy torch scikit-learn
```

2. Start the API server:

```bash
cd "Anomaly Model/Model_Serving_Retraining_Service"
python fastapi_app.py
```

The server will run at http://localhost:8000. You can access the API documentation at http://localhost:8000/docs.

## API Endpoints

### GET /

Returns basic information about the model.

**Response:**
```json
{
  "message": "Weather Anomaly Prediction API",
  "model_type": "lstm",
  "stride": 1,
  "window_size": 12,
  "validation_mae": 1.5575,
  "test_mae": 1.6243,
  "input_features": 52
}
```

### POST /predict

Predicts precipitation anomalies based on 12 months of weather data.

**Request Body:**

```json
{
  "data": [
    {
      "AVALANCHE": 0.0,
      "BLIZZARD": 0.0,
      "COASTAL_FLOOD": 0.0,
      "COLD/WIND_CHILL": 0.0,
      "DEBRIS_FLOW": 0.0,
      "DENSE_FOG": 0.0,
      "DENSE_SMOKE": 0,
      "DROUGHT": 0.0,
      "DUST_DEVIL": 0.0,
      "DUST_STORM": 0,
      "EXCESSIVE_HEAT": 0.0,
      "ASTRONOMICAL_LOW_TIDE": 0.0,
      "EXTREME_COLD/WIND_CHILL": 0.0,
      "FLASH_FLOOD": 0,
      "FLOOD": 0,
      "FREEZING_FOG": 0.0,
      "FROST/FREEZE": 0.0,
      "FUNNEL_CLOUD": 0,
      "HAIL": 0,
      "HEAT": 0,
      "HEAVY_RAIN": 0,
      "HEAVY_SNOW": 0,
      "HIGH_SURF": 0.0,
      "HIGH_WIND": 0,
      "HURRICANE_(TYPHOON)": 0.0,
      "ICE_STORM": 0.0,
      "LAKE-EFFECT_SNOW": 0.0,
      "LAKESHORE_FLOOD": 0.0,
      "LIGHTNING": 0,
      "MARINE_HAIL": 0,
      "MARINE_HIGH_WIND": 0.0,
      "MARINE_STRONG_WIND": 0,
      "MARINE_THUNDERSTORM_WIND": 0,
      "RIP_CURRENT": 0.0,
      "SEICHE": 0.0,
      "SLEET": 0.0,
      "SNEAKERWAVE": 0,
      "STORM_SURGE/TIDE": 0.0,
      "STRONG_WIND": 0,
      "THUNDERSTORM_WIND": 0,
      "TORNADO": 0,
      "TROPICAL_DEPRESSION": 0.0,
      "TROPICAL_STORM": 0.0,
      "TSUNAMI": 0.0,
      "VOLCANIC_ASH": 0,
      "WATERSPOUT": 0.0,
      "WILDFIRE": 0.0,
      "WINTER_STORM": 0,
      "WINTER_WEATHER": 0,
      "precip_12m_mean": 4.2,
      "precip_12m_std": 1.8,
      "precip_12m_z": 0.3,
      "season_sin": 0.5,
      "season_cos": 0.866
    },
    ... // Repeat for all 12 months
  ]
}
```

The API requires 12 months of data, each containing the features shown above. The data should be ordered chronologically, with the most recent month last. Note that PRECIPITATION is not included as an input feature since it's what we're predicting.

**Response:**

```json
{
  "prediction": 1.25,
  "is_anomaly": true,
  "threshold": 1.0
}
```

- `prediction`: The predicted precipitation value for the next month (in inches)
- `is_anomaly`: Boolean indicating if the prediction exceeds the anomaly threshold
- `threshold`: The threshold value that defines an anomaly (1.0 inch)

## Example Usage

### Python Example

```python
import requests
import json

# Load 12 months of weather data
with open("sample_weather_data.json", "r") as f:
    weather_data = json.load(f)

# Send prediction request
response = requests.post(
    "http://localhost:8000/predict",
    json={"data": weather_data}
)

# Print result
result = response.json()
print(f"Predicted precipitation: {result['prediction']} inches")
print(f"Is anomaly: {result['is_anomaly']}")
```

## Model Information

The API serves an LSTM model trained on historical weather data for precipitation anomaly detection:

- Model type: LSTM
- Window size: 12 months
- Input features: Weather events, statistical metrics, and seasonal indicators
- Target: Next month's precipitation
- Anomaly threshold: 1.0 inch of precipitation

## Data Format

The input data follows the schema from the processed_weather_data.csv file, containing:

1. Weather event frequencies (AVALANCHE, BLIZZARD, etc.)
2. Statistical metrics (precip_12m_mean, precip_12m_std, precip_12m_z)
3. Seasonal indicators (season_sin, season_cos)

Each data point represents one month of aggregated weather data. PRECIPITATION itself is not included as an input feature since that's what the model is predicting. 
