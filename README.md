# ReTune Project

This project collects and integrates environmental and market data from multiple sources to power predictive models for monthly precipitation forecasting, extreme weather anomaly detection, and Kalshi event market predictions. The final meta-model combines outputs from all three models to improve accuracy and can execute trades automatically through Kalshi’s API.

## Data Sources

1. **NOAA Dataset**: Historical daily weather data including precipitation totals, temperature, humidity, wind speed, and pressure.
2. **OpenWeatherMap API**: 5-day forecast data used to simulate future conditions.
3. **NCEI Global Hourly Data**: Hourly surface observations (e.g., pressure, humidity) aggregated into daily summaries for feature generation.
4. **Kalshi API**: Historical and real-time market pricing data for weather-related contracts, and endpoints to place trades.

## Project Structure

- `Integration_Layer/`: Interfaces with weather and market APIs, handles data processing, and trade logic
  - `noaa_client.py`: Client for retrieving NOAA daily weather data  
  - `openweather_client.py`: Retrieves short-term forecasts from OpenWeatherMap  
  - `ncei_client.py`: Retrieves and aggregates hourly data from NCEI  
  - `data_processor.py`: Merges all sources into a single structured dataset with engineered features  
- `Precipitation Model/`: LSTM model that predicts total monthly precipitation using historical weather features  
- `Market Model/`: LSTM or Random Forest model trained on Kalshi pricing trends to predict likely YES/NO market outcomes  
- `Anomaly Model/`: Binary classifier to identify the likelihood of extreme or anomalous weather events (e.g., storms, unusual rainfall)  
- `meta_model.py`: Aggregates predictions from all models to generate a final decision  
- `main.py`: FastAPI app that serves model endpoints for training, prediction, and trading  
- `config.py`: Stores configuration variables and API endpoints  
- `integration_logs.log`: Logs API fetch status, training progress, and trade decisions  

## Setup

1. Install dependencies:
pip install -r requirements.txt

2. Create a `.env` file with your API keys:
```
NOAA_TOKEN=your_noaa_token
NCEI_TOKEN=your_ncei_token
OPENWEATHER_API_KEY=your_openweather_api_key
KALSHI_API_KEY=your_kalshi_api_key
```

4. Run the API server:
```
uvicorn main:app --reload
```

This launches the FastAPI server exposing endpoints for training, updating, predicting, and trading.

## Endpoints

- `POST /train-model`: Trains or retrains all models using historical data.
- `POST /update-model`: Adds recent data and incrementally updates models.
- `POST /predict`: Runs all models and outputs combined prediction with trade recommendation.
- `POST /kalshi-order`: Places a trade on Kalshi (optional, depends on prediction and confidence thresholds).

## Output

The system returns a JSON object containing:

- Predicted total precipitation for the upcoming month
- Likelihood of a weather anomaly (e.g., rainstorm)
- Kalshi market movement prediction (YES/NO)
- Suggested Kalshi trade action (YES/NO/HOLD)
- Confidence scores and raw outputs from each individual model

## Example Use

Run the full pipeline for forecasting and trading:
```
curl -X POST http://localhost:8000/predict
```

## Notes

- Historical weather data is cached after first download for faster re-runs.
- Predictions use engineered features including sine/cosine time encoding, weather lags, and rolling stats.
- Model retraining can be scheduled via cron or manually triggered via the `/train-model` endpoint.

  # ReTune Installation Guide

This guide provides installation and setup instructions for all components of the ReTune system.

## Table of Contents
1. [Anomaly Model](#anomaly-model)
2. [Precipitation Model](#precipitation-model)
3. [Market Model](#market-model)
4. [Integration Layer](#integration-layer)

## Anomaly Model

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps
1. Navigate to the Anomaly Model directory:
   ```
   cd "Anomaly Model"
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the data processor first to prepare the data:
   ```
   python data_processing.py
   ```

4. Start the FastAPI application:
   ```
   python fastapi_app.py
   ```

5. The API should now be running on http://localhost:8000

## Precipitation Model

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps
1. Navigate to the Precipitation Model directory:
   ```
   cd "Precipitation Model"
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the data processor to prepare the data:
   ```
   python convert_to_monthly.py
   ```

4. Start the API:
   ```
   python api.py
   ```

5. The API should now be running on http://localhost:8001

## Market Model

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps
1. Navigate to the Market Model directory:
   ```
   cd "Market Model"
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the data processor:
   ```
   python data_processor.py
   ```

4. Start the API:
   ```
   python app.py
   ```

5. The API should now be running on http://localhost:8002

## Integration Layer

### Prerequisites
- Python 3.8 or higher
- pip package manager
- cron (for scheduled tasks)

### Installation Steps
1. Navigate to the Integration Layer directory:
   ```
   cd Integration_Layer
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up the model retraining cron job:
   ```
   # Make the script executable
   chmod +x model_retraining.py
   # Edit your crontab
   crontab -e
   # Add the following line to run retraining on the 1st of each month at 2am
   0 2 1 * * /usr/bin/python3 /full/path/to/Integration_Layer/model_retraining.py --cron >> /full/path/to/cron_retraining.log 2>&1
   ```

5. To manually trigger a retraining:
   ```
   python model_retraining.py
   ```

6. To make market betting predictions using the integrated models:
   ```
   python model_integration.py
   ```

## Complete System Setup

To set up the entire system, follow these steps in order:

1. Install and start the Anomaly Model
2. Install and start the Precipitation Model
3. Install and start the Market Model
4. Set up the Integration Layer

Once all services are running, the Integration Layer can communicate with each model for predictions and automated retraining.

## Troubleshooting

If you encounter any issues:

1. Check that all required dependencies are installed
2. Verify that each API is running on the expected port
3. Check the log files in each directory for error messages
4. Ensure environment variables are correctly set (check .env files if present)

