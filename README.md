
# ReTune Project

This project collects and integrates environmental and market data from multiple sources to power predictive models for weather forecasting, anomaly detection, and Kalshi event market predictions. The final meta model combines outputs from individual models and executes trades automatically via Kalshi’s API.

## Data Sources

1. **NOAA Dataset**: Historical daily rainfall totals, temperature, humidity, wind speed, pressure, and other data.
2. **OpenWeatherMap/WeatherAPI**: Current weather conditions and short-term forecasts.
3. **NCEI Global Hourly Data**: Hourly surface observations like wind speed, pressure, humidity, and other weather conditions.

## Project Structure

- `Integration_Layer/`: Modules for interacting with weather and market APIs and has Trade logic and Kalshi API interaction for placing market bets 
  - `noaa_client.py`: Client for NOAA API  
  - `openweather_client.py`: Client for OpenWeatherMap API  
  - `ncei_client.py`: Client for NCEI Global Hourly Data  
  - `data_processor.py`: Processes and combines data from different sources  
- `Precipitation Model/`: LSTM model for predicting monthly precipitation  
- `Market Model/`: LSTM model for detecting Kalshi market pricing trends  
- `Anomaly Model/`: Model to identify extreme or anomalous weather patterns  
- `main.py`: Main script to run the data pipeline  
- `config.py`: Configuration settings for API keys, paths, and constants  
- `integration_logs.log`: Log file tracking API and model integration status  

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your API keys:
   ```
   OPENWEATHER_API_KEY=your_openweather_api_key
   WEATHERAPI_KEY=your_weatherapi_key
   ```

3. Run the data collection script:
   ```
   python main.py
   ```

## Output

The script will generate CSV files containing processed weather and market data, which are ready to be used as input for model training and Kalshi market integration.

