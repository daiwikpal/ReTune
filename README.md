# NYC Precipitation Prediction Model

This project collects weather data from multiple sources to train an LSTM model for predicting monthly precipitation in New York City.

## Data Sources

1. **NOAA Dataset**: Historical daily rainfall totals, temperature, humidity, wind speed, pressure, and other data.
2. **OpenWeatherMap/WeatherAPI**: Current weather conditions and short-term forecasts.
3. **NCEI Global Hourly Data**: Hourly surface observations like wind speed, pressure, humidity, and other weather conditions.

## Project Structure

- `weather_data/`: Module for interacting with weather APIs
  - `noaa_client.py`: Client for NOAA API
  - `openweather_client.py`: Client for OpenWeatherMap API
  - `ncei_client.py`: Client for NCEI Global Hourly Data
  - `data_processor.py`: Processes and combines data from different sources
- `main.py`: Main script to run the data collection and processing
- `config.py`: Configuration settings for the project
- `model.py`: LSTM model for precipitation prediction

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

The script will generate a CSV file with processed weather data ready for training the LSTM model.
