"""
Configuration settings for the NYC Precipitation Prediction Model.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# NYC coordinates
NYC_LAT = 40.7128
NYC_LON = -74.0060

# Time periods
HISTORICAL_START_DATE = "2010-01-01"
HISTORICAL_END_DATE = "2023-12-31"
FORECAST_DAYS = 7

# NOAA API settings
NOAA_BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2"
NOAA_TOKEN = os.getenv("NOAA_TOKEN")
NOAA_DATASET_ID = "GHCND"  # Global Historical Climatology Network Daily
NOAA_STATION_ID = "GHCND:USW00094728"  # Central Park, NY

# OpenWeatherMap API settings
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# WeatherAPI settings
WEATHERAPI_BASE_URL = "http://api.weatherapi.com/v1"
WEATHERAPI_KEY = os.getenv("WEATHERAPI_KEY")

# NCEI Global Hourly Data settings
NCEI_BASE_URL = "https://www.ncei.noaa.gov/access/services/data/v1"
NCEI_DATASET = "global-hourly"
NCEI_STATION = "72503"  # LaGuardia Airport, NY

# Output settings
DATA_DIR = "data"
OUTPUT_FILE = "data/nyc_weather_data.csv"


# Feature settings
FEATURES = [
    "date",
    "precipitation",
    "temperature_max",
    "temperature_min",
    "temperature_avg",
    "humidity",
    "wind_speed",
    "pressure",
    "dew_point",
    "cloud_cover",
]

# Model settings
SEQUENCE_LENGTH = 12  # Number of months to use as input
TRAIN_SPLIT = 0.8  # Percentage of data to use for training
