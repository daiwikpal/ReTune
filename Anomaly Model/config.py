"""
Configuration settings for the NYC Weather Anomaly Prediction Model.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# NYC coordinates
NYC_LAT = 40.7128
NYC_LON = -74.0060

# NOAA API settings (legacy)
NOAA_BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2"
NOAA_TOKEN = os.getenv("NOAA_API_TOKEN")
NOAA_DATASET_ID = "GHCND"  # Global Historical Climatology Network Daily
NOAA_STATION_ID = "GHCND:USW00094728"  # Central Park, NY

# NCEI Global Hourly Data settings
NCEI_BASE_URL = "https://www.ncei.noaa.gov/access/services/data/v1"
NCEI_DATASET = "global-hourly"
NCEI_STATION = "72503"  # LaGuardia Airport, NY

# Data directory settings
HISTORICAL_DATA_DIR = "Historical Data"
OUTPUT_FILE = "processed_weather_data.csv" 