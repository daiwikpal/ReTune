"""
Configuration settings for the NYC Precipitation Prediction Model
and the Kalshi Market Prediction Model.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# -----------------------------------------------------------------------------
# NYC Precipitation Model Settings
# -----------------------------------------------------------------------------
NYC_LAT = 40.7128
NYC_LON = -74.0060

# Time periods for weather data
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

# Output settings for precipitation model
DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "nyc_weather_data.csv")

# Feature settings for precipitation model
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

# Model settings for precipitation model
SEQUENCE_LENGTH = 30  # Number of days to use as input
TRAIN_SPLIT = 0.8     # Percentage of data for training

# -----------------------------------------------------------------------------
# Kalshi Market Model Settings
# -----------------------------------------------------------------------------
# Demo API root from the documentation
KALSHI_API_ROOT = "https://demo-api.kalshi.co/trade-api/v2"

# Market data output file
MARKET_OUTPUT_FILE = os.path.join(DATA_DIR, "market_data.csv")

# Date range
MARKET_START_DATE = "2024-01-01"
MARKET_END_DATE = "2024-12-31"

# Model settings for market model
MARKET_SEQUENCE_LENGTH = 7  # e.g., use a 7-day sequence for market data
MARKET_TRAIN_SPLIT = 0.8

# Demo API credentials for Kalshi Market Model
KALSHI_API_KEY_ID = "1b9bb232-205c-4796-a00d-daee36caf1a4"
KALSHI_PRIVATE_KEY = """-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEArcBeVs9oONHzI8r0Uz2czracFpNYJiyyRCLUpcHC3eetUvG0
Xt4t1YDe9DYxJKfl/op0paaCUWYiGbR4ui/aT5NnmpwWsj1W/xpkrxfycUEwoMxw
8GsmDuETiuyG93KA88DqmoyLke9pCIq/wWv9ne/HCYuib2UM38qmWJ50RNbYN/uw
/u/hV2/93tAUdATwBWev61DthyN6ZpTTis/TjgsxetwPUcHZJ7AkD8vZtKqFvJBX
kTcf84pqWmIM8rZJirHpUTrZ0oYQNFIUan2peCUDmVrsZsl1LuunxercGZtixvvl
JXHgaNAmgfU8Q17vztoGqnUOz9knuR6mtF3/owIDAQABAoIBACZRVzzDztNBEeLh
3lIBM3Su9utPoVAB6Wp3V/gaLBRuBF5XaZTGYMtF+WVYEixoTqN8+06Q0wqKgJi0
oS9GYFnOQznolGTIfEgUFQI8QL5TRxsfGwUaTDFZ18p6cSVe/itDbzmYJyuO5dX5
a2KHo03SVE25mqAAY/9ynI7Mmw2XCtafWwGkJTLgRIT+UMGiDw4F8joA5pfi1NTa
52/sNfn8vwKe9NSXWjXhOsinS/sQ9dkE26I8iWd5G978W61akbkiGHG66TyFKfKp
O54p8BA9UF2mYCIpN8aceNbOGznmSsmSwYUyiDM1fCXxBW05ug4V2TMjxFN8833u
o8hdD8ECgYEAzQ7wC45kmA84uLVsJXuySPmkgpfYfoSjDtCtCnjknD92z9zkDpDR
MiAeti/uXV79ioHjbYCvPLkWcd2Kj1eccj9zXSNN2D1LTNLVuA8i0svsg3nAX/v2
XJ6STtRibbp/CUTgf7q0+fzkrnwUbuPEVe72eo5GLebbR9MNxG3UdGECgYEA2Opp
DSEQpaup7zcw/xtqLix+yMpR08xUO3vWUM36xPyFqYYOr00vYFEaWns0ABwg6Wkk
CaBxSu9y5typrJWxuZznDg0ZHHH9+6gjWZ9BSwGAB+ilaQAVE5JqwD0GYoEFatnq
mKkIOYQRUHaUj41aUS95JIt4zXmTGn52c/GbsoMCgYAxzKrnY5FyjF/OG+FDySKn
LlmRjab6MuQWuP8NSAbdG1yTZqXME8d6UOqkfEd7TZJtjNXaxiHIsXqN9Kut3C0W
Yep2eBhzp48d/SYCKUrfvr6Vv3/Ez8ApBimqE1JEK8KmUZ/j5UgGXjB7X47mz1Hj
PMGHSf4pL6OZcdwqFDJAgQKBgQDLWh43x7qyhZcfAq+1SP8m4GHPXRSPKSwCQ9ss
D547E70+qsWThBmZAw9gqcWbIMOd7gpx1+694HLoiQ+sEv31U2ms24yiBR+k0ACr
4Ue1yGc9gtWm9QPNQGNBazRUHj506Gwsx7JFMVGGDFTDqzFzkLzSDMqoXoQQv2PO
2D8tfwKBgDPGB8DpXHichGyNCLjJk64kJu5n0eaISlA026PImPCpNYjlD3FhNIao
3D+wNz3wECcQ9CkmctPlB0SaxTG3q6YF700lk7TK9fdQeHzMaso9A7ylljL33/ST
hkM1CSXt81h+4l2g79EXrd3/NvJ5rEd6KzcijNBmyTgQTil4jWP8
-----END RSA PRIVATE KEY-----"""