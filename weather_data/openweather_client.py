"""
Client for interacting with the OpenWeatherMap API.
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import logging
from typing import Dict, List, Any, Optional

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenWeatherClient:
    """
    Client for fetching current weather conditions and forecasts from OpenWeatherMap API.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenWeatherMap API client.
        
        Args:
            api_key: OpenWeatherMap API key. If None, uses the key from config.
        """
        self.base_url = config.OPENWEATHER_BASE_URL
        self.api_key = api_key or config.OPENWEATHER_API_KEY
        
        if not self.api_key:
            logger.warning("OpenWeatherMap API key not provided. Set OPENWEATHER_API_KEY in .env file.")
    
    def get_current_weather(self, lat: float = None, lon: float = None) -> Dict[str, Any]:
        """
        Get current weather conditions for a location.
        
        Args:
            lat: Latitude of the location
            lon: Longitude of the location
            
        Returns:
            Dictionary with current weather data
        """
        if lat is None:
            lat = config.NYC_LAT
        if lon is None:
            lon = config.NYC_LON
            
        endpoint = f"{self.base_url}/weather"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "imperial"  # Use imperial units for consistency with NOAA data
        }
        
        response = self._make_request(endpoint, params)
        return response
    
    def get_forecast(self, lat: float = None, lon: float = None, days: int = 5) -> pd.DataFrame:
        """
        Get weather forecast for a location.
        
        Args:
            lat: Latitude of the location
            lon: Longitude of the location
            days: Number of days to forecast (max 5 for free tier)
            
        Returns:
            DataFrame with forecast data
        """
        if lat is None:
            lat = config.NYC_LAT
        if lon is None:
            lon = config.NYC_LON
            
        endpoint = f"{self.base_url}/forecast"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "imperial"
        }
        
        response = self._make_request(endpoint, params)
        
        if not response or "list" not in response:
            logger.warning("No forecast data available")
            return pd.DataFrame()
        
        # Extract forecast data
        forecast_data = []
        for item in response["list"]:
            forecast_time = datetime.fromtimestamp(item["dt"])
            
            # Only include forecasts up to the specified number of days
            if forecast_time > datetime.now() + timedelta(days=days):
                continue
                
            forecast_item = {
                "date": forecast_time,
                "temperature": item["main"]["temp"],
                "feels_like": item["main"]["feels_like"],
                "temperature_min": item["main"]["temp_min"],
                "temperature_max": item["main"]["temp_max"],
                "pressure": item["main"]["pressure"],
                "humidity": item["main"]["humidity"],
                "wind_speed": item["wind"]["speed"],
                "wind_direction": item["wind"]["deg"],
                "clouds": item["clouds"]["all"],
                "weather_main": item["weather"][0]["main"],
                "weather_description": item["weather"][0]["description"],
            }
            
            # Add precipitation if available
            if "rain" in item and "3h" in item["rain"]:
                forecast_item["precipitation_3h"] = item["rain"]["3h"]
            else:
                forecast_item["precipitation_3h"] = 0
                
            forecast_data.append(forecast_item)
        
        # Convert to DataFrame
        df = pd.DataFrame(forecast_data)
        
        # Add day column for aggregation
        df["day"] = df["date"].dt.date
        
        # Aggregate by day
        daily_forecast = df.groupby("day").agg({
            "temperature": "mean",
            "temperature_min": "min",
            "temperature_max": "max",
            "pressure": "mean",
            "humidity": "mean",
            "wind_speed": "mean",
            "clouds": "mean",
            "precipitation_3h": "sum"  # Sum 3-hour precipitation to get daily total
        }).reset_index()
        
        # Rename precipitation column
        daily_forecast = daily_forecast.rename(columns={"precipitation_3h": "precipitation"})
        
        # Convert precipitation from mm to inches for consistency with NOAA data
        daily_forecast["precipitation"] = daily_forecast["precipitation"] / 25.4
        
        # Convert day back to datetime
        daily_forecast["date"] = pd.to_datetime(daily_forecast["day"])
        daily_forecast = daily_forecast.drop(columns=["day"])
        
        return daily_forecast
    
    def get_historical_data(self, lat: float = None, lon: float = None, 
                           start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """
        Get historical weather data for a location.
        Note: This requires a paid subscription to OpenWeatherMap.
        
        Args:
            lat: Latitude of the location
            lon: Longitude of the location
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            DataFrame with historical weather data
        """
        if lat is None:
            lat = config.NYC_LAT
        if lon is None:
            lon = config.NYC_LON
            
        if start_date is None:
            start_date = datetime.now() - timedelta(days=5)
        if end_date is None:
            end_date = datetime.now()
            
        # This endpoint requires a paid subscription
        endpoint = f"{self.base_url}/history/city"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "imperial",
            "start": int(start_date.timestamp()),
            "end": int(end_date.timestamp()),
        }
        
        response = self._make_request(endpoint, params)
        
        if not response or "list" not in response:
            logger.warning("No historical data available or paid subscription required")
            return pd.DataFrame()
        
        # Extract historical data
        historical_data = []
        for item in response["list"]:
            data_time = datetime.fromtimestamp(item["dt"])
            
            data_item = {
                "date": data_time,
                "temperature": item["main"]["temp"],
                "feels_like": item["main"].get("feels_like"),
                "temperature_min": item["main"]["temp_min"],
                "temperature_max": item["main"]["temp_max"],
                "pressure": item["main"]["pressure"],
                "humidity": item["main"]["humidity"],
                "wind_speed": item["wind"]["speed"],
                "wind_direction": item["wind"]["deg"],
                "clouds": item["clouds"]["all"],
                "weather_main": item["weather"][0]["main"],
                "weather_description": item["weather"][0]["description"],
            }
            
            # Add precipitation if available
            if "rain" in item and "1h" in item["rain"]:
                data_item["precipitation_1h"] = item["rain"]["1h"]
            else:
                data_item["precipitation_1h"] = 0
                
            historical_data.append(data_item)
        
        # Convert to DataFrame
        df = pd.DataFrame(historical_data)
        
        # Add day column for aggregation
        df["day"] = df["date"].dt.date
        
        # Aggregate by day
        daily_data = df.groupby("day").agg({
            "temperature": "mean",
            "temperature_min": "min",
            "temperature_max": "max",
            "pressure": "mean",
            "humidity": "mean",
            "wind_speed": "mean",
            "clouds": "mean",
            "precipitation_1h": "sum"  # Sum hourly precipitation to get daily total
        }).reset_index()
        
        # Rename precipitation column
        daily_data = daily_data.rename(columns={"precipitation_1h": "precipitation"})
        
        # Convert precipitation from mm to inches for consistency with NOAA data
        daily_data["precipitation"] = daily_data["precipitation"] / 25.4
        
        # Convert day back to datetime
        daily_data["date"] = pd.to_datetime(daily_data["day"])
        daily_data = daily_data.drop(columns=["day"])
        
        return daily_data
    
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a request to the OpenWeatherMap API with error handling.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            JSON response as dictionary
        """
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                logger.error("Invalid OpenWeatherMap API key. Please check your key.")
            elif response.status_code == 429:
                logger.error("OpenWeatherMap API rate limit exceeded. Try again later.")
            else:
                logger.error(f"HTTP error: {e}")
            return {}
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return {}
            
        except ValueError as e:
            logger.error(f"JSON parsing error: {e}")
            return {}


if __name__ == "__main__":
    # Example usage
    client = OpenWeatherClient()
    
    # Get current weather
    current = client.get_current_weather()
    print("Current Weather:", current)
    
    # Get forecast
    forecast = client.get_forecast(days=3)
    print("\nForecast:")
    print(forecast)
