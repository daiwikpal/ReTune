"""
Client for interacting with the NCEI Global Hourly Data API.
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import logging
from typing import Dict, List, Any, Optional
import io

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NCEIClient:
    """
    Client for fetching hourly surface observations from NCEI Global Hourly Data.
    """
    
    def __init__(self):
        """
        Initialize the NCEI Global Hourly Data client.
        """
        self.base_url = config.NCEI_BASE_URL
        self.dataset = config.NCEI_DATASET
        self.station = config.NCEI_STATION
    
    def get_hourly_data(self, 
                       start_date: str, 
                       end_date: str, 
                       station_id: str = None) -> pd.DataFrame:
        """
        Get hourly weather data for a specific time period and station.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            station_id: ID of the weather station
            
        Returns:
            DataFrame with hourly weather data
        """
        if not station_id:
            station_id = self.station
            
        # Format dates for NCEI API
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        
        # NCEI API has a limit on the time range, so we'll fetch data in chunks
        max_days_per_request = 30
        
        all_data = []
        current_start = start_date_obj
        
        while current_start <= end_date_obj:
            # Calculate end date for this chunk
            current_end = min(current_start + timedelta(days=max_days_per_request), end_date_obj)
            
            # Format dates for API request
            current_start_str = current_start.strftime("%Y-%m-%d")
            current_end_str = current_end.strftime("%Y-%m-%d")
            
            logger.info(f"Fetching NCEI data from {current_start_str} to {current_end_str}")
            
            # Construct request parameters
            params = {
                "dataset": self.dataset,
                "stations": station_id,
                "startDate": current_start_str,
                "endDate": current_end_str,
                "format": "csv",
                "includeAttributes": "true",
                "includeStationName": "true",
                "includeStationLocation": "true",
                "units": "standard",  # Use standard units
            }
            
            # Make the request
            response = self._make_request(self.base_url, params)
            
            if response:
                # Parse CSV data
                try:
                    df = pd.read_csv(io.StringIO(response))
                    all_data.append(df)
                except Exception as e:
                    logger.error(f"Error parsing CSV data: {e}")
            
            # Move to next chunk
            current_start = current_end + timedelta(days=1)
            
            # Respect rate limits
            time.sleep(1)
        
        if not all_data:
            logger.warning(f"No data found for the specified parameters")
            return pd.DataFrame()
        
        # Combine all chunks
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Process the data
        processed_df = self._process_hourly_data(combined_df)
        
        return processed_df
    
    def get_daily_aggregated_data(self, 
                                 start_date: str, 
                                 end_date: str, 
                                 station_id: str = None) -> pd.DataFrame:
        """
        Get daily aggregated weather data from hourly observations.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            station_id: ID of the weather station
            
        Returns:
            DataFrame with daily aggregated weather data
        """
        hourly_data = self.get_hourly_data(start_date, end_date, station_id)
        
        if hourly_data.empty:
            return pd.DataFrame()
        
        # Create date column without time
        hourly_data["date"] = pd.to_datetime(hourly_data["DATE"]).dt.date
        
        # Aggregate by date
        daily_data = hourly_data.groupby("date").agg({
            "TMP": "mean",  # Mean temperature
            "DEW": "mean",  # Mean dew point
            "SLP": "mean",  # Mean sea level pressure
            "WND_SPEED": "mean",  # Mean wind speed
            "WND_DIRECTION": lambda x: self._mean_direction(x),  # Mean wind direction
            "CIG": "mean",  # Mean ceiling height
            "VIS": "mean",  # Mean visibility
            "PRECIP": "sum",  # Total precipitation
            "RH": "mean",  # Mean relative humidity
        }).reset_index()
        
        # Rename columns for clarity
        daily_data = daily_data.rename(columns={
            "TMP": "temperature_avg",
            "DEW": "dew_point",
            "SLP": "pressure",
            "WND_SPEED": "wind_speed",
            "WND_DIRECTION": "wind_direction",
            "CIG": "ceiling_height",
            "VIS": "visibility",
            "PRECIP": "precipitation",
            "RH": "humidity",
        })
        
        # Convert date to datetime
        daily_data["date"] = pd.to_datetime(daily_data["date"])
        
        return daily_data
    
    def _process_hourly_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process hourly data from NCEI.
        
        Args:
            df: DataFrame with raw NCEI data
            
        Returns:
            Processed DataFrame
        """
        # Create a copy to avoid modifying the original
        processed = df.copy()
        
        # Convert date to datetime
        processed["DATE"] = pd.to_datetime(processed["DATE"])
        
        # Process temperature (in tenths of degrees C)
        if "TMP" in processed.columns:
            processed["TMP"] = pd.to_numeric(processed["TMP"], errors="coerce") / 10.0
            # Convert to Fahrenheit for consistency
            processed["TMP"] = processed["TMP"] * 9/5 + 32
        
        # Process dew point (in tenths of degrees C)
        if "DEW" in processed.columns:
            processed["DEW"] = pd.to_numeric(processed["DEW"], errors="coerce") / 10.0
            # Convert to Fahrenheit for consistency
            processed["DEW"] = processed["DEW"] * 9/5 + 32
        
        # Process sea level pressure (in tenths of hPa)
        if "SLP" in processed.columns:
            processed["SLP"] = pd.to_numeric(processed["SLP"], errors="coerce") / 10.0
        
        # Process wind data
        if "WND" in processed.columns:
            # WND format: direction,direction_quality,type_code,speed,speed_quality
            processed["WND_DIRECTION"] = processed["WND"].str.split(",").str[0].astype(float)
            processed["WND_SPEED"] = processed["WND"].str.split(",").str[3].astype(float) / 10.0  # Convert to m/s
            # Convert to mph for consistency
            processed["WND_SPEED"] = processed["WND_SPEED"] * 2.237
        
        # Process precipitation (in mm)
        if "AA1" in processed.columns:
            # AA1 format: period,depth,condition,quality
            processed["PRECIP"] = processed["AA1"].str.split(",").str[1].replace("", "0").astype(float) / 10.0
            # Convert to inches for consistency
            processed["PRECIP"] = processed["PRECIP"] / 25.4
        else:
            processed["PRECIP"] = 0
        
        # Process ceiling height (in meters)
        if "CIG" in processed.columns:
            processed["CIG"] = pd.to_numeric(processed["CIG"], errors="coerce")
        
        # Process visibility (in meters)
        if "VIS" in processed.columns:
            processed["VIS"] = pd.to_numeric(processed["VIS"], errors="coerce")
        
        # Calculate relative humidity from temperature and dew point
        if "TMP" in processed.columns and "DEW" in processed.columns:
            processed["RH"] = self._calculate_rh(processed["TMP"], processed["DEW"])
        
        return processed
    
    def _calculate_rh(self, temp_f: pd.Series, dew_f: pd.Series) -> pd.Series:
        """
        Calculate relative humidity from temperature and dew point (both in Fahrenheit).
        
        Args:
            temp_f: Temperature in Fahrenheit
            dew_f: Dew point in Fahrenheit
            
        Returns:
            Relative humidity as percentage
        """
        # Convert to Celsius for the calculation
        temp_c = (temp_f - 32) * 5/9
        dew_c = (dew_f - 32) * 5/9
        
        # Calculate saturation vapor pressure
        svp = 6.11 * 10.0 ** (7.5 * temp_c / (237.3 + temp_c))
        
        # Calculate actual vapor pressure
        avp = 6.11 * 10.0 ** (7.5 * dew_c / (237.3 + dew_c))
        
        # Calculate relative humidity
        rh = (avp / svp) * 100
        
        return rh
    
    def _mean_direction(self, directions: pd.Series) -> float:
        """
        Calculate the mean of circular wind directions.
        
        Args:
            directions: Series of wind directions in degrees
            
        Returns:
            Mean wind direction in degrees
        """
        import numpy as np
        
        # Convert to radians
        rad = np.radians(directions)
        
        # Calculate mean of sin and cos components
        sin_mean = np.nanmean(np.sin(rad))
        cos_mean = np.nanmean(np.cos(rad))
        
        # Convert back to degrees
        degrees = np.degrees(np.arctan2(sin_mean, cos_mean))
        
        # Ensure result is in [0, 360)
        return (degrees + 360) % 360
    
    def _make_request(self, url: str, params: Dict[str, Any]) -> str:
        """
        Make a request to the NCEI API with error handling.
        
        Args:
            url: API URL
            params: Query parameters
            
        Returns:
            Response text (CSV data)
        """
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.text
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e}")
            return ""
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return ""


if __name__ == "__main__":
    # Example usage
    client = NCEIClient()
    
    # Get hourly data for a short period
    hourly_data = client.get_hourly_data("2022-01-01", "2022-01-03")
    print(hourly_data.head())
    
    # Get daily aggregated data
    daily_data = client.get_daily_aggregated_data("2022-01-01", "2022-01-10")
    print(daily_data.head())
