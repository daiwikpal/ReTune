"""
Client for interacting with the NOAA Climate Data Online (CDO) API.
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

class NOAAClient:
    """
    Client for fetching historical weather data from NOAA's Climate Data Online API.
    """
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize the NOAA API client.
        
        Args:
            token: NOAA API token. If None, uses the token from config.
        """
        self.base_url = config.NOAA_BASE_URL
        self.token = token or config.NOAA_TOKEN
        
        if not self.token:
            logger.warning("NOAA API token not provided. Set NOAA_TOKEN in .env file.")
        
        self.headers = {
            "token": self.token
        }
    
    def get_stations(self, location_id: str = None, extent: str = None) -> List[Dict[str, Any]]:
        """
        Get weather stations based on location or bounding box.
        
        Args:
            location_id: ID of the location to search for stations
            extent: Bounding box for the search (minLat, minLon, maxLat, maxLon)
            
        Returns:
            List of station metadata
        """
        endpoint = f"{self.base_url}/stations"
        params = {
            "datasetid": config.NOAA_DATASET_ID,
            "limit": 1000,
        }
        
        if location_id:
            params["locationid"] = location_id
        
        if extent:
            params["extent"] = extent
            
        response = self._make_request(endpoint, params)
        return response.get("results", [])
    
    def get_data_types(self) -> List[Dict[str, Any]]:
        """
        Get available data types from NOAA API.
        
        Returns:
            List of data types metadata
        """
        endpoint = f"{self.base_url}/datatypes"
        params = {
                "datasetid": config.NOAA_DATASET_ID,
                "limit": 1000,
            }
            
        response = self._make_request(endpoint, params)
        return response.get("results", [])
        
    def get_daily_data(self, start_date: str, end_date: str, station_id: str = None, data_types: List[str] = None) -> pd.DataFrame:
        if not station_id:
            station_id = config.NOAA_STATION_ID

        if not data_types:
            data_types = ["PRCP", "TMAX", "TMIN", "TAVG", "AWND"]

        # valid_types = [dt['id'] for dt in self.get_data_types()]
        valid_types = ['PRCP', 'TMAX', 'TMIN', 'TAVG', 'AWND']
        unsupported = [dt for dt in data_types if dt not in valid_types]
        data_types = [dt for dt in data_types if dt in valid_types]
        if unsupported:
            logger.warning(f"Skipping unsupported datatypes: {unsupported}")

        all_results = []
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        current = start

        while current <= end:
            chunk_start = current
            chunk_end = min(chunk_start + timedelta(days=29), end)  # 30-day chunk

            params = {
                "datasetid": config.NOAA_DATASET_ID,
                "stationid": station_id,
                "startdate": chunk_start.strftime("%Y-%m-%d"),
                "enddate": chunk_end.strftime("%Y-%m-%d"),
                "datatypeid": ",".join(data_types),
                "units": "standard",
                "limit": 1000,
                "offset": 0,
            }
            logger.info(f"Fetching NOAA data from {params['startdate']} to {params['enddate']}...")
            chunk_results = []

            while True:
                params["offset"] = len(chunk_results)
                response = self._make_request(f"{self.base_url}/data", params)
                results = response.get("results", [])
                if not results:
                    break
                chunk_results.extend(results)
                if len(results) < 1000:
                    break
                time.sleep(2)

            all_results.extend(chunk_results)
            current = chunk_end + timedelta(days=1)

        if not all_results:
            logger.warning(f"No data found for the specified parameters between {start_date} and {end_date}.")
            return pd.DataFrame()

        df = pd.DataFrame(all_results)
        if df.empty:
            return df

        pivot_df = df.pivot_table(
            index="date",
            columns="datatype",
            values="value",
            aggfunc="first"
        ).reset_index()

        pivot_df["date"] = pd.to_datetime(pivot_df["date"])
        pivot_df = pivot_df.sort_values("date")

        for col in pivot_df.columns:
            if col != "date":
                pivot_df[col] = pd.to_numeric(pivot_df[col], errors="coerce")

        if "PRCP" in pivot_df.columns:
            pivot_df["PRCP"] = pivot_df["PRCP"] / 10.0 / 25.4  # Convert from tenths of mm to inches
        if "TMAX" in pivot_df.columns:
            pivot_df["TMAX"] = (pivot_df["TMAX"] / 10.0) * 9/5 + 32
        if "TMIN" in pivot_df.columns:
            pivot_df["TMIN"] = (pivot_df["TMIN"] / 10.0) * 9/5 + 32
        if "TAVG" in pivot_df.columns:
            pivot_df["TAVG"] = (pivot_df["TAVG"] / 10.0) * 9/5 + 32

        return pivot_df

    
    def get_monthly_precipitation(self, 
                                 start_date: str, 
                                 end_date: str, 
                                 station_id: str = None) -> pd.DataFrame:
        """
        Get monthly precipitation totals.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            station_id: ID of the weather station
            
        Returns:
            DataFrame with monthly precipitation totals
        """
        daily_data = self.get_daily_data(start_date, end_date, station_id, ["PRCP"])
        
        if daily_data.empty:
            return pd.DataFrame()
        
        #year and month columns
        daily_data["year"] = daily_data["date"].dt.year
        daily_data["month"] = daily_data["date"].dt.month
        monthly_data = daily_data.groupby(["year", "month"])["PRCP"].sum().reset_index()
        monthly_data["date"] = pd.to_datetime(monthly_data[["year", "month"]].assign(day=1))
        monthly_data = monthly_data.rename(columns={"PRCP": "precipitation"})
        monthly_data = monthly_data[["date", "year", "month", "precipitation"]]
        
        return monthly_data
    
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a request to the NOAA API with error handling and rate limiting.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            JSON response as dictionary
        """
        try:
            response = requests.get(endpoint, headers=self.headers, params=params)
            
            if response.status_code == 429:
                logger.warning("Rate limited by NOAA API.")
                time.sleep(5) # wait 5 seconds because of rate lim!
                return self._make_request(endpoint, params)
                
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 400 and "Invalid Token" in response.text:
                logger.error("Invalid NOAA API token. Please check your token.")
            elif response.status_code == 503:
                logger.error("NOAA API service unavailable. Try again later.")
            else:
                logger.error(f"HTTP error: {e}")
            return {"results": []}
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return {"results": []}
            
        except ValueError as e:
            logger.error(f"JSON parsing error: {e}")
            return {"results": []}


if __name__ == "__main__":
    # Example usage
    client = NOAAClient()
    
    # Get data for Central Park, NY for 2022
    data = client.get_daily_data("2022-01-01", "2022-01-10")
    print(data.head())
    
    #Get monthly precipitation totals
    monthly_data = client.get_monthly_precipitation("2020-01-01", "2022-12-31")
    print(monthly_data.head())
