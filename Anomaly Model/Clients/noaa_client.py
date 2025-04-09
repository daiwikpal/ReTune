import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv


class NOAAClient:
    """Client for interacting with NOAA's Climate Data Online API"""
    
    def __init__(self, token: Optional[str] = None):
        """Initialize the NOAA client with an API token"""
        self.base_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2"
        self.pathToEnv = os.path.join(os.path.dirname(__file__), '..', '.env')
        load_dotenv(dotenv_path=self.pathToEnv)
        self.token = token or os.getenv("NOAA_API_TOKEN")
        if not self.token:
            raise ValueError("NOAA API token is required. Set it in the constructor or as NOAA_API_TOKEN environment variable")
        
        self.headers = {
            "token": self.token
        }
        
        # Central Park Station ID for NYC precipitation data
        self.nyc_station_id = "GHCND:USW00094728"  # Central Park Station
    
    def get_monthly_precipitation(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch monthly precipitation data for NYC (Central Park Station)
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with monthly precipitation data
        """
        endpoint = f"{self.base_url}/data"
        
        params = {
            "datasetid": "GHCND",  # Global Historical Climatology Network Daily
            "stationid": self.nyc_station_id,
            "datatypeid": "PRCP",  # Precipitation
            "startdate": start_date,
            "enddate": end_date,
            "limit": 1000,
            "units": "standard"
        }
        
        print(f"Requesting data from {endpoint} with params: {params}")  # Debugging output
        
        try:
            response = requests.get(endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data.get('results'):
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data['results'])
            
            # Convert date to datetime and set as index
            df['date'] = pd.to_datetime(df['date'])
            
            # Convert precipitation from tenths of mm to inches
            df['value'] = df['value'] / 254  # Convert from tenths of mm to inches
            
            # Group by month and sum precipitation
            monthly_df = df.groupby(df['date'].dt.to_period('M'))['value'].sum().reset_index()
            monthly_df['date'] = monthly_df['date'].astype(str)
            monthly_df.columns = ['TimeStamp', 'PRECIPITATION']
            
            return monthly_df
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching precipitation data: {str(e)}")
            return pd.DataFrame()
    
    def get_historical_monthly_precipitation(self, years: int = 20) -> pd.DataFrame:
        """
        Fetch historical monthly precipitation data for the specified number of years
        
        Args:
            years: Number of years of historical data to fetch (default: 20)
            
        Returns:
            DataFrame with monthly precipitation data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        # Debugging output for current date verification
        print(f"Current system date: {datetime.now().strftime('%Y-%m-%d')}")
        
        # Ensure end_date does not exceed the current date
        end_date = datetime.now()  # Set end_date to the current date
        
        # Debugging output for date verification
        print(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        return self.get_monthly_precipitation(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        ) 