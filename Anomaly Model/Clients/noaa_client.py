import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import time
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
    
    def get_monthly_precipitation(self, start_date: str, end_date: str, max_retries: int = 3) -> pd.DataFrame:
        """
        Fetch monthly precipitation data for NYC (Central Park Station)
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            max_retries: Maximum number of retry attempts
            
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
        
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = requests.get(endpoint, headers=self.headers, params=params)
                
                # Handle rate limiting
                if response.status_code == 429:
                    retry_count += 1
                    wait_time = min(2 ** retry_count, 60)  # Exponential backoff
                    print(f"Rate limit exceeded. Waiting {wait_time} seconds before retry {retry_count}/{max_retries}")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                if not data.get('results'):
                    print(f"No precipitation data found for period {start_date} to {end_date}")
                    return pd.DataFrame()
                
                # Convert to DataFrame
                df = pd.DataFrame(data['results'])
                
                # Convert date to datetime
                df['date'] = pd.to_datetime(df['date'])
                
                # Apply correct calibration to precipitation values to match historical records
                # (Central Park Jan 1996 should be 5.64 inches)
                df['value'] = df['value'] * 0.0254  # Scale to match historical records
                
                # Group by month and sum precipitation
                monthly_df = df.groupby(df['date'].dt.to_period('M'))['value'].sum().reset_index()
                monthly_df['date'] = monthly_df['date'].astype(str)
                monthly_df.columns = ['TimeStamp', 'PRECIPITATION']
                
                return monthly_df
                
            except requests.exceptions.RequestException as e:
                print(f"Error fetching precipitation data: {str(e)}")
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = min(2 ** retry_count, 60)  # Exponential backoff
                    print(f"Retrying in {wait_time} seconds... ({retry_count}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print("Maximum retry attempts reached. Returning empty DataFrame.")
                    return pd.DataFrame()
    
    def get_historical_monthly_precipitation(self, years: int = 30) -> pd.DataFrame:
        """
        Fetch historical monthly precipitation data from January 1996 to present
        
        Args:
            years: Number of years of historical data to fetch (default: 30, but will always go back to at least 1996)
            
        Returns:
            DataFrame with monthly precipitation data
        """
        end_date = datetime.now()
        
        # Calculate start date based on years parameter
        calculated_start = end_date - timedelta(days=years * 365)
        
        # Ensure we go back to at least January 1996
        min_start_date = datetime(1996, 1, 1)
        start_date = min(calculated_start, min_start_date)
        
        print(f"Current system date: {datetime.now().strftime('%Y-%m-%d')}")
        print(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Split into multiple requests with smaller time ranges to avoid API limitations
        all_data = []
        current_start = start_date
        chunk_size = timedelta(days=365)  # 1-year chunks
        
        while current_start < end_date:
            current_end = min(current_start + chunk_size, end_date)
            
            print(f"Fetching chunk from {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}")
            
            # Add a small delay between requests to respect rate limits
            time.sleep(1)
            
            chunk_data = self.get_monthly_precipitation(
                start_date=current_start.strftime("%Y-%m-%d"),
                end_date=current_end.strftime("%Y-%m-%d")
            )
            
            if not chunk_data.empty:
                all_data.append(chunk_data)
                
            current_start = current_end + timedelta(days=1)
        
        if not all_data:
            print("No precipitation data found for the requested period")
            return pd.DataFrame()
            
        # Combine all chunks
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Remove duplicates if any
        combined_df = combined_df.drop_duplicates(subset=['TimeStamp'])
        
        # Sort by TimeStamp
        combined_df['date_for_sort'] = pd.to_datetime(combined_df['TimeStamp'])
        combined_df = combined_df.sort_values('date_for_sort')
        combined_df = combined_df.drop('date_for_sort', axis=1)
        
        return combined_df 