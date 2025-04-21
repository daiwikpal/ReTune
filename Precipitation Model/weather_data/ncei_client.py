import requests
import pandas as pd
from datetime import datetime, timedelta, date
import os
import time
import calendar
from typing import List, Dict, Optional, Any, Union
import io
from dotenv import load_dotenv
import sys
import logging

# Add parent directory to path to allow importing config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Set up logging
logger = logging.getLogger(__name__)

class NCEIClient:
    """Client for interacting with NCEI's Climate Data Online API"""
    
    def __init__(self, token: Optional[str] = None, log_level: int = logging.INFO):
        """
        Initialize the NCEI client with an API token
        
        Args:
            token: API token for NCEI (optional if in environment)
            log_level: Logging level (default: INFO)
        """
        # Configure logger
        self._configure_logger(log_level)
        
        self.base_url = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
        self.pathToEnv = os.path.join(os.path.dirname(__file__), '..', '.env')
        load_dotenv(dotenv_path=self.pathToEnv)
        self.token = token or os.getenv("NOAA_TOKEN")
        if not self.token:
            logger.error("API token not found in constructor or environment")
            raise ValueError("API token is required. Set it in the constructor or as NOAA_TOKEN environment variable")
        
        # Station ID for precipitation data
        self.station_id = "GHCND:USW00094728"  # Central Park, NY
        self.dataset_id = "GHCND"  # Global Historical Climatology Network Daily
        self.datatype_id = "PRCP"  # Precipitation
        self.page_limit = 1000  # API page size limit
        
        logger.info(f"NCEI Client initialized (Station: {self.station_id})")
    
    def _configure_logger(self, log_level: int):
        """Configure the logger with appropriate handlers and formatting"""
        logger.setLevel(log_level)
        
        # Clear existing handlers if any
        if logger.handlers:
            logger.handlers = []
            
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(ch)
    
    def get_monthly_precipitation(self, start_date: Union[str, date], end_date: Union[str, date]) -> pd.DataFrame:
        """
        Fetch monthly precipitation data for NYC (Central Park)
        
        Args:
            start_date: Start date in YYYY-MM-DD format or date object
            end_date: End date in YYYY-MM-DD format or date object
            
        Returns:
            DataFrame with monthly precipitation data (in inches)
        """
        # Convert string dates to date objects if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
            
        logger.info(f"Requesting NCEI precipitation data from {start_date} to {end_date}")
        
        
        # Create DataFrame with monthly data
        monthly_data = []
        
        # Get the first day of the month for the start date
        current_month_start = date(start_date.year, start_date.month, 1)
        logger.info(f"Processing data month by month starting from {current_month_start}")
        
        month_count = 0
        while current_month_start <= end_date:
            # Calculate the last day of the current month
            if current_month_start.month == 12:
                next_month = date(current_month_start.year + 1, 1, 1)
            else:
                next_month = date(current_month_start.year, current_month_start.month + 1, 1)
            
            current_month_end = next_month - timedelta(days=1)
            
            # Only process if month is within our date range
            if current_month_end >= start_date:
                # Adjust range for partial months
                range_start = max(start_date, current_month_start)
                range_end = min(end_date, current_month_end)
                
                logger.debug(f"Processing month: {current_month_start.year}-{current_month_start.month:02d}")
                
                try:
                    # Get precipitation for this month
                    month_precip = self._get_total_precipitation(range_start, range_end)
                    
                    # Add to our monthly data
                    monthly_data.append({
                        'TimeStamp': f"{current_month_start.year}-{current_month_start.month:02d}",
                        'PRECIPITATION': month_precip
                    })
                    
                    month_count += 1
                    logger.debug(f"Month {current_month_start.year}-{current_month_start.month:02d}: {month_precip:.2f} inches")
                    
                    # Log progress every few months
                    if month_count % 5 == 0:
                        logger.info(f"Processed {month_count} months so far")
                        
                except Exception as e:
                    logger.error(f"Error getting precipitation for {current_month_start.year}-{current_month_start.month:02d}: {str(e)}")
            
            # Move to next month
            if current_month_start.month == 12:
                current_month_start = date(current_month_start.year + 1, 1, 1)
            else:
                current_month_start = date(current_month_start.year, current_month_start.month + 1, 1)
        
        # Convert to DataFrame
        if monthly_data:
            df = pd.DataFrame(monthly_data)
            logger.info(f"Created monthly precipitation DataFrame with {len(df)} months of data")
            return df
        else:
            logger.warning(f"No precipitation data found for period {start_date} to {end_date}")
            return pd.DataFrame()
    
    def get_historical_monthly_precipitation(self, years: int = 30) -> pd.DataFrame:
        """
        Fetch historical monthly precipitation data from January 1996 to present
        
        Args:
            years: Number of years of historical data to fetch (default: 30, but will always go back to at least 1996)
            
        Returns:
            DataFrame with monthly precipitation data (in inches)
        """
        end_date = date.today()
        
        # Calculate start date based on years parameter
        calculated_start = date(end_date.year - years, end_date.month, 1)
        
        # Ensure we go back to at least January 1996
        min_start_date = date(1996, 1, 1)
        start_date = min(calculated_start, min_start_date)
        
        logger.info(f"Current system date: {datetime.now().strftime('%Y-%m-%d')}")
        logger.info(f"Fetching NCEI historical data from {start_date} to {end_date} ({years} years)")
        
        # Split into multiple requests with smaller time ranges to avoid API limitations
        all_data = []
        current_start = start_date
        
        # Process one year at a time to avoid timeout issues
        total_years = end_date.year - start_date.year + 1
        year_count = 0
        
        while current_start < end_date:
            year_count += 1
            # End date is either Dec 31 of current year or today if it's the current year
            if current_start.year == end_date.year:
                current_end = end_date
            else:
                current_end = date(current_start.year, 12, 31)
            
            logger.info(f"Fetching year {current_start.year} data ({year_count}/{total_years})")
            
            try:
                # Get data for this year
                chunk_data = self.get_monthly_precipitation(current_start, current_end)
                
                if not chunk_data.empty:
                    logger.info(f"Got data for year {current_start.year}: {len(chunk_data)} months")
                    all_data.append(chunk_data)
                else:
                    logger.warning(f"No data found for year {current_start.year}")
            except Exception as e:
                logger.error(f"Error processing year {current_start.year}: {str(e)}")
            
            # Move to the next year
            current_start = date(current_start.year + 1, 1, 1)
        
        if not all_data:
            logger.error("No precipitation data found for the entire requested period")
            return pd.DataFrame()
            
        # Combine all chunks
        logger.info(f"Combining data from {len(all_data)} years")
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Remove duplicates if any
        dupe_count = combined_df.duplicated(subset=['TimeStamp']).sum()
        if dupe_count > 0:
            logger.info(f"Removing {dupe_count} duplicate monthly records")
            combined_df = combined_df.drop_duplicates(subset=['TimeStamp'])
        
        # Sort by TimeStamp
        combined_df['date_for_sort'] = pd.to_datetime(combined_df['TimeStamp'])
        combined_df = combined_df.sort_values('date_for_sort')
        combined_df = combined_df.drop('date_for_sort', axis=1)
        
        logger.info(f"Final historical precipitation dataset contains {len(combined_df)} months from {combined_df['TimeStamp'].min()} to {combined_df['TimeStamp'].max()}")
        return combined_df
    
    def _get_total_precipitation(self, start: date, end: date) -> float:
        """
        Get the total precipitation in inches between start and end dates (inclusive).
        
        Args:
            start: Start date
            end: End date
            
        Returns:
            Total precipitation in inches
        """
        if start > end:
            logger.error(f"Invalid date range: {start} > {end}")
            raise ValueError("Start date must be before or equal to end date")
        
        logger.debug(f"Fetching precipitation data for {start} to {end}")
        
        headers = {"token": self.token}
        params = {
            "datasetid": self.dataset_id,
            "datatypeid": self.datatype_id,
            "stationid": self.station_id,
            "startdate": start.isoformat(),
            "enddate": end.isoformat(),
            "units": "standard",  # Gets data in tenths of inches
            "limit": self.page_limit,
        }
        
        total_inches = 0.0
        offset = 1
        page_count = 0
        
        while True:
            page_count += 1
            params["offset"] = offset
            try:
                logger.debug(f"Making API request with offset {offset} (page {page_count})")
                response = requests.get(self.base_url, headers=headers, params=params, timeout=30)
                
                # For debugging
                logger.debug(f"Request → {response.request.method} {response.request.url}")
                
                response.raise_for_status()
                data = response.json()
                
                results = data.get("results", [])
                if not results:
                    logger.debug("No results found in response")
                else:
                    
                    page_precip = sum(r["value"] for r in results) 
                    total_inches += page_precip
                    logger.debug(f"Page {page_count} has {len(results)} records, {page_precip:.2f} inches")
                
                if len(results) < self.page_limit:
                    logger.debug(f"End of pagination reached after {page_count} pages")
                    break
                    
                offset += self.page_limit
                
                # Respect rate limits with a small delay
                time.sleep(0.5)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"API request failed: {str(e)}")
                # Return partial data if we've already got some
                if total_inches > 0:
                    logger.warning(f"Returning partial data ({total_inches:.2f} inches)")
                    return total_inches
                raise
        
        logger.debug(f"Total precipitation for {start} to {end}: {total_inches:.2f} inches")
        return total_inches
    
    def get_month_total(self, year: int, month: int) -> float:
        """
        Get total precipitation in inches for a specific month and year.
        
        Args:
            year: Year (e.g., 2024)
            month: Month (1-12)
            
        Returns:
            Total precipitation in inches for the month
        """
        if month not in range(1, 13):
            logger.error(f"Invalid month: {month}")
            raise ValueError("Month must be 1-12")
            
        start_date = date(year, month, 1)
        end_date = date(year, month, calendar.monthrange(year, month)[1])
        
        logger.info(f"Getting precipitation total for {year}-{month:02d}")
        return self._get_total_precipitation(start_date, end_date) 