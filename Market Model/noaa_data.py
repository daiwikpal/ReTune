"""
Module to collect NOAA historical precipitation data.
"""
import os
import logging
import requests
import pandas as pd
from datetime import datetime
import config

logger = logging.getLogger(__name__)

def collect_noaa_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Collect historical precipitation data from NOAA.
    Returns:
        A DataFrame with NOAA data.
    """
    base_url = config.NOAA_BASE_URL
    token = config.NOAA_TOKEN
    headers = {"token": token}
    endpoint = f"{base_url}/data"
    params = {
        "datasetid": config.NOAA_DATASET_ID,
        "stationid": config.NOAA_STATION_ID,
        "startdate": start_date,
        "enddate": end_date,
        "limit": 1000,
        "units": "standard"
    }
    try:
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        if "results" in data:
            df = pd.DataFrame(data["results"])
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            return df
    except Exception as e:
        logger.error(f"Error collecting NOAA data: {e}")
    return pd.DataFrame()

def combine_market_and_noaa_data(market_df: pd.DataFrame, noaa_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine market data and NOAA precipitation data on the date column.
    Still to refine right now really basic and just combines them
    """
    if 'date' not in market_df.columns or 'date' not in noaa_df.columns:
        return market_df
    combined_df = pd.merge(market_df, noaa_df, on='date', how='inner')
    return combined_df
