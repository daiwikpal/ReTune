"""
Data processor for combining and preparing weather data for the LSTM model.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
from typing import Dict, List, Any, Optional, Tuple

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from weather_data.noaa_client import NOAAClient
from weather_data.openweather_client import OpenWeatherClient
from weather_data.ncei_client import NCEIClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Processor for combining and preparing weather data from different sources.
    """
    
    def __init__(self):
        """
        Initialize the data processor.
        """
        self.noaa_client = None
        self.openweather_client = OpenWeatherClient()
        self.ncei_client = NCEIClient()
        
        # Create data directory if it doesn't exist
        os.makedirs(config.DATA_DIR, exist_ok=True)
    
    def collect_historical_data(self, 
                                start_date: str = None, 
                                end_date: str = None) -> pd.DataFrame:
        """
        Collect historical weather data from NOAA only.
        """
        if os.path.exists(config.OUTPUT_FILE):
            logger.info(f"Found existing data file at {config.OUTPUT_FILE}, loading instead of refetching")
            return pd.read_csv(config.OUTPUT_FILE)

        if start_date is None:
            start_date = config.HISTORICAL_START_DATE
        if end_date is None:
            end_date = config.HISTORICAL_END_DATE

        logger.info(f"Collecting historical data from {start_date} to {end_date}")

        # Initialize NOAA client
        if self.noaa_client is None:
            self.noaa_client = NOAAClient()

        # Fetch NOAA daily data
        logger.info("Fetching NOAA daily data...")
        noaa_data = self.noaa_client.get_daily_data(start_date, end_date)

        if noaa_data.empty:
            logger.warning("NOAA data is empty!")
            return pd.DataFrame()

        if "date" not in noaa_data.columns:
            logger.warning("NOAA data missing 'date' column. Attempting to create it from index...")
            noaa_data["date"] = pd.to_datetime(noaa_data.index)

        debug_path = os.path.join(config.DATA_DIR, "_debug_noaa.csv")
        noaa_data.to_csv(debug_path, index=False)
        logger.info(f"Saved NOAA debug data to: {debug_path}")

        # Rename columns
        noaa_renamed = noaa_data.rename(columns={
            "PRCP": "precipitation",
            "TMAX": "temperature_max",
            "TMIN": "temperature_min",
            "TAVG": "temperature_avg",
            "AWND": "wind_speed",
        })

        if "date" not in noaa_renamed.columns:
            raise ValueError("NOAA data is missing the 'date' column even after attempt to fix.")

        return noaa_renamed.sort_values("date")


    
    def collect_forecast_data(self, days: int = None) -> pd.DataFrame:
        """
        Collect forecast data from OpenWeatherMap.
        
        Args:
            days: Number of days to forecast
            
        Returns:
            DataFrame with forecast data
        """
        if days is None:
            days = config.FORECAST_DAYS
            
        logger.info(f"Collecting forecast data for the next {days} days")
        
        forecast_data = self.openweather_client.get_forecast(days=days)
        
        return forecast_data
    
    def prepare_data_for_model(self, 
                             historical_data: pd.DataFrame, 
                             forecast_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Prepare combined data for the LSTM model.
        
        Args:
            historical_data: Historical weather data
            forecast_data: Forecast data (optional)
            
        Returns:
            DataFrame with prepared data
        """
        data = historical_data.copy()
        
        if forecast_data is not None and not forecast_data.empty:
            if 'date' in forecast_data.columns and 'date' in data.columns:
                forecast_data = forecast_data[~forecast_data["date"].isin(data["date"])]
            else:
                # Log warning if 'date' column is missing
                logger.warning("'date' column missing in either historical_data or forecast_data. "
                              "Skipping duplicate date filtering.")
                # If 'date' is missing in forecast_data, try to create it from index if it's a DatetimeIndex
                if 'date' not in forecast_data.columns and isinstance(forecast_data.index, pd.DatetimeIndex):
                    forecast_data = forecast_data.reset_index().rename(columns={'index': 'date'})
                # If 'date' is missing in data, try to create it from index if it's a DatetimeIndex
                if 'date' not in data.columns and isinstance(data.index, pd.DatetimeIndex):
                    data = data.reset_index().rename(columns={'index': 'date'})
                    
            data = pd.concat([data, forecast_data], ignore_index=True)
        
        # Ensure 'date' column exists before sorting
        if 'date' in data.columns:
            data = data.sort_values("date")
        else:
            logger.warning("'date' column not found in combined data. Cannot sort by date.")

        data = self._fill_missing_values(data)
        data = self._create_features(data)
        data = self._select_features(data)
        
        return data
    
    def create_sequences(self, 
                        data: pd.DataFrame, 
                        sequence_length: int = None,
                        target_column: str = "precipitation") -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM model training.
        
        Args:
            data: Prepared data
            sequence_length: Number of time steps in each sequence
            target_column: Column to predict
            
        Returns:
            Tuple of (X, y) arrays for model training
        """
        if sequence_length is None:
            sequence_length = config.SEQUENCE_LENGTH
            
        data = data.sort_values("date")
        features = data.drop(columns=["date"])
        normalized_data, scalers = self._normalize_data(features)
        X, y = [], []
        for i in range(len(normalized_data) - sequence_length):
            X.append(normalized_data[i:i+sequence_length])
            y.append(normalized_data[i+sequence_length, features.columns.get_loc(target_column)])
        
        return np.array(X), np.array(y), scalers
    
    def save_data(self, data: pd.DataFrame, filename: str = None) -> str:
        """
        Save processed data to CSV.
        
        Args:
            data: Data to save
            filename: Output filename
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            filename = config.OUTPUT_FILE

        # build absolute path to <project_root>/filename
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        absolute_path = os.path.join(project_root, filename)

        # ensure parent dir exists
        os.makedirs(os.path.dirname(absolute_path), exist_ok=True)

        # write CSV
        data.to_csv(absolute_path, index=False)
        logger.info(f"Data saved to {absolute_path}")

        return absolute_path
    
    def _combine_historical_data(self, 
                                noaa_data: pd.DataFrame, 
                                ncei_data: pd.DataFrame) -> pd.DataFrame:
        """
        Combine historical data from NOAA and NCEI, avoiding duplication from fallback logic.
        """
        # If both are empty, return empty
        if noaa_data.empty and ncei_data.empty:
            logger.warning("Both NOAA and NCEI data are empty.")
            return pd.DataFrame()

        # Check if NOAA data is mostly missing or single row duplicated
        if noaa_data.empty or noaa_data.drop(columns=["date"], errors="ignore").nunique().sum() <= 2:
            logger.warning("NOAA data looks mostly constant or empty. Skipping it.")
            noaa_data = pd.DataFrame()

        # Same check for NCEI
        if ncei_data.empty or ncei_data.drop(columns=["date"], errors="ignore").nunique().sum() <= 2:
            logger.warning("NCEI data looks mostly constant or empty. Skipping it.")
            ncei_data = pd.DataFrame()

        if noaa_data.empty:
            return ncei_data
        if ncei_data.empty:
            return noaa_data

        # Rename NOAA columns
        noaa_renamed = noaa_data.rename(columns={
            "PRCP": "precipitation",
            "TMAX": "temperature_max",
            "TMIN": "temperature_min",
            "TAVG": "temperature_avg",
            "AWND": "wind_speed",
        })

        # Merge on date
        merged = pd.merge(noaa_renamed, ncei_data, on="date", how="outer", suffixes=("_noaa", "_ncei"))
        combined = pd.DataFrame()
        combined["date"] = merged["date"]

        # Combine with preference for NOAA
        combined["precipitation"] = merged["precipitation_noaa"].fillna(merged["precipitation"])
        combined["temperature_max"] = merged["temperature_max_noaa"].fillna(merged["temperature_avg"] + 5)
        combined["temperature_min"] = merged["temperature_min_noaa"].fillna(merged["temperature_avg"] - 5)
        combined["temperature_avg"] = merged["temperature_avg_noaa"].fillna(merged["temperature_avg"])
        combined["humidity"] = merged["humidity"]
        combined["wind_speed"] = merged["wind_speed_noaa"].fillna(merged["wind_speed"])
        combined["pressure"] = merged["pressure"]
        combined["dew_point"] = merged["dew_point"]

        return combined.sort_values("date")

    
    def _fill_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values in the dataset.
        
        Args:
            data: DataFrame with potentially missing values
            
        Returns:
            DataFrame with filled values
        """
        # Make a copy to avoid modifying the original
        filled_data = data.copy()
        
        # Forward fill and backward fill for missing values
        # for col in filled_data.columns:
        #     if col != "date":
        #         # First try forward fill
        #         filled_data[col] = filled_data[col].fillna(method="ffill")
        #         # Then try backward fill for any remaining NaNs
        #         filled_data[col] = filled_data[col].fillna(method="bfill")
                
        #         # If still have NaNs, fill with column mean
        #         if filled_data[col].isna().any():
        #             col_mean = filled_data[col].mean()
        #             filled_data[col] = filled_data[col].fillna(col_mean)

        for col in filled_data.columns:
            if col != "date":
                if filled_data[col].isna().all():
                    logger.warning(f"All values missing for column: {col}. Leaving as NaN.")
                    continue
                filled_data[col] = filled_data[col].ffill().bfill()
        return filled_data
    
    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features for the model.

        Args:
            data: DataFrame with basic features

        Returns:
            DataFrame with additional features
        """
        featured_data = data.copy()

        # Ensure 'date' column is datetime
        if "date" in featured_data.columns:
            if not np.issubdtype(featured_data["date"].dtype, np.datetime64):
                featured_data["date"] = pd.to_datetime(featured_data["date"], errors="coerce")
            
            featured_data["month"] = featured_data["date"].dt.month
            featured_data["day"] = featured_data["date"].dt.day
            featured_data["dayofyear"] = featured_data["date"].dt.dayofyear
            featured_data["season"] = featured_data["month"].apply(
                lambda x: 1 if x in [12, 1, 2]
                        else 2 if x in [3, 4, 5]
                        else 3 if x in [6, 7, 8]
                        else 4
            )
        else:
            logger.warning("'date' column not found. Cannot create date-based features.")
        if "temperature_max" in featured_data.columns and "temperature_min" in featured_data.columns:
            featured_data["temperature_range"] = (
                featured_data["temperature_max"] - featured_data["temperature_min"]
            )

        if "precipitation" in featured_data.columns and not featured_data["precipitation"].isna().all():
            featured_data["precipitation_lag1"] = featured_data["precipitation"].shift(1)
            featured_data["precipitation_lag2"] = featured_data["precipitation"].shift(2)
            featured_data["precipitation_lag3"] = featured_data["precipitation"].shift(3)
            featured_data["precipitation_lag7"] = featured_data["precipitation"].shift(7)
            featured_data["precipitation_rolling_mean_7d"] = (
                featured_data["precipitation"].rolling(window=7).mean()
            )
            featured_data["precipitation_rolling_max_7d"] = (
                featured_data["precipitation"].rolling(window=7).max()
            )
        else:
            logger.warning("Cannot compute lag/rolling features: 'precipitation' column missing or empty.")

        featured_data = self._fill_missing_values(featured_data)

        return featured_data

    
    def _select_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Select and rename features for the model.
        
        Args:
            data: DataFrame with all features
            
        Returns:
            DataFrame with selected features
        """
        features_to_keep = ["date", "precipitation"]
        
        possible_features = [
            "temperature_max", "temperature_min", "temperature_avg", 
            "humidity", "wind_speed", "pressure", "dew_point",
            "temperature_range", "precipitation_lag1", "precipitation_lag7",
            "precipitation_rolling_mean_7d", "month", "season"
        ]
        
        for feature in possible_features:
            if feature in data.columns:
                features_to_keep.append(feature)
        
        selected_data = data[features_to_keep]
        
        return selected_data
    
    def _normalize_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """
        Normalize the data for the LSTM model.
        
        Args:
            data: DataFrame to normalize
            
        Returns:
            Tuple of (normalized_data, scalers)
        """
        from sklearn.preprocessing import MinMaxScaler
        
        scalers = {}
        normalized_data = np.zeros(data.shape)
        
        for i, column in enumerate(data.columns):
            scaler = MinMaxScaler(feature_range=(0, 1))
            normalized_data[:, i] = scaler.fit_transform(data[column].values.reshape(-1, 1)).flatten()
            scalers[column] = scaler
        
        return normalized_data, scalers

    def collect_ncei_data(self, start_date: str, end_date: str, save_path: str = "data/ncei_weather_data.csv") -> pd.DataFrame:
        """
        Collect historical monthly precipitation data from NCEI and save to a separate file.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            save_path: Path to save the CSV
            
        Returns:
            DataFrame with NCEI monthly precipitation data
        """
        logger.info(f"Collecting NCEI data from {start_date} to {end_date}")
        
        df = self.ncei_client.get_historical_monthly_precipitation(years=3)
        
        if df.empty:
            logger.warning("No NCEI data returned.")
            return df
        
        # standardized output!
        df = df.rename(columns={
            "TimeStamp": "date",
            "PRECIPITATION": "precipitation"
        })
        df["date"] = pd.to_datetime(df["date"])

        # Save it to file
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        absolute_save = os.path.join(project_root, save_path)
        os.makedirs(os.path.dirname(absolute_save), exist_ok=True)
        df.to_csv(absolute_save, index=False)
        logger.info(f"NCEI data saved to {absolute_save}")

        return df

    def test_save_csv(self, save_path: str = "data/test_output.csv") -> None:
        """
        Test saving a dummy DataFrame to verify directory path works.
        """
        dummy_df = pd.DataFrame({
            "date": ["2024-01-01", "2024-02-01"],
            "precipitation": [1.23, 4.56]
        })
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        dummy_df.to_csv(save_path, index=False)
        logger.info(f"✅ Dummy data saved to {save_path}")



if __name__ == "__main__":
    processor = DataProcessor()
    historical_data = processor.collect_historical_data("2020-01-01", "2022-12-31")
    forecast_data = processor.collect_forecast_data(days=7)
    prepared_data = processor.prepare_data_for_model(historical_data, forecast_data)
    processor.save_data(prepared_data, "data/nyc_weather_data.csv")
    X, y, scalers = processor.create_sequences(prepared_data, sequence_length=30)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    # Collect and save NCEI data separately
    ncei_data = processor.collect_ncei_data("2020-01-01", "2022-12-31")
    print(f"NCEI Data shape: {ncei_data.shape}")

