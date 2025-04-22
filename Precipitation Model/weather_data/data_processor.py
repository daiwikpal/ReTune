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

    def _aggregate_noaa_monthly(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert NOAA *daily* dataframe to a *monthly* dataframe
        with comprehensive aggregations for each feature.
        - Sum: precipitation
        - Mean/Min/Max: temperature, humidity, wind_speed, pressure
        - Cyclical encoding: month_sin, month_cos
        
        Missing columns get filled with NaN so .agg() will never KeyError.
        """
        if daily_df.empty:
            return pd.DataFrame()

        df = daily_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        # Columns to aggregate with different methods
        mean_cols = ["humidity", "wind_speed", "pressure"]
        temp_cols = ["temperature_max", "temperature_min"]
        
        # Ensure all needed columns exist
        for col in mean_cols + temp_cols:
            if col not in df.columns:
                df[col] = np.nan
                logger.warning(f"[aggregate] NOAA daily missing '{col}', filling with NaN")
        
        # Define aggregation map
        agg_map = {
            "precipitation": "sum"  # Sum precipitation
        }
        
        # Add mean aggregations
        for col in mean_cols:
            agg_map[col] = ["mean", "min", "max"]
            
        # Add temperature aggregations
        for col in temp_cols:
            agg_map[col] = ["mean", "min", "max"]
        
        # Perform aggregation
        monthly = df.resample("MS").agg(agg_map)
        
        # Flatten multi-index columns
        monthly.columns = ['_'.join(col).strip() for col in monthly.columns.values]
        
        # Reset index to get date column back
        monthly = monthly.reset_index()
        
        # Add temperature range (using averages)
        monthly["temperature_range"] = (
            monthly["temperature_max_mean"] - monthly["temperature_min_mean"]
        )
        
        # Add month number
        monthly["month"] = monthly["date"].dt.month
        
        # Add cyclical month encoding
        monthly["month_cos"] = np.cos(2 * np.pi * monthly["month"] / 12)
        monthly["month_sin"] = np.sin(2 * np.pi * monthly["month"] / 12)
        
        # Add season
        monthly["season"] = monthly["month"].apply(
            lambda x: 1 if x in [12, 1, 2]  # Winter
                    else 2 if x in [3, 4, 5]  # Spring
                    else 3 if x in [6, 7, 8]  # Summer
                    else 4  # Fall
        )
        
        return monthly
        
    def generate_monthly_data(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive monthly data by aggregating daily data.
        
        This method takes daily weather data and creates aggregations with:
        - Monthly sum for precipitation (as precipitation_sum)
        - Monthly mean/min/max for temperature, humidity, wind_speed, pressure
        - Temperature range calculations
        - Cyclical month encoding (sin/cos)
        - Seasonal information
        
        Args:
            daily_data: DataFrame with daily weather data
            
        Returns:
            DataFrame with monthly aggregated data
        """
        if daily_data.empty:
            logger.warning("Daily data is empty, cannot generate monthly data")
            return pd.DataFrame()
            
        # Ensure date column is datetime
        daily_data = daily_data.copy()
        daily_data["date"] = pd.to_datetime(daily_data["date"])
        
        # Set date as index for resampling
        daily_data = daily_data.set_index("date")
        
        # Columns to aggregate
        sum_cols = ["precipitation"]
        mean_cols = ["humidity", "wind_speed", "pressure"]
        temp_cols = ["temperature_max", "temperature_min"]
        
        # Create aggregation dictionary
        agg_dict = {}
        
        # Add sum aggregations
        for col in sum_cols:
            if col in daily_data.columns:
                agg_dict[col] = "sum"
                
        # Add mean, min, max aggregations for meteorological columns
        for col in mean_cols:
            if col in daily_data.columns:
                agg_dict[col] = ["mean", "min", "max"]
                
        # Add temperature aggregations
        for col in temp_cols:
            if col in daily_data.columns:
                agg_dict[col] = ["mean", "min", "max"]
        
        # Perform monthly aggregation
        monthly_data = daily_data.resample("MS").agg(agg_dict)
        
        # Flatten multi-index columns
        monthly_data.columns = ['_'.join(col).strip() for col in monthly_data.columns.values]
        
        # Reset index to get date as column
        monthly_data = monthly_data.reset_index()
        
        # Rename precipitation_sum if it exists
        if "precipitation_sum" in monthly_data.columns:
            logger.info("Found precipitation_sum column")
        elif "precipitation_sum" not in monthly_data.columns and "precipitation" in daily_data.columns:
            # Create precipitation_sum if it doesn't exist but we have daily precipitation
            logger.info("Renaming precipitation to precipitation_sum")
            monthly_data = monthly_data.rename(columns={"precipitation": "precipitation_sum"})
        
        # Calculate temperature range metrics
        if "temperature_max_mean" in monthly_data.columns and "temperature_min_mean" in monthly_data.columns:
            monthly_data["temperature_range_mean"] = (
                monthly_data["temperature_max_mean"] - monthly_data["temperature_min_mean"]
            )
            
        if "temperature_max_max" in monthly_data.columns and "temperature_min_min" in monthly_data.columns:
            monthly_data["temperature_range_extreme"] = (
                monthly_data["temperature_max_max"] - monthly_data["temperature_min_min"]
            )
        
        # Add month
        monthly_data["month"] = monthly_data["date"].dt.month
        
        # Add cyclical month encoding
        monthly_data["month_cos"] = np.cos(2 * np.pi * monthly_data["month"] / 12)
        monthly_data["month_sin"] = np.sin(2 * np.pi * monthly_data["month"] / 12)
        
        # Add season
        monthly_data["season"] = monthly_data["month"].apply(
            lambda x: 1 if x in [12, 1, 2]  # Winter
                    else 2 if x in [3, 4, 5]  # Spring
                    else 3 if x in [6, 7, 8]  # Summer
                    else 4  # Fall
        )
        
        # Add lag features for precipitation
        if "precipitation_sum" in monthly_data.columns:
            monthly_data["precipitation_lag1"] = monthly_data["precipitation_sum"].shift(1)
            monthly_data["precipitation_lag3"] = monthly_data["precipitation_sum"].shift(3)
            monthly_data["precipitation_lag6"] = monthly_data["precipitation_sum"].shift(6)
            monthly_data["precipitation_rolling_mean_3m"] = monthly_data["precipitation_sum"].rolling(window=3).mean()
            monthly_data["precipitation_rolling_mean_6m"] = monthly_data["precipitation_sum"].rolling(window=6).mean()
        
        return monthly_data

    def collect_historical_data(self, 
                                start_date: str = None, 
                                end_date: str = None) -> pd.DataFrame:
        """
        Collect and prepare historical weather data:
        1. Collect daily NOAA data
        2. Convert to monthly aggregates
        3. Get NCEI monthly precipitation
        4. Add NCEI precipitation as precipitation_sum column
        """
        if start_date is None:
            start_date = config.HISTORICAL_START_DATE
        if end_date is None:
            end_date = config.HISTORICAL_END_DATE

        logger.info(f"Collecting historical data from {start_date} to {end_date}")

        # Step 1: Collect daily data from NOAA
        if self.noaa_client is None:
            self.noaa_client = NOAAClient()

        logger.info("Fetching NOAA daily data...")
        noaa_daily = self.noaa_client.get_daily_data(start_date, end_date)

        if noaa_daily.empty:
            logger.warning("NOAA daily data is empty!")
            return pd.DataFrame()

        # Ensure we have a date column
        if "date" not in noaa_daily.columns:
            logger.warning("NOAA data missing 'date' column. Creating it from index...")
            noaa_daily["date"] = pd.to_datetime(noaa_daily.index)

        # Rename NOAA columns
        noaa_daily = noaa_daily.rename(columns={
            "PRCP": "precipitation",
            "TMAX": "temperature_max",
            "TMIN": "temperature_min",
            "TAVG": "temperature_avg",
            "AWND": "wind_speed",
        })

        # Save daily data for debugging
        debug_path = os.path.join(config.DATA_DIR, "_debug_noaa_daily.csv")
        noaa_daily.to_csv(debug_path, index=False)
        logger.info(f"Saved NOAA daily data to: {debug_path}")

        # Step 2: Convert daily data to monthly
        logger.info("Converting daily data to monthly aggregates...")
        monthly_data = self.convert_to_monthly(noaa_daily)
        
        if monthly_data.empty:
            logger.warning("Monthly conversion resulted in empty data!")
            return pd.DataFrame()
            
        # Step 3: Get NCEI monthly precipitation data
        logger.info("Fetching NCEI monthly precipitation data...")
        ncei_data = self.ncei_client.get_monthly_precipitation(start_date, end_date)
        
        if not ncei_data.empty:
            # Process NCEI data
            ncei_data = (
                ncei_data
                .rename(columns={"TimeStamp": "date", "PRECIPITATION": "precipitation_sum"})
                .assign(date=lambda df: pd.to_datetime(df["date"]))
                .sort_values("date")
            )
            
            # Step 4: Merge NCEI precipitation data with monthly aggregates
            logger.info("Merging NCEI precipitation data with monthly aggregates...")
            # If we already have precipitation_sum from the monthly conversion, replace it with NCEI data
            if "precipitation_sum" in monthly_data.columns:
                monthly_data = monthly_data.drop(columns=["precipitation_sum"])
                
            # Merge the datasets
            combined_data = pd.merge(monthly_data, ncei_data[["date", "precipitation_sum"]], 
                                    on="date", how="left")
            
            # Fill any missing values in precipitation_sum with calculated values from monthly_data
            if "precipitation_sum" in combined_data.columns and combined_data["precipitation_sum"].isna().any():
                if "precipitation_sum" in monthly_data.columns:
                    mask = combined_data["precipitation_sum"].isna()
                    combined_data.loc[mask, "precipitation_sum"] = monthly_data.loc[mask, "precipitation_sum"]
        else:
            logger.warning("NCEI data is empty, using only NOAA monthly aggregates")
            combined_data = monthly_data
            # If no NCEI data, make sure we have precipitation_sum (renamed from precipitation if needed)
            if "precipitation_sum" not in combined_data.columns and "precipitation" in combined_data.columns:
                combined_data = combined_data.rename(columns={"precipitation": "precipitation_sum"})
        
        # Save the combined monthly data
        combined_path = os.path.join(config.DATA_DIR, "combined_monthly_data.csv")
        combined_data.to_csv(combined_path, index=False)
        logger.info(f"Saved combined monthly data to: {combined_path}")
        
        return combined_data

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
    
    def convert_to_monthly(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert daily weather data to monthly aggregated data.
        
        This is a convenience method that provides a simple interface to generate
        monthly aggregated data from daily data.
        
        Args:
            daily_data: DataFrame with daily weather data
            
        Returns:
            DataFrame with monthly aggregated data
        """
        logger.info("Converting daily data to monthly aggregated data")
        return self.generate_monthly_data(daily_data)

    def prepare_data_for_model(self, 
                             historical_data: pd.DataFrame, 
                             forecast_data: pd.DataFrame = None,
                             monthly: bool = False) -> pd.DataFrame:
        """
        Prepare combined data for the LSTM model.
        
        Args:
            historical_data: Historical weather data
            forecast_data: Forecast data (optional)
            monthly: Whether to convert data to monthly aggregates (default: False)
            
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
            data["date"] = pd.to_datetime(data["date"], errors="coerce")
            data = data.sort_values("date")
        else:
            logger.warning("'date' column not found in combined data. Cannot sort by date.")

        # Optionally convert to monthly data
        if monthly:
            logger.info("Converting to monthly data for model preparation")
            data = self.convert_to_monthly(data)
        else:
            # Only apply these to daily data
            data = self._fill_missing_values(data)
            data = self._create_features(data)
            
        data = self._select_features(data)
        
        return data
    
    
    def save_data(self, data: pd.DataFrame, filename: str | None = None) -> str:
        """
        Save a DataFrame to <project_root>/data/<filename>
        """
        if filename is None:
            filename = config.OUTPUT_FILE               # already absolute
        else:
            filename = os.path.join(
                config.DATA_DIR, os.path.basename(filename)
            )

        os.makedirs(config.DATA_DIR, exist_ok=True)
        data.to_csv(filename, index=False)
        logger.info(f"Data saved to {filename}")
        return filename
    
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
            featured_data["month_cos"] = np.cos(2 * np.pi * featured_data["month"] / 12)
            featured_data["month_sin"] = np.sin(2 * np.pi * featured_data["month"] / 12)
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
        # Make a copy to avoid modifying the original
        selected_data = data.copy()
        
        # Always keep date
        features_to_keep = ["date"]
        
        # Determine if we're working with monthly or daily data by checking for columns with aggregation suffixes
        is_monthly = any(col.endswith(('_mean', '_min', '_max', '_sum')) for col in selected_data.columns)
        
        # Target column - handle both daily and monthly formats
        if "precipitation" in selected_data.columns:
            features_to_keep.append("precipitation")
        elif "precipitation_sum" in selected_data.columns:
            features_to_keep.append("precipitation_sum")
        
        # Daily data features
        daily_features = [
            "temperature_max", "temperature_min",
            "humidity", "wind_speed", "pressure",
            "temperature_range",
            "precipitation_lag1", "precipitation_lag2", "precipitation_lag3", "precipitation_lag7",
            "precipitation_rolling_mean_7d", "precipitation_rolling_max_7d",
            "season", "month_cos", "month_sin",
        ]
        
        # Monthly data features
        monthly_features = [
            # Temperature features
            "temperature_max_mean", "temperature_max_min", "temperature_max_max",
            "temperature_min_mean", "temperature_min_min", "temperature_min_max",
            # Weather metrics
            "humidity_mean", "humidity_min", "humidity_max",
            "wind_speed_mean", "wind_speed_min", "wind_speed_max",
            "pressure_mean", "pressure_min", "pressure_max",
            # Derived temperature features
            "temperature_range_mean", "temperature_range_extreme",
            # Precipitation lag and rolling features
            "precipitation_lag1", "precipitation_lag3", "precipitation_lag6",
            "precipitation_rolling_mean_3m", "precipitation_rolling_mean_6m",
            # Seasonal features
            "season", "month_cos", "month_sin",
        ]
        
        # Select appropriate features based on data type
        features_to_check = monthly_features if is_monthly else daily_features
        
        # Add features that exist in the data
        for feat in features_to_check:
            if feat in selected_data.columns:
                features_to_keep.append(feat)
        
        # Return only selected features
        return selected_data[features_to_keep]
    
    
    def create_sequences(self, 
                         data: pd.DataFrame, 
                         sequence_length: int = None,
                         target_column: str = "precipitation_sum",
                         exclude_target_from_features: bool = False) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Transform dataframe into sequences for LSTM model, and normalize data.
        
        Args:
            data: DataFrame with a 'date' column and feature columns
            sequence_length: How many timesteps in each input sequence (default=config.SEQUENCE_LENGTH)
            target_column: Column to predict (default="precipitation_sum")
            exclude_target_from_features: If True, don't include target column in the feature set
        
        Returns:
            X, y, scalers - where X and y are numpy arrays, and scalers is a dict of scalers
        """
        if sequence_length is None:
            sequence_length = config.SEQUENCE_LENGTH
        
        # Ensure data is sorted by date
        data = data.copy().sort_values('date')
        
        # Get feature columns (all columns except date and optionally target)
        if exclude_target_from_features:
            feature_columns = [col for col in data.columns if col != 'date' and col != target_column]
            # Make sure we still have the target column in the data for prediction
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in the data")
            # Add the target column to the end for sequence creation
            all_columns = feature_columns + [target_column]
        else:
            # Keep old behavior for backward compatibility
            feature_columns = [col for col in data.columns if col != 'date']
            all_columns = feature_columns
            
        # Create normalizers for all columns
        X_norm, scalers = self._normalize_data(data[all_columns])
        
        sequences = []
        targets = []
        
        # For each possible sequence
        for i in range(len(data) - sequence_length):
            if exclude_target_from_features:
                # Only include feature columns in the sequence (exclude target)
                feature_indices = list(range(len(feature_columns)))
                seq = X_norm[i:i+sequence_length, feature_indices]
                
                # Get target from the separate target column at the end
                target_idx = len(all_columns) - 1
            else:
                # Original behavior: use all columns
                seq = X_norm[i:i+sequence_length, :]
                target_idx = all_columns.index(target_column)
                
            # Get target value (the next time step's target column)
            target = X_norm[i+sequence_length, target_idx]
            
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets), scalers

    
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

    def collect_ncei_data(self,
                          start_date: str,
                          end_date: str,
                          save_path: str = "data/ncei_weather_data.csv") -> pd.DataFrame:
        """
        Pull NCEI *monthly* precipitation AND enrich it with
        aggregated NOAA features, then save a single combined CSV.

        The resulting file has one row per month with comprehensive monthly aggregations:
        - Monthly sum for precipitation (as precipitation_sum)
        - Monthly mean/min/max for temperature, humidity, wind_speed, pressure
        - Temperature range calculations
        - Cyclical month encoding (sin/cos)
        - Seasonal and lag information
        """
        # Fetch & prepare the NCEI frame
        logger.info(f"Collecting NCEI precipitation {start_date} → {end_date}")
        # only fetch the monthly totals for our window (fast!)
        ncei = self.ncei_client.get_monthly_precipitation(start_date, end_date)
        if ncei.empty:
            logger.warning("No NCEI data returned.")
            return ncei

        ncei = (
            ncei
            .rename(columns={"TimeStamp": "date", "PRECIPITATION": "precipitation_sum"})
            .assign(date=lambda df: pd.to_datetime(df["date"]))
            .sort_values("date")
        )
        # restrict to your window
        ncei = ncei[(ncei["date"] >= start_date) & (ncei["date"] <= end_date)]

        # Load/cache NOAA daily so we don't hammer the API each run
        debug_daily = os.path.join(config.DATA_DIR, "_debug_noaa.csv")
        if os.path.exists(debug_daily):
            logger.info("🛠 Loading NOAA daily from cache (_debug_noaa.csv), skipping API")
            noaa_daily = pd.read_csv(debug_daily, parse_dates=["date"])
        else:
            logger.info("Collecting NOAA daily data for feature enrichment …")
            if self.noaa_client is None:
                self.noaa_client = NOAAClient()
            noaa_daily = self.noaa_client.get_daily_data(start_date, end_date)
            noaa_daily.to_csv(debug_daily, index=False)
            logger.info(f"  → cached raw NOAA daily to {debug_daily}")

        # Rename NOAA columns
        if "date" not in noaa_daily.columns:
            noaa_daily["date"] = pd.to_datetime(noaa_daily.index)
        noaa_daily = noaa_daily.rename(columns={
            "PRCP": "precipitation",
            "TMAX": "temperature_max",
            "TMIN": "temperature_min",
            "TAVG": "temperature_avg",
            "AWND": "wind_speed",
        })
        
        # Generate comprehensive monthly data from NOAA daily data
        noaa_monthly = self.generate_monthly_data(noaa_daily)
        
        # If we have precipitation in both datasets, prioritize NCEI's precipitation
        if "precipitation_sum" in noaa_monthly.columns:
            noaa_monthly = noaa_monthly.drop(columns=["precipitation_sum"])

        # Merge NCEI + NOAA monthlies
        merged = pd.merge(ncei, noaa_monthly, on="date", how="left")
        
        # Add any missing features if they weren't created in the monthly aggregation
        # Add month if missing
        if "month" not in merged.columns:
            merged["month"] = merged["date"].dt.month
        
        # Add season if missing
        if "season" not in merged.columns:
            merged["season"] = merged["month"].map(
                lambda m: 1 if m in [12,1,2]
                          else 2 if m in [3,4,5]
                          else 3 if m in [6,7,8]
                          else 4
            )
        
        # Add cyclical month encodings if missing
        if "month_cos" not in merged.columns:
            merged["month_cos"] = np.cos(2 * np.pi * merged["month"] / 12)
        if "month_sin" not in merged.columns:
            merged["month_sin"] = np.sin(2 * np.pi * merged["month"] / 12)

        # Add lag features if missing
        if "precipitation_sum" in merged.columns:
            for lag in (1, 2, 3, 6):
                lag_col = f"precipitation_lag{lag}"
                if lag_col not in merged.columns:
                    merged[lag_col] = merged["precipitation_sum"].shift(lag)

            # Add rolling statistics if missing
            if "precipitation_rolling_mean_3m" not in merged.columns:
                merged["precipitation_rolling_mean_3m"] = merged["precipitation_sum"].rolling(window=3).mean()
            if "precipitation_rolling_mean_6m" not in merged.columns:
                merged["precipitation_rolling_mean_6m"] = merged["precipitation_sum"].rolling(window=6).mean()

        # Drop any rows with NaNs from the shifts
        merged = merged.dropna(subset=["precipitation_sum"]).reset_index(drop=True)

        # Save & return
        save_path = os.path.join(config.DATA_DIR, os.path.basename(save_path))
        os.makedirs(config.DATA_DIR, exist_ok=True)
        merged.to_csv(save_path, index=False)
        logger.info(f"Combined NCEI + NOAA monthly data saved → {save_path}")
        logger.info("Monthly dataset contains: wind_speed (mph), temperature_max/min (°F), and precipitation (inches)")

        return merged

    def collect_ncei_daily_data(self,
                            start_date: str,
                            end_date: str,
                            save_path: str = "data/ncei_daily_weather_data.csv") -> None:
        """
        Collect daily weather data from NCEI for the specified date range and save to file.
        Also aggregates the daily data to monthly (wind_speed and temperature_max) and
        adds monthly precipitation data from NCEI historical data.
        
        For large date ranges, this method breaks the request into smaller chunks
        that the NCEI API can handle (typically 1 year at a time).
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            save_path: Path to save the monthly data (default: "data/ncei_daily_weather_data.csv")
        """
        # Convert start_date and end_date to datetime objects for consistent handling
        if isinstance(start_date, str):
            start_dt = pd.to_datetime(start_date)
        else:
            start_dt = start_date
            
        if isinstance(end_date, str):
            end_dt = pd.to_datetime(end_date)
        else:
            end_dt = end_date
        
        # Ensure end date isn't in the future
        current_date = pd.to_datetime(datetime.now().date())
        if end_dt > current_date:
            logger.warning(f"End date {end_dt.strftime('%Y-%m-%d')} is in the future. Restricting to current date.")
            end_dt = current_date
            
        # Format dates for precise filtering
        start_year_month = start_dt.strftime('%Y-%m')
        end_year_month = end_dt.strftime('%Y-%m')
        
        logger.info(f"Collecting NCEI daily data from {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
        
        # Calculate the total time span in days
        time_span_days = (end_dt - start_dt).days
        logger.info(f"Total time span: {time_span_days} days")
        
        # For large time ranges, we need to split the requests into smaller chunks
        # NCEI API typically handles shorter periods better
        all_daily_data = []
        
        # Define smaller chunk size (120 days instead of 365)
        CHUNK_SIZE_DAYS = 120
        
        # Add retry logic for API requests
        MAX_RETRIES = 3
        RETRY_DELAY = 5  # seconds
        
        if time_span_days > CHUNK_SIZE_DAYS:
            logger.info(f"Large time range detected. Breaking into chunks of {CHUNK_SIZE_DAYS} days")
            
            # Create chunks of dates
            current_start = start_dt
            while current_start <= end_dt:
                # Calculate end of this chunk (or use final end_dt if it's closer)
                current_end = min(current_start + pd.Timedelta(days=CHUNK_SIZE_DAYS-1), end_dt)
                
                logger.info(f"Fetching chunk: {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}")
                
                # Implement retry logic with exponential backoff
                chunk_data = pd.DataFrame()
                success = False
                
                for retry in range(MAX_RETRIES):
                    try:
                        # Get data for this chunk with increased timeout
                        chunk_data = self.ncei_client.get_daily_data(
                            current_start.strftime('%Y-%m-%d'),
                            current_end.strftime('%Y-%m-%d'),
                            timeout=60  # Increased timeout to 60 seconds
                        )
                        
                        if not chunk_data.empty:
                            # Standardize column names
                            chunk_data = chunk_data.rename(columns={
                                "DATE": "date",
                                "PRCP": "precipitation",
                                "TMAX": "temperature_max",
                                "TMIN": "temperature_min",
                                "TAVG": "temperature_avg",
                                "AWND": "wind_speed"
                            })
                            
                            # Success, break out of retry loop
                            success = True
                            break
                        else:
                            logger.warning(f"Empty data returned for chunk {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}, retry {retry+1}/{MAX_RETRIES}")
                    
                    except Exception as e:
                        logger.error(f"Error fetching chunk {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}, retry {retry+1}/{MAX_RETRIES}: {str(e)}")
                        
                        # If this isn't the last retry, wait before trying again with exponential backoff
                        if retry < MAX_RETRIES - 1:
                            backoff_time = RETRY_DELAY * (2 ** retry)
                            logger.info(f"Waiting {backoff_time} seconds before retrying...")
                            import time
                            time.sleep(backoff_time)
                
                if success and not chunk_data.empty:
                    # Add this chunk to our collection
                    all_daily_data.append(chunk_data)
                    logger.info(f"Chunk data retrieved: {len(chunk_data)} records")
                else:
                    logger.warning(f"Failed to retrieve data for chunk {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')} after {MAX_RETRIES} retries")
                
                # Add a small delay between chunks to avoid overwhelming the API
                import time
                time.sleep(1)
                
                # Move to next chunk
                current_start = current_end + pd.Timedelta(days=1)
            
            # Combine all chunks
            if all_daily_data:
                daily_data = pd.concat(all_daily_data, ignore_index=True)
                logger.info(f"Combined data from all chunks: {len(daily_data)} records")
            else:
                logger.error("Failed to retrieve any data from any chunk")
                # Return without saving anything
                return
        else:
            # For smaller time ranges, use a single request
            try:
                daily_data = self.ncei_client.get_daily_data(
                    start_dt.strftime('%Y-%m-%d'),
                    end_dt.strftime('%Y-%m-%d')
                )
                
                if daily_data.empty:
                    logger.warning("No NCEI daily data returned.")
                    # Return without saving anything
                    return
                    
                # Process the data: standardize column names
                daily_data = daily_data.rename(columns={
                    "DATE": "date",
                    "PRCP": "precipitation",
                    "TMAX": "temperature_max",
                    "TMIN": "temperature_min",
                    "TAVG": "temperature_avg",
                    "AWND": "wind_speed"
                })
            except Exception as e:
                logger.error(f"Error fetching daily data: {str(e)}")
                # Return without saving anything
                return
        
        # Ensure date column is datetime
        daily_data["date"] = pd.to_datetime(daily_data["date"])
        
        # Strict filtering to ensure only data within the date range is included
        daily_data = daily_data[(daily_data["date"] >= start_dt) & (daily_data["date"] <= end_dt)]
        logger.info(f"Filtered daily data to date range: {len(daily_data)} days")
        
        # Create additional features
        daily_data = self._create_features(daily_data)
        
        # Fill missing values
        daily_data = self._fill_missing_values(daily_data)
        
        # Save the daily data to a fixed path
        daily_save_path = os.path.join(config.DATA_DIR, "ncei_daily_weather_data_raw.csv")
        os.makedirs(config.DATA_DIR, exist_ok=True)
        daily_data.to_csv(daily_save_path, index=False)
        logger.info(f"NCEI daily data saved to {daily_save_path}")
        
        # Aggregate only wind_speed and temperature_max to monthly
        logger.info("Aggregating wind_speed and temperature_max to monthly data...")
        
        # Create a copy of daily data with only the columns we need
        monthly_data = daily_data.copy()
        
        # Set date as index for resampling
        monthly_data = monthly_data.set_index("date")
        
        # Define the columns to aggregate and their aggregation methods
        agg_dict = {
            "wind_speed": ["mean", "min", "max"],
            "temperature_max": ["mean", "min", "max"],
            "temperature_min": ["mean", "min", "max"]  # Added temperature_min
        }
        
        # Perform monthly aggregation
        monthly_agg = monthly_data.resample("MS").agg(agg_dict)
        
        # Flatten multi-index columns
        monthly_agg.columns = ['_'.join(col).strip() for col in monthly_agg.columns.values]
        
        # Reset index to get date as column
        monthly_agg = monthly_agg.reset_index()
        
        # Add cyclical month encoding
        monthly_agg["month_cos"] = np.cos(2 * np.pi * monthly_agg["date"].dt.month / 12)
        monthly_agg["month_sin"] = np.sin(2 * np.pi * monthly_agg["date"].dt.month / 12)
        
        # Get monthly precipitation data from NCEI client
        logger.info("Calculating monthly precipitation data from daily data...")
        
        # Create TimeStamp column in the format expected by NCEIClient
        monthly_agg['TimeStamp'] = monthly_agg['date'].dt.strftime('%Y-%m')
        
        start_time = datetime.now()
        
        # Instead of querying NCEI again, calculate monthly precipitation from daily data
        if 'precipitation' in daily_data.columns:
            logger.info("Computing monthly precipitation by aggregating daily data")
            
            # Convert daily data's date to month format for aggregation
            daily_data_copy = daily_data.copy()
            daily_data_copy['TimeStamp'] = daily_data_copy['date'].dt.strftime('%Y-%m')
            
            # Group by month and sum precipitation
            monthly_precip = daily_data_copy.groupby('TimeStamp')['precipitation'].sum().reset_index()
            monthly_precip['precipitation'] = monthly_precip['precipitation'] * 10
            monthly_precip = monthly_precip.rename(columns={'precipitation': 'PRECIPITATION'})
            
            logger.info(f"Calculated monthly precipitation for {len(monthly_precip)} months from daily data (values in inches)")
            
            # Merge with monthly aggregates
            monthly_agg = pd.merge(
                monthly_agg,
                monthly_precip,
                on='TimeStamp',
                how='left'
            )
            
            # Check for missing values
            missing_count = monthly_agg['PRECIPITATION'].isna().sum()
            if missing_count > 0:
                logger.warning(f"Missing precipitation values for {missing_count} months after aggregation")
                
                # Use forward/backward fill for any missing months
                logger.info("Using forward fill for missing values...")
                monthly_agg['PRECIPITATION'] = monthly_agg['PRECIPITATION'].fillna(method='ffill')
                
                remaining_missing = monthly_agg['PRECIPITATION'].isna().sum()
                if remaining_missing > 0:
                    logger.info(f"Using backward fill for remaining {remaining_missing} values...")
                    monthly_agg['PRECIPITATION'] = monthly_agg['PRECIPITATION'].fillna(method='bfill')
                
                # If there are still missing values, use synthetic data
                final_missing = monthly_agg['PRECIPITATION'].isna().sum()
                if final_missing > 0:
                    logger.warning(f"Using synthetic data for remaining {final_missing} values")
                    nan_indices = monthly_agg['PRECIPITATION'].isna()
                    
                    # Generate seasonal synthetic data only for months in our range
                    for idx in monthly_agg[nan_indices].index:
                        month = monthly_agg.loc[idx, 'date'].month
                        
                        # Create seasonal pattern with some randomness
                        if 3 <= month <= 8:  # Spring and Summer (March to August)
                            precip = np.random.uniform(0.01, 0.025)  # Higher precipitation
                        else:  # Fall and Winter
                            precip = np.random.uniform(0.005, 0.015)  # Lower precipitation
                        
                        monthly_agg.loc[idx, 'PRECIPITATION'] = precip
            
            logger.info(f"Successfully combined monthly precipitation data")
        else:
            logger.warning("No precipitation data in daily records, generating synthetic data")
            
            # Generate synthetic monthly precipitation data only for our date range
            monthly_agg['PRECIPITATION'] = monthly_agg.apply(
                lambda row: np.random.uniform(0.01, 0.025) if 3 <= row['date'].month <= 8 
                           else np.random.uniform(0.005, 0.015),
                axis=1
            )
            logger.info(f"Generated synthetic precipitation data for {len(monthly_agg)} months")
        
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        logger.info(f"Processed precipitation data in {elapsed:.1f} seconds")
        
        # Save the monthly data to the specified save_path
        monthly_save_path = os.path.join(config.DATA_DIR, os.path.basename(save_path))
        monthly_agg.to_csv(monthly_save_path, index=False)
        logger.info(f"Monthly aggregated data saved to {monthly_save_path}")
        logger.info("Monthly dataset contains: wind_speed (mph), temperature_max/min (°F), and precipitation (inches)")


if __name__ == "__main__":
    processor = DataProcessor()
    
    # Test the new NCEI daily data collection
    print("\n=== Testing NCEI Daily Data Collection ===")
    start_date = "2020-01-01"
    end_date = "2025-02-27"
    processor.collect_ncei_daily_data(start_date, end_date, "data/monthly_weather_data.csv")
    

