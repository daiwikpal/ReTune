import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import math

from Clients.vtec_client import VTECClient
from Clients.ncei_client import NCEIClient

class AlertDataProcessor:
    """
    Process alert data from VTEC API and prepare it for model training.
    """
    
    # Mapping between phenomena codes and column names in processed_weather_data.csv
    PHENOMENA_MAPPING = {
    'AV': 'AVALANCHE',
    'BZ': 'BLIZZARD',
    'CF': 'COASTAL_FLOOD',
    'WC': 'COLD/WIND_CHILL',
    'DF': 'DEBRIS_FLOW',
    'FG': 'DENSE_FOG',
    'SM': 'DENSE_SMOKE',
    'DR': 'DROUGHT',
    'DU': 'DUST_STORM',
    'DS': 'DUST_DEVIL',
    'EH': 'EXCESSIVE_HEAT',
    'AS': 'ASTRONOMICAL_LOW_TIDE',
    'EC': 'EXTREME_COLD/WIND_CHILL',
    'FF': 'FLASH_FLOOD',
    'FL': 'FLOOD',
    'ZF': 'FREEZING_FOG',
    'FR': 'FROST/FREEZE',
    'FC': 'FUNNEL_CLOUD',
    'SV': 'HAIL',  # includes hail, but technically "Severe Thunderstorm"
    'HT': 'HEAT',
    'HR': 'HEAVY_RAIN',
    'SN': 'HEAVY_SNOW',
    'SU': 'HIGH_SURF',
    'HW': 'HIGH_WIND',
    'HU': 'HURRICANE_(TYPHOON)',
    'IS': 'ICE_STORM',
    'LE': 'LAKE-EFFECT_SNOW',
    'LS': 'LAKESHORE_FLOOD',
    'LT': 'LIGHTNING',
    'MH': 'MARINE_HAIL',
    'MW': 'MARINE_HIGH_WIND',
    'MS': 'MARINE_STRONG_WIND',
    'MA': 'MARINE_THUNDERSTORM_WIND',
    'RP': 'RIP_CURRENT',
    'SQ': 'SEICHE',
    'SL': 'SLEET',
    'SV': 'THUNDERSTORM_WIND',
    'TO': 'TORNADO',
    'TD': 'TROPICAL_DEPRESSION',
    'TR': 'TROPICAL_STORM',
    'TS': 'TSUNAMI',
    'VA': 'VOLCANIC_ASH',
    'WF': 'WILDFIRE',
    'WS': 'WINTER_STORM',
    'WW': 'WINTER_WEATHER'
}

    
    def __init__(self, existing_data_path: str):
        """
        Initialize with path to existing processed weather data.
        
        Parameters:
        -----------
        existing_data_path : str
            Path to the existing processed_weather_data.csv file
        """
        self.vtec_client = VTECClient()
        self.ncei_client = NCEIClient()
        self.existing_data = pd.read_csv(existing_data_path)
        
        # Extract the columns we need to populate when processing VTEC data
        self.event_columns = [col for col in self.existing_data.columns 
                             if col not in ('TimeStamp', 'PRECIPITATION', 
                                           'precip_12m_mean', 'precip_12m_std', 
                                           'precip_12m_z', 'season_sin', 'season_cos')]
    
    def fetch_alerts(self, begints: str, endts: str, wfos: List[str]) -> Dict[str, Any]:
        """
        Fetch alerts from the VTEC API.
        
        Parameters:
        -----------
        begints : str
            Start date in YYYY-MM-DD format
        endts : str
            End date in YYYY-MM-DD format
        wfos : List[str]
            List of Weather Forecast Office codes
            
        Returns:
        --------
        Dict[str, Any]
            Raw JSON response from the VTEC API
        """
        return self.vtec_client.get_alerts(begints=begints, endts=endts, wfos=wfos)
    
    def fetch_precipitation(self, begints: str, endts: str) -> pd.DataFrame:
        """
        Fetch precipitation data from the NCEI API.
        
        Parameters:
        -----------
        begints : str
            Start date in YYYY-MM-DD format
        endts : str
            End date in YYYY-MM-DD format
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with monthly precipitation data
        """
        return self.ncei_client.get_monthly_precipitation(begints, endts)
    
    def process_alerts(self, alerts_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Process raw alert data into a format compatible with processed_weather_data.csv.
        
        Parameters:
        -----------
        alerts_data : Dict[str, Any]
            Raw JSON response from the VTEC API
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with processed alert data
        """
        if not alerts_data or 'data' not in alerts_data:
            return pd.DataFrame()
        
        # Convert alerts data to DataFrame
        alerts_df = pd.DataFrame(alerts_data['data'])
        
        # Extract month from utc_issue date
        alerts_df['month'] = pd.to_datetime(alerts_df['utc_issue']).dt.strftime('%Y-%m')
        
        # Count occurrences of each phenomena by month
        monthly_counts = {}
        
        for _, row in alerts_df.iterrows():
            month = row['month']
            phenomena = row['phenomena']
            
            if month not in monthly_counts:
                monthly_counts[month] = {col: 0 for col in self.event_columns}
            
            # Map phenomena to the appropriate column
            if phenomena in self.PHENOMENA_MAPPING:
                column = self.PHENOMENA_MAPPING[phenomena]
                monthly_counts[month][column] += 1
        
        # Convert to DataFrame
        result_df = pd.DataFrame.from_dict(monthly_counts, orient='index')
        result_df.index.name = 'TimeStamp'
        result_df.reset_index(inplace=True)
        
        # Fill NaN values with 0
        for col in self.event_columns:
            if col not in result_df.columns:
                result_df[col] = 0
            else:
                result_df[col] = result_df[col].fillna(0)
        
        return result_df
    
    def add_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add seasonal sine and cosine features based on month.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with TimeStamp column
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with added seasonal features
        """
        # Parse month from TimeStamp
        df['month'] = pd.to_datetime(df['TimeStamp']).dt.month
        
        # Convert month to seasonal features
        df['season_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
        df['season_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
        
        # Drop temporary month column
        df.drop('month', axis=1, inplace=True)
        
        return df
    
    def merge_with_existing_data(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Merge new alert data with existing processed data.
        
        Parameters:
        -----------
        new_data : pd.DataFrame
            New processed alert data
            
        Returns:
        --------
        pd.DataFrame
            Merged DataFrame ready for model use
        """
        # Make a copy of existing data
        combined_df = self.existing_data.copy()
        
        # Identify new months that aren't in the existing data
        existing_months = set(combined_df['TimeStamp'])
        new_months = [m for m in new_data['TimeStamp'] if m not in existing_months]
        
        if not new_months:
            return combined_df
        
        # Filter new data to only include new months
        new_data_filtered = new_data[new_data['TimeStamp'].isin(new_months)].copy()
        
        # Add appropriate placeholder values for columns not in new_data
        for col in combined_df.columns:
            if col not in new_data_filtered.columns and col != 'PRECIPITATION':
                new_data_filtered[col] = 0
        
        # Add placeholder for PRECIPITATION if it's not already in the DataFrame
        if 'PRECIPITATION' not in new_data_filtered.columns:
            # Try to fetch precipitation data for new months
            self.add_precipitation_data(new_data_filtered)
        
        # Ensure all columns from original data are present
        for col in combined_df.columns:
            if col not in new_data_filtered.columns:
                new_data_filtered[col] = np.nan
        
        # Make sure all required columns are present with the same order
        new_data_filtered = new_data_filtered[combined_df.columns]
        
        # Fill remaining NaN values with appropriate defaults
        # For event columns, use 0
        for col in self.event_columns:
            if col in new_data_filtered.columns:
                new_data_filtered[col] = new_data_filtered[col].fillna(0)
        
        # Append new data
        combined_df = pd.concat([combined_df, new_data_filtered], ignore_index=True)
        
        # Sort by timestamp
        combined_df.sort_values('TimeStamp', inplace=True)
        
        return combined_df
    
    def add_precipitation_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add precipitation data to new months.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with TimeStamp column
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with added precipitation data
        """
        # Extract unique year-month values from TimeStamp
        df['TimeStamp'] = df['TimeStamp'].astype(str)
        months_to_fetch = df['TimeStamp'].unique()
        
        if not months_to_fetch.size:
            return df
        
        # Convert to start and end dates for the NCEI API
        earliest_month = min(months_to_fetch)
        latest_month = max(months_to_fetch)
        
        # Add day component to create dates
        start_date = f"{earliest_month}-01"
        # Calculate the last day of the latest month
        year, month = map(int, latest_month.split('-'))
        if month == 12:
            next_month_year = year + 1
            next_month = 1
        else:
            next_month_year = year
            next_month = month + 1
        
        end_date = f"{next_month_year}-{next_month:02d}-01"
        
        try:
            # Fetch precipitation data
            precip_df = self.fetch_precipitation(start_date, end_date)
            
            # If we got data, merge it with our DataFrame
            if not precip_df.empty:
                # Create a mapping from TimeStamp to PRECIPITATION
                precip_map = dict(zip(precip_df['TimeStamp'], precip_df['PRECIPITATION']))
                
                # Add precipitation values to our DataFrame
                df['PRECIPITATION'] = df['TimeStamp'].map(precip_map)
            else:
                print(f"No precipitation data found for {start_date} to {end_date}")
                # Use historical average as fallback for missing precipitation data
                if not self.existing_data.empty and 'PRECIPITATION' in self.existing_data.columns:
                    historical_avg = self.existing_data['PRECIPITATION'].mean()
                    print(f"Using historical average precipitation: {historical_avg:.2f}")
                    df['PRECIPITATION'] = historical_avg
                else:
                    # If no historical data available, use a reasonable default for NYC
                    print("Using default precipitation value")
                    df['PRECIPITATION'] = 4.0  # Average monthly precipitation for NYC is ~4 inches
        except Exception as e:
            print(f"Error fetching precipitation data: {e}")
            # Use historical average as fallback
            if not self.existing_data.empty and 'PRECIPITATION' in self.existing_data.columns:
                historical_avg = self.existing_data['PRECIPITATION'].mean()
                print(f"Using historical average precipitation: {historical_avg:.2f}")
                df['PRECIPITATION'] = historical_avg
            else:
                # Default value
                df['PRECIPITATION'] = 4.0
            
        return df
    
    def calculate_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate rolling statistical features for precipitation.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with PRECIPITATION column
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with calculated statistical features
        """
        # Ensure the DataFrame is sorted by TimeStamp
        df['date_for_sort'] = pd.to_datetime(df['TimeStamp'])
        df.sort_values('date_for_sort', inplace=True)
        df.drop('date_for_sort', axis=1, inplace=True)
        
        # Fill any remaining NaN values in PRECIPITATION column
        if df['PRECIPITATION'].isna().any():
            # Use historical average for missing values
            historical_avg = df['PRECIPITATION'].mean()
            df['PRECIPITATION'] = df['PRECIPITATION'].fillna(historical_avg)
        
        # Calculate 12-month rolling mean and std
        df['precip_12m_mean'] = df['PRECIPITATION'].rolling(window=12, min_periods=1).mean()
        df['precip_12m_std'] = df['PRECIPITATION'].rolling(window=12, min_periods=1).std()
        
        # Calculate z-score using the rolling statistics
        # Handle division by zero by replacing 0 std with 1
        df['precip_12m_z'] = (df['PRECIPITATION'] - df['precip_12m_mean']) / df['precip_12m_std'].replace(0, 1)
        
        # Ensure no NaN values remain in any of the columns
        for col in df.columns:
            if df[col].isna().any():
                if col in ['precip_12m_mean', 'PRECIPITATION']:
                    # For precipitation-related columns, use mean
                    df[col] = df[col].fillna(df[col].mean())
                elif col in ['precip_12m_std']:
                    # For std columns, use mean of non-zero values or 1
                    mean_std = df[col][df[col] > 0].mean()
                    df[col] = df[col].fillna(mean_std if not pd.isna(mean_std) else 1.0)
                elif col in ['precip_12m_z']:
                    # For z-scores, use 0 (meaning "average")
                    df[col] = df[col].fillna(0)
                elif col in ['season_sin', 'season_cos']:
                    # For seasonal features, recalculate if needed
                    if 'season_sin' in df.columns and df['season_sin'].isna().any():
                        month_num = pd.to_datetime(df['TimeStamp']).dt.month
                        df.loc[df['season_sin'].isna(), 'season_sin'] = np.sin(2 * np.pi * month_num / 12.0)
                    if 'season_cos' in df.columns and df['season_cos'].isna().any():
                        month_num = pd.to_datetime(df['TimeStamp']).dt.month
                        df.loc[df['season_cos'].isna(), 'season_cos'] = np.cos(2 * np.pi * month_num / 12.0)
                else:
                    # For other columns (event columns), use 0
                    df[col] = df[col].fillna(0)
        
        return df
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Save processed data to CSV.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Processed DataFrame
        output_path : str
            Path to save the CSV file
        """
        df.to_csv(output_path, index=False)
        
    def process_and_merge_alerts(self, 
                               begints: str, 
                               endts: str, 
                               wfos: List[str],
                               output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Main method to fetch, process, and merge alert data with existing data.
        
        Parameters:
        -----------
        begints : str
            Start date in YYYY-MM-DD format
        endts : str
            End date in YYYY-MM-DD format
        wfos : List[str]
            List of Weather Forecast Office codes
        output_path : str, optional
            Path to save the processed data
            
        Returns:
        --------
        pd.DataFrame
            Processed and merged DataFrame
        """
        # Fetch alerts
        alerts_data = self.fetch_alerts(begints, endts, wfos)
        
        # Process alerts
        processed_df = self.process_alerts(alerts_data)
        
        # Generate monthly records for each month in the date range if none found
        if processed_df.empty:
            start_date = datetime.strptime(begints, "%Y-%m-%d")
            end_date = datetime.strptime(endts, "%Y-%m-%d")
            
            # Create a range of months between start_date and end_date
            months = []
            current_date = start_date.replace(day=1)
            while current_date <= end_date:
                months.append(current_date.strftime("%Y-%m"))
                if current_date.month == 12:
                    current_date = current_date.replace(year=current_date.year + 1, month=1)
                else:
                    current_date = current_date.replace(month=current_date.month + 1)
            
            # Create empty records for each month
            empty_records = []
            for month in months:
                record = {'TimeStamp': month}
                for col in self.event_columns:
                    record[col] = 0
                empty_records.append(record)
            
            if empty_records:
                processed_df = pd.DataFrame(empty_records)
        
        # Add seasonal features
        processed_df = self.add_seasonal_features(processed_df)
        
        # Merge with existing data
        merged_df = self.merge_with_existing_data(processed_df)
        
        # Calculate statistical features 
        merged_df = self.calculate_statistical_features(merged_df)
        
        # Verify there are no missing values
        missing_values = merged_df.isna().sum().sum()
        if missing_values > 0:
            print(f"Warning: There are still {missing_values} missing values in the data")
            print("Columns with missing values:")
            for col in merged_df.columns:
                if merged_df[col].isna().any():
                    print(f"  {col}: {merged_df[col].isna().sum()} missing values")
            
            # Final fallback - replace any remaining NaNs
            merged_df = merged_df.fillna(0)
        
        # Save if output path is provided
        if output_path:
            self.save_processed_data(merged_df, output_path)
        
        return merged_df 