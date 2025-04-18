import pandas as pd
import numpy as np
from datetime import datetime
import os
import glob
from Clients.noaa_client import NOAAClient
from Clients.ncei_client import NCEIClient
import random
import config
import logging

# Set up logging
logger = logging.getLogger(__name__)

def configure_logging(log_level=logging.INFO):
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
    
    # Also log to a file
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"data_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    fh = logging.FileHandler(log_file)
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    logger.info(f"Logging configured. Log file: {log_file}")
    return log_file

def get_data_dir():
    """Helper function to handle different working directory scenarios"""
    # If we're in the Anomaly Model directory
    if os.path.isdir(config.HISTORICAL_DATA_DIR):
        return f'{config.HISTORICAL_DATA_DIR}/*.csv'
    # If we're in the root directory
    elif os.path.isdir(f'Anomaly Model/{config.HISTORICAL_DATA_DIR}'):
        return f'Anomaly Model/{config.HISTORICAL_DATA_DIR}/*.csv'
    else:
        logger.warning(f"WARNING: Could not find {config.HISTORICAL_DATA_DIR} directory")
        return None

def get_output_path(filename):
    """Helper function to get the correct path for output files"""
    # If we're in the Anomaly Model directory
    if os.path.isdir(config.HISTORICAL_DATA_DIR):
        return filename
    # If we're in the root directory
    elif os.path.isdir(f'Anomaly Model/{config.HISTORICAL_DATA_DIR}'):
        return os.path.join('Anomaly Model', filename)
    else:
        logger.warning(f"WARNING: Could not find {config.HISTORICAL_DATA_DIR} directory")
        return filename

def process_weather_data(include_precipitation: bool = True, use_synthetic_if_api_fails: bool = True, log_level=logging.INFO):
    """
    Process historical weather data and optionally add precipitation data
    
    Args:
        include_precipitation: Whether to include precipitation data
        use_synthetic_if_api_fails: Whether to generate synthetic precipitation data if API fails
        log_level: Logging level
        
    Returns:
        DataFrame with processed weather data
    """
    # Configure logging
    log_file = configure_logging(log_level)
    logger.info("Starting weather data processing")
    logger.info(f"Parameters: include_precipitation={include_precipitation}, use_synthetic_if_api_fails={use_synthetic_if_api_fails}")
    
    # Get all CSV files in the historical data directory
    data_path = get_data_dir()
    if not data_path:
        logger.error("Could not find Historical Data directory")
        logger.error(f"Current working directory: {os.getcwd()}")
        return pd.DataFrame()
        
    csv_files = glob.glob(data_path)
    logger.info(f"Found {len(csv_files)} CSV files in {data_path}")
    
    if not csv_files:
        logger.error(f"No CSV files found in {data_path}")
        logger.error(f"Current working directory: {os.getcwd()}")
        return pd.DataFrame()
    
    processed_dfs = []
    file_count = 0
    total_files = len(csv_files)
    
    for csv_file in sorted(csv_files):
        file_count += 1
        logger.info(f"Processing file {file_count}/{total_files}: {os.path.basename(csv_file)}")
        
        try:
            df = pd.read_csv(csv_file)
            logger.info(f"  Read {len(df)} rows from {os.path.basename(csv_file)}")
            
            # Check if required columns exist
            required_columns = ['STATE', 'BEGIN_YEARMONTH', 'BEGIN_DAY', 'BEGIN_TIME', 'EVENT_TYPE', 'EVENT_ID']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"  Missing required columns in {os.path.basename(csv_file)}: {missing_columns}")
                continue
            
            target_states = ['NEW YORK', 'CONNECTICUT', 'PENNSYLVANIA', 'NEW JERSEY']
            df = df[df['STATE'].isin(target_states)]
            logger.debug(f"  After filtering states: {len(df)} rows")
            
            if len(df) == 0:
                logger.warning(f"  No data for target states in {os.path.basename(csv_file)}")
                continue
            
            df['BEGIN_DATE'] = pd.to_datetime(
                df['BEGIN_YEARMONTH'].astype(str) + 
                df['BEGIN_DAY'].astype(str).str.zfill(2) + 
                df['BEGIN_TIME'].astype(str).str.zfill(4),
                format='%Y%m%d%H%M'
            )
            
            df['TimeStamp'] = df['BEGIN_DATE'].dt.to_period('M')
            
            df['EVENT_TYPE'] = df['EVENT_TYPE'].str.upper().str.replace(' ', '_')
            
            pivot_df = pd.pivot_table(
                df,
                values='EVENT_ID',
                index='TimeStamp',
                columns='EVENT_TYPE',
                aggfunc='count',
                fill_value=0
            )
            
            logger.debug(f"  Created pivot table with shape: {pivot_df.shape}")
            processed_dfs.append(pivot_df)
            
        except Exception as e:
            logger.error(f"Error processing {os.path.basename(csv_file)}: {str(e)}", exc_info=True)
            continue
    
    if not processed_dfs:
        logger.error("No data was successfully processed. Check file paths and data format.")
        return pd.DataFrame()
    
    logger.info(f"Concatenating {len(processed_dfs)} DataFrames...")
    combined_df = pd.concat(processed_dfs)
    combined_df = combined_df.groupby(level=0).sum()
    
    anomaly_types = [
        'AVALANCHE', 'BLIZZARD', 'COASTAL_FLOOD', 'COLD/WIND_CHILL',
        'DEBRIS_FLOW', 'DENSE_FOG', 'DENSE_SMOKE', 'DROUGHT',
        'DUST_DEVIL', 'DUST_STORM', 'EXCESSIVE_HEAT', 'ASTRONOMICAL_LOW_TIDE',
        'EXTREME_COLD/WIND_CHILL', 'FLASH_FLOOD', 'FLOOD', 'FREEZING_FOG',
        'FROST/FREEZE', 'FUNNEL_CLOUD', 'HAIL', 'HEAT', 'HEAVY_RAIN',
        'HEAVY_SNOW', 'HIGH_SURF', 'HIGH_WIND', 'HURRICANE_(TYPHOON)',
        'ICE_STORM', 'LAKE-EFFECT_SNOW', 'LAKESHORE_FLOOD', 'LIGHTNING',
        'MARINE_HAIL', 'MARINE_HIGH_WIND', 'MARINE_STRONG_WIND',
        'MARINE_THUNDERSTORM_WIND', 'RIP_CURRENT', 'SEICHE', 'SLEET',
        'SNEAKERWAVE', 'STORM_SURGE/TIDE', 'STRONG_WIND',
        'THUNDERSTORM_WIND', 'TORNADO', 'TROPICAL_DEPRESSION',
        'TROPICAL_STORM', 'TSUNAMI', 'VOLCANIC_ASH', 'WATERSPOUT',
        'WILDFIRE', 'WINTER_STORM', 'WINTER_WEATHER'
    ]
    
    logger.info("Adding missing anomaly type columns...")
    missing_anomalies = []
    for anomaly in anomaly_types:
        if anomaly not in combined_df.columns:
            combined_df[anomaly] = 0
            missing_anomalies.append(anomaly)
    
    if missing_anomalies:
        logger.debug(f"Added {len(missing_anomalies)} missing anomaly columns: {', '.join(missing_anomalies)}")
    
    combined_df = combined_df.reset_index()
    
    combined_df['TimeStamp'] = combined_df['TimeStamp'].astype(str)
    
    # Convert TimeStamp to datetime for filtering
    combined_df['date'] = pd.to_datetime(combined_df['TimeStamp'], format='%Y-%m')
    
    # Filter data after January 1996
    min_date = pd.to_datetime('1996-01-01')
    initial_rows = len(combined_df)
    combined_df = combined_df[combined_df['date'] >= min_date]
    logger.info(f"Filtered to dates >= 1996-01-01: {len(combined_df)} rows (removed {initial_rows - len(combined_df)} rows)")
    
    # Remove the temporary date column
    combined_df = combined_df.drop(columns=['date'])
    
    # Get precipitation data
    if include_precipitation:
        got_real_data = False
        try:
            logger.info("Initializing NCEI client for precipitation data")
            ncei_client = NCEIClient(log_level=log_level)
            
            # Get all available historical precipitation data
            logger.info("Fetching real precipitation data from NCEI API...")
            start_time = datetime.now()
            precip_df = ncei_client.get_historical_monthly_precipitation(years=30)
            end_time = datetime.now()
            elapsed = (end_time - start_time).total_seconds()
            
            if not precip_df.empty:
                logger.info(f"Successfully retrieved precipitation data ({len(precip_df)} months) in {elapsed:.1f} seconds")
                
                logger.info("Merging precipitation data with weather events...")
                combined_df = pd.merge(
                    combined_df,
                    precip_df,
                    on='TimeStamp',
                    how='left'
                )
                
                logger.info(f"After merging: {len(combined_df)} rows")
                missing_count = combined_df['PRECIPITATION'].isna().sum()
                logger.info(f"Missing precipitation values: {missing_count}/{len(combined_df)} ({missing_count/len(combined_df)*100:.1f}%)")
                
                # If we still have NaN values after merging, handle them
                if combined_df['PRECIPITATION'].isna().any():
                    logger.warning(f"Filling {missing_count} missing precipitation values")
                    logger.info("Using forward fill method first...")
                    combined_df['PRECIPITATION'] = combined_df['PRECIPITATION'].fillna(method='ffill')
                    
                    remaining_missing = combined_df['PRECIPITATION'].isna().sum()
                    if remaining_missing > 0:
                        logger.info(f"Using backward fill for remaining {remaining_missing} values...")
                        combined_df['PRECIPITATION'] = combined_df['PRECIPITATION'].fillna(method='bfill')
                    
                    # If there are still NaNs, fill with random values in a reasonable range
                    final_missing = combined_df['PRECIPITATION'].isna().sum()
                    if final_missing > 0 and use_synthetic_if_api_fails:
                        logger.warning(f"Still have {final_missing} missing values, using synthetic data")
                        nan_indices = combined_df['PRECIPITATION'].isna()
                        combined_df.loc[nan_indices, 'PRECIPITATION'] = np.random.uniform(0.005, 0.02, size=nan_indices.sum())
                        logger.info(f"Filled {nan_indices.sum()} missing values with synthetic data")
                
                got_real_data = True
                logger.info("Successfully integrated precipitation data")
            else:
                logger.warning("NCEI API returned no precipitation data")
                
        except Exception as e:
            logger.error(f"Error fetching precipitation data: {str(e)}", exc_info=True)
        
        # If we couldn't get data from the API and synthetic data is enabled
        if not got_real_data and use_synthetic_if_api_fails:
            logger.warning("Generating synthetic precipitation data for visualization purposes...")
            # Get all unique TimeStamps
            timestamps = sorted(combined_df['TimeStamp'].unique())
            
            # Create seasonal patterns - higher precipitation in spring/summer
            synthetic_data = []
            for ts in timestamps:
                year_month = ts.split('-')
                if len(year_month) == 2:
                    year, month = year_month
                    month = int(month)
                    
                    # Create seasonal pattern with some randomness
                    if 3 <= month <= 8:  # Spring and Summer (March to August)
                        precip = np.random.uniform(0.01, 0.025)  # Higher precipitation
                    else:  # Fall and Winter
                        precip = np.random.uniform(0.005, 0.015)  # Lower precipitation
                    
                    synthetic_data.append({
                        'TimeStamp': ts,
                        'PRECIPITATION': precip
                    })
            
            if synthetic_data:
                precip_df = pd.DataFrame(synthetic_data)
                combined_df = pd.merge(
                    combined_df,
                    precip_df,
                    on='TimeStamp',
                    how='left'
                )
                logger.info(f"Added synthetic precipitation data for {len(synthetic_data)} months")
            else:
                combined_df['PRECIPITATION'] = np.nan
                logger.warning("Failed to create synthetic data")
    else:
        # Even if precipitation isn't included, create empty column
        logger.info("Precipitation data not requested, adding empty column")
        combined_df['PRECIPITATION'] = np.nan
    
    # Convert TimeStamp to datetime for calculations
    combined_df['date_temp'] = pd.to_datetime(combined_df['TimeStamp'], format='%Y-%m')
    
    # Calculate seasonal indicators
    logger.info("Adding seasonal indicators...")
    # Extract month from TimeStamp
    combined_df['month'] = combined_df['date_temp'].dt.month
    
    # Calculate seasonal sine and cosine
    combined_df['angle'] = 2 * np.pi * (combined_df['month'] / 12)
    combined_df['season_sin'] = np.sin(combined_df['angle'])
    combined_df['season_cos'] = np.cos(combined_df['angle'])
    
    # Sort by date for rolling calculations
    combined_df = combined_df.sort_values('date_temp')
    
    # Calculate 12-month rolling stats if precipitation data is available
    if 'PRECIPITATION' in combined_df.columns and not combined_df['PRECIPITATION'].isna().all():
        logger.info("Calculating 12-month rolling precipitation statistics...")
        
        # Calculate 12-month rolling mean and standard deviation
        combined_df['precip_12m_mean'] = combined_df['PRECIPITATION'].rolling(window=12).mean()
        combined_df['precip_12m_std'] = combined_df['PRECIPITATION'].rolling(window=12).std()
        
        # Calculate 12-month rolling z-score
        combined_df['precip_12m_z'] = (combined_df['PRECIPITATION'] - combined_df['precip_12m_mean']) / combined_df['precip_12m_std']
        
        # Log counts of new columns
        missing_counts = {
            'precip_12m_mean': combined_df['precip_12m_mean'].isna().sum(),
            'precip_12m_std': combined_df['precip_12m_std'].isna().sum(),
            'precip_12m_z': combined_df['precip_12m_z'].isna().sum()
        }
        
        logger.info(f"New columns added with {missing_counts['precip_12m_mean']} NaN values in rolling mean (first 11 months)")
        logger.info(f"Z-score has {missing_counts['precip_12m_z']} NaN values")
    else:
        # Add empty columns for consistency
        combined_df['precip_12m_mean'] = np.nan
        combined_df['precip_12m_std'] = np.nan
        combined_df['precip_12m_z'] = np.nan
        logger.warning("No precipitation data available, adding empty rolling statistics columns")
    
    # Remove temporary columns
    combined_df = combined_df.drop(columns=['date_temp', 'month', 'angle'])
    
    # Define column order with all new columns
    column_order = ['TimeStamp'] + anomaly_types + ['PRECIPITATION', 
                                                   'precip_12m_mean', 
                                                   'precip_12m_std', 
                                                   'precip_12m_z',
                                                   'season_sin',
                                                   'season_cos']
    
    # Only include columns that actually exist in the DataFrame
    column_order = [col for col in column_order if col in combined_df.columns]
    combined_df = combined_df[column_order]
    
    logger.info(f"Final dataset has {len(combined_df)} rows and {len(combined_df.columns)} columns")
    logger.info(f"Date range: {combined_df['TimeStamp'].min()} to {combined_df['TimeStamp'].max()}")
    
    # Log some precipitation statistics if available
    if 'PRECIPITATION' in combined_df.columns and not combined_df['PRECIPITATION'].isna().all():
        precip_stats = combined_df['PRECIPITATION'].describe()
        logger.info(f"Precipitation statistics (inches):")
        logger.info(f"  Min: {precip_stats['min']:.3f}")
        logger.info(f"  Max: {precip_stats['max']:.3f}")
        logger.info(f"  Mean: {precip_stats['mean']:.3f}")
        logger.info(f"  Median: {precip_stats['50%']:.3f}")
    
    logger.info(f"Processing complete. Log file: {log_file}")
    return combined_df

def count_ny_blizzards_2024():
    """Count the number of blizzards in New York in 2024"""
    # Configure logging
    configure_logging()
    logger.info("Counting NY blizzards for 2024")
    
    data_path = get_data_dir()
    if not data_path:
        logger.error("Could not find Historical Data directory")
        return 0
        
    # Get the 2024 file based on the pattern
    files_2024 = glob.glob(data_path.replace('*.csv', 'd2024_*.csv'))
    if not files_2024:
        logger.error("No 2024 data file found")
        return 0
        
    file_2024 = files_2024[0]
    logger.info(f"Reading 2024 data from {file_2024}")
    
    df = pd.read_csv(file_2024)
    
    ny_blizzards = df[
        (df['STATE'] == 'NEW YORK') & 
        (df['EVENT_TYPE'] == 'Blizzard') & 
        (df['YEAR'] == 2024)
    ]
    
    count = len(ny_blizzards)
    logger.info(f"Number of blizzards in New York in 2024: {count}")
    
    if not ny_blizzards.empty:
        logger.info("Blizzard details:")
        for _, row in ny_blizzards.iterrows():
            logger.info(f"Date: {row['BEGIN_YEARMONTH']}-{row['BEGIN_DAY']}")
            logger.info(f"Location: {row['CZ_NAME']}")
            logger.info(f"Event Narrative: {row['EVENT_NARRATIVE']}")
    
    return count

# Example usage:
if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("Starting weather data processing script")
    logger.info("=" * 80)
    
    # Try to get real precipitation data but use synthetic data if API fails
    processed_data = process_weather_data(include_precipitation=True, use_synthetic_if_api_fails=True)
    
    if not processed_data.empty:
        output_file = get_output_path(config.OUTPUT_FILE)
        processed_data.to_csv(output_file, index=False)
        logger.info(f"Processed data has been saved to {output_file}")
        logger.info("First few rows of the processed data:")
        logger.info("\n" + processed_data.head().to_string())
    else:
        logger.error("No data was processed. Cannot save output file.")
    
    logger.info("=" * 80)
    logger.info("Script execution complete")
    logger.info("=" * 80)