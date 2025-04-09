import pandas as pd
import numpy as np
from datetime import datetime
import os
import glob
from Clients.noaa_client import NOAAClient
import random

def process_weather_data(include_precipitation: bool = True):
    # Get all CSV files in the historical data directory
    csv_files = glob.glob('Anomaly Model/historical data/*.csv')
    
    processed_dfs = []
    
    for csv_file in sorted(csv_files):
        print(f"Processing {os.path.basename(csv_file)}...")
        
        try:
            df = pd.read_csv(csv_file)
            
            target_states = ['NEW YORK', 'CONNECTICUT', 'PENNSYLVANIA', 'NEW JERSEY']
            df = df[df['STATE'].isin(target_states)]
            
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
            
            processed_dfs.append(pivot_df)
            
        except Exception as e:
            print(f"Error processing {os.path.basename(csv_file)}: {str(e)}")
            continue
    
    if not processed_dfs:
        raise ValueError("No data was successfully processed")
    
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
    
    for anomaly in anomaly_types:
        if anomaly not in combined_df.columns:
            combined_df[anomaly] = 0
    
    combined_df = combined_df.reset_index()
    
    combined_df['TimeStamp'] = combined_df['TimeStamp'].astype(str)
    
    if include_precipitation:
        try:
            noaa_client = NOAAClient()
            precip_df = noaa_client.get_historical_monthly_precipitation(years=20)
            
            if not precip_df.empty:
                combined_df = pd.merge(
                    combined_df,
                    precip_df,
                    on='TimeStamp',
                    how='left'
                )
                
                combined_df['PRECIPITATION'] = combined_df['PRECIPITATION'].fillna(method='ffill')
            else:
                print("Warning: Could not fetch precipitation data")
                combined_df['PRECIPITATION'] = np.nan
        except Exception as e:
            print(f"Error fetching precipitation data: {str(e)}")
            combined_df['PRECIPITATION'] = np.nan
    else:
        combined_df['PRECIPITATION'] = [random.uniform(0, 10) for _ in range(len(combined_df))]
    
    column_order = ['TimeStamp'] + anomaly_types
    column_order.append('PRECIPITATION')
    
    column_order = [col for col in column_order if col in combined_df.columns]
    combined_df = combined_df[column_order]
    
    return combined_df

def count_ny_blizzards_2024():
    df = pd.read_csv('Anomaly Model/historical data/StormEvents_details-ftp_v1.0_d2024_c20250401.csv')
    
    ny_blizzards = df[
        (df['STATE'] == 'NEW YORK') & 
        (df['EVENT_TYPE'] == 'Blizzard') & 
        (df['YEAR'] == 2024)
    ]
    
    print(f"\nNumber of blizzards in New York in 2024: {len(ny_blizzards)}")
    
    if not ny_blizzards.empty:
        print("\nBlizzard details:")
        for _, row in ny_blizzards.iterrows():
            print(f"Date: {row['BEGIN_YEARMONTH']}-{row['BEGIN_DAY']}")
            print(f"Location: {row['CZ_NAME']}")
            print(f"Event Narrative: {row['EVENT_NARRATIVE']}\n")
    
    return len(ny_blizzards)

# Example usage:
if __name__ == "__main__":
    print("Starting to process all historical weather data...")
    processed_data = process_weather_data(include_precipitation=False)
    output_file = 'Anomaly Model/processed_weather_data.csv'
    processed_data.to_csv(output_file, index=False)
    print(f"\nProcessed data has been saved to {output_file}")
    print("\nFirst few rows of the processed data:")
    print(processed_data.head())