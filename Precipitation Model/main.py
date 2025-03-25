"""
Main script for NYC precipitation prediction model.
"""
import os
import argparse
import logging
from datetime import datetime

import config
from weather_data.data_processor import DataProcessor
from model import train_precipitation_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main function to run the NYC precipitation prediction model.
    """
    parser = argparse.ArgumentParser(description="NYC Precipitation Prediction Model")
    parser.add_argument("--collect-only", action="store_true", help="Only collect data without training the model")
    parser.add_argument("--train-only", action="store_true", help="Only train the model using existing data")
    parser.add_argument("--start-date", type=str, default=config.HISTORICAL_START_DATE, help="Start date for historical data (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=config.HISTORICAL_END_DATE, help="End date for historical data (YYYY-MM-DD)")
    parser.add_argument("--forecast-days", type=int, default=config.FORECAST_DAYS, help="Number of days to forecast")
    parser.add_argument("--output", type=str, default=config.OUTPUT_FILE, help="Output file path for collected data")
    
    args = parser.parse_args()
    
    # Create data directory if it doesn't exist
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    # Collect data if needed
    if not args.train_only:
        logger.info("Collecting weather data...")
        processor = DataProcessor()
        
        # Collect historical data
        historical_data = processor.collect_historical_data(args.start_date, args.end_date)
        
        # Collect forecast data
        forecast_data = processor.collect_forecast_data(args.forecast_days)
        
        # Prepare data for model
        prepared_data = processor.prepare_data_for_model(historical_data, forecast_data)
        
        # Save to CSV
        output_file = processor.save_data(prepared_data, args.output)
        logger.info(f"Data collection completed. Data saved to {output_file}")
    
    # Train model if needed
    if not args.collect_only:
        logger.info("Training precipitation prediction model...")
        model = train_precipitation_model(args.output)
        logger.info("Model training completed.")

if __name__ == "__main__":
    main()
