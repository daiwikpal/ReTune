"""
Main script for the Kalshi Market Prediction Model.
Handles data collection and model training.
"""
import os
import argparse
import logging
import config
from market_data.data_processor import MarketDataProcessor
#from market_model import train_market_model  # Ensure this module is implemented

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main function to run the Kalshi Market Prediction Model.
    """
    parser = argparse.ArgumentParser(description="Kalshi Market Prediction Model")
    parser.add_argument("--collect-only", action="store_true", help="Collect market data only, do not train the model")
    parser.add_argument("--train-only", action="store_true", help="Train the model using existing market data, do not collect new data")
    parser.add_argument("--output", type=str, default=config.MARKET_OUTPUT_FILE, help="Output CSV file path for market data")
    
    args = parser.parse_args()
    os.makedirs(config.DATA_DIR, exist_ok=True)

    # Data collection step
    if not args.train_only:
        logger.info("Collecting market data...")
        processor = MarketDataProcessor()
        df = processor.collect_market_data()
        if df.empty:
            logger.error("No market data collected. Exiting.")
            return
        output_file = processor.save_data(df, args.output)
        logger.info(f"Market data saved to {output_file}")

    # Model training step
    if not args.collect_only:
        logger.info("Training market prediction model...")
        model = train_market_model(args.output)
        logger.info("Market model training completed.")

if __name__ == "__main__":
    main()
