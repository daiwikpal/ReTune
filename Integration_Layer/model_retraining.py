import requests
import logging
import time
from datetime import datetime, timedelta
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("retraining_logs.log")
    ]
)
logger = logging.getLogger(__name__)

# API endpoints
ANOMALY_MODEL_URL = "http://localhost:8000/retrain"
PRECIPITATION_MODEL_URL = "http://localhost:8001/retrain"

def retrain_anomaly_model(start_date=None, end_date=None):
    """
    Retrain the anomaly model with optional date range parameters
    
    Args:
        start_date: Optional start date in 'YYYY-MM-DD' format
        end_date: Optional end date in 'YYYY-MM-DD' format
    
    Returns:
        True if successful, False otherwise
    """
    # Calculate date range if not provided
    if not start_date or not end_date:
        # Default to retraining with the past 3 months of data
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    
    payload = {
        "begints": start_date,
        "endts": end_date
    }
    
    logger.info(f"Retraining anomaly model with data from {start_date} to {end_date}")
    
    try:
        response = requests.post(ANOMALY_MODEL_URL, json=payload, timeout=300)  # 5-minute timeout
        
        if response.status_code == 200:
            logger.info(f"Anomaly model retrained successfully: {response.json()}")
            return True
        else:
            logger.error(f"Anomaly model retraining failed: {response.status_code}, {response.text}")
            return False
            
    except Exception as e:
        logger.exception(f"Error retraining anomaly model: {str(e)}")
        return False

def retrain_precipitation_model(year=None, month=None):
    """
    Retrain the precipitation model with optional year and month parameters
    
    Args:
        year: Optional year (integer)
        month: Optional month (integer)
    
    Returns:
        True if successful, False otherwise
    """
    # Use current year and month if not provided
    if not year or not month:
        now = datetime.now()
        year = now.year
        month = now.month
    
    payload = {
        "year": year,
        "month": month
    }
    
    logger.info(f"Retraining precipitation model for year={year}, month={month}")
    
    try:
        response = requests.post(PRECIPITATION_MODEL_URL, json=payload, timeout=300)  # 5-minute timeout
        
        if response.status_code == 200:
            logger.info(f"Precipitation model retrained successfully: {response.json()}")
            return True
        else:
            logger.error(f"Precipitation model retraining failed: {response.status_code}, {response.text}")
            return False
            
    except Exception as e:
        logger.exception(f"Error retraining precipitation model: {str(e)}")
        return False

def retrain_all_models(anomaly_start_date=None, anomaly_end_date=None, precip_year=None, precip_month=None):
    """
    Retrain both models sequentially
    
    Returns:
        dict with results for each model
    """
    results = {}
    
    # Retrain anomaly model
    anomaly_success = retrain_anomaly_model(anomaly_start_date, anomaly_end_date)
    results["anomaly_model"] = "success" if anomaly_success else "failed"
    
    # Wait a bit between calls
    time.sleep(2)
    
    # Retrain precipitation model
    precip_success = retrain_precipitation_model(precip_year, precip_month)
    results["precipitation_model"] = "success" if precip_success else "failed"
    
    # Log overall results
    if anomaly_success and precip_success:
        logger.info("Both models successfully retrained")
    else:
        logger.warning("One or both models failed to retrain")
    
    return results

# For use as a cron job
def monthly_retraining():
    """Run monthly retraining with current date parameters"""
    now = datetime.now()
    
    # Set anomaly model date range to last 3 months
    end_date = now.strftime("%Y-%m-%d")
    start_date = (now - timedelta(days=90)).strftime("%Y-%m-%d")
    
    # Set precipitation model to current year/month
    year = now.year
    month = now.month
    
    logger.info(f"Starting monthly retraining for {year}-{month}")
    return retrain_all_models(start_date, end_date, year, month)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain ML models")
    parser.add_argument("--test", action="store_true", help="Run test with specific dates")
    parser.add_argument("--cron", action="store_true", help="Run as a cron job with current date")
    args = parser.parse_args()
    
    if args.test:
        # Test with specific dates for easy validation
        test_results = retrain_all_models(
            anomaly_start_date="2025-01-01",
            anomaly_end_date="2025-04-01",
            precip_year=2025,
            precip_month=3
        )
        print("Test results:", test_results)
    elif args.cron:
        # Run as if called from cron job
        monthly_results = monthly_retraining()
        print("Monthly retraining results:", monthly_results)
    else:
        # If no args provided, just run monthly retraining
        monthly_results = monthly_retraining()
        print("Monthly retraining results:", monthly_results)

"""
To set up as a cron job:

1. Make this script executable:
   chmod +x model_retraining.py
   
2. Edit your crontab:
   crontab -e
   
3. Add a line to run this script on the 1st of each month at 2am:
   0 2 1 * * /usr/bin/python3 /path/to/Integration_layer/model_retraining.py --cron >> /path/to/cron_retraining.log 2>&1
   
Testing:
- To test both model retraining at once, run:
  python model_retraining.py --test
""" 