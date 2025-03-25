"""
Data Processor for Kalshi Market Data using the GET /markets endpoint.
This implementation follows the official documentation:
https://trading-api.readme.io/reference/getmarkets-1
"""
import time
import base64
import logging
import pandas as pd
import numpy as np
import requests
from typing import Dict, Tuple
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketDataProcessor:
    """
    Handles data collection and preprocessing for Kalshi market data using demo credentials.
    """
    def __init__(self):
        # Load demo credentials from config
        self.api_key_id = config.KALSHI_API_KEY_ID
        self.private_key_str = config.KALSHI_PRIVATE_KEY
        self.private_key = serialization.load_pem_private_key(
            self.private_key_str.encode('utf-8'),
            password=None,
            backend=default_backend()
        )
        # Set API root from config
        self.api_root = config.KALSHI_API_ROOT

    def _generate_headers(self, method: str, path: str) -> Dict[str, str]:
        """
        Generate the required authentication headers for Kalshi API.
        """
        timestamp_ms = str(int(time.time() * 1000))
        message = timestamp_ms + method.upper() + path
        signature = self._sign_message(message)
        headers = {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "Content-Type": "application/json"
        }
        return headers

    def _sign_message(self, message: str) -> str:
        """
        Sign the given message using RSA-PSS with SHA256 and return the base64-encoded signature.
        """
        message_bytes = message.encode('utf-8')
        signature = self.private_key.sign(
            message_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH
            ),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode('utf-8')

    def collect_market_data(self) -> pd.DataFrame:
        """
        Collect market data using the GET /markets endpoint as per the documentation.
        Returns:
            A DataFrame containing the list of markets.
        """
        path = "/markets"
        url = self.api_root + path
        headers = self._generate_headers("GET", path)
        logger.info("Collecting market data from Kalshi API using GET /markets ...")
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data_json = response.json()
            # According to the docs, the JSON response contains a "markets" key.
            if isinstance(data_json, dict) and "markets" in data_json:
                markets_list = data_json["markets"]
            else:
                markets_list = data_json
            df = pd.DataFrame(markets_list)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            logger.info("Market data successfully collected.")
            return df
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
        except Exception as err:
            logger.error(f"An error occurred: {err}")
        return pd.DataFrame()

    def prepare_data_for_model(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare the collected market data for model training.
        """
        logger.info("Preparing market data for the LSTM model...")
        df = df.sort_values("date").reset_index(drop=True)
        required_columns = ["date", "market_price", "open_interest", "trading_volume"]
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            logger.error(f"Missing required columns: {missing}")
        return df

    def save_data(self, df: pd.DataFrame, output_file: str) -> str:
        """
        Save the processed market data to a CSV file.
        """
        df.to_csv(output_file, index=False)
        return output_file

    def create_sequences(
        self, 
        data: pd.DataFrame, 
        sequence_length: int, 
        target_column: str = "market_price"
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, any]]:
        """
        Convert the DataFrame into sequences for LSTM training.
        """
        logger.info("Creating sequences for market data...")
        feature_cols = [col for col in data.columns if col != "date"]
        if target_column not in feature_cols:
            feature_cols.append(target_column)
        data_array = data[feature_cols].values
        X, y = [], []
        for i in range(len(data_array) - sequence_length):
            seq_x = data_array[i : i + sequence_length, :]
            seq_y = data_array[i + sequence_length, feature_cols.index(target_column)]
            X.append(seq_x)
            y.append(seq_y)
        X = np.array(X)
        y = np.array(y)
        scalers = {}  # Add scaling logic if needed
        return X, y, scalers