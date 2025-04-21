import time
import base64
import logging
import pandas as pd
import requests
from typing import Dict
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend
import config

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

class MarketDataProcessor:
    """
    Client for fetching market data and candlesticks from the Kalshi API.
    """

    def __init__(self):
        # ensure key_data is bytes
        key_data = config.KALSHI_PRIVATE_KEY
        if isinstance(key_data, str):
            key_data = key_data.encode("utf-8")
        self.private_key = serialization.load_pem_private_key(
            key_data,
            password=None,
            backend=default_backend(),
        )
        self.api_key = config.KALSHI_API_KEY_ID
        self.api_root = config.KALSHI_API_ROOT

    def _headers(self, method: str, path: str) -> Dict[str, str]:
        ts = str(int(time.time() * 1000))
        msg = ts + method.upper() + path
        sig = self.private_key.sign(
            msg.encode("utf-8"),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY":       self.api_key,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
            "Content-Type":            "application/json",
        }

    def candlesticks(
        self,
        series_ticker: str,
        market_ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 1440,
    ) -> pd.DataFrame:
        path = f"/series/{series_ticker}/markets/{market_ticker}/candlesticks"
        resp = requests.get(
            self.api_root + path,
            headers=self._headers("GET", path),
            params={
                "start_ts":        start_ts,
                "end_ts":          end_ts,
                "period_interval": period_interval,
            },
        )
        resp.raise_for_status()
        body = resp.json()
        candles = body.get("candlesticks") or body.get("candles") or []
        if not candles:
            return pd.DataFrame()

        df = pd.json_normalize(candles, sep=".")
        df.rename(columns={"end_period_ts": "time"}, inplace=True)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        return df

    def get_market_info(self, market_ticker: str) -> dict:
        path = "/markets"
        resp = requests.get(
            self.api_root + path,
            headers=self._headers("GET", path),
            params={"ticker": market_ticker},
        )
        resp.raise_for_status()
        return resp.json().get("markets", [{}])[0]

    def collect_market_data(self) -> pd.DataFrame:
        path = "/markets"
        cursor = None
        all_markets = []

        while True:
            params = {"limit": 1000}
            if cursor:
                params["cursor"] = cursor
            resp = requests.get(
                self.api_root + path,
                headers=self._headers("GET", path),
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()
            all_markets.extend(data.get("markets", []))
            cursor = data.get("cursor")
            if not cursor:
                break

        df = pd.DataFrame(all_markets)
        if "close_time" in df.columns:
            df["close_time"] = pd.to_datetime(df["close_time"], unit="s", utc=True)
        logger.info("Collected %d markets", len(df))
        return df
