import requests
import json
import math
import logging
import uuid
import pprint
from datetime import datetime
from typing import Dict, Optional

import kalshi_python
from kalshi_python.models import CreateOrderRequest

# ──────────────────────────────────────────────────────────────────────────────
# Logging configuration
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("integration_logs.log")
    ]
)
logger = logging.getLogger(__name__)

# ────────────────────────────────
# Kalshi Configuration
# ────────────────────────────────
config = kalshi_python.Configuration()
config.host = "https://demo-api.kalshi.co/trade-api/v2"

kalshi_api = kalshi_python.ApiInstance(
    email="Shubhampalak2@gmail.com",
    password="shubham1519P!",
    configuration=config,
)

# Ticker for this month’s “Rain in NYC” contract
MARKET_TICKER = "KXRAINNYCM-25APR-4"

# ──────────────────────────────────────────────────────────────────────────────
# ModelIntegrator
# ──────────────────────────────────────────────────────────────────────────────
class ModelIntegrator:
    """Integration layer for anomaly, precipitation, and market-trend models."""

    def __init__(
        self,
        anomaly_model_url: str       = "http://localhost:8000",
        precipitation_model_url: str = "http://localhost:8001",
        market_model_url: str        = "http://localhost:8002"
    ):
        self.anomaly_model_url       = anomaly_model_url
        self.precipitation_model_url = precipitation_model_url
        self.market_model_url        = market_model_url
        logger.info(
            f"Initialized ModelIntegrator with anomaly at {anomaly_model_url}, "
            f"precip at {precipitation_model_url}, market at {market_model_url}"
        )

    def get_anomaly_prediction(self, target_month: str) -> Optional[float]:
        try:
            endpoint = f"{self.anomaly_model_url}/predict_simple"
            payload = {"target_month": target_month}
            logger.info(f"Requesting anomaly prediction for {target_month}")
            response = requests.post(endpoint, json=payload)
            if response.status_code == 200:
                prediction = response.json().get("prediction")
                logger.info(f"Received anomaly prediction: {prediction}")
                return float(prediction)
            else:
                logger.error(f"Anomaly model HTTP {response.status_code}: {response.text}")
        except Exception as e:
            logger.exception(f"Error getting anomaly prediction: {e}")
        return None

    def get_precipitation_prediction(self, target_month: str) -> Optional[float]:
        try:
            endpoint = f"{self.precipitation_model_url}/predict"
            year, month = target_month.split("-")
            payload = {"year": int(year), "month": int(month)}
            logger.info(f"Requesting precipitation prediction for {target_month}")
            response = requests.post(endpoint, json=payload)
            if response.status_code == 200:
                prediction = response.json().get("predicted_monthly_precipitation_inches")
                logger.info(f"Received precipitation prediction: {prediction}")
                return float(prediction)
            else:
                logger.error(f"Precipitation model HTTP {response.status_code}: {response.text}")
        except Exception as e:
            logger.exception(f"Error getting precipitation prediction: {e}")
        return None

    def get_market_prediction(self) -> Optional[float]:
        try:
            endpoint = f"{self.market_model_url}/predict-live"
            logger.info("Requesting market model forecast")
            response = requests.get(endpoint)
            if response.status_code == 200:
                p = response.json().get("forecast_price")
                logger.info(f"Received market forecast: {p}")
                return float(p)
            else:
                logger.error(f"Market model HTTP {response.status_code}: {response.text}")
        except Exception as e:
            logger.exception(f"Error fetching market prediction: {e}")
        return None

    def _rain_cdf(self, mu: float, threshold: float, sigma: float) -> float:
        z = (mu - threshold) / (sigma * math.sqrt(2))
        return 0.5 * (1 + math.erf(z))

    def get_integrated_prediction(
        self,
        target_month: str,
        threshold: float = 4.0,
        sigma: float     = 1.2
    ) -> Dict:
        mu_anom   = self.get_anomaly_prediction(target_month)
        mu_precip = self.get_precipitation_prediction(target_month)
        if mu_anom is None or mu_precip is None:
            return {"success": False, "error": "Failed to fetch one of the weather models"}

        p_anom   = self._rain_cdf(mu_anom, threshold, sigma)
        p_precip = self._rain_cdf(mu_precip, threshold, sigma)

        p_market = self.get_market_prediction()
        if p_market is None:
            return {"success": False, "error": "Failed to fetch market model"}

        # 70% anomaly + 25% precip + 5% market
        weight_anom   = 0.70
        weight_precip = 0.25
        weight_market = 0.05
        p_final = (
            weight_anom   * p_anom +
            weight_precip * p_precip +
            weight_market * p_market
        )

        suggested_action = "go long" if p_final > p_market else "go short"

        return {
            "success": True,
            "raw_inches": {
                "anomaly":       mu_anom,
                "precipitation": mu_precip
            },
            "model_probabilities": {
                "P_anomaly": p_anom,
                "P_precip":  p_precip,
                "P_market":  p_market,
                "P_final":   p_final
            },
            "suggested_action": suggested_action,
            "timestamp": datetime.utcnow().isoformat()
        }

# ──────────────────────────────────────────────────────────────────────────────
# Kalshi Trading Logic (MARKET ORDER)
# ──────────────────────────────────────────────────────────────────────────────
def make_kalshi_trade(suggested_action: str, count: int = 10):
    """
    Place a market order on Kalshi for YES or NO at the current price.
    """
    side = "yes" if suggested_action == "go long" else "no"
    logger.info(f"Placing MARKET trade: side={side.upper()}, count={count}")

    try:
        status = kalshi_api.get_exchange_status()
        if not status.trading_active:
            logger.warning("Exchange not active; skipping trade.")
            return

        # market order: omit yes_price/no_price, set type="market"
        order = CreateOrderRequest(
            ticker=MARKET_TICKER,
            client_order_id=str(uuid.uuid4()),
            type="market",
            action="buy",
            side=side,
            count=count
        )

        response = kalshi_api.create_order(order)
        logger.info("Order submitted:")
        pprint.pprint(response)
    except Exception as e:
        logger.exception(f"Trade failed: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Main Execution
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    integrator = ModelIntegrator(
        anomaly_model_url="http://localhost:8000",
        precipitation_model_url="http://localhost:8001",
        market_model_url="http://localhost:8002"
    )

    target = "2025-03"  # or datetime.now().strftime("%Y-%m")
    result = integrator.get_integrated_prediction(target)
    print(json.dumps(result, indent=2))

    if result.get("success"):
        make_kalshi_trade(result["suggested_action"], count=10)
