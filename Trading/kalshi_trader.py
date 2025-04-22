import uuid
import pprint
from datetime import datetime

import kalshi_python
from kalshi_python.models import CreateOrderRequest

# ────────────────────────────────
# CONFIGURATION
# ────────────────────────────────

# Switch to "https://trading-api.kalshi.com/trade-api/v2" for production
config = kalshi_python.Configuration()
config.host = "https://demo-api.kalshi.co/trade-api/v2"

# Kalshi login credentials
kalshi_api = kalshi_python.ApiInstance(
    email="Shubhampalak2@gmail.com",
    password="shubham1519P!",
    configuration=config,
)

# Example market ticker (change to the current month's ticker as needed)
MARKET_TICKER = "KXRAINNYCM-25APR-4"

# Example model output
final_prob = 0.82  # ← Replace this with your integration layer output


# ────────────────────────────────
# TRADING FUNCTION
# ────────────────────────────────

def make_kalshi_trade(prob: float, threshold: float = 0.55, count: int = 10):
    """
    Buys YES or NO on Kalshi depending on the prediction probability.
    Uses a high limit price (99c) to simulate a market order.
    """
    action = "yes" if prob >= threshold else "no"
    print(f"📈 Model says P(rain ≥ 4\") = {prob:.2%} → buying {action.upper()}")

    try:
        # Check exchange status
        status = kalshi_api.get_exchange_status()
        if not status.trading_active:
            print("Exchange is not active — skipping order.")
            return

        # Submit the order
        order = CreateOrderRequest(
            ticker=MARKET_TICKER,
            client_order_id=str(uuid.uuid4()),
            type="limit",
            action="buy",
            side=action,
            yes_price=99 if action == "yes" else None,
            no_price=99 if action == "no" else None,
            count=count,
        )

        response = kalshi_api.create_order(order)
        print("✅ Order submitted:")
        pprint.pprint(response)

    except Exception as e:
        print(f" Order failed: {e}")


# ────────────────────────────────
# EXECUTE TRADE
# ────────────────────────────────

if __name__ == "__main__":
    make_kalshi_trade(final_prob)
