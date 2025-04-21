import requests
import time
import base64
import pandas as pd
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend
from datetime import datetime, timedelta, timezone

API_ROOT = "https://api.elections.kalshi.com/trade-api/v2"
API_KEY_ID = "19d43008-559b-4359-9c5e-e93e2286e381"
PRIVATE_KEY_PEM = b"""
-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA8ReGisw/QoS0+2nkRyzTeMHyRgxoM+rHlJcg/fdSknE+uaff
DPdBi6VeKQRGYJOWW/ZWRqSEif/xaBPGxAOCDv2YEbyk7QL85onryORsTMfqdZnX
W1qjcIUljt1EWbtz7MaQtEfz4kQ3ak1akomgSwVFkwUPNjN8k6rzv4f4kEQX/tUHT
UKxnGKiDhA1+Y6MFCTQpqljiupb3+Iqh8Jp0NrgF7iPBgofRm9QQFyj7KkzuSC82
DiWr7p89j7mwi5xkOHjzpI3thrtRfFrhtSI4D48sYA7NI7i7J8dvJCyDemUxG+yw
ap537gGl8RH7RXEqsF3/pFSttOmarwU8JFtsQIDAQABAoIBAFH5+dmAEZKApTlHjE
NXTqCk8cg3t8bPcgVoTeONERqSPw3JsuIpwLPHCvlPZs634ExsunFSx25VBLSq5M8
JklOejvKY/ktI1M1e4Dly0hBZebN7F+sMKr67x0WInxET2BsOeW2Tea3aHC2dF8rz
2PkpVbrj4YRAhX+AhVj+1tnWZH/BwimeXzUZ7iYSVmaXIsFloGNzc5CD0fo/rTRo+
Qig239qGF26QE8kdZ8cscnQ4aH4f+8kt0WXipZUo+CwH8w1VVfQn0BCXL5zBgopgf
OMEquWWOZs8qacRJirDlL0/dvtDTimo2QocPHtruGpVPXzO98/aW9HkS+4QE9RYAE
CgYEA/zqarveutMPG7WzUyu9K8Z7a0hisIunXhnK5oL2PSkW2QUbJltgpIOcvslba
r+LLAENqXJPxsFpY60wRlEy6N37Bpy8iXpmcEJDmqhYmXNqUl9B8W0bBkJZSWNARv
1ja7UoRbq8iEXvDeI+jrEmP5F1eVci9BY2i6fqdrWhUqDECgYEA8dH81sGPSoLwrg
8BppVjzOhZXis9Xf450YrqmdOAxBSEwEQp1jj4IVNiTVEadV2EL9fAAQ7H863aVFf
LJImI/d6Zt3xmEiyDrPQXV0bed2rrDfnFjtB2mOzCjdD5d+ZTb3OgMbfHXKP3/RI0
6NiYaSgGnF5p0nvBS+i6Fx7lPYECgYEAl3RFWlKsH6SVpUqRs6LwTBVCMK5nZ4hwV
t45fUM/hol7r5y7/4FiVp4Z1bBosTrZz9wxf6JjJ41Vert3KxOk5U2YyQbVVGG/FK
2H06K1PSCQUM5tHRUNxCkP0JgYD/5AW9M6KP5QLyPMSPyj2ZcFhjRJyIoIcQUtib8
oVkC1C+ECgYBY7bVCsNnmN/MUv5YG0edqwoOi+tnNiCFxKtoNiddPxI/xON/91OOD
NevvrQTC5oonIfuNKAdmWKfy3nppTF7hpYv4CzpqMo6V+wneYcSMO/iHIjSBya3jn
SYLCy/C5SQH14iw3/nj2rTnD1v/yS97dGnB9YKLElHKfaJ5wDXjgQKBgQDvbGOqKb
Y9fjyZNE2JZy20AE8qt82Ugb0j45GiihmTe8CDw8RdKmICr8RCZuo7uhbww8X6hCf
XSpod3nMLE8jhRPAWw2JcNKhhnM47WDoxav8yo/aS+XDjiYKuc8ChpW0Aozo2M/Jd
rWXONn475rWTnmelHqttgHkHhjpnrZQXFQ==
-----END RSA PRIVATE KEY-----
"""

SERIES_TICKER = "KXRAINNYCM"
START_DATE = datetime(2024, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime.now(timezone.utc) - timedelta(days=1)
PERIOD_INTERVAL = 1440  # daily candlesticks

def get_auth_headers(method: str, path: str) -> dict:
    timestamp = str(int(time.time() * 1000))
    message = timestamp + method.upper() + path
    private_key = serialization.load_pem_private_key(
        PRIVATE_KEY_PEM, password=None, backend=default_backend()
    )
    signature = private_key.sign(
        message.encode(),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH
        ),
        hashes.SHA256()
    )
    return {
        "KALSHI-ACCESS-KEY": API_KEY_ID,
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
        "Content-Type": "application/json"
    }

def fetch_candlesticks(series_ticker, market_ticker, start_ts, end_ts, period_interval):
    path = f"/series/{series_ticker}/markets/{market_ticker}/candlesticks"
    url = API_ROOT + path
    params = {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "period_interval": period_interval
    }
    headers = get_auth_headers("GET", path)
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    return data.get("candlesticks", [])

def main():
    markets_df = pd.read_csv("KXHIGHNY_settled_markets.csv")
    market_tickers = markets_df["ticker"].unique()
    print(f"✅ Loaded {len(market_tickers)} unique market tickers.")

    all_candlesticks = []
    for idx, ticker in enumerate(market_tickers, 1):
        print(f"\nFetching candlesticks for {ticker} ({idx}/{len(market_tickers)})...")

        candles = fetch_candlesticks(
            SERIES_TICKER,
            ticker,
            int(START_DATE.timestamp()),
            int(END_DATE.timestamp()),
            PERIOD_INTERVAL
        )

        if candles:
            for candle in candles:
                price_data = candle.get("price", {})
                if all(key in price_data and price_data[key] is not None for key in ["open", "high", "low", "close"]):
                    close_price = price_data["close"] / 100      
                    label = 1 if close_price >= 0.90 else 0
                    all_candlesticks.append({
                        "market_ticker": ticker,
                        "timestamp": candle.get("end_period_ts"),
                        "open": price_data["open"] / 100,
                        "high": price_data["high"] / 100,
                        "low": price_data["low"] / 100,
                        "close": price_data["close"] / 100,
                        "volume": candle.get("volume", 0),
                        "open_interest": candle.get("open_interest", 0),
                        "label": label

                    })
                else:
                    print(f"⚠️ Skipped incomplete price data for {ticker} at timestamp {candle.get('end_period_ts')}")

            print(f"✅ {len(candles)} candlesticks fetched.")
        else:
            print("⚠️ No candlesticks found for this market ticker.")

    if all_candlesticks:
        candles_df = pd.DataFrame(all_candlesticks)
        candles_df["timestamp"] = pd.to_datetime(candles_df["timestamp"], unit='s')
        candles_df.to_csv("KXHIGHNY_candlesticks.csv", index=False)
        print(f"\n📌 Total {len(candles_df)} candlesticks saved to 'KXHIGHNY_candlesticks.csv'")
    else:
        print("⚠️ No candlestick data was fetched for any market.")

if __name__ == "__main__":
    main()
