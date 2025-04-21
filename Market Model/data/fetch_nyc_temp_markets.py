import requests
import time
import base64
import pandas as pd
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend
from datetime import datetime, timedelta, timezone

# Your Kalshi credentials
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
START_DATE    = datetime(2024, 1, 1, tzinfo=timezone.utc)
END_DATE      = datetime.now(timezone.utc) - timedelta(days=1)

def to_timestamp(dt: datetime) -> int:
    return int(dt.timestamp())

def get_auth_headers(method: str, path: str) -> dict:
    ts = str(int(time.time() * 1000))
    msg = ts + method.upper() + path
    key = serialization.load_pem_private_key(PRIVATE_KEY_PEM, password=None, backend=default_backend())
    sig = key.sign(
        msg.encode(),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH
        ),
        hashes.SHA256()
    )
    return {
        "KALSHI-ACCESS-KEY":       API_KEY_ID,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        "Content-Type":            "application/json"
    }

def fetch_markets(series_ticker: str, start_date: datetime, end_date: datetime, status: str = 'settled'):
    markets = []
    limit = 1000
    cursor = None

    while True:
        params = {
            'series_ticker':  series_ticker,
            'status':         status,
            'min_close_ts':   to_timestamp(start_date),
            'max_close_ts':   to_timestamp(end_date),
            'limit':          limit
        }
        if cursor:
            params['cursor'] = cursor

        path = "/markets"
        headers = get_auth_headers("GET", path)
        resp = requests.get(API_ROOT + path, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()

        page = data.get("markets", [])
        markets.extend(page)
        print(f"✅ Fetched {len(page)} markets (Total so far: {len(markets)})")

        cursor = data.get("cursor")
        if not cursor:
            break

    # Build DataFrame and filter for only the “-4” market
    df = pd.DataFrame(markets)
    if df.empty:
        print("⚠️ No markets found for given criteria.")
        return

    # Keep only rows whose ticker ends with "-4"
    df_filtered = df[df['ticker'].str.endswith('-4')].copy()
    if df_filtered.empty:
        print("⚠️ No ‘-4’ markets found in the results.")
    else:
        out_csv = f"{series_ticker}_4inch_settled_markets.csv"
        df_filtered.to_csv(out_csv, index=False)
        print(f"\n📌 Saved {len(df_filtered)} ‘-4’ market(s) to '{out_csv}'")

if __name__ == "__main__":
    fetch_markets(SERIES_TICKER, START_DATE, END_DATE)
