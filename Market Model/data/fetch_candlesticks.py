import requests
import time
import base64
import pandas as pd
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend
from datetime import datetime, timedelta, timezone
import re

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
# period_interval=1440 gives one 1‑day bar
DAILY_INTERVAL = 1440

# ─── AUTH HELPERS ─────────────────────────────────────────────────────────
def _now_ms() -> str:
    return str(int(time.time() * 1000))

def get_auth_headers(method: str, path: str) -> dict:
    ts  = _now_ms()
    msg = ts + method.upper() + path
    key = serialization.load_pem_private_key(PRIVATE_KEY_PEM, password=None, backend=default_backend())
    sig = key.sign(
        msg.encode(),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256()
    )
    return {
        "KALSHI-ACCESS-KEY":       API_KEY_ID,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        "Content-Type":            "application/json"
    }

# ─── STEP 1: FETCH ALL “-4” MARKETS ────────────────────────────────────────
def fetch_settled_markets(series: str, start: datetime, end: datetime) -> pd.DataFrame:
    all_mkts = []
    cursor   = None
    path     = "/markets"

    while True:
        params = {
            "series_ticker": series,
            "status":        "settled",
            "min_close_ts":  int(start.timestamp()),
            "max_close_ts":  int(end.timestamp()),
            "limit":         1000
        }
        if cursor:
            params["cursor"] = cursor

        resp = requests.get(API_ROOT + path,
                            headers=get_auth_headers("GET", path),
                            params=params)
        resp.raise_for_status()
        data = resp.json()
        page = data.get("markets", [])
        all_mkts.extend(page)
        cursor = data.get("cursor")
        if not cursor:
            break

    df = pd.DataFrame(all_mkts)
    # keep only tickers ending in "-4"
    return df[df["ticker"].str.endswith("-4")].reset_index(drop=True)

# ─── STEP 2: DAILY CANDLE HELPERS ─────────────────────────────────────────
def fetch_candles(series: str, market: str, start_ts: int, end_ts: int, interval: int):
    path = f"/series/{series}/markets/{market}/candlesticks"
    resp = requests.get(API_ROOT + path,
                        headers=get_auth_headers("GET", path),
                        params={
                            "start_ts":        start_ts,
                            "end_ts":          end_ts,
                            "period_interval": interval
                        })
    resp.raise_for_status()
    return resp.json().get("candlesticks", [])

# ─── STEP 3: MAIN LOOP ────────────────────────────────────────────────────
def main():
    # fetch all 4‑inch markets from Feb 2024 → today
    START = datetime(2024, 2, 1, tzinfo=timezone.utc)
    END   = datetime.now(timezone.utc)
    df4   = fetch_settled_markets(SERIES_TICKER, START, END)
    print(f"Found {len(df4)} 4‑inch markets:\n", df4["ticker"].tolist())

    rows = []
    for _, mkt in df4.iterrows():
        ticker = mkt["ticker"]
        title  = mkt["title"]  # ex: "Rain in NYC in Mar 2025?"
        # extract month & year from title
        mo, yr = re.search(r"in (\w+) (\d{4})", title).groups()
        month  = datetime.strptime(mo, "%b").month
        year   = int(yr)
        # calendar bounds for that month:
        month_start = datetime(year, month, 1, tzinfo=timezone.utc)
        next_month  = (month_start + timedelta(days=32)).replace(day=1)
        month_end   = next_month  # exclusive

        print(f"\n➡️  {ticker}: fetching {mo} {year} daily bars…")

        # pull daily bars across the full month in one go
        candles = fetch_candles(
            SERIES_TICKER,
            ticker,
            int(month_start.timestamp()),
            int(month_end.timestamp()),
            DAILY_INTERVAL
        )

        for c in candles:
            p = c.get("price", {})
            dt = pd.to_datetime(c["end_period_ts"], unit="s", utc=True).floor("D")
            # safely divide by 100 only when not None
            o = p.get("open")
            h = p.get("high")
            l = p.get("low")
            x = p.get("close")
            rows.append({
                "date":           dt.date(),
                "ticker":         ticker,
                "open":           (o / 100) if o is not None else None,
                "high":           (h / 100) if h is not None else None,
                "low":            (l / 100) if l is not None else None,
                "close":          (x / 100) if x is not None else None,
                "volume":         c.get("volume", 0),
                "open_interest":  c.get("open_interest", 0)
            })

    # assemble, reindex to ensure no gaps
    df_all = pd.DataFrame(rows)
    df_all = df_all.sort_values(["ticker","date"])
    out    = []

    # for each ticker, reindex daily
    for tck, sub in df_all.groupby("ticker"):
        mo = sub["date"].iloc[0].month
        yr = sub["date"].iloc[0].year
        full_idx = pd.date_range(
            datetime(yr, mo, 1),
            (datetime(yr, mo, 1) + timedelta(days=32)).replace(day=1) - timedelta(days=1),
            freq="D"
        ).date
        sub2 = sub.set_index("date").reindex(full_idx).assign(ticker=tck).rename_axis("date").reset_index()
        out.append(sub2)

    final = pd.concat(out, ignore_index=True)
    final.to_csv("KXRAINNYCM_4inch_daily.csv", index=False, date_format="%Y-%m-%d")
    print(f"\n✅ Wrote {len(final)} rows → KXRAINNYCM_4inch_daily.csv")

if __name__ == "__main__":
    main()