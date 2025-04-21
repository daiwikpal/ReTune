"""
Export every settled binary bracket in Kalshi’s KXHIGHNY series
to data/nyc_daily_high_dataset.csv
"""

import os, time, base64, requests, pandas as pd
from typing import Dict, List
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend
import re
from datetime import datetime, date

# ─── credentials ───────────────────────────────────────────────────────
ROOT = "https://api.elections.kalshi.com/trade-api/v2"
API_KEY = "19d43008-559b-4359-9c5e-e93e2286e381"
PRIVATE_KEY = b"""
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

SERIES   = "KXHIGHNY"
OUT_DIR  = "data"
OUT_FILE = os.path.join(OUT_DIR, "nyc_daily_high_dataset.csv")

# ─── auth helper ───────────────────────────────────────────────────────
_priv = serialization.load_pem_private_key(PRIVATE_KEY, None, default_backend())
def signed_hdrs(method: str, path: str) -> Dict[str, str]:
    ts  = str(int(time.time()*1000))
    sig = base64.b64encode(_priv.sign(
            (ts + method.upper() + path).encode(),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                         salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256())).decode()
    return {
        "KALSHI-ACCESS-KEY":       API_KEY,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": sig,
        "Content-Type":            "application/json"
    }

# ─── helpers ───────────────────────────────────────────────────────────
def settled_events() -> List[str]:
    evts, url = [], f"{ROOT}/events?series_ticker={SERIES}&status=settled&limit=200"
    while url:
        path = url.split(".com")[-1]
        r = requests.get(url, headers=signed_hdrs("GET", path), timeout=10)
        r.raise_for_status()
        js  = r.json()
        evts += [e["event_ticker"] for e in js["events"]]
        url  = js.get("next_url")
    return evts

def settled_markets(event_tkr: str) -> List[str]:
    # correct endpoint (bulk /markets) with event filter + status
    q   = f"/series/{SERIES}/markets?event_ticker={event_tkr}&status=settled&limit=1000"
    url = ROOT + q
    r   = requests.get(url, headers=signed_hdrs("GET", q), timeout=10)
    if r.status_code in (400,404):
        return []
    r.raise_for_status()
    return [m["ticker"] for m in r.json()["markets"] if "-B" in m["ticker"]]

def daily_candle(tkr: str):
    p = f"/series/{SERIES}/markets/{tkr}/candlesticks"
    r = requests.get(
            ROOT+p, headers=signed_hdrs("GET", p),
            params={"start_ts":0, "end_ts":int(time.time()), "period_interval":1440},
            timeout=10)
    if r.status_code in (400,404):
        return None
    r.raise_for_status()
    cdls = r.json().get("candlesticks") or r.json().get("candles") or []
    return cdls[-1] if cdls else None

def strike_F(tkr: str) -> float:
    return float(tkr.split("-B")[-1])

def ticker_date(event_tkr: str) -> date:
    """
    Parse either format:
      • KXHIGHNY-25APR20   (DDDMMMYY)
      • KXHIGHNY-2025-04-20
    """
    suffix = event_tkr.split("-", 1)[-1]          # drop series prefix

    # new style YYYY-MM-DD
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", suffix):
        return datetime.strptime(suffix, "%Y-%m-%d").date()

    # old style DDMMMYY  → 25APR20
    if re.fullmatch(r"\d{2}[A-Z]{3}\d{2}", suffix):
        return datetime.strptime(suffix, "%d%b%y").date()

    raise ValueError(f"Unrecognised event ticker format: {event_tkr}")
# ─── main ──────────────────────────────────────────────────────────────
def build():
    rows = []
    print("Querying settled events …")
    for evt in settled_events():
        d = ticker_date(evt)
        mkts = settled_markets(evt)
        if not mkts:
            print(f"  {d}: no settled markets")
            continue
        print(f"  {d}: {len(mkts)} markets")
        for tkr in mkts:
            c = daily_candle(tkr)
            if not c: 
                continue
            close_c = c["price"]["close"]
            rows.append({
                "date":          d.isoformat(),
                "ticker":        tkr,
                "strike_F":      strike_F(tkr),
                "settle_price":  close_c/100.0,
                "volume":        c["volume"],
                "open_interest": c["open_interest"],
                "label":         1 if close_c>=99 else 0
            })

    if not rows:
        print("No data collected — check credentials / API status.")
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame(rows).sort_values(["date","strike_F"]).to_csv(OUT_FILE, index=False)
    print(f"✅  CSV written → {OUT_FILE}  ({len(rows)} rows)")

if __name__ == "__main__":
    build()
