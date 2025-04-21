# candles_smoke.py
import time, base64, json, requests, socket, sys
from datetime import datetime, timedelta, timezone
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

# ─── credentials ───────────────────────────────────────────────
API_KEY_ID  = "19d43008-559b-4359-9c5e-e93e2286e381"
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

# ─── query parameters ───────────────────────────────────────────
SERIES  = "KXHIGHNY"
MARKET  = "KXHIGHNY-25APR20-B72.5"
PERIOD  = 1                       # 1‑minute (closest to 15m)
ny_tz   = timezone(timedelta(hours=-4))
yday_230 = (datetime.now(ny_tz).replace(hour=14, minute=30, second=0, microsecond=0)
            - timedelta(days=1))
START   = int(yday_230.timestamp())           # seconds
END     = int(time.time())

# ─── helpers ────────────────────────────────────────────────────
def build_headers(path: str):
    ts  = str(int(time.time() * 1000))
    msg = ts + "GET" + path
    priv= serialization.load_pem_private_key(PRIVATE_KEY, None, default_backend())
    sig = base64.b64encode(priv.sign(
            msg.encode(),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                         salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256()
          )).decode()
    return {
        "KALSHI-ACCESS-KEY":       API_KEY_ID,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": sig,
        "Content-Type":            "application/json"
    }

def try_host(host):
    path = f"/trade-api/v2/series/{SERIES}/markets/{MARKET}/candlesticks"
    url  = f"https://{host}{path}"
    hdrs = build_headers(path)
    params = {"start_ts": START, "end_ts": END, "period_interval": PERIOD}
    return requests.get(url, headers=hdrs, params=params, timeout=10)

# ─── attempt trading host, then elections host ─────────────────
for host in ["trading-api.kalshi.com", "api.elections.kalshi.com"]:
    try:
        socket.gethostbyname(host)            # DNS lookup
        print(f"Trying {host} …")
        r = try_host(host)
        print("HTTP", r.status_code)
        print(json.dumps(r.json(), indent=2))
        sys.exit()
    except socket.gaierror:
        print(f"DNS failed for {host}")
    except Exception as e:
        print(f"Error with {host}: {e}")

print("Both hosts failed – check DNS / firewall.")
