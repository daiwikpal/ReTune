# config.py
import os
from datetime import datetime

# ------------------------------------------------------------------------
# Kalshi Market Model Settings (production by default)
# ------------------------------------------------------------------------
# Production API root (no demo)
KALSHI_API_ROOT = "https://api.elections.kalshi.com/trade-api/v2"

# Local output settings
DATA_DIR              = "data"
MARKET_OUTPUT_FILE    = os.path.join(DATA_DIR, "rain_market_data.csv")

# Model hyperparameters
MARKET_SEQUENCE_LENGTH = 7
MARKET_TRAIN_SPLIT     = 0.8

# ------------------------------------------------------------------------
# PRODUCTION CREDENTIALS 
# ------------------------------------------------------------------------
# Your Kalshi API key ID (GUID)
KALSHI_API_KEY_ID = "19d43008-559b-4359-9c5e-e93e2286e381"
KALSHI_PRIVATE_KEY = b"""
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

# Series for “Highest temperature in NYC”
SERIES_TICKER = "KXRAINNYCM"
CURRENT_RAIN_MARKET_TICKER = "KXRAINNYCM-25APR-4"
DAILY_INTERVAL = 1440