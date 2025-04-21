import pandas as pd
import numpy as np
import json

df = pd.read_csv("data/ncei_weather_data.csv", parse_dates=["date"])

df["month"]  = df["date"].dt.month
df["season"] = df["month"].map(lambda m:
    1 if m in [12,1,2] else
    2 if m in [3,4,5]   else
    3 if m in [6,7,8]   else 4
)

for lag in (1,2,3,7):
    df[f"precipitation_lag{lag}"] = df["precipitation"].shift(lag)

df["precipitation_rolling_mean_7d"] = df["precipitation"].rolling(window=7, min_periods=1).mean()
df["precipitation_rolling_max_7d"]  = df["precipitation"].rolling(window=7, min_periods=1).max()

df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)

FEATURE_COLUMNS = [
    "precipitation",
    "month",
    "season",
    "precipitation_lag1",
    "precipitation_lag2",
    "precipitation_lag3",
]
clean = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

last12 = clean.sort_values("date").iloc[-12:].copy()

last12 = last12.where(pd.notnull(last12), None)

fields = [
    "date",
    "precipitation",
    "temperature_max",
    "temperature_min",
    "humidity",
    "wind_speed",
    "pressure",
    "temperature_range",
    "month",
    "season",
    "month_cos",
    "month_sin",
    "precipitation_lag1",
    "precipitation_lag2",
    "precipitation_lag3",
    "precipitation_lag7",
    "precipitation_rolling_mean_7d",
    "precipitation_rolling_max_7d",
]

# 10) Build your payload
records = []
for _, row in last12.iterrows():
    rec = {}
    for col in fields:
        val = row[col]
        if col == "date" and isinstance(val, pd.Timestamp):
            val = val.strftime("%Y-%m-%d")
        rec[col] = val
    records.append(rec)

payload = {"recent": records}

# 11) Print it
print(json.dumps(payload, indent=2))
