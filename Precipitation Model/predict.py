"""
Simple wrapper that autoregressively forecasts any future month.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from model import PrecipitationModel, FEATURE_COLUMNS
import config

def predict_precip_for_month(target_month: str) -> float:
    """
    target_month: "YYYY-MM"

    Loads the trained model + scalers, then:
     1) grabs last 12 cleaned rows
     2) computes how many months ahead
     3) loops forecast_next() that many times, each time appending the newly
        forecasted row (updating month, season, and lags) so the next step
        sees the synthetic data
    """
    # 1) Load the model & scalers
    m = PrecipitationModel(sequence_length=12)
    m.load()    # matches the .save() above

    # 2) Load & clean the full dataset
    df = pd.read_csv(config.NCEI_DATA_FILE, parse_dates=["date"]).sort_values("date")
    df["month"]  = df.date.dt.month
    df["season"] = df.month.map(lambda m:
        1 if m in [12,1,2] else
        2 if m in [3,4,5]  else
        3 if m in [6,7,8]  else 4
    )
    for lag in (1,2,3):
        df[f"precipitation_lag{lag}"] = df.precipitation.shift(lag)
    clean = df.dropna().reset_index(drop=True)

    # 3) Determine how many months ahead
    last_date = clean.date.iloc[-1].replace(day=1)
    target    = pd.to_datetime(f"{target_month}-01")
    months_ahead = ((target.year - last_date.year) * 12 +
                    (target.month - last_date.month))
    if months_ahead < 1:
        raise ValueError("Target month must be after the most recent data")

    # 4) Seed the loop with the final 12 rows
    recent = clean[["date"] + FEATURE_COLUMNS].iloc[-12:].copy()

    # 5) Autoregressively step forward
    next_precip = None
    for _ in range(months_ahead):
        next_precip = m.forecast_next(recent)
        # build a synthetic “next row”
        new_date = recent.date.iloc[-1] + pd.offsets.MonthBegin(1)
        month    = new_date.month
        season   = (1 if month in [12,1,2] else
                    2 if month in [3,4,5]  else
                    3 if month in [6,7,8]  else 4)
        # shift lags
        l1 = next_precip
        l2 = recent.precipitation_lag1.iloc[-1]
        l3 = recent.precipitation_lag2.iloc[-1]
        # append
        new_row = {
            "date": new_date,
            "precipitation":      next_precip,
            "month":              month,
            "season":             season,
            "precipitation_lag1": l1,
            "precipitation_lag2": l2,
            "precipitation_lag3": l3,
        }
        recent = recent.append(new_row, ignore_index=True).iloc[1:]
    return round(float(next_precip), 4)

if __name__ == "__main__":
    nm = (datetime.today().replace(day=1) + pd.offsets.MonthEnd(1)).strftime("%Y-%m")
    print(f"Forecast for {nm}:", predict_precip_for_month(nm))




