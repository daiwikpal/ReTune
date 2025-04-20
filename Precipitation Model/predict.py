from datetime import datetime
import math
import os
import numpy as np
import pandas as pd
import config
from weather_data.data_processor import DataProcessor
from model import PrecipitationModel
from model import FEATURE_COLUMNS

def predict_precip_for_month(target_month: str) -> float:
    seq_len = config.SEQUENCE_LENGTH
    model = PrecipitationModel(sequence_length=seq_len)
    model.load_model()

    ncei_path = os.path.join(config.DATA_DIR, "ncei_weather_data.csv")
    df = pd.read_csv(config.NCEI_DATA_FILE, parse_dates=["date"]).sort_values("date")

    # No resampling – file is already monthly
    df["month"]  = df["date"].dt.month
    df["season"] = df["month"].map(lambda m:
                    1 if m in [12,1,2] else
                    2 if m in [3,4,5] else
                    3 if m in [6,7,8] else 4)
    for lag in (1, 2, 3):
        df[f"precipitation_lag{lag}"] = df["precipitation"].shift(lag)

    monthly = df.dropna().reset_index(drop=True)
    monthly_selected = monthly[["date"] + FEATURE_COLUMNS]
    proc = DataProcessor()
    X_all, y_all, scalers = proc.create_sequences(
        monthly_selected,
        sequence_length=seq_len,
        target_column="precipitation"
    )

    last_seq = X_all[-1]
    last_date = monthly["date"].iloc[-1].replace(day=1)

    # compute how many months ahead
    target = pd.to_datetime(f"{target_month}-01")
    months_ahead = (target.year - last_date.year) * 12 + (target.month - last_date.month)

    if months_ahead < 1:
        raise ValueError("Target month must be in the future")

    # autoregressive prediction loop
    for _ in range(months_ahead):
        y_next_norm = model.predict(last_seq.reshape(1, *last_seq.shape))
        y_next = y_next_norm.flatten()[0]

        # build the next timestep (normalized)
        precip_index = FEATURE_COLUMNS.index("precipitation")
        base = last_seq[-1].copy()        # last timestep’s features
        base[precip_index] = y_next       # inject the prediction

        # slide the window forward by one month
        last_seq = np.vstack([last_seq[1:], base])

    # final inverse transform
    total_rain = scalers["precipitation"].inverse_transform([[y_next]])[0][0]
    if not math.isfinite(total_rain):
        raise ValueError("Model produced NaN or infinite precipitation")

    return round(float(total_rain), 4)


if __name__ == "__main__":
    next_month = (datetime.today().replace(day=1) + pd.offsets.MonthEnd(1)).strftime("%Y-%m")
    rain = predict_precip_for_month(next_month)
    print(f"Forecast for {next_month}: {rain:.3f} in. of rain")


