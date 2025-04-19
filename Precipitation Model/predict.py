from datetime import datetime
import numpy as np
import pandas as pd
import config
from weather_data.data_processor import DataProcessor
from model import PrecipitationModel

def predict_precip_for_month(target_month: str) -> float:
    seq_len = config.SEQUENCE_LENGTH
    model = PrecipitationModel(sequence_length=seq_len)
    model.load_model()

    df = pd.read_csv(config.OUTPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])
    monthly = (
        df.set_index("date")
          .resample("M")
          .agg({
              "precipitation": "sum",
              "temperature_max": "mean",
              "temperature_min": "mean",
              "humidity": "mean",
              "wind_speed": "mean",
              "pressure": "mean"
          })
          .reset_index()
    )

    monthly["month"] = monthly["date"].dt.month
    monthly["season"] = monthly["month"].map(
        lambda m: 1 if m in [12,1,2]
                  else 2 if m in [3,4,5]
                  else 3 if m in [6,7,8]
                  else 4
    )
    for lag in (1, 2, 3):
        monthly[f"precipitation_lag{lag}"] = monthly["precipitation"].shift(lag)

    monthly = monthly.dropna().reset_index(drop=True)

    proc = DataProcessor()
    X_all, y_all, scalers = proc.create_sequences(
        monthly,
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

        # construct the new input step
        new_step = [0.0] * last_seq.shape[1]
        precip_index = list(scalers.keys()).index("precipitation")
        new_step[precip_index] = y_next
        last_seq = np.vstack([last_seq[1:], new_step])

    # final inverse transform
    total_rain = scalers["precipitation"].inverse_transform([[y_next]])[0][0]
    return round(float(total_rain), 4)


if __name__ == "__main__":
    next_month = (datetime.today().replace(day=1) + pd.offsets.MonthEnd(1)).strftime("%Y-%m")
    rain = predict_precip_for_month(next_month)
    print(f"Forecast for {next_month}: {rain:.3f} in. of rain")


