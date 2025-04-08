import pandas as pd
import numpy as np
from datetime import timedelta, date
from market_model_full import train_market_model
import os
import config

# Ensure data folder exists
os.makedirs(config.DATA_DIR, exist_ok=True)

# Generate 100 days of mock market data
start = date(2024, 12, 30)
rows = []
for i in range(100):
    d = start + timedelta(days=i)
    price = round(0.42 + 0.03 * np.sin(i / 4), 3)         # fake market_price
    oi = np.random.randint(1500, 2500)                   # open_interest
    volume = np.random.randint(100, 400)                 # trading_volume
    rows.append([d, price, oi, volume])

df = pd.DataFrame(rows, columns=["date", "market_price", "open_interest", "trading_volume"])
df.to_csv(config.MARKET_OUTPUT_FILE, index=False)

print(f"✅ Mock market data created at {config.MARKET_OUTPUT_FILE}")

train_market_model(config.MARKET_OUTPUT_FILE)
import os
import config
from market_model_full import (
    collect_and_prepare_market_data_only,
    train_market_model,
    predict_market
)


def run_pipeline():
    print("\n🚀 Running Kalshi Market Trend Model Pipeline...")

    # Check if mock data exists, if not create it
    if not os.path.exists(config.MARKET_OUTPUT_FILE):
        print("📁 No market_data.csv found. Generating mock data...")
        from datetime import timedelta, date
        import pandas as pd
        import numpy as np

        os.makedirs(config.DATA_DIR, exist_ok=True)
        start = date(2024, 12, 30)
        rows = []
        for i in range(100):
            d = start + timedelta(days=i)
            price = round(0.35 + 0.1 * np.sin(i / 4), 3)  # stays between 0.25 and 0.45
            oi = np.random.randint(1500, 2500)
            volume = np.random.randint(100, 400)
            rows.append([d, price, oi, volume])

        df = pd.DataFrame(rows, columns=["date", "market_price", "open_interest", "trading_volume"])
        df.to_csv(config.MARKET_OUTPUT_FILE, index=False)
        print("✅ Mock market data created.")

    # Train model
    print("🧠 Training model...")
    train_market_model(config.MARKET_OUTPUT_FILE)
    print("✅ Model trained.")

    # Make test prediction
    print("🔮 Making a test prediction...")
    example_input = [0.46, 2100, 320]  # Replace with real data if available
    prediction = predict_market(example_input)
    print("📈 Prediction result:")
    for key, value in prediction.items():
        print(f"   {key}: {value}")


if __name__ == "__main__":
    run_pipeline()
    