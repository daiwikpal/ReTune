import tensorflow as tf
import pandas as pd
from model import PrecipitationModel, FEATURE_COLUMNS
import config

# 1) Load the latest saved model+scalers from disk
m = PrecipitationModel(sequence_length=12)
m.load()  
print("🔍 Loaded model.input_shape:", m.model.input_shape)

# 2) Check your FEATURE_COLUMNS
print("🔍 FEATURE_COLUMNS (len={}):".format(len(FEATURE_COLUMNS)), FEATURE_COLUMNS)

# 3) Build a dummy DataFrame exactly like your payload
#    (replace config.NCEI_DATA_FILE with your real CSV if you want)
df = pd.read_csv(config.NCEI_DATA_FILE, parse_dates=["date"]).sort_values("date")
# take last 12 rows, pick only date + FEATURE_COLUMNS
df_in = df[["date"] + FEATURE_COLUMNS].iloc[-12:].copy()
print("🔍 df_in.columns:", list(df_in.columns))
print("🔍 df_in.values.shape:", df_in.values.shape)

# 4) Try a dummy predict to force the same error
try:
    _ = m.model.predict(
        df_in[FEATURE_COLUMNS].values.reshape((1,12,len(FEATURE_COLUMNS)))
    )
    print("✅ Dummy predict succeeded with shape (1,12,{})".format(len(FEATURE_COLUMNS)))
except Exception as e:
    print("❌ Dummy predict failed:", e)
