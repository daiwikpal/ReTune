# check_paths.py
import os
import time
import config
from model import PrecipitationModel, FEATURE_COLUMNS

# 1) Print out your config paths
print("CONFIG:")
print("  DATA_DIR       =", config.DATA_DIR)
print("  MODEL H5 PATH  =", os.path.join(config.DATA_DIR, "models", "precip_model.h5"))
print("  SCALERS .npy   =", os.path.join(config.DATA_DIR, "models", "scalers.npy"))
print()

# 2) Check if they exist, and their modification times
h5_path   = os.path.join(config.DATA_DIR, "models", "precip_model.h5")
npy_path  = os.path.join(config.DATA_DIR, "models", "scalers.npy")

for p in (h5_path, npy_path):
    if os.path.exists(p):
        print(f"{p}  → exists,  last modified:", time.ctime(os.path.getmtime(p)))
    else:
        print(f"{p}  → DOES NOT EXIST")
print()

# 3) Try loading the model and print its input_shape
try:
    m = PrecipitationModel(sequence_length=12)
    m.load()
    print("✔ Loaded model.input_shape:", m.model.input_shape)
except Exception as e:
    print("❌ Loading error:", e)
print()

# 4) Confirm FEATURE_COLUMNS
print(f"FEATURE_COLUMNS (len={len(FEATURE_COLUMNS)}):", FEATURE_COLUMNS)
