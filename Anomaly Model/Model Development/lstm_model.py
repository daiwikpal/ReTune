"""
rainfall_seq_models.py
Train & evaluate GRU and LSTM regressors on 12‑month windows of
Central Park precipitation data.

Runs both:
    • stride = 1   (overlapping windows)
    • stride = 12  (non‑overlapping)

Reports best validation MAE (in inches) for every combo.
"""

from __future__ import annotations
import argparse
import math
from pathlib import Path
import pickle
import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------------- #
DATA_FILE = Path(__file__).parent / "processed_weather_data.csv"

CONFIG = dict(
    window_size=12,     # months per sample
    hidden_size=16,
    dropout=0.1,        # Reduced dropout
    batch_size=16,
    lr=1e-4,           # Reduced learning rate
    train_split=0.7,
    val_split=0.15,    # Added validation split (remaining 0.15 is for test)
    clip_grad=1.0,      # Add gradient clipping
)
# --------------------------------------------------------------------------- #
class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float().unsqueeze(-1)

    def __len__(self): return len(self.x)

    def __getitem__(self, idx): return self.x[idx], self.y[idx]


def make_windows(
    df: pd.DataFrame,
    window: int,
    stride: int,
    feature_cols: list[str],
    target_col: str = "PRECIPITATION",
):
    xs, ys = [], []
    for start in range(0, len(df) - window, stride):
        end = start + window
        xs.append(df.iloc[start:end][feature_cols].values)
        ys.append(df.iloc[end][target_col])
    return np.stack(xs) if xs else np.array([]), np.array(ys) if ys else np.array([])


def preprocess_data(df: pd.DataFrame):
    """Clean and prepare features, returning processed dataframe and selected features"""
    # Create a deep copy to avoid modifying original
    df_clean = df.copy()
    
    # Make sure TimeStamp is excluded from features
    feature_cols = [c for c in df.columns if c not in ("TimeStamp", "PRECIPITATION")]
    
    return df_clean, feature_cols


def clean_and_split_data(df: pd.DataFrame):
    """Split data chronologically, then clean each split appropriately"""
    # Convert TimeStamp to datetime and filter for dates after December 1996
    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"])
    df = df[df["TimeStamp"] > pd.to_datetime("1996-12-31")]
    
    # First create chronological train/val/test splits
    # Split data into train, validation, and test sets chronologically
    train_end_idx = int(len(df) * CONFIG["train_split"])
    val_end_idx = int(len(df) * (CONFIG["train_split"] + CONFIG["val_split"]))
    
    train_df = df.iloc[:train_end_idx].copy()
    val_df = df.iloc[train_end_idx:val_end_idx].copy()
    test_df = df.iloc[val_end_idx:].copy()
    
    print(f"Train set: {len(train_df)} samples (dates: {train_df['TimeStamp'].min().date()} to {train_df['TimeStamp'].max().date()})")
    print(f"Val set: {len(val_df)} samples (dates: {val_df['TimeStamp'].min().date()} to {val_df['TimeStamp'].max().date()})")
    print(f"Test set: {len(test_df)} samples (dates: {test_df['TimeStamp'].min().date()} to {test_df['TimeStamp'].max().date()})")
    
    # Preprocess to get feature columns
    _, feature_cols = preprocess_data(train_df)
    
    # Clean train data and compute statistics
    for col in feature_cols:
        # Replace NaN values with median
        if train_df[col].isna().any():
            median_val = train_df[col].median()
            train_df[col] = train_df[col].fillna(median_val)
            val_df[col] = val_df[col].fillna(median_val)  # Use train median
            test_df[col] = test_df[col].fillna(median_val)  # Use train median
        
        # Handle extreme outliers (values beyond 3 std devs) using training data stats
        mean, std = train_df[col].mean(), train_df[col].std()
        upper_bound = mean + 3*std
        lower_bound = mean - 3*std
        
        train_df.loc[train_df[col] > upper_bound, col] = upper_bound
        train_df.loc[train_df[col] < lower_bound, col] = lower_bound
        
        val_df.loc[val_df[col] > upper_bound, col] = upper_bound
        val_df.loc[val_df[col] < lower_bound, col] = lower_bound
        
        test_df.loc[test_df[col] > upper_bound, col] = upper_bound
        test_df.loc[test_df[col] < lower_bound, col] = lower_bound
    
    return train_df, val_df, test_df, feature_cols


def create_dataloaders(train_df, val_df, test_df, feature_cols, stride):
    """Create dataloaders without data leakage"""
    # Create windows for each set separately
    X_train, y_train = make_windows(train_df, CONFIG["window_size"], stride, feature_cols)
    X_val, y_val = make_windows(val_df, CONFIG["window_size"], stride, feature_cols)
    X_test, y_test = make_windows(test_df, CONFIG["window_size"], stride, feature_cols)
    
    if len(X_train) == 0:
        raise ValueError(f"Not enough training data for window size {CONFIG['window_size']} and stride {stride}")
    
    # Print dataset sizes
    print(f"Training windows: {len(X_train)}")
    print(f"Validation windows: {len(X_val)}")
    print(f"Test windows: {len(X_test)}")
    
    # Fit scaler only on training data
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_flat)
    
    # Transform each dataset with the same scaler
    X_train = scaler.transform(X_train_flat).reshape(X_train.shape)
    
    if len(X_val) > 0:
        X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    
    if len(X_test) > 0:
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    # Check for NaN or infinite values
    for name, data in [("train", X_train), ("validation", X_val), ("test", X_test)]:
        if len(data) > 0 and (np.isnan(data).any() or np.isinf(data).any()):
            print(f"WARNING: NaN or infinite values found in {name} data after scaling!")
            if name == "train":
                X_train = np.nan_to_num(X_train)
            elif name == "validation":
                X_val = np.nan_to_num(X_val)
            else:
                X_test = np.nan_to_num(X_test)
    
    # Create dataloaders
    train_dl = DataLoader(SequenceDataset(X_train, y_train), batch_size=CONFIG["batch_size"], shuffle=True)
    val_dl = None if len(X_val) == 0 else DataLoader(SequenceDataset(X_val, y_val), batch_size=CONFIG["batch_size"])
    test_dl = None if len(X_test) == 0 else DataLoader(SequenceDataset(X_test, y_test), batch_size=CONFIG["batch_size"])
    
    return train_dl, val_dl, test_dl, X_train.shape[-1], scaler


# --------------------------------------------------------------------------- #
class Regressor(nn.Module):
    def __init__(self, typ: str, in_sz: int):
        super().__init__()
        rnn = nn.LSTM if typ == "lstm" else nn.GRU
        self.rnn = rnn(in_sz, CONFIG["hidden_size"], batch_first=True)
        self.dropout = nn.Dropout(CONFIG["dropout"])
        self.fc = nn.Linear(CONFIG["hidden_size"], 1)
        
        # Initialize weights
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.dropout(out[:, -1, :])  # Apply dropout after RNN
        return self.fc(out)      # last time‑step


# --------------------------------------------------------------------------- #
def train(train_dl, val_dl, model, dev, max_ep=100, patience=10):
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    loss_fn = nn.MSELoss()
    best, wait = math.inf, patience
    best_state = model.state_dict()  # Initialize best_state with current model state

    for ep in range(1, max_ep + 1):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            pred = model(xb)
            
            # Check for NaN in predictions
            if torch.isnan(pred).any():
                print("NaN detected in predictions, skipping batch")
                continue
                
            loss = loss_fn(pred, yb)
            
            # Check for NaN in loss
            if torch.isnan(loss).any():
                print("NaN detected in loss, skipping batch")
                continue
                
            loss.backward()
            
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["clip_grad"])
            
            opt.step()

        # If no validation set is available, use training loss
        if val_dl is None:
            print(f"Epoch {ep:03d} - No validation set available")
            wait -= 1
            if wait == 0:
                print("Early stopping")
                break
            continue
            
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(dev), yb.to(dev)
                pred = model(xb)
                if not torch.isnan(pred).any():
                    batch_mae = torch.mean(torch.abs(pred - yb)).item()
                    val_losses.append(batch_mae)
            
            val_mae = np.mean(val_losses) if val_losses else float('inf')

        print(f"Epoch {ep:03d}   val_MAE={val_mae:.4f}")
        if val_mae < best - 1e-6 and not math.isnan(val_mae):
            best, wait, best_state = val_mae, patience, model.state_dict()
        else:
            wait -= 1
            if wait == 0:
                print("Early stopping")
                break

    model.load_state_dict(best_state)
    return best


def evaluate(model, dataloader, device):
    """Evaluate model on the given dataloader"""
    if dataloader is None:
        return float('inf')
        
    model.eval()
    test_losses = []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            if not torch.isnan(pred).any():
                batch_mae = torch.mean(torch.abs(pred - yb)).item()
                test_losses.append(batch_mae)
    
    return np.mean(test_losses) if test_losses else float('inf')


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-epochs", type=int, default=1000)
    ap.add_argument("--patience",   type=int, default=10)
    args = ap.parse_args()

    torch.manual_seed(0);  np.random.seed(0)
    
    # Load data
    df = pd.read_csv(DATA_FILE).sort_values("TimeStamp").reset_index(drop=True)
    
    # Clean and split data chronologically
    train_df, val_df, test_df, feature_cols = clean_and_split_data(df)
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory for models if it doesn't exist
    models_dir = Path(__file__).parent / "saved_models"
    os.makedirs(models_dir, exist_ok=True)

    # Store results for comparison
    results = []

    for stride in (1, 12):
        print(f"\n=== Creating datasets with stride={stride} ===")
        # Create dataloaders with proper train/val/test split
        train_dl, val_dl, test_dl, in_sz, scaler = create_dataloaders(
            train_df, val_df, test_df, feature_cols, stride
        )
        
        for typ in ("gru", "lstm"):
            print(f"\n=== {typ.upper()}  stride={stride}")
            model = Regressor(typ, in_sz)
            
            # Train the model
            val_mae = train(train_dl, val_dl, model, dev,
                         max_ep=args.max_epochs, patience=args.patience)
            print(f"Best VAL MAE: {val_mae:.4f} inches")
            
            # Evaluate on test data
            test_mae = evaluate(model, test_dl, dev)
            print(f"TEST MAE: {test_mae:.4f} inches")
            
            # Save model info
            model_info = {
                'type': typ,
                'stride': stride,
                'val_mae': val_mae,
                'test_mae': test_mae,
                'in_size': in_sz,
                'model': model,
                'scaler': scaler
            }
            results.append(model_info)
    
    # Find best model by validation score
    best_model_info = min(results, key=lambda x: x['val_mae'])
    
    # Save the best model (selected by validation MAE)
    model_filename = f"{best_model_info['type']}_stride{best_model_info['stride']}_valMAE{best_model_info['val_mae']:.4f}.pkl"
    model_path = models_dir / model_filename
    
    # Save the model state dict and metadata
    save_dict = {
        'model_state': best_model_info['model'].state_dict(),
        'model_info': {
            'type': best_model_info['type'],
            'stride': best_model_info['stride'],
            'val_mae': best_model_info['val_mae'],
            'test_mae': best_model_info['test_mae'],
            'in_size': best_model_info['in_size'],
            'config': CONFIG
        },
        'scaler': best_model_info['scaler']  # Save the scaler for future transformations
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(save_dict, f)
    
    print(f"\nBest model saved to {model_path}")
    print(f"Best model: {best_model_info['type'].upper()} with stride={best_model_info['stride']}")
    print(f"Validation MAE: {best_model_info['val_mae']:.4f} inches")
    print(f"Test MAE: {best_model_info['test_mae']:.4f} inches")


def load_model(model_path):
    with open(model_path, 'rb') as f:
        save_dict = pickle.load(f)
    
    model_info = save_dict['model_info']
    model = Regressor(model_info['type'], model_info['in_size'])
    model.load_state_dict(save_dict['model_state'])
    
    return model, model_info, save_dict.get('scaler')


if __name__ == "__main__":
    main()