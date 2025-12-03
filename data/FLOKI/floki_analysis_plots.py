import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------- CONFIG -------------
CANDLE_CSV = "Coinbase_FLOKIUSD_5min.csv"
PUMPS_CSV = "floki_pumps_by_candle.csv"
DUMPS_CSV = "floki_dumps_by_candle.csv"
TEST_FRACTION = 0.2
# ----------------------------------

# ---------- LOAD CANDLE DATA ----------
if not os.path.exists(CANDLE_CSV):
    raise FileNotFoundError(f"Missing {CANDLE_CSV} in current folder.")

df = pd.read_csv(CANDLE_CSV)
# assume Coinbase-style columns: unix, low, high, open, close, volume, date
if "unix" in df.columns:
    df["timestamp"] = pd.to_datetime(df["unix"], unit="s")
elif "date" in df.columns:
    df["timestamp"] = pd.to_datetime(df["date"])
else:
    raise ValueError("No recognizable time column in FLOKI candle CSV.")

df = df.sort_values("timestamp").reset_index(drop=True)

# ---------- LOAD PUMPS / DUMPS IF PRESENT ----------
pumps = pd.read_csv(PUMPS_CSV) if os.path.exists(PUMPS_CSV) else None
dumps = pd.read_csv(DUMPS_CSV) if os.path.exists(DUMPS_CSV) else None

# ---------- BASIC FEATURES ----------
df["return"] = df["close"].pct_change()
df["vol_change"] = df["volume"].pct_change()
df["price_diff"] = df["close"].diff()
df["rolling_volatility"] = df["close"].rolling(30).std()

df = df.dropna().reset_index(drop=True)

features = ["return", "vol_change", "price_diff", "rolling_volatility"]
X = df[features].values
y = df["close"].values

n = len(df)
test_size = int(TEST_FRACTION * n)
train_size = n - test_size

X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:]
y_test = y[train_size:]

# ---------- SIMPLE LINEAR REGRESSION (NO SKLEARN) ----------
# Add bias term
X_train_aug = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test_aug = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

# Closed-form solution: beta = (X^T X)^-1 X^T y
XtX = X_train_aug.T @ X_train_aug
Xty = X_train_aug.T @ y_train
beta = np.linalg.inv(XtX) @ Xty  # (num_features+1,)

y_train_pred = X_train_aug @ beta
y_test_pred = X_test_aug @ beta

# ---------- METRICS (MAE, RMSE, R^2) ----------
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

train_mae = mae(y_train, y_train_pred)
train_rmse = rmse(y_train, y_train_pred)
train_r2 = r2(y_train, y_train_pred)

test_mae = mae(y_test, y_test_pred)
test_rmse = rmse(y_test, y_test_pred)
test_r2 = r2(y_test, y_test_pred)

print("\n===== SIMPLE LINEAR MODEL METRICS (NO SKLEARN) =====")
print(f"Train MAE:  {train_mae:.8f}")
print(f"Train RMSE: {train_rmse:.8f}")
print(f"Train R²:   {train_r2:.4f}")
print("-------------------------------------------")
print(f"Test MAE:   {test_mae:.8f}")
print(f"Test RMSE:  {test_rmse:.8f}")
print(f"Test R²:    {test_r2:.4f}")
print("====================================================\n")

# ---------- PLOT 1: PRICE + PUMPS / DUMPS ----------
plt.figure(figsize=(14, 6))
plt.plot(df["timestamp"], df["close"], label="FLOKI Close", linewidth=0.8)

if pumps is not None and len(pumps) > 0 and "start_time" in pumps.columns:
    pump_times = pd.to_datetime(pumps["start_time"])
    plt.scatter(pump_times, pumps.get("start_close", np.nan),
                label="Pump Windows", s=20)
if dumps is not None and len(dumps) > 0 and "start_time" in dumps.columns:
    dump_times = pd.to_datetime(dumps["start_time"])
    plt.scatter(dump_times, dumps.get("start_close", np.nan),
                label="Dump Windows", s=20)

plt.title("FLOKI Price with Detected Pump/Dump Windows")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.savefig("FLOKI_price_pumps_dumps.png")
plt.close()

# ---------- PLOT 2: VOLUME ----------
plt.figure(figsize=(14, 4))
plt.plot(df["timestamp"], df["volume"])
plt.title("FLOKI Volume Over Time")
plt.xlabel("Time")
plt.ylabel("Volume")
plt.tight_layout()
plt.savefig("FLOKI_volume.png")
plt.close()

# ---------- PLOT 3: TEST SET PREDICTIONS ----------
plt.figure(figsize=(14, 5))
plt.plot(y_test, label="Actual", linewidth=0.8)
plt.plot(y_test_pred, label="Predicted", linewidth=0.8)
plt.title("Linear Model – Test Set Price Prediction")
plt.xlabel("Test Index (Time Order)")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.savefig("FLOKI_test_predictions.png")
plt.close()

# ---------- PLOT 4: ERROR DISTRIBUTION ----------
errors = y_test - y_test_pred
plt.figure(figsize=(8, 5))
plt.hist(errors, bins=40)
plt.title("Prediction Error Distribution (Actual - Predicted)")
plt.xlabel("Error")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("FLOKI_prediction_error_distribution.png")
plt.close()

# ---------- PLOT 5: COEFFICIENT MAGNITUDE (FEATURE IMPORTANCE-LIKE) ----------
coef = beta[1:]  # exclude bias
plt.figure(figsize=(8, 5))
plt.bar(features, np.abs(coef))
plt.title("Linear Model Coefficient Magnitudes")
plt.ylabel("|Coefficient|")
plt.tight_layout()
plt.savefig("FLOKI_feature_coeff_magnitude.png")
plt.close()

print("Saved plots:")
print(" - FLOKI_price_pumps_dumps.png")
print(" - FLOKI_volume.png")
print(" - FLOKI_test_predictions.png")
print(" - FLOKI_prediction_error_distribution.png")
print(" - FLOKI_feature_coeff_magnitude.png")
