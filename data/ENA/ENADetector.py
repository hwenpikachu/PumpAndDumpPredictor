import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from math import ceil

# --- Load & prep ---
df = pd.read_csv('Coinbase_ENAUSD_5min.csv')
df = df.sort_values('unix').reset_index(drop=True)

# --- Feature engineering (5-min candles) ---
# Candle body return (open->close) and close-to-close return
df['ret_body'] = (df['close'] - df['open']) / np.where(df['open'] == 0, np.nan, df['open'])
df['ret_cc'] = df['close'].pct_change()
df['range'] = (df['high'] - df['low']) / np.where(df['open'] == 0, np.nan, df['open'])
df['vol_change'] = df['volume'].pct_change()

# Rolling volatility (1 trading day ~ 288 candles)
df['volatility_288'] = df['ret_cc'].rolling(288).std()

# Drop initial NaNs from pct_change/rolling
df_model = df.dropna().copy()

# --- Option B thresholds (Z-score + absolute minimum) ---
ret_mean, ret_std = df_model['ret_body'].mean(), df_model['ret_body'].std(ddof=0)
vol_mean, vol_std = df_model['vol_change'].mean(), df_model['vol_change'].std(ddof=0)

# Sensitivity B
abs_min_ret = 0.01  # 1% candle body
z_k = 2.0

label_B = (
    (df_model['ret_body'] > ret_mean + z_k * ret_std) &
    (df_model['vol_change'] > vol_mean + z_k * vol_std) &
    (df_model['ret_body'] > abs_min_ret)
).astype(int)
df_model['pump_label_B'] = label_B

num_pos_B = int(df_model['pump_label_B'].sum())

# If B finds zero positives, fall back to slightly higher sensitivity for training only
used_sensitivity = 'B'
if num_pos_B == 0:
    z_k_train = 1.5
    abs_min_ret_train = 0.005
    label_train = (
        (df_model['ret_body'] > ret_mean + z_k_train * ret_std) &
        (df_model['vol_change'] > vol_mean + z_k_train * vol_std) &
        (df_model['ret_body'] > abs_min_ret_train)
    ).astype(int)
    used_sensitivity = 'C (fallback for training)'
else:
    label_train = df_model['pump_label_B']

df_model['pump_label_train'] = label_train
num_pos_train = int(df_model['pump_label_train'].sum())

# Features
features = ['ret_body', 'ret_cc', 'range', 'vol_change', 'volatility_288']
X = df_model[features].copy()
y = df_model['pump_label_train'].copy()

# --- Time-aware split ensuring positives in train ---
N = len(df_model)
min_test = max(1, int(0.1 * N))  # at least 10% test
default_split = int(0.8 * N)

pos_idx = np.where(y.values == 1)[0]
if len(pos_idx) > 0:
    last_pos = int(pos_idx.max())
    split_idx = max(default_split, last_pos + 1)  # ensure last positive is inside train
    split_idx = min(split_idx, N - min_test)      # ensure some test remains
else:
    split_idx = default_split

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# --- Scale for LR ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

reports = {}

# --- Train Logistic Regression if both classes present in train ---
if y_train.nunique() >= 2 and y_test.nunique() >= 2:
    lr = LogisticRegression(max_iter=2000, class_weight='balanced')
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_proba = lr.predict_proba(X_test_scaled)[:, 1]

    reports['LR_confusion'] = confusion_matrix(y_test, lr_pred).tolist()
    reports['LR_report'] = classification_report(y_test, lr_pred, output_dict=True)
    df_model.loc[df_model.index[split_idx:], 'lr_pred'] = lr_pred
    df_model.loc[df_model.index[split_idx:], 'lr_proba'] = lr_proba
else:
    reports['LR_confusion'] = None
    reports['LR_report'] = None
    df_model['lr_pred'] = np.nan
    df_model['lr_proba'] = np.nan

# --- Random Forest ---
if y_train.nunique() >= 2 and y_test.nunique() >= 2:
    rf = RandomForestClassifier(n_estimators=400, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]

    reports['RF_confusion'] = confusion_matrix(y_test, rf_pred).tolist()
    reports['RF_report'] = classification_report(y_test, rf_pred, output_dict=True)
    reports['RF_feature_importances'] = dict(zip(features, rf.feature_importances_))

    df_model.loc[df_model.index[split_idx:], 'rf_pred'] = rf_pred
    df_model.loc[df_model.index[split_idx:], 'rf_proba'] = rf_proba
else:
    reports['RF_confusion'] = None
    reports['RF_report'] = None
    reports['RF_feature_importances'] = None
    df_model['rf_pred'] = np.nan
    df_model['rf_proba'] = np.nan

# --- Isolation Forest (unsupervised) ---
iso = IsolationForest(
    n_estimators=300,
    contamination=0.01,  # assume ~1% of points are anomalies
    random_state=42
)
iso.fit(X_train)  # fit on train window only
iso_score = iso.decision_function(X)  # higher = more normal
iso_pred = iso.predict(X)  # -1 anomaly, 1 normal
df_model['iso_score'] = iso_score
df_model['iso_anomaly'] = (iso_pred == -1).astype(int)

import pandas as pd
import numpy as np

def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute minimal features if not already present."""
    out = df.copy()
    if 'date' not in out.columns:
        out['date'] = pd.to_datetime(out['unix'], unit='s', utc=True)
    if 'ret_body' not in out.columns:
        out['ret_body'] = (out['close'] - out['open']) / np.where(out['open'] == 0, np.nan, out['open'])
    if 'ret_cc' not in out.columns:
        out['ret_cc'] = out['close'].pct_change()
    if 'range' not in out.columns:
        out['range'] = (out['high'] - out['low']) / np.where(out['open'] == 0, np.nan, out['open'])
    if 'vol_change' not in out.columns:
        out['vol_change'] = out['volume'].pct_change()
    if 'volatility_288' not in out.columns:
        out['volatility_288'] = out['ret_cc'].rolling(288).std()
    return out.dropna().copy()

def _thresholds_for_sensitivity(sensitivity: str):
    """Return (z_k, min_abs_ret) for pump/dump based on sensitivity."""
    s = sensitivity.upper()
    if s == 'A':        # strict (few events)
        return 3.0, 0.02   # 3σ and 2%
    elif s == 'C':      # loose (many events)
        return 1.5, 0.005  # 1.5σ and 0.5%
    else:               # default 'B'
        return 2.0, 0.01   # 2σ and 1%

def _zscore_series(x: pd.Series, window=None):
    """Global z-score if window is None; else rolling (local) z-score."""
    if window is None:
        mu, sig = x.mean(), x.std(ddof=0)
        return (x - mu) / (sig if sig > 0 else 1.0)
    else:
        mu = x.rolling(window).mean()
        sig = x.rolling(window).std(ddof=0)
        sig = sig.replace(0, np.nan)
        return (x - mu) / sig

def generate_event_reports(
    df: pd.DataFrame,
    sensitivity: str = 'B',
    rolling_window=None,   # e.g., 864 for ~3 days @ 5m; None = global
    lookahead_minutes: int = 12 * 60      # link first dump within 12h after pump
):
    """
    Detects pump/dump 5m candles and writes per-candle, per-day, and pump->dump sequence CSVs.
    sensitivity: 'A' (strict), 'B' (default), 'C' (loose)
    rolling_window: None for global z-scores, or int window length in candles for local z-scores
    """
    dfx = _ensure_features(df).sort_values('unix').reset_index(drop=True)

    # Choose sensitivity thresholds
    z_k, min_abs_ret = _thresholds_for_sensitivity(sensitivity)

    # Compute z-scores either globally or rolling-window
    z_ret = _zscore_series(dfx['ret_body'], rolling_window)
    z_vol = _zscore_series(dfx['vol_change'], rolling_window)

    # Pump & dump rules
    pump_mask = (z_ret > z_k) & (z_vol > z_k) & (dfx['ret_body'] > min_abs_ret)
    dump_mask = (z_ret < -z_k) & (z_vol > z_k) & (dfx['ret_body'] < -min_abs_ret)

    pumps_df = dfx.loc[pump_mask, ['date','open','high','low','close','volume','ret_body','vol_change']].copy()
    dumps_df = dfx.loc[dump_mask, ['date','open','high','low','close','volume','ret_body','vol_change']].copy()

    # Ensure datetime type for .dt accessor
    if not pumps_df.empty:
        pumps_df['date'] = pd.to_datetime(pumps_df['date'], utc=True)
        pumps_df['day'] = pumps_df['date'].dt.date
        pumps_by_day = (pumps_df.groupby('day')
                        .agg(first_time=('date','min'),
                             last_time=('date','max'),
                             count=('date','count'),
                             max_ret=('ret_body','max'),
                             max_volchg=('vol_change','max'))
                        .reset_index()
                        .sort_values('first_time'))
    else:
        pumps_by_day = pd.DataFrame(columns=['day','first_time','last_time','count','max_ret','max_volchg'])

    if not dumps_df.empty:
        dumps_df['date'] = pd.to_datetime(dumps_df['date'], utc=True)
        dumps_df['day'] = dumps_df['date'].dt.date
        dumps_by_day = (dumps_df.groupby('day')
                        .agg(first_time=('date','min'),
                             last_time=('date','max'),
                             count=('date','count'),
                             min_ret=('ret_body','min'),
                             max_volchg=('vol_change','max'))
                        .reset_index()
                        .sort_values('first_time'))
    else:
        dumps_by_day = pd.DataFrame(columns=['day','first_time','last_time','count','min_ret','max_volchg'])


    # Link pump -> first dump within lookahead window
    sequences = []
    if not pumps_df.empty and not dumps_df.empty:
        for _, r in pumps_df.iterrows():
            after = dumps_df[dumps_df['date'] > r['date']]
            within = after[after['date'] <= r['date'] + pd.to_timedelta(lookahead_minutes, unit='m')]
            if not within.empty:
                first = within.iloc[0]
                sequences.append({
                    'pump_time': r['date'],
                    'pump_ret': r['ret_body'],
                    'dump_time': first['date'],
                    'dump_ret': first['ret_body'],
                    'minutes_between': (first['date'] - r['date']).total_seconds() / 60.0
                })
    seq_df = pd.DataFrame(sequences, columns=['pump_time','pump_ret','dump_time','dump_ret','minutes_between'])
    if not seq_df.empty:
        seq_df = seq_df.sort_values('pump_time')

    # Write files
    pumps_path = 'ena_pumps_by_candle.csv'
    dumps_path = 'ena_dumps_by_candle.csv'
    pumps_day_path = 'ena_pumps_by_day.csv'
    dumps_day_path = 'ena_dumps_by_day.csv'
    seq_path = 'ena_pump_then_dump_sequences.csv'

    pumps_df.to_csv(pumps_path, index=False)
    dumps_df.to_csv(dumps_path, index=False)
    pumps_by_day.to_csv(pumps_day_path, index=False)
    dumps_by_day.to_csv(dumps_day_path, index=False)
    seq_df.to_csv(seq_path, index=False)

    # Console summary
    print({
        "sensitivity": sensitivity,
        "rolling_window": rolling_window,
        "pumps_found": int(len(pumps_df)),
        "dumps_found": int(len(dumps_df)),
        "days_with_pumps": int(len(pumps_by_day)),
        "days_with_dumps": int(len(dumps_by_day)),
        "sequences_found": int(len(seq_df)),
        "files": {
            "pumps_by_candle": pumps_path,
            "dumps_by_candle": dumps_path,
            "pumps_by_day": pumps_day_path,
            "dumps_by_day": dumps_day_path,
            "pump_then_dump_sequences": seq_path
        }
    })


# --- Merge back date/time for readability ---
out_cols = ['unix', 'date', 'open', 'high', 'low', 'close', 'volume'] + \
           features + ['pump_label_B', 'pump_label_train',
                       'lr_pred', 'lr_proba', 'rf_pred', 'rf_proba',
                       'iso_score', 'iso_anomaly']

# Keep only columns that exist (in case LR/RF not trained)
out_cols = [c for c in out_cols if c in df_model.columns]

out_df = df_model[out_cols].copy()

# Save to file
out_path = 'ENA_5min_pump_predictions.csv'
out_df.to_csv(out_path, index=False)

# Prepare a compact summary dict
summary = {
    "rows_total": int(len(df)),
    "rows_model": int(len(df_model)),
    "positives_option_B": int(num_pos_B),
    "positives_used_for_training": int(num_pos_train),
    "train_size": int(len(X_train)),
    "test_size": int(len(X_test)),
    "used_sensitivity_for_training": used_sensitivity,
    "LR_confusion": reports['LR_confusion'],
    "RF_confusion": reports['RF_confusion'],
    "RF_feature_importances": reports['RF_feature_importances']
}

summary, out_path

generate_event_reports(df, sensitivity='B', rolling_window=None) 
