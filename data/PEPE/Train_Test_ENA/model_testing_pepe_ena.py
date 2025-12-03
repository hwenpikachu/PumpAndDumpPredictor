import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------------------------------------
# Shared helpers: features + pump/dump detection
# -------------------------------------------------------

def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute minimal features if not already present and ensure proper dtypes."""
    out = df.copy()

    # Always ensure 'date' is a proper datetime (UTC)
    if 'date' in out.columns:
        out['date'] = pd.to_datetime(out['date'], utc=True, errors='coerce')
    else:
        out['date'] = pd.to_datetime(out['unix'], unit='s', utc=True)

    # Core features
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

    # Drop rows where we don't have the needed features or date
    out = out.dropna(subset=['date', 'ret_body', 'ret_cc', 'range', 'vol_change'])

    return out.reset_index(drop=True)


def _quantile_params(sensitivity: str):
    """
    Quantile-based thresholds for pump/dump:
    - ret_hi_q: upper quantile for pump returns
    - ret_lo_q: lower quantile for dump returns
    - vol_q:   upper quantile for volume change
    - min_abs_ret: minimum absolute return to avoid tiny moves
    """
    s = sensitivity.upper()
    if s == 'A':      # strict (few events)
        return {
            "ret_hi_q": 0.995,
            "ret_lo_q": 0.005,
            "vol_q": 0.99,
            "min_abs_ret": 0.02,   # 2%
        }
    elif s == 'C':    # loose (many events)
        return {
            "ret_hi_q": 0.97,
            "ret_lo_q": 0.03,
            "vol_q": 0.95,
            "min_abs_ret": 0.005,  # 0.5%
        }
    else:             # default 'B' (medium)
        return {
            "ret_hi_q": 0.99,
            "ret_lo_q": 0.01,
            "vol_q": 0.97,
            "min_abs_ret": 0.01,   # 1%
        }


def generate_event_reports(
    df: pd.DataFrame,
    sensitivity: str = 'B',
    rolling_window: int | None = None,    # kept for compatibility, currently unused
    lookahead_minutes: int = 12 * 60,     # link first dump within 12h after pump
    prefix: str = "pepe"
):
    """
    Detects pump/dump 5m candles and writes per-candle, per-day, and pump->dump sequence CSVs.
    Uses quantile-based thresholds so you actually get some events.

    sensitivity: 'A' (strict), 'B' (default), 'C' (loose)
    prefix: string used to name output files, e.g. 'pepe' or 'ena'
    """
    dfx = _ensure_features(df).sort_values('unix').reset_index(drop=True)

    # Get quantile parameters
    params = _quantile_params(sensitivity)
    ret_hi_q = params["ret_hi_q"]
    ret_lo_q = params["ret_lo_q"]
    vol_q = params["vol_q"]
    min_abs_ret = params["min_abs_ret"]

    # Compute quantile thresholds on the whole sample
    ret_hi_thr = dfx['ret_body'].quantile(ret_hi_q)
    ret_lo_thr = dfx['ret_body'].quantile(ret_lo_q)
    vol_thr = dfx['vol_change'].quantile(vol_q)

    # Pump: large positive return + large volume change
    pump_mask = (
        (dfx['ret_body'] > max(min_abs_ret, ret_hi_thr)) &
        (dfx['vol_change'] > vol_thr)
    )

    # Dump: large negative return + large volume change
    dump_mask = (
        (dfx['ret_body'] < min(-min_abs_ret, ret_lo_thr)) &
        (dfx['vol_change'] > vol_thr)
    )

    pumps_df = dfx.loc[pump_mask, ['date','open','high','low','close','volume','ret_body','vol_change']].copy()
    dumps_df = dfx.loc[dump_mask, ['date','open','high','low','close','volume','ret_body','vol_change']].copy()

    # Per-day summaries (UTC)
    if not pumps_df.empty:
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

    # Output paths depend on prefix
    pumps_path = f'{prefix}_pumps_by_candle.csv'
    dumps_path = f'{prefix}_dumps_by_candle.csv'
    pumps_day_path = f'{prefix}_pumps_by_day.csv'
    dumps_day_path = f'{prefix}_dumps_by_day.csv'
    seq_path = f'{prefix}_pump_then_dump_sequences.csv'

    pumps_df.to_csv(pumps_path, index=False)
    dumps_df.to_csv(dumps_path, index=False)
    pumps_by_day.to_csv(pumps_day_path, index=False)
    dumps_by_day.to_csv(dumps_day_path, index=False)
    seq_df.to_csv(seq_path, index=False)

    print({
        "asset_prefix": prefix,
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

    return pumps_path


# -------------------------------------------------------
# Build forecast dataset for any asset
# -------------------------------------------------------

def build_forecast_dataset(
    price_csv: str,
    pumps_csv: str,
    horizon_candles: int = 6   # forecast next 6 candles (30 minutes at 5m)
) -> pd.DataFrame:
    """
    Given price candles + pump events for a single asset, build a dataset with:
      - features: ret_body, ret_cc, range, vol_change, volatility_288
      - is_pump_candle: 1 if this candle itself is a pump
      - pump_forecast_label: 1 if any pump happens in the next H candles
    """
    df = pd.read_csv(price_csv)
    df = df.sort_values("unix").reset_index(drop=True)

    # Feature engineering
    df["ret_body"] = (df["close"] - df["open"]) / np.where(df["open"] == 0, np.nan, df["open"])
    df["ret_cc"] = df["close"].pct_change()
    df["range"] = (df["high"] - df["low"]) / np.where(df["open"] == 0, np.nan, df["open"])
    df["vol_change"] = df["volume"].pct_change()
    df["volatility_288"] = df["ret_cc"].rolling(288).std()
    df["date"] = pd.to_datetime(df["unix"], unit="s", utc=True)

    features = ["ret_body", "ret_cc", "range", "vol_change", "volatility_288"]

    df_model = df.dropna(subset=features + ["date"]).copy()
    df_model = df_model.reset_index(drop=True)

    # Load detected pump events
    pumps_true = pd.read_csv(pumps_csv)
    pumps_true["date"] = pd.to_datetime(pumps_true["date"], utc=True, errors="coerce")
    pump_times = set(pumps_true["date"].dropna().unique())

    # Mark actual pump candles
    df_model["is_pump_candle"] = df_model["date"].isin(pump_times).astype(int)

    # Create forecast label: pump occurs in next H candles?
    H = horizon_candles
    is_pump = df_model["is_pump_candle"].values.astype(bool)
    N = len(df_model)
    forecast_label = np.zeros(N, dtype=int)

    for i in range(N - 1):
        j_end = min(N, i + 1 + H)
        if is_pump[i+1:j_end].any():
            forecast_label[i] = 1

    df_model["pump_forecast_label"] = forecast_label

    # For compatibility / clarity
    df_model["pump_label_train"] = df_model["pump_forecast_label"]

    # Some quick info
    total_pos = int(df_model["pump_forecast_label"].sum())
    print(f"{price_csv}: forecast positives = {total_pos}, negatives = {len(df_model) - total_pos}")

    return df_model


# -------------------------------------------------------
# Cross-asset training: PEPE -> ENA
# -------------------------------------------------------

def cross_asset_training(
    pepe_price_csv: str = "Coinbase_PEPEUSD_5min.csv",
    ena_price_csv: str = "Coinbase_ENAUSD_5min.csv",
    horizon_candles: int = 6
):
    # 1) Detect events on both assets
    df_pepe_raw = pd.read_csv(pepe_price_csv).sort_values("unix").reset_index(drop=True)
    df_ena_raw  = pd.read_csv(ena_price_csv).sort_values("unix").reset_index(drop=True)

    print("Detecting pump/dump events for PEPE...")
    pepe_pumps_csv = generate_event_reports(df_pepe_raw, sensitivity='B', prefix="pepe")

    print("Detecting pump/dump events for ENA...")
    ena_pumps_csv = generate_event_reports(df_ena_raw, sensitivity='B', prefix="ena")

    # 2) Build forecast datasets
    print("\nBuilding forecast dataset for PEPE (train)...")
    df_pepe = build_forecast_dataset(
        price_csv=pepe_price_csv,
        pumps_csv=pepe_pumps_csv,
        horizon_candles=horizon_candles
    )

    print("\nBuilding forecast dataset for ENA (test)...")
    df_ena = build_forecast_dataset(
        price_csv=ena_price_csv,
        pumps_csv=ena_pumps_csv,
        horizon_candles=horizon_candles
    )

    features = ["ret_body", "ret_cc", "range", "vol_change", "volatility_288"]

    X_train = df_pepe[features]
    y_train = df_pepe["pump_forecast_label"].astype(int)

    X_test = df_ena[features]
    y_test = df_ena["pump_forecast_label"].astype(int)

    print("\n--- Dataset sizes (cross-asset) ---")
    print(f"PEPE train size: {len(X_train)}, positives: {int(y_train.sum())}")
    print(f"ENA  test size: {len(X_test)}, positives: {int(y_test.sum())}")

    # 3) Scale features for LR
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Init prediction columns for saving
    df_pepe["lr_pred"] = np.nan
    df_pepe["lr_proba"] = np.nan
    df_pepe["rf_pred"] = np.nan
    df_pepe["rf_proba"] = np.nan

    df_ena["lr_pred"] = np.nan
    df_ena["lr_proba"] = np.nan
    df_ena["rf_pred"] = np.nan
    df_ena["rf_proba"] = np.nan

    # 4) Logistic Regression: train on PEPE, test on ENA
    if y_train.nunique() >= 2 and y_test.nunique() >= 1:
        lr = LogisticRegression(max_iter=2000, class_weight="balanced")
        lr.fit(X_train_scaled, y_train)
        lr_pred = lr.predict(X_test_scaled)
        lr_proba = lr.predict_proba(X_test_scaled)[:, 1]

        print("\n=== Logistic Regression (train=PEPE, test=ENA) ===")
        print(classification_report(y_test, lr_pred, digits=4))
        print("Confusion matrix:\n", confusion_matrix(y_test, lr_pred))

        df_ena["lr_pred"] = lr_pred
        df_ena["lr_proba"] = lr_proba
    else:
        print("\n[LR] Not enough class variety in PEPE train or ENA test.")

    # 5) Random Forest: train on PEPE, test on ENA
    if y_train.nunique() >= 2 and y_test.nunique() >= 1:
        rf = RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            class_weight="balanced"
        )
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_proba = rf.predict_proba(X_test)[:, 1]

        print("\n=== Random Forest (train=PEPE, test=ENA) ===")
        print(classification_report(y_test, rf_pred, digits=4))
        print("Confusion matrix:\n", confusion_matrix(y_test, rf_pred))

        feature_importances = dict(zip(features, rf.feature_importances_))
        print("\nRF feature importances (train=PEPE):", feature_importances)

        df_ena["rf_pred"] = rf_pred
        df_ena["rf_proba"] = rf_proba
    else:
        print("\n[RF] Not enough class variety in PEPE train or ENA test.")

    # 6) Save train + test CSVs
    train_out_cols = (
        ["unix", "date", "open", "high", "low", "close", "volume",
         "is_pump_candle", "pump_forecast_label"] + features
    )
    train_out_cols += ["lr_pred", "lr_proba", "rf_pred", "rf_proba"]
    train_out_cols = [c for c in train_out_cols if c in df_pepe.columns]

    test_out_cols = train_out_cols  # same schema if columns exist
    test_out_cols = [c for c in test_out_cols if c in df_ena.columns]

    pepe_out_path = "PEPE_5min_pump_forecast_trainset.csv"
    ena_out_path = "ENA_5min_pump_forecast_from_PEPE_model.csv"

    df_pepe[train_out_cols].to_csv(pepe_out_path, index=False)
    df_ena[test_out_cols].to_csv(ena_out_path, index=False)

    print(f"\nSaved PEPE training dataset to {pepe_out_path}")
    print(f"Saved ENA test predictions (from PEPE model) to {ena_out_path}")


# -------------------------------------------------------
# Main
# -------------------------------------------------------

if __name__ == "__main__":
    cross_asset_training(
        pepe_price_csv="Coinbase_PEPEUSD_5min.csv",
        ena_price_csv="Coinbase_ENAUSD_5min.csv",
        horizon_candles=6   # 30 minutes lookahead at 5m candles
    )
