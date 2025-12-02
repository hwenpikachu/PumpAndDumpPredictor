"""
ShibDetector.py

Read 5-minute SHIB/USD candles from Coinbase_SHIBUSD_5min.csv,
label pump / dump events, and export CSVs similar to the PEPE repo:

    shib_pumps_by_candle.csv
    shib_pumps_by_day.csv
    shib_dumps_by_candle.csv
    shib_dumps_by_day.csv
    shib_pump_then_dump_sequences.csv

Run AFTER you have some data:
    python ShibDetector.py
"""

from pathlib import Path

import pandas as pd

ROOT_CSV = Path("Coinbase_SHIBUSD_5min.csv")  # from SHIB/ folder
OUT_DIR = Path(".")  # current SHIB folder

# Detection hyper-params (tweak if you want)
WINDOW_CANDLES = 12          # 12 * 5min = 60 minutes
PUMP_PCT = 0.15              # +15% in the window
DUMP_PCT = -0.15             # -15% in the window
VOLUME_MULTIPLIER = 2.0      # volume spike vs rolling median
ROLL_VOL_WINDOW = 288        # 24 hours of 5-min candles (24*60/5)


def load_candles() -> pd.DataFrame:
    if not ROOT_CSV.exists():
        raise FileNotFoundError(
            f"Could not find {ROOT_CSV}. "
            "Run Coinbase_SHIBUSD_5min.csv collector first."
        )

    df = pd.read_csv(ROOT_CSV)

    # parse timestamp
    if "datetime" in df.columns:
        df["timestamp"] = pd.to_datetime(df["datetime"])
    elif "timestamp_ms" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms")
    elif "time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["time"], unit="s")
    elif "date" in df.columns:
        # this matches your SHIB / PEPE collector output
        df["timestamp"] = pd.to_datetime(df["date"], utc=True)
    else:
        raise ValueError("No recognizable time column in SHIB CSV.")

    df = df.sort_values("timestamp").reset_index(drop=True)

    needed_cols = {"open", "high", "low", "close", "volume"}
    missing = needed_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    return df



def add_volume_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling median volume used as baseline for "volume spike".
    """
    df["vol_median_24h"] = df["volume"].rolling(
        ROLL_VOL_WINDOW, min_periods=50
    ).median()
    df["vol_median_24h"] = df["vol_median_24h"].fillna(df["volume"].median())
    return df


def detect_events(df: pd.DataFrame):
    pumps = []
    dumps = []

    i = 0
    n = len(df)

    while i < n - WINDOW_CANDLES:
        if i % 50000 == 0 and i > 0:
            print(f"  scanned {i}/{n} candles...")

        window = df.iloc[i : i + WINDOW_CANDLES]
        start_row = window.iloc[0]
        start_price = float(start_row["close"])
        peak_price = float(window["high"].max())
        trough_price = float(window["low"].min())
        pct_up = (peak_price - start_price) / start_price
        pct_down = (trough_price - start_price) / start_price
        total_vol = float(window["volume"].sum())
        base_vol = float(df.loc[df.index[i], "vol_median_24h"])
        start_ts = start_row["timestamp"]
        end_ts = window["timestamp"].iloc[-1]

        if pct_up >= PUMP_PCT and total_vol >= VOLUME_MULTIPLIER * base_vol:
            pumps.append(
                {
                    "start_time": start_ts,
                    "end_time": end_ts,
                    "start_close": start_price,
                    "peak_high": peak_price,
                    "pct_change": pct_up,
                    "total_volume": total_vol,
                    "baseline_volume": base_vol,
                }
            )
            i += WINDOW_CANDLES
            continue

        if pct_down <= DUMP_PCT and total_vol >= VOLUME_MULTIPLIER * base_vol:
            dumps.append(
                {
                    "start_time": start_ts,
                    "end_time": end_ts,
                    "start_close": start_price,
                    "trough_low": trough_price,
                    "pct_change": pct_down,
                    "total_volume": total_vol,
                    "baseline_volume": base_vol,
                }
            )
            i += WINDOW_CANDLES
            continue

        i += 1

    pumps_df = pd.DataFrame(pumps)
    dumps_df = pd.DataFrame(dumps)

    return pumps_df, dumps_df



def export_by_candle_and_day(pumps_df: pd.DataFrame, dumps_df: pd.DataFrame):
    if not pumps_df.empty:
        pumps_df.to_csv(OUT_DIR / "shib_pumps_by_candle.csv", index=False)

        pumps_df["date"] = pumps_df["start_time"].dt.date
        pumps_by_day = (
            pumps_df.groupby("date")
            .agg(
                num_pumps=("start_time", "count"),
                max_pct_change=("pct_change", "max"),
                avg_pct_change=("pct_change", "mean"),
                total_volume=("total_volume", "sum"),
            )
            .reset_index()
        )
        pumps_by_day.to_csv(OUT_DIR / "shib_pumps_by_day.csv", index=False)

    if not dumps_df.empty:
        dumps_df.to_csv(OUT_DIR / "shib_dumps_by_candle.csv", index=False)

        dumps_df["date"] = dumps_df["start_time"].dt.date
        dumps_by_day = (
            dumps_df.groupby("date")
            .agg(
                num_dumps=("start_time", "count"),
                min_pct_change=("pct_change", "min"),
                avg_pct_change=("pct_change", "mean"),
                total_volume=("total_volume", "sum"),
            )
            .reset_index()
        )
        dumps_by_day.to_csv(OUT_DIR / "shib_dumps_by_day.csv", index=False)


def export_pump_dump_sequences(pumps_df: pd.DataFrame, dumps_df: pd.DataFrame):
    """
    Simple pump-then-dump sequences:
    first dump within 24h after each pump.
    """
    if pumps_df.empty or dumps_df.empty:
        return

    sequences = []
    for _, pump in pumps_df.iterrows():
        start = pump["start_time"]
        horizon = start + pd.Timedelta(hours=24)

        mask = (dumps_df["start_time"] > start) & (
            dumps_df["start_time"] <= horizon
        )
        eligible_dumps = dumps_df[mask].sort_values("start_time")

        if eligible_dumps.empty:
            continue

        dump = eligible_dumps.iloc[0]

        sequences.append(
            {
                "pump_start": pump["start_time"],
                "pump_end": pump["end_time"],
                "pump_pct_change": pump["pct_change"],
                "pump_total_volume": pump["total_volume"],
                "dump_start": dump["start_time"],
                "dump_end": dump["end_time"],
                "dump_pct_change": dump["pct_change"],
                "dump_total_volume": dump["total_volume"],
            }
        )

    if sequences:
        seq_df = pd.DataFrame(sequences)
        seq_df.to_csv(
            OUT_DIR / "shib_pump_then_dump_sequences.csv", index=False
        )


def main():
    print("Loading SHIB candles...")
    df = load_candles()
    print(f"Loaded {len(df)} candles.")

    print("Computing rolling volume baseline...")
    df = add_volume_baseline(df)
    print("Volume baseline ready.")

    print("Scanning for pump/dump windows...")
    pumps_df, dumps_df = detect_events(df)
    print("Finished scan.")
    print(f"Detected {len(pumps_df)} pump windows and {len(dumps_df)} dump windows.")

    print("Exporting CSVs...")
    export_by_candle_and_day(pumps_df, dumps_df)
    export_pump_dump_sequences(pumps_df, dumps_df)
    print("Done. Files written in the SHIB/ folder.")



if __name__ == "__main__":
    main()
