import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-GUI backend
import matplotlib.pyplot as plt


def _load_data(price_csv, pumps_csv, dumps_csv):
    """Helper to load and parse datetime columns."""
    df = pd.read_csv(price_csv)
    pumps = pd.read_csv(pumps_csv)
    dumps = pd.read_csv(dumps_csv)

    # Ensure datetime format
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    else:
        df["date"] = pd.to_datetime(df["unix"], unit="s", utc=True, errors="coerce")

    if not pumps.empty:
        pumps["date"] = pd.to_datetime(pumps["date"], utc=True, errors="coerce")
    if not dumps.empty:
        dumps["date"] = pd.to_datetime(dumps["date"], utc=True, errors="coerce")

    return df, pumps, dumps


def plot_price_with_events(
    price_csv="Coinbase_PEPEUSD_5min.csv",
    pumps_csv="pepe_pumps_by_candle.csv",
    dumps_csv="pepe_dumps_by_candle.csv",
    out_path="pepe_price_with_events.png"
):
    df, pumps, dumps = _load_data(price_csv, pumps_csv, dumps_csv)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["date"], df["close"], label="PEPE close (5m)", color="blue", linewidth=1.2, zorder=1)

    # Pump markers (GREEN, triangles up, on top)
    if not pumps.empty:
        ax.scatter(
            pumps["date"], pumps["close"],
            marker="^", s=40, color="green",
            edgecolor="black", linewidth=0.5,
            label="Pump candles",
            zorder=3
        )

    # Dump markers (RED, triangles down, on top)
    if not dumps.empty:
        ax.scatter(
            dumps["date"], dumps["close"],
            marker="v", s=40, color="red",
            edgecolor="black", linewidth=0.5,
            label="Dump candles",
            zorder=3
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Price (USD)")
    ax.set_title("PEPE 5m Price with Pump (green) and Dump (red) Markers")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved price-with-events plot to {out_path}")


def plot_events_per_day(
    pumps_csv="pepe_pumps_by_candle.csv",
    dumps_csv="pepe_dumps_by_candle.csv",
    out_path="pepe_events_per_day.png"
):
    pumps = pd.read_csv(pumps_csv)
    dumps = pd.read_csv(dumps_csv)

    if not pumps.empty:
        pumps["date"] = pd.to_datetime(pumps["date"], utc=True, errors="coerce")
        pumps["day"] = pumps["date"].dt.date
        pump_counts = pumps.groupby("day")["date"].count()
    else:
        pump_counts = pd.Series(dtype=int)

    if not dumps.empty:
        dumps["date"] = pd.to_datetime(dumps["date"], utc=True, errors="coerce")
        dumps["day"] = dumps["date"].dt.date
        dump_counts = dumps.groupby("day")["date"].count()
    else:
        dump_counts = pd.Series(dtype=int)

    days = sorted(set(pump_counts.index).union(dump_counts.index))
    if not days:
        print("No events to plot per-day counts for.")
        return

    pump_y = [pump_counts.get(d, 0) for d in days]
    dump_y = [dump_counts.get(d, 0) for d in days]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(days, pump_y, label="Pumps per day")
    ax.plot(days, dump_y, label="Dumps per day")

    ax.set_xlabel("Day (UTC)")
    ax.set_ylabel("Count")
    ax.set_title("Pump/Dump Counts Per Day")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved events-per-day plot to {out_path}")


# ---------- FIRST MONTH VERSIONS ----------

def _first_month_mask(df):
    """Return mask for rows within the first month of data."""
    min_date = df["date"].min().normalize()
    cutoff = min_date + pd.DateOffset(months=1)
    return (df["date"] >= min_date) & (df["date"] < cutoff)


def plot_price_with_events_first_month(
    price_csv="Coinbase_PEPEUSD_5min.csv",
    pumps_csv="pepe_pumps_by_candle.csv",
    dumps_csv="pepe_dumps_by_candle.csv",
    out_path="pepe_price_with_events_first_month.png"
):
    df, pumps, dumps = _load_data(price_csv, pumps_csv, dumps_csv)

    mask_price = _first_month_mask(df)
    df_m = df.loc[mask_price]

    # filter pumps/dumps to same window
    if not pumps.empty:
        pumps_m = pumps.loc[_first_month_mask(pumps)]
    else:
        pumps_m = pumps

    if not dumps.empty:
        dumps_m = dumps.loc[_first_month_mask(dumps)]
    else:
        dumps_m = dumps

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_m["date"], df_m["close"], label="PEPE close (5m)", color="blue", linewidth=1.2, zorder=1)

    if not pumps_m.empty:
        ax.scatter(
            pumps_m["date"], pumps_m["close"],
            marker="^", s=40, color="green",
            edgecolor="black", linewidth=0.5,
            label="Pump candles",
            zorder=3
        )

    if not dumps_m.empty:
        ax.scatter(
            dumps_m["date"], dumps_m["close"],
            marker="v", s=40, color="red",
            edgecolor="black", linewidth=0.5,
            label="Dump candles",
            zorder=3
        )

    ax.set_xlabel("Time (first month)")
    ax.set_ylabel("Price (USD)")
    ax.set_title("PEPE 5m Price with Pump/Dump Markers (First Month)")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved first-month price-with-events plot to {out_path}")


def plot_events_per_day_first_month(
    pumps_csv="pepe_pumps_by_candle.csv",
    dumps_csv="pepe_dumps_by_candle.csv",
    out_path="pepe_events_per_day_first_month.png"
):
    pumps = pd.read_csv(pumps_csv)
    dumps = pd.read_csv(dumps_csv)

    if not pumps.empty:
        pumps["date"] = pd.to_datetime(pumps["date"], utc=True, errors="coerce")
        mask_p = _first_month_mask(pumps)
        pumps = pumps.loc[mask_p]
        pumps["day"] = pumps["date"].dt.date
        pump_counts = pumps.groupby("day")["date"].count()
    else:
        pump_counts = pd.Series(dtype=int)

    if not dumps.empty:
        dumps["date"] = pd.to_datetime(dumps["date"], utc=True, errors="coerce")
        mask_d = _first_month_mask(dumps)
        dumps = dumps.loc[mask_d]
        dumps["day"] = dumps["date"].dt.date
        dump_counts = dumps.groupby("day")["date"].count()
    else:
        dump_counts = pd.Series(dtype=int)

    days = sorted(set(pump_counts.index).union(dump_counts.index))
    if not days:
        print("No events to plot per-day counts for (first month).")
        return

    pump_y = [pump_counts.get(d, 0) for d in days]
    dump_y = [dump_counts.get(d, 0) for d in days]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(days, pump_y, label="Pumps per day")
    ax.plot(days, dump_y, label="Dumps per day")

    ax.set_xlabel("Day (UTC, first month)")
    ax.set_ylabel("Count")
    ax.set_title("Pump/Dump Counts Per Day (First Month)")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved first-month events-per-day plot to {out_path}")


if __name__ == "__main__":
    # Full-history plots
    plot_price_with_events()
    plot_events_per_day()

    # First-month-only plots
    plot_price_with_events_first_month()
    plot_events_per_day_first_month()
