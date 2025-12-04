import pandas as pd
import requests
from datetime import datetime, timedelta
import time

BASE = "https://api.exchange.coinbase.com"

def fetch_candles_with_timeout(product_id="FARTCOIN-USD", granularity=300, max_minutes=5):
    """
    Fetch 5-minute candles for the given product.
    Automatically stops after 'max_minutes' minutes (default 5).
    """

    all_rows = []

    # Timer start
    start_clock = time.time()
    timeout_secs = max_minutes * 60

    # End timestamp = now
    end_time = datetime.utcnow()
    delta = timedelta(seconds=granularity * 300)  # ~25 hours per request

    while True:
        # Check timeout
        if time.time() - start_clock > timeout_secs:
            print(f"\n TIMEOUT: Stopping after {max_minutes} minutes.\n")
            break

        # Calculate start for this window
        start_time = end_time - delta

        url = f"{BASE}/products/{product_id}/candles"
        params = {
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "granularity": granularity
        }

        r = requests.get(url, params=params)

        if r.status_code != 200:
            print("Error:", r.status_code, r.text[:200])
            break

        rows = r.json()

        # No more rows -> stop
        if not rows:
            print("No more data returned; stopping.")
            break

        all_rows.extend(rows)

        print(f"Fetched {len(rows)} rows (ending at {end_time}) — total: {len(all_rows)}")

        # Move window back
        end_time = start_time

        # Respect Coinbase rate limits
        time.sleep(0.35)

    # Build DataFrame
    df = pd.DataFrame(all_rows, columns=["unix", "low", "high", "open", "close", "volume"])
    df["date"] = pd.to_datetime(df["unix"], unit="s", utc=True)
    df = df.sort_values("unix")

    return df


if __name__ == "__main__":
    print("Fetching 5-minute FARTCOIN-USD candles (up to 5 minutes runtime)...\n")

    df = fetch_candles_with_timeout(
        product_id="FARTCOIN-USD",
        granularity=300,     # 5 minutes
        max_minutes=5        # <= Stop after 5 minutes
    )

    print(f"Fetched {len(df)} rows total.")

    df.to_csv("Coinbase_FARTUSD_5min.csv", index=False)
    print("\n Saved as Coinbase_FARTUSD_5min.csv")
