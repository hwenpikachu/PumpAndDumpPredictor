import pandas as pd
import requests

# Testing api call

BASE = "https://api.exchange.coinbase.com"  # correct base

def fetch_daily_data(symbol: str):
    """
    symbol like 'BTC/USD' or 'BTC-USD'
    writes Coinbase_<PAIR>_dailydata.csv with daily candles
    """
    # normalize to product_id form e.g., BTC-USD
    if "/" in symbol:
        base, quote = symbol.split("/")
        product_id = f"{base}-{quote}"
    else:
        product_id = symbol

    url = f"{BASE}/products/{product_id}/candles"
    params = {"granularity": 86400}  # 1 day

    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        print("Did not receive OK response from Coinbase API")
        print(r.status_code, r.text[:300])
        return

    # response is list of [time, low, high, open, close, volume]
    rows = r.json()
    if not rows:
        print("No candle data returned.")
        return

    cols = ["unix", "low", "high", "open", "close", "volume"]
    df = pd.DataFrame(rows, columns=cols)

    # API returns newest-first; make oldest-first and add helpful cols
    df.sort_values("unix", inplace=True)
    df["date"] = pd.to_datetime(df["unix"], unit="s", utc=True).dt.tz_convert("America/New_York")
    # fiat volume approximation (close * base-volume)
    df["vol_fiat"] = df["volume"] * df["close"]

    outname = f"Coinbase_{product_id.replace('-', '')}_dailydata.csv"
    df.to_csv(outname, index=False)
    print(f"Wrote {len(df)} rows to {outname}")
    

if __name__ == "__main__":
    # example
    fetch_daily_data("BTC/USD")
