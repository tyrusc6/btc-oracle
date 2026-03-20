"""
historical_data.py — Download historical BTC/USD trade data from Kraken REST API.

Downloads tick-by-tick trade data for training the ML model.
No API key required (public endpoint).

Usage:
    python historical_data.py
    python historical_data.py --start 2025-01-01 --end 2025-03-01

Output:
    .tmp/xbtusd_trades_2025-01-01_2025-03-01.parquet
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

KRAKEN_BASE = "https://api.kraken.com/0/public"
KRAKEN_PAIR = "XBTUSD"
SLEEP_BETWEEN_CALLS = 1.2


def _date_to_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _date_to_ns(date_str: str) -> int:
    return _date_to_ms(date_str) * 1_000_000


def download_trades(pair: str, start_date: str, end_date: str, output_path: str) -> pd.DataFrame:
    """
    Download all trades for `pair` between start_date and end_date from Kraken.
    Paginates using 'since' (nanoseconds).
    Saves to output_path as Parquet.
    """
    start_ns = _date_to_ns(start_date)
    end_ms = _date_to_ms(end_date)
    end_ns = end_ms * 1_000_000

    if start_ns >= end_ns:
        print(f"ERROR: start_date ({start_date}) must be before end_date ({end_date})")
        sys.exit(1)

    print(f"Downloading {pair} trades from Kraken: {start_date} -> {end_date}")
    print(f"Output: {output_path}")

    all_trades = []
    since_ns = start_ns

    days = (end_ms - _date_to_ms(start_date)) / 86_400_000
    estimated_calls = max(1, int(days * 100))
    pbar = tqdm(total=estimated_calls, unit="call", desc="Downloading", ascii=True)

    stalled_count = 0

    while True:
        url = f"{KRAKEN_BASE}/Trades"
        params = {"pair": pair, "since": since_ns, "count": 1000}

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"\nNetwork error: {e}. Retrying in 10s...")
            time.sleep(10)
            continue

        if data.get("error"):
            errs = data["error"]
            print(f"\nKraken API error: {errs}. Retrying in 15s...")
            time.sleep(15)
            continue

        result = data["result"]
        pair_key = next(k for k in result if k != "last")
        trades_raw = result[pair_key]
        last_ns = int(result["last"])

        for t in trades_raw:
            ts_ms = int(float(t[2]) * 1000)
            if ts_ms >= end_ms:
                last_ns = end_ns + 1
                break
            all_trades.append({
                "timestamp": ts_ms,
                "price": float(t[0]),
                "qty": float(t[1]),
                "side": "buy" if t[3] == "b" else "sell",
            })

        pbar.update(1)

        if last_ns >= end_ns:
            break

        if last_ns <= since_ns:
            stalled_count += 1
            if stalled_count >= 3:
                print(f"\nWARN: No progress after 3 attempts at since={since_ns}. Stopping.")
                break
            time.sleep(5)
        else:
            stalled_count = 0
            since_ns = last_ns

        time.sleep(SLEEP_BETWEEN_CALLS)

    pbar.close()

    if not all_trades:
        print("ERROR: No trades downloaded. Check pair name and date range.")
        sys.exit(1)

    df = pd.DataFrame(all_trades)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["price"] = df["price"].astype("float32")
    df["qty"] = df["qty"].astype("float32")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, compression="snappy", index=False)

    print(f"\nDownloaded {len(df):,} trades")
    print(f"Date range: {pd.to_datetime(df['timestamp'].min(), unit='ms', utc=True)} -> "
          f"{pd.to_datetime(df['timestamp'].max(), unit='ms', utc=True)}")
    side_counts = df["side"].value_counts()
    print(f"Buy trades: {side_counts.get('buy', 0):,} | Sell trades: {side_counts.get('sell', 0):,}")
    print(f"Saved to: {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)")

    return df


def load_or_download(pair: str = KRAKEN_PAIR, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Load from cache if available, otherwise download from Kraken."""
    start_date = start_date or os.getenv("BACKTEST_START_DATE", "2025-01-01")
    end_date = end_date or os.getenv("BACKTEST_END_DATE", "2025-03-01")

    fname = f"{pair.lower()}_trades_{start_date}_{end_date}.parquet"
    output_path = os.path.join(".tmp", fname)

    if os.path.exists(output_path):
        print(f"Loading cached data: {output_path}")
        df = pd.read_parquet(output_path)
        print(f"Loaded {len(df):,} trades")
        print(f"Date range: {pd.to_datetime(df['timestamp'].min(), unit='ms', utc=True)} -> "
              f"{pd.to_datetime(df['timestamp'].max(), unit='ms', utc=True)}")
        return df

    return download_trades(pair, start_date, end_date, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download BTC/USD trades from Kraken")
    parser.add_argument("--pair", default=KRAKEN_PAIR, help="Kraken pair (default: XBTUSD)")
    parser.add_argument("--start", default=os.getenv("BACKTEST_START_DATE", "2025-01-01"))
    parser.add_argument("--end", default=os.getenv("BACKTEST_END_DATE", "2025-03-01"))
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    args = parser.parse_args()

    fname = f"{args.pair.lower()}_trades_{args.start}_{args.end}.parquet"
    output_path = os.path.join(".tmp", fname)

    if args.force and os.path.exists(output_path):
        os.remove(output_path)
        print(f"Removed cached file: {output_path}")

    load_or_download(args.pair, args.start, args.end)
