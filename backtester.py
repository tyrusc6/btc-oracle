"""
BTC Oracle - Backtester V4
Pulls max data from Kraken using 1-min candles (720 per request),
resamples to 15-min, tests Bollinger Extremes + Anti-Momentum strategies.
"""

import json, os, time, requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta


def fetch_1min_candles():
    """Pull 1-minute candles in chunks to get weeks of data."""
    print("Fetching 1-minute BTC candles from Kraken (multiple chunks)...")
    all_candles = []
    
    # Start from now and go backwards in 12-hour chunks (720 1-min candles each)
    end_time = int(datetime.now(timezone.utc).timestamp())
    
    for chunk in range(20):  # 20 chunks × 12 hours = 10 days of 1-min data
        since = end_time - ((chunk + 1) * 720 * 60)
        try:
            resp = requests.get(f"https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=1&since={since}", timeout=15)
            data = resp.json()
            if data.get("error") and len(data["error"]) > 0:
                print(f"  Kraken error: {data['error']}")
                break
            candles = data.get("result", {}).get("XXBTZUSD", [])
            if candles:
                all_candles.extend(candles)
                dt = datetime.fromtimestamp(int(candles[0][0]), tz=timezone.utc)
                print(f"  Chunk {chunk+1}: {len(candles)} candles from {dt.strftime('%Y-%m-%d %H:%M')}")
            time.sleep(1.5)  # rate limit
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(3)
    
    if not all_candles:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_candles, columns=["ts","o","h","l","c","vwap","vol","cnt"])
    for col in ["o","h","l","c","vol"]:
        df[col] = df[col].astype(float)
    df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="s", utc=True)
    df = df.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
    
    # Resample to 15-min candles
    print(f"\n  Raw: {len(df)} 1-min candles")
    df = df.set_index("ts")
    resampled = df.resample("15min").agg({
        "o": "first", "h": "max", "l": "min", "c": "last", "vol": "sum"
    }).dropna().reset_index()
    
    print(f"  Resampled: {len(resampled)} 15-min candles")
    print(f"  From {resampled['ts'].iloc[0]} to {resampled['ts'].iloc[-1]}")
    return resampled


def fetch_15min_direct():
    """Also pull 15-min candles directly for comparison."""
    print("\nAlso fetching 15-min candles directly...")
    try:
        resp = requests.get("https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=15", timeout=15)
        data = resp.json()
        candles = data.get("result", {}).get("XXBTZUSD", [])
        if candles:
            df = pd.DataFrame(candles, columns=["ts","o","h","l","c","vwap","vol","cnt"])
            for col in ["o","h","l","c","vol"]:
                df[col] = df[col].astype(float)
            df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="s", utc=True)
            print(f"  Got {len(df)} 15-min candles directly")
            return df
    except:
        pass
    return pd.DataFrame()


def calc_rsi(prices, period=14):
    if len(prices) < period + 1: return None
    d = np.diff(prices)
    g = np.where(d > 0, d, 0)
    l = np.where(d < 0, -d, 0)
    ag = np.mean(g[:period])
    al = np.mean(l[:period])
    for i in range(period, len(g)):
        ag = (ag * (period-1) + g[i]) / period
        al = (al * (period-1) + l[i]) / period
    if al == 0: return 100.0 if ag > 0 else 50.0
    return 100 - (100 / (1 + ag/al))


def calc_bb(prices, period=20, std=2):
    if len(prices) < period: return None, None, None
    s = pd.Series(prices)
    mid = float(s.rolling(period).mean().iloc[-1])
    sd = float(s.rolling(period).std().iloc[-1])
    return mid + std*sd, mid, mid - std*sd


# ========== STRATEGIES ==========

def strategy_bollinger_extremes(df, idx):
    """Mean reversion at Bollinger Band extremes."""
    if idx < 25: return None, None, False
    p = df["c"].values[:idx+1].astype(float)
    upper, mid, lower = calc_bb(p)
    if upper is None: return None, None, False
    
    current = p[-1]
    bb_pos = (current - lower) / (upper - lower) if upper != lower else 0.5
    
    if bb_pos > 0.85:
        # Near upper band - expect pullback
        confidence = min(0.95, 0.6 + (bb_pos - 0.85) * 2.0)
        return "DOWN", confidence, True
    elif bb_pos < 0.15:
        # Near lower band - expect bounce
        confidence = min(0.95, 0.6 + (0.15 - bb_pos) * 2.0)
        return "UP", confidence, True
    return None, None, False


def strategy_anti_momentum(df, idx):
    """Fade recent momentum - bet against last 3 candles."""
    if idx < 5: return None, None, False
    p = df["c"].values[:idx+1].astype(float)
    
    chg3 = p[-1] - p[-4] if len(p) >= 4 else 0
    chg1 = p[-1] - p[-2] if len(p) >= 2 else 0
    
    # Both short and medium momentum must agree for signal
    if chg3 > 0 and chg1 > 0:
        # Been going up - bet on reversal
        strength = abs(chg3) / p[-1] * 100  # as pct
        confidence = min(0.85, 0.55 + strength * 5)
        return "DOWN", confidence, True
    elif chg3 < 0 and chg1 < 0:
        # Been going down - bet on reversal
        strength = abs(chg3) / p[-1] * 100
        confidence = min(0.85, 0.55 + strength * 5)
        return "UP", confidence, True
    return None, None, False


def strategy_combined(df, idx):
    """Bollinger extremes + anti-momentum combined. Only trade when both agree."""
    bb_sig, bb_conf, bb_trade = strategy_bollinger_extremes(df, idx)
    am_sig, am_conf, am_trade = strategy_anti_momentum(df, idx)
    
    # Both must fire and agree
    if bb_trade and am_trade and bb_sig == am_sig:
        conf = (bb_conf + am_conf) / 2 + 0.05  # bonus for agreement
        return bb_sig, min(0.95, conf), True
    
    # Bollinger alone at extreme (bb_pos < 0.1 or > 0.9)
    if bb_trade and bb_conf >= 0.75:
        return bb_sig, bb_conf, True
    
    return None, None, False


def strategy_rsi_extreme(df, idx):
    """RSI at extreme levels - mean reversion."""
    if idx < 20: return None, None, False
    p = df["c"].values[:idx+1].astype(float)
    rsi = calc_rsi(p)
    if rsi is None: return None, None, False
    
    if rsi > 75:
        conf = min(0.90, 0.55 + (rsi - 75) * 0.02)
        return "DOWN", conf, True
    elif rsi < 25:
        conf = min(0.90, 0.55 + (25 - rsi) * 0.02)
        return "UP", conf, True
    return None, None, False


def strategy_mega_combined(df, idx):
    """All mean reversion signals combined. Most selective."""
    bb_sig, bb_conf, bb_trade = strategy_bollinger_extremes(df, idx)
    am_sig, am_conf, am_trade = strategy_anti_momentum(df, idx)
    rsi_sig, rsi_conf, rsi_trade = strategy_rsi_extreme(df, idx)
    
    signals = []
    if bb_trade: signals.append((bb_sig, bb_conf))
    if am_trade: signals.append((am_sig, am_conf))
    if rsi_trade: signals.append((rsi_sig, rsi_conf))
    
    if len(signals) < 2:
        return None, None, False
    
    # At least 2 must agree
    up_count = sum(1 for s, c in signals if s == "UP")
    down_count = sum(1 for s, c in signals if s == "DOWN")
    
    if up_count >= 2:
        avg_conf = np.mean([c for s, c in signals if s == "UP"])
        return "UP", min(0.95, avg_conf + 0.05), True
    elif down_count >= 2:
        avg_conf = np.mean([c for s, c in signals if s == "DOWN"])
        return "DOWN", min(0.95, avg_conf + 0.05), True
    
    return None, None, False


# ========== RUN ==========

def test_strategy(df, name, strategy_func):
    total = len(df)
    wins = losses = skips = 0
    results = []
    
    for i in range(25, total - 1):
        sig, conf, trade = strategy_func(df, i)
        if not trade:
            skips += 1
            continue
        
        cur = float(df.iloc[i]["c"])
        nxt = float(df.iloc[i+1]["c"])
        went_up = nxt > cur
        outcome = "WIN" if (went_up == (sig == "UP")) else "LOSS"
        
        if outcome == "WIN": wins += 1
        else: losses += 1
        results.append({"sig": sig, "conf": round(conf, 3), "outcome": outcome})
    
    traded = wins + losses
    wr = wins / traded * 100 if traded > 0 else 0
    print(f"  {name:30s} | WR: {wr:5.1f}% | {wins:3d}W/{losses:3d}L | {traded:4d} trades ({traded/(total-26)*100:.0f}%)")
    return {"name": name, "wr": round(wr, 1), "wins": wins, "losses": losses, "trades": traded, "skips": skips}


def load_historical_15m(path):
    """Load historical trades and resample to 15-min candles."""
    print(f"Loading historical data from {path}...")
    trades = pd.read_parquet(path)
    print(f"  Loaded {len(trades):,} trades")

    trades["datetime"] = pd.to_datetime(trades["timestamp"], unit="ms", utc=True)
    trades = trades.set_index("datetime").sort_index()

    candles = trades["price"].resample("15min").ohlc()
    candles["vol"] = trades["qty"].resample("15min").sum()
    candles = candles.dropna(subset=["open"])
    candles = candles.reset_index()
    candles = candles.rename(columns={"open": "o", "high": "h", "low": "l", "close": "c", "datetime": "ts"})
    print(f"  Built {len(candles)} 15-min candles")
    print(f"  From {candles['ts'].iloc[0]} to {candles['ts'].iloc[-1]}")
    return candles


def run():
    print("=" * 60)
    print("BACKTESTER V4 - MEAN REVERSION STRATEGIES")
    print("=" * 60)

    # Try historical data first (much more samples)
    hist_path = ".tmp/xbtusd_trades_2024-09-01_2025-03-01.parquet"
    hist_path_2m = ".tmp/xbtusd_trades_2025-01-01_2025-03-01.parquet"

    if os.path.exists(hist_path):
        df = load_historical_15m(hist_path)
    elif os.path.exists(hist_path_2m):
        df = load_historical_15m(hist_path_2m)
    else:
        print("No historical data found, fetching live from Kraken...")
        df_1m = fetch_1min_candles()
        df_15m = fetch_15min_direct()
        df = df_1m if not df_1m.empty and len(df_1m) > len(df_15m) else df_15m

    if df.empty:
        print("No data!"); return

    print(f"\nTesting strategies on {len(df)} candles")
    print("=" * 60)
    
    strategies = [
        ("Bollinger Extremes", strategy_bollinger_extremes),
        ("Anti-Momentum", strategy_anti_momentum),
        ("RSI Extreme", strategy_rsi_extreme),
        ("BB + Anti-Mom Combined", strategy_combined),
        ("Mega Combined (2 of 3)", strategy_mega_combined),
    ]
    
    results = []
    for name, func in strategies:
        r = test_strategy(df, name, func)
        results.append(r)
    
    results.sort(key=lambda x: x["wr"], reverse=True)
    
    print(f"\n{'='*60}")
    print("RANKING BY WIN RATE")
    print("=" * 60)
    for i, r in enumerate(results):
        print(f"  #{i+1}: {r['name']:30s} {r['wr']}% ({r['wins']}/{r['trades']})")
    
    print(f"\n{'='*60}")
    best = results[0]
    print(f">>> BEST: {best['name']} at {best['wr']}% ({best['wins']}/{best['trades']} trades)")
    print("=" * 60)
    
    with open("backtest_v4_meanrev.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to backtest_v4_meanrev.json")


if __name__ == "__main__":
    run()
