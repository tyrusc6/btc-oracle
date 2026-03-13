"""
BTC Oracle - Technical Indicators (Production Grade)
Uses Kraken OHLC candles for accurate calculations.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import requests
import db


def fetch_recent_ticks(minutes=60):
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    data = db.select("tick_data", f"recorded_at=gte.{cutoff}&order=recorded_at.asc&limit=2000")
    if data:
        df = pd.DataFrame(data)
        df["recorded_at"] = pd.to_datetime(df["recorded_at"])
        return df
    return pd.DataFrame()


def get_current_price():
    """Get current BTC price directly from Kraken API."""
    try:
        resp = requests.get("https://api.kraken.com/0/public/Ticker?pair=XBTUSD", timeout=10)
        data = resp.json()
        if data.get("result"):
            return float(data["result"]["XXBTZUSD"]["c"][0])
    except:
        pass
    return None


def fetch_kraken_ohlc(interval=1, count=200):
    """Fetch OHLC candles from Kraken."""
    try:
        resp = requests.get(f"https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval={interval}", timeout=10)
        data = resp.json()
        if data.get("result"):
            candles = data["result"].get("XXBTZUSD", [])
            if candles:
                df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "vwap", "volume", "count"])
                for col in ["open", "high", "low", "close", "vwap", "volume"]:
                    df[col] = df[col].astype(float)
                df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
                return df.tail(count)
    except Exception as e:
        print(f"  Error fetching Kraken OHLC: {e}")
    return pd.DataFrame()


def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return None
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    return 100 - (100 / (1 + avg_gain / avg_loss))


def calculate_macd(prices, fast=12, slow=26, signal=9):
    if len(prices) < slow + signal:
        return None, None, None
    s = pd.Series(prices)
    macd_line = s.ewm(span=fast, adjust=False).mean() - s.ewm(span=slow, adjust=False).mean()
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float((macd_line - signal_line).iloc[-1])


def calculate_bollinger(prices, period=20, std_dev=2):
    if len(prices) < period:
        return None, None, None
    s = pd.Series(prices)
    mid = float(s.rolling(period).mean().iloc[-1])
    std = float(s.rolling(period).std().iloc[-1])
    return mid + std_dev * std, mid, mid - std_dev * std


def calculate_ema(prices, period):
    if len(prices) < period:
        return None
    return float(pd.Series(prices).ewm(span=period, adjust=False).mean().iloc[-1])


def calculate_stoch_rsi(prices, period=14):
    """Optimized StochRSI - calculate RSI series efficiently."""
    if len(prices) < period * 2 + 3:
        return None, None
    
    # Calculate full RSI series at once
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    rsi_values = []
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi_values.append(100.0 if avg_gain > 0 else 50.0)
        else:
            rsi_values.append(100 - (100 / (1 + avg_gain / avg_loss)))
    
    if len(rsi_values) < period:
        return None, None
    
    s = pd.Series(rsi_values)
    lowest = s.rolling(period).min()
    highest = s.rolling(period).max()
    diff = highest - lowest
    diff = diff.replace(0, np.nan)
    stoch = ((s - lowest) / diff).rolling(3).mean()
    d = stoch.rolling(3).mean()
    
    k_val = float(stoch.iloc[-1]) * 100 if pd.notna(stoch.iloc[-1]) else None
    d_val = float(d.iloc[-1]) * 100 if pd.notna(d.iloc[-1]) else None
    return round(k_val, 2) if k_val is not None else None, round(d_val, 2) if d_val is not None else None


def calculate_atr(highs, lows, closes, period=14):
    if len(closes) < period + 1:
        return None
    tr = [max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1])) for i in range(1, len(closes))]
    return float(np.mean(tr[-period:])) if len(tr) >= period else None


def calculate_obv_trend(prices, volumes):
    if len(prices) < 10:
        return None
    obv = [0]
    for i in range(1, len(prices)):
        v = volumes[i] if not np.isnan(volumes[i]) and volumes[i] > 0 else 0
        if prices[i] > prices[i-1]:
            obv.append(obv[-1] + v)
        elif prices[i] < prices[i-1]:
            obv.append(obv[-1] - v)
        else:
            obv.append(obv[-1])
    return "RISING" if np.mean(obv[-5:]) > np.mean(obv[-10:-5]) else "FALLING" if len(obv) >= 10 else None


def detect_trend(prices):
    """Multi-method trend detection."""
    if len(prices) < 20:
        return "UNKNOWN", 0
    
    current = prices[-1]
    sma_10 = np.mean(prices[-10:])
    sma_20 = np.mean(prices[-20:])
    slope = np.polyfit(np.arange(20), prices[-20:], 1)[0]
    
    recent_5 = prices[-5:]
    prev_5 = prices[-10:-5]
    higher_highs = max(recent_5) > max(prev_5)
    higher_lows = min(recent_5) > min(prev_5)
    
    pct = ((prices[-1] - prices[-min(30, len(prices))]) / prices[-min(30, len(prices))]) * 100
    
    score = 0
    score += 1 if current > sma_10 else -1
    score += 1 if current > sma_20 else -1
    score += 1 if sma_10 > sma_20 else -1
    score += 1 if slope > 0 else -1
    score += 1 if higher_highs and higher_lows else (-1 if not higher_highs and not higher_lows else 0)
    
    if score >= 3: trend = "STRONG_UPTREND"
    elif score >= 1: trend = "UPTREND"
    elif score <= -3: trend = "STRONG_DOWNTREND"
    elif score <= -1: trend = "DOWNTREND"
    else: trend = "SIDEWAYS"
    
    return trend, round(pct, 4)


def get_all_indicators():
    print("Calculating indicators...")
    
    df_1m = fetch_kraken_ohlc(interval=1, count=200)
    df_5m = fetch_kraken_ohlc(interval=5, count=100)
    
    if df_1m.empty or len(df_1m) < 30:
        print("  Not enough OHLC data from Kraken")
        return None
    
    prices = df_1m["close"].values.astype(float)
    highs = df_1m["high"].values.astype(float)
    lows = df_1m["low"].values.astype(float)
    volumes = df_1m["volume"].values.astype(float)
    current_price = float(prices[-1])
    
    rsi = calculate_rsi(prices)
    macd, macd_sig, macd_hist = calculate_macd(prices)
    bb_upper, bb_middle, bb_lower = calculate_bollinger(prices)
    stoch_k, stoch_d = calculate_stoch_rsi(prices)
    atr = calculate_atr(highs, lows, prices)
    obv = calculate_obv_trend(prices, volumes)
    roc = round(((prices[-1] - prices[-11]) / prices[-11]) * 100, 4) if len(prices) >= 11 else None
    vwap = float(np.sum(prices * volumes) / np.sum(volumes)) if np.sum(volumes) > 0 else None
    ema_9 = calculate_ema(prices, 9)
    ema_21 = calculate_ema(prices, 21)
    sma_50 = float(np.mean(prices[-50:])) if len(prices) >= 50 else None
    momentum = float(prices[-1] - prices[-11]) if len(prices) >= 11 else None
    
    trend_1m, trend_pct = detect_trend(prices)
    trend_5m = "UNKNOWN"
    if not df_5m.empty and len(df_5m) >= 20:
        trend_5m, _ = detect_trend(df_5m["close"].values.astype(float))
    
    bb_position = None
    if bb_upper is not None and bb_lower is not None and bb_upper != bb_lower:
        bb_position = round((current_price - bb_lower) / (bb_upper - bb_lower), 4)
    
    indicators = {
        "current_price": current_price,
        "rsi": round(rsi, 2) if rsi is not None else None,
        "macd": round(macd, 4) if macd is not None else None,
        "macd_signal": round(macd_sig, 4) if macd_sig is not None else None,
        "macd_histogram": round(macd_hist, 4) if macd_hist is not None else None,
        "bollinger_upper": round(bb_upper, 2) if bb_upper is not None else None,
        "bollinger_middle": round(bb_middle, 2) if bb_middle is not None else None,
        "bollinger_lower": round(bb_lower, 2) if bb_lower is not None else None,
        "bollinger_position": bb_position,
        "ema_9": round(ema_9, 2) if ema_9 is not None else None,
        "ema_21": round(ema_21, 2) if ema_21 is not None else None,
        "sma_50": round(sma_50, 2) if sma_50 is not None else None,
        "ema_crossover": "BULLISH" if ema_9 is not None and ema_21 is not None and ema_9 > ema_21 else "BEARISH" if ema_9 is not None and ema_21 is not None else None,
        "momentum": round(momentum, 2) if momentum is not None else None,
        "rate_of_change": roc,
        "vwap": round(vwap, 2) if vwap is not None else None,
        "price_vs_vwap": "ABOVE" if vwap is not None and current_price > vwap else "BELOW" if vwap is not None else None,
        "stoch_rsi_k": stoch_k,
        "stoch_rsi_d": stoch_d,
        "atr": round(atr, 2) if atr is not None else None,
        "obv_trend": obv,
        "trend_1m": trend_1m,
        "trend_5m": trend_5m,
        "trend_pct_change": trend_pct,
        "volume_24h": float(volumes[-1]) if len(volumes) > 0 else None,
        "tick_count": len(df_1m),
        "data_span_minutes": len(df_1m)
    }
    
    print(f"  ${current_price:,.2f} | RSI: {indicators['rsi']} | Trend: {trend_1m}/{trend_5m} | OBV: {obv}")
    return indicators


if __name__ == "__main__":
    r = get_all_indicators()
    if r:
        for k, v in r.items():
            print(f"  {k}: {v}")
