"""
BTC Oracle - Technical Indicators Calculator
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import db


def fetch_recent_ticks(minutes=60):
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    params = f"recorded_at=gte.{cutoff}&order=recorded_at.asc"
    data = db.select("tick_data", params)
    if data:
        df = pd.DataFrame(data)
        df["recorded_at"] = pd.to_datetime(df["recorded_at"])
        return df
    return pd.DataFrame()


def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return None
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_macd(prices, fast=12, slow=26, signal=9):
    if len(prices) < slow + signal:
        return None, None, None
    series = pd.Series(prices)
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float(histogram.iloc[-1])


def calculate_bollinger_bands(prices, period=20, std_dev=2):
    if len(prices) < period:
        return None, None, None
    series = pd.Series(prices)
    middle = series.rolling(window=period).mean().iloc[-1]
    std = series.rolling(window=period).std().iloc[-1]
    return float(middle + std_dev * std), float(middle), float(middle - std_dev * std)


def calculate_ema(prices, period):
    if len(prices) < period:
        return None
    return float(pd.Series(prices).ewm(span=period, adjust=False).mean().iloc[-1])


def calculate_sma(prices, period):
    if len(prices) < period:
        return None
    return float(np.mean(prices[-period:]))


def calculate_momentum(prices, period=10):
    if len(prices) < period + 1:
        return None
    return float(prices[-1] - prices[-period - 1])


def calculate_vwap(prices, volumes):
    if len(prices) == 0 or len(volumes) == 0:
        return None
    prices = np.array(prices, dtype=float)
    volumes = np.array(volumes, dtype=float)
    valid = ~np.isnan(volumes) & (volumes > 0)
    if not np.any(valid):
        return None
    return float(np.sum(prices[valid] * volumes[valid]) / np.sum(volumes[valid]))


def calculate_stoch_rsi(prices, period=14, smooth_k=3, smooth_d=3):
    """Stochastic RSI - more sensitive than regular RSI."""
    if len(prices) < period * 2:
        return None, None
    rsi_values = []
    for i in range(period, len(prices)):
        rsi = calculate_rsi(prices[:i+1], period)
        if rsi is not None:
            rsi_values.append(rsi)
    if len(rsi_values) < period:
        return None, None
    series = pd.Series(rsi_values)
    lowest = series.rolling(period).min()
    highest = series.rolling(period).max()
    stoch_rsi = (series - lowest) / (highest - lowest)
    k = stoch_rsi.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    return round(float(k.iloc[-1]) * 100, 2) if not pd.isna(k.iloc[-1]) else None, \
           round(float(d.iloc[-1]) * 100, 2) if not pd.isna(d.iloc[-1]) else None


def calculate_atr(prices, period=14):
    """Average True Range - volatility measure."""
    if len(prices) < period + 1:
        return None
    true_ranges = []
    for i in range(1, len(prices)):
        tr = abs(prices[i] - prices[i-1])
        true_ranges.append(tr)
    if len(true_ranges) < period:
        return None
    return float(np.mean(true_ranges[-period:]))


def calculate_obv_trend(prices, volumes):
    """On Balance Volume trend direction."""
    if len(prices) < 10 or len(volumes) < 10:
        return None
    obv = [0]
    for i in range(1, len(prices)):
        vol = volumes[i] if volumes[i] and not np.isnan(volumes[i]) else 0
        if prices[i] > prices[i-1]:
            obv.append(obv[-1] + vol)
        elif prices[i] < prices[i-1]:
            obv.append(obv[-1] - vol)
        else:
            obv.append(obv[-1])
    # OBV trend: compare last 5 avg to previous 5 avg
    if len(obv) >= 10:
        recent_avg = np.mean(obv[-5:])
        prev_avg = np.mean(obv[-10:-5])
        return "RISING" if recent_avg > prev_avg else "FALLING"
    return None


def calculate_price_rate_of_change(prices, period=10):
    """Rate of change - speed of price movement."""
    if len(prices) < period + 1:
        return None
    return round(((prices[-1] - prices[-period-1]) / prices[-period-1]) * 100, 4)


def get_all_indicators():
    print("Calculating indicators...")
    df = fetch_recent_ticks(minutes=120)

    if df.empty or len(df) < 30:
        print(f"Not enough tick data yet ({len(df)} ticks). Need at least 30.")
        return None

    prices = df["price"].values.astype(float)
    vol_array = []
    for v in df["volume"].values:
        try:
            vol_array.append(float(v) if v is not None else 0.0)
        except:
            vol_array.append(0.0)
    vol_array = np.array(vol_array)

    current_price = float(prices[-1])
    rsi = calculate_rsi(prices)
    macd, macd_sig, macd_hist = calculate_macd(prices)
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(prices)
    stoch_k, stoch_d = calculate_stoch_rsi(prices)
    atr = calculate_atr(prices)
    obv_trend = calculate_obv_trend(prices, vol_array)
    roc = calculate_price_rate_of_change(prices)

    # Bollinger Band position (0 = at lower, 1 = at upper)
    bb_position = None
    if bb_upper and bb_lower and bb_upper != bb_lower:
        bb_position = round((current_price - bb_lower) / (bb_upper - bb_lower), 4)

    indicators = {
        "current_price": current_price,
        "rsi": round(rsi, 2) if rsi else None,
        "macd": round(macd, 4) if macd else None,
        "macd_signal": round(macd_sig, 4) if macd_sig else None,
        "macd_histogram": round(macd_hist, 4) if macd_hist else None,
        "bollinger_upper": round(bb_upper, 2) if bb_upper else None,
        "bollinger_middle": round(bb_middle, 2) if bb_middle else None,
        "bollinger_lower": round(bb_lower, 2) if bb_lower else None,
        "bollinger_position": bb_position,
        "ema_9": round(calculate_ema(prices, 9), 2) if calculate_ema(prices, 9) else None,
        "ema_21": round(calculate_ema(prices, 21), 2) if calculate_ema(prices, 21) else None,
        "sma_50": round(calculate_sma(prices, 50), 2) if calculate_sma(prices, 50) else None,
        "momentum": round(calculate_momentum(prices), 2) if calculate_momentum(prices) else None,
        "vwap": round(calculate_vwap(prices, vol_array), 2) if calculate_vwap(prices, vol_array) else None,
        "stoch_rsi_k": stoch_k,
        "stoch_rsi_d": stoch_d,
        "atr": round(atr, 2) if atr else None,
        "obv_trend": obv_trend,
        "rate_of_change": roc,
        "price_vs_vwap": "ABOVE" if calculate_vwap(prices, vol_array) and current_price > calculate_vwap(prices, vol_array) else "BELOW" if calculate_vwap(prices, vol_array) else None,
        "ema_crossover": "BULLISH" if calculate_ema(prices, 9) and calculate_ema(prices, 21) and calculate_ema(prices, 9) > calculate_ema(prices, 21) else "BEARISH" if calculate_ema(prices, 9) and calculate_ema(prices, 21) else None,
        "volume_24h": float(vol_array[-1]) if len(vol_array) > 0 else None,
        "tick_count": len(df),
        "data_span_minutes": (df["recorded_at"].max() - df["recorded_at"].min()).total_seconds() / 60
    }

    print(f"  Price: ${current_price:,.2f} | RSI: {indicators['rsi']} | StochRSI: {stoch_k} | OBV: {obv_trend} | Ticks: {indicators['tick_count']}")
    return indicators


if __name__ == "__main__":
    result = get_all_indicators()
    if result:
        print("Indicators calculated!")
    else:
        print("Need more data. Run collector.py first.")
