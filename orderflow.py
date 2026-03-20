"""
orderflow.py — Pure order flow computation functions.

Stateless: works identically on historical batch data and live rolling windows.

Data schema (Kraken normalized):
  timestamp  — Unix ms
  price      — float
  qty        — float (BTC quantity)
  side       — "buy" or "sell" (aggressor side, directly from Kraken)
"""

import numpy as np
import pandas as pd


def label_trade_side(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure 'side', 'buy_vol', and 'sell_vol' columns are present.

    Handles two input formats:
      - Kraken: 'side' column already present ("buy"/"sell") — used directly
      - Legacy: 'is_buyer_maker' column — mapped (False=buy, True=sell)
    """
    df = df.copy()
    if "side" not in df.columns:
        df["side"] = np.where(df["is_buyer_maker"], "sell", "buy")
    df["buy_vol"] = np.where(df["side"] == "buy", df["qty"], 0.0)
    df["sell_vol"] = np.where(df["side"] == "sell", df["qty"], 0.0)
    return df


def compute_cvd(df: pd.DataFrame, window_seconds: int = 300) -> pd.Series:
    """
    Cumulative Volume Delta over a rolling time window.

    CVD = cumulative sum of (buy_vol - sell_vol).
    Divergence between CVD direction and price direction is the core signal:
      - Price up + CVD down → distribution, expect reversal down
      - Price down + CVD up → absorption, expect bounce up
    """
    df = df.copy()
    if "buy_vol" not in df.columns:
        df = label_trade_side(df)

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("datetime").sort_index()

    df["delta"] = df["buy_vol"] - df["sell_vol"]

    window = f"{window_seconds}s"
    cvd = df["delta"].rolling(window, min_periods=1).sum()
    return cvd


def compute_delta_per_candle(df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """
    Resample trades into OHLCV candles with order flow metrics.

    Returns DataFrame with columns:
      open, high, low, close, volume, buy_vol, sell_vol,
      delta (buy_vol - sell_vol), delta_pct (delta / volume)
    """
    df = df.copy()
    if "buy_vol" not in df.columns:
        df = label_trade_side(df)

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("datetime").sort_index()

    candles = df["price"].resample(freq).ohlc()
    candles["volume"] = df["qty"].resample(freq).sum()
    candles["buy_vol"] = df["buy_vol"].resample(freq).sum()
    candles["sell_vol"] = df["sell_vol"].resample(freq).sum()
    candles["trade_count"] = df["qty"].resample(freq).count()
    candles["delta"] = candles["buy_vol"] - candles["sell_vol"]
    candles["delta_pct"] = candles["delta"] / candles["volume"].replace(0, np.nan)

    candles = candles.dropna(subset=["open"])
    return candles


def detect_large_trades(
    df: pd.DataFrame, threshold_btc: float = 1.0
) -> pd.DataFrame:
    """
    Flag and aggregate large trades (potential smart money / institutional flow).

    Adds columns: is_large, large_buy_vol, large_sell_vol.
    """
    df = df.copy()
    if "buy_vol" not in df.columns:
        df = label_trade_side(df)

    df["is_large"] = df["qty"] >= threshold_btc
    df["large_buy_vol"] = np.where(df["is_large"] & (df["side"] == "buy"), df["qty"], 0.0)
    df["large_sell_vol"] = np.where(df["is_large"] & (df["side"] == "sell"), df["qty"], 0.0)
    return df


def compute_absorption(
    candles: pd.DataFrame, price_move_threshold: float = 0.001
) -> pd.Series:
    """
    Detect absorption: high volume bars where price barely moved.

    When large volume appears but price doesn't move, the opposing side
    is absorbing (e.g., strong sell pressure absorbed by buyers → bullish).

    Returns:
        pd.Series of absorption_score = volume / abs(close - open).
        High score = high volume relative to price move (strong absorption).
        Returns 0 when price move exceeds threshold (no absorption).
    """
    price_move = (candles["close"] - candles["open"]).abs()
    price_move_pct = price_move / candles["open"].replace(0, np.nan)

    absorption = np.where(
        price_move_pct < price_move_threshold,
        candles["volume"] / (price_move + 1e-8),
        0.0,
    )
    return pd.Series(absorption, index=candles.index, name="absorption_score")


def compute_orderbook_imbalance(
    bids: list, asks: list, depth: int = 10
) -> float:
    """
    Order book imbalance = bid_depth / (bid_depth + ask_depth).

    Values:
      > 0.6 → bid-heavy, buying pressure (bullish)
      < 0.4 → ask-heavy, selling pressure (bearish)
      ~0.5  → balanced
    """
    bid_vol = sum(float(b[1]) for b in bids[:depth])
    ask_vol = sum(float(a[1]) for a in asks[:depth])
    total = bid_vol + ask_vol
    if total == 0:
        return 0.5
    return bid_vol / total


def compute_rolling_large_trade_volumes(
    df: pd.DataFrame,
    window_seconds: int = 300,
    threshold_btc: float = 1.0,
) -> pd.DataFrame:
    """
    Rolling aggregation of large buy/sell volumes over a time window.

    Returns DataFrame with columns: large_buy_vol_roll, large_sell_vol_roll,
    large_trade_imbalance (large_buy / (large_buy + large_sell)).
    """
    df = df.copy()
    if "large_buy_vol" not in df.columns:
        df = detect_large_trades(df, threshold_btc)

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("datetime").sort_index()

    window = f"{window_seconds}s"
    result = pd.DataFrame(index=df.index)
    result["large_buy_vol_roll"] = df["large_buy_vol"].rolling(window, min_periods=1).sum()
    result["large_sell_vol_roll"] = df["large_sell_vol"].rolling(window, min_periods=1).sum()

    total = result["large_buy_vol_roll"] + result["large_sell_vol_roll"]
    result["large_trade_imbalance"] = result["large_buy_vol_roll"] / total.replace(0, np.nan)
    result["large_trade_imbalance"] = result["large_trade_imbalance"].fillna(0.5)
    return result
