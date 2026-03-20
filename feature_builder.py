"""
feature_builder.py — Feature engineering for BTC 15-minute direction prediction.

Combines order flow, technical indicators, and microstructure signals into
a flat feature vector per 1-minute candle. This is the interface between
raw trade data and the ML model.

CRITICAL: No lookahead bias. Features at time t use only data <= t.
The target label (price_up_15m) uses t+15, which is the prediction target, not a feature.

Usage:
    python feature_builder.py
    python feature_builder.py --start 2025-01-01 --end 2025-03-01
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from orderflow import (
    label_trade_side,
    compute_delta_per_candle,
    detect_large_trades,
    compute_absorption,
    compute_orderbook_imbalance,
)

load_dotenv()

MIN_LOOKBACK_CANDLES = 30


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _compute_rolling_vwap(candles: pd.DataFrame, period: int = 20) -> pd.Series:
    typical_price = (candles["high"] + candles["low"] + candles["close"]) / 3
    tp_vol = typical_price * candles["volume"]
    return tp_vol.rolling(period, min_periods=1).sum() / candles["volume"].rolling(period, min_periods=1).sum()


def compute_technical_features(candles: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicator columns to candles DataFrame."""
    df = candles.copy()

    df["rsi_14"] = _compute_rsi(df["close"], 14)

    # Bollinger Bands (20, 2)
    bb_mid = df["close"].rolling(20, min_periods=1).mean()
    bb_std = df["close"].rolling(20, min_periods=1).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    band_width = (bb_upper - bb_lower).replace(0, np.nan)
    df["bb_position"] = (df["close"] - bb_lower) / band_width

    # Rolling VWAP deviation
    vwap = _compute_rolling_vwap(df, period=20)
    df["vwap_dev"] = (df["close"] - vwap) / vwap.replace(0, np.nan)

    # Rate of change
    df["roc_5m"] = df["close"].pct_change(5)
    df["roc_10m"] = df["close"].pct_change(10)

    # Price position relative to recent range
    df["high_20m"] = df["high"].rolling(20, min_periods=1).max()
    df["low_20m"] = df["low"].rolling(20, min_periods=1).min()
    range_20m = (df["high_20m"] - df["low_20m"]).replace(0, np.nan)
    df["price_vs_high_20m"] = (df["close"] - df["high_20m"]) / range_20m
    df["price_vs_low_20m"] = (df["close"] - df["low_20m"]) / range_20m

    # === NEW FEATURES (12 additional indicators) ===

    # MACD(12,26,9) histogram — trend momentum
    ema_12 = df["close"].ewm(span=12, min_periods=1).mean()
    ema_26 = df["close"].ewm(span=26, min_periods=1).mean()
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, min_periods=1).mean()
    df["macd_histogram"] = macd_line - macd_signal

    # StochRSI K — momentum oscillator for mean reversion
    rsi = df["rsi_14"]
    rsi_min = rsi.rolling(14, min_periods=1).min()
    rsi_max = rsi.rolling(14, min_periods=1).max()
    rsi_range = (rsi_max - rsi_min).replace(0, np.nan)
    df["stoch_rsi_k"] = ((rsi - rsi_min) / rsi_range).rolling(3, min_periods=1).mean()

    # EMA crossover — EMA(9) - EMA(21) normalized by price
    ema_9 = df["close"].ewm(span=9, min_periods=1).mean()
    ema_21 = df["close"].ewm(span=21, min_periods=1).mean()
    df["ema_cross"] = (ema_9 - ema_21) / df["close"].replace(0, np.nan)

    # Momentum — 11-period raw price change
    df["momentum_11"] = df["close"].diff(11)

    # OBV slope — On-Balance Volume trend over 10 periods
    obv_direction = np.where(df["close"].diff() > 0, 1, np.where(df["close"].diff() < 0, -1, 0))
    obv = (obv_direction * df["volume"]).cumsum()
    obv_series = pd.Series(obv, index=df.index)
    df["obv_slope"] = obv_series.diff(10) / obv_series.rolling(10, min_periods=1).mean().replace(0, np.nan)

    # ATR(14) normalized — volatility context
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=1).mean()
    df["atr_14"] = atr / df["close"].replace(0, np.nan)

    # ROC 15m — matches our prediction window
    df["roc_15m"] = df["close"].pct_change(15)

    # Volume ratio — recent vs average (spike detection)
    vol_5m = df["volume"].rolling(5, min_periods=1).sum()
    vol_20m = df["volume"].rolling(20, min_periods=1).mean() * 5
    df["volume_ratio_5m"] = vol_5m / vol_20m.replace(0, np.nan)

    # Delta ratio — buy-sell pressure intensity
    if "delta" in df.columns:
        delta_5m = df["delta"].rolling(5, min_periods=1).sum()
        vol_5m_total = df["volume"].rolling(5, min_periods=1).sum()
        df["delta_ratio_5m"] = delta_5m / vol_5m_total.replace(0, np.nan)
    else:
        df["delta_ratio_5m"] = 0.0

    # Large trade ratio — institutional activity
    if "large_buy_vol" in df.columns and "large_sell_vol" in df.columns:
        large_vol_5m = (df["large_buy_vol"] + df["large_sell_vol"]).rolling(5, min_periods=1).sum()
        total_vol_5m = df["volume"].rolling(5, min_periods=1).sum()
        df["large_trade_ratio"] = large_vol_5m / total_vol_5m.replace(0, np.nan)
    else:
        df["large_trade_ratio"] = 0.0

    # Price momentum normalized by ATR
    df["price_momentum_5m"] = df["close"].diff(5) / atr.replace(0, np.nan)

    # Candle body ratio — body vs total range (conviction indicator)
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["candle_body_ratio"] = (df["close"] - df["open"]).abs() / candle_range

    return df


def compute_microstructure_features(trades_in_window: pd.DataFrame, candle_time) -> dict:
    """Compute microstructure features from raw trades within a 1-minute window."""
    if trades_in_window.empty:
        return {
            "uptick_ratio": 0.5,
            "spread_proxy": 0.0,
            "trade_count_1m": 0,
            "hour_utc": candle_time.hour,
            "minute_utc": float(candle_time.minute),
        }

    prices = trades_in_window["price"].values
    upticks = np.sum(np.diff(prices) > 0) if len(prices) > 1 else 0
    total_ticks = len(prices) - 1 if len(prices) > 1 else 1
    uptick_ratio = upticks / total_ticks

    spread_proxy = float(prices.max() - prices.min()) if len(prices) > 0 else 0.0

    return {
        "uptick_ratio": uptick_ratio,
        "spread_proxy": spread_proxy,
        "trade_count_1m": len(trades_in_window),
        "hour_utc": candle_time.hour,
        "minute_utc": float(candle_time.minute),
    }


def build_feature_vector(candles, raw_trades, orderbook_imbalance=0.5, large_trade_threshold=1.0):
    """
    Build a single feature vector from the last N candles and raw trades.
    Returns pd.Series with named features, or None if insufficient data.
    """
    if len(candles) < MIN_LOOKBACK_CANDLES:
        return None

    current = candles.iloc[-1]
    candle_time = candles.index[-1]

    # CVD over different windows
    if "delta" in candles.columns:
        cvd_5m = candles["delta"].iloc[-5:].sum()
        cvd_15m = candles["delta"].iloc[-15:].sum()
        cvd_1h = candles["delta"].iloc[-60:].sum() if len(candles) >= 60 else candles["delta"].sum()
    else:
        cvd_5m = cvd_15m = cvd_1h = 0.0

    # CVD divergence
    price_roc_5m = (candles["close"].iloc[-1] - candles["close"].iloc[-6]) / candles["close"].iloc[-6] if len(candles) >= 6 else 0.0
    cvd_divergence = 1.0 if (price_roc_5m > 0) != (cvd_5m > 0) and abs(price_roc_5m) > 0.0005 else 0.0

    # Large trade metrics over 5m window
    if "large_buy_vol" in candles.columns:
        large_buy_5m = candles["large_buy_vol"].iloc[-5:].sum()
        large_sell_5m = candles["large_sell_vol"].iloc[-5:].sum()
        large_total_5m = large_buy_5m + large_sell_5m
        large_trade_imbalance = large_buy_5m / large_total_5m if large_total_5m > 0 else 0.5
    else:
        large_buy_5m = large_sell_5m = 0.0
        large_trade_imbalance = 0.5

    # Absorption score
    absorption_score = float(candles["absorption_score"].iloc[-1]) if "absorption_score" in candles.columns else 0.0

    # Microstructure features
    micro = compute_microstructure_features(raw_trades, candle_time)

    features = {
        "cvd_5m": cvd_5m,
        "cvd_15m": cvd_15m,
        "cvd_1h": cvd_1h,
        "delta_1m": float(current.get("delta", 0.0)),
        "delta_pct_1m": float(current.get("delta_pct", 0.0)),
        "large_buy_vol_5m": large_buy_5m,
        "large_sell_vol_5m": large_sell_5m,
        "large_trade_imbalance": large_trade_imbalance,
        "absorption_score": absorption_score,
        "ob_imbalance": orderbook_imbalance,
        "cvd_divergence": cvd_divergence,
        "rsi_14": float(current.get("rsi_14", 50.0)),
        "bb_position": float(current.get("bb_position", 0.5)),
        "vwap_dev": float(current.get("vwap_dev", 0.0)),
        "roc_5m": float(current.get("roc_5m", 0.0)),
        "roc_10m": float(current.get("roc_10m", 0.0)),
        "price_vs_high_20m": float(current.get("price_vs_high_20m", 0.0)),
        "price_vs_low_20m": float(current.get("price_vs_low_20m", 0.0)),
        "uptick_ratio": micro["uptick_ratio"],
        "spread_proxy": micro["spread_proxy"],
        "trade_count_1m": micro["trade_count_1m"],
        "hour_utc": micro["hour_utc"],
        "minute_utc": micro["minute_utc"],
        "volume_1m": float(current.get("volume", 0.0)),
        "buy_vol_1m": float(current.get("buy_vol", 0.0)),
        "sell_vol_1m": float(current.get("sell_vol", 0.0)),
        # New features
        "macd_histogram": float(current.get("macd_histogram", 0.0)),
        "stoch_rsi_k": float(current.get("stoch_rsi_k", 0.5)),
        "ema_cross": float(current.get("ema_cross", 0.0)),
        "momentum_11": float(current.get("momentum_11", 0.0)),
        "obv_slope": float(current.get("obv_slope", 0.0)),
        "atr_14": float(current.get("atr_14", 0.0)),
        "roc_15m": float(current.get("roc_15m", 0.0)),
        "volume_ratio_5m": float(current.get("volume_ratio_5m", 1.0)),
        "delta_ratio_5m": float(current.get("delta_ratio_5m", 0.0)),
        "large_trade_ratio": float(current.get("large_trade_ratio", 0.0)),
        "price_momentum_5m": float(current.get("price_momentum_5m", 0.0)),
        "candle_body_ratio": float(current.get("candle_body_ratio", 0.5)),
    }

    return pd.Series(features)


def build_historical_features(trades_df, output_path, large_trade_threshold=1.0):
    """
    Build the full feature dataset from historical trade data.
    Walks through history in 1-minute steps with no lookahead.
    """
    print("Labeling trade sides...")
    trades_df = label_trade_side(trades_df)
    trades_df = detect_large_trades(trades_df, large_trade_threshold)

    print("Building 1-minute candles...")
    candles = compute_delta_per_candle(trades_df, freq="1min")

    # Add large trade aggregates to candles
    trades_df["datetime"] = pd.to_datetime(trades_df["timestamp"], unit="ms", utc=True)
    trades_df = trades_df.set_index("datetime").sort_index()

    candles["large_buy_vol"] = trades_df["large_buy_vol"].resample("1min").sum().reindex(candles.index, fill_value=0)
    candles["large_sell_vol"] = trades_df["large_sell_vol"].resample("1min").sum().reindex(candles.index, fill_value=0)

    print("Computing absorption scores...")
    candles["absorption_score"] = compute_absorption(candles)

    print("Computing technical indicators...")
    candles = compute_technical_features(candles)

    print(f"Building features for {len(candles)} candles (dropping first {MIN_LOOKBACK_CANDLES} for warmup)...")

    feature_rows = []
    candle_times = candles.index.tolist()

    for i in tqdm(range(MIN_LOOKBACK_CANDLES, len(candles)), desc="Feature engineering"):
        candle_time = candle_times[i]
        lookback_candles = candles.iloc[max(0, i - 60):i + 1]

        window_end = candle_time
        window_start = candle_time - pd.Timedelta(minutes=1)
        raw_trades_window = trades_df.loc[
            (trades_df.index >= window_start) & (trades_df.index < window_end)
        ].reset_index(drop=True)

        fvec = build_feature_vector(
            lookback_candles,
            raw_trades_window,
            orderbook_imbalance=0.5,
            large_trade_threshold=large_trade_threshold,
        )

        if fvec is None:
            continue

        fvec["candle_time"] = candle_time
        feature_rows.append(fvec)

    print(f"\nBuilt {len(feature_rows)} feature vectors")

    features_df = pd.DataFrame(feature_rows).set_index("candle_time")

    # Build target: will price be higher 15 minutes from now?
    print("Computing 15-minute forward labels...")
    close_prices = candles["close"]

    targets = []
    for t in features_df.index:
        future_t = t + pd.Timedelta(minutes=15)
        if future_t in close_prices.index:
            future_price = close_prices.loc[future_t]
            current_price = close_prices.loc[t]
            targets.append(1 if future_price > current_price else 0)
        else:
            targets.append(np.nan)

    features_df["price_up_15m"] = targets
    features_df = features_df.dropna(subset=["price_up_15m"])
    features_df["price_up_15m"] = features_df["price_up_15m"].astype(int)

    # Subsample every 15 minutes to eliminate overlapping prediction windows.
    pre_subsample = len(features_df)
    features_df = features_df.iloc[::15]
    print(f"Subsampled: {pre_subsample} -> {len(features_df)} non-overlapping 15-min windows")

    print(f"Final dataset: {len(features_df)} rows")
    print(f"Label distribution: {features_df['price_up_15m'].value_counts().to_dict()}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features_df.to_parquet(output_path, compression="snappy")
    print(f"Saved features to: {output_path}")

    return features_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ML feature dataset from historical trades")
    parser.add_argument("--start", default=os.getenv("BACKTEST_START_DATE", "2025-01-01"))
    parser.add_argument("--end", default=os.getenv("BACKTEST_END_DATE", "2025-03-01"))
    parser.add_argument("--symbol", default="XBTUSD")
    args = parser.parse_args()

    trades_path = os.path.join(".tmp", f"{args.symbol.lower()}_trades_{args.start}_{args.end}.parquet")
    if not os.path.exists(trades_path):
        print(f"ERROR: Trade data not found at {trades_path}")
        print("Run: python historical_data.py first")
        sys.exit(1)

    print(f"Loading trade data from {trades_path}...")
    trades_df = pd.read_parquet(trades_path)

    output_path = os.path.join(".tmp", f"{args.symbol.lower()}_features_{args.start}_{args.end}.parquet")
    build_historical_features(trades_df, output_path)
