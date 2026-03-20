"""
lgbm_signal.py — LightGBM ML order flow signal for BTC Oracle.

Loads the trained LightGBM model and generates a prediction signal
by fetching fresh Kraken trade data and computing the same 26 features
used during training.

Integrated as Tier 7 in scoring_model.py.
Uses orderflow.py for shared order flow computations.
"""

import os
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from orderflow import label_trade_side, detect_large_trades

# Cached model (loaded once at startup)
_model = None
_model_features = None


def load_model(path="model.pkl"):
    """Load model once and cache it. Returns True if successful."""
    global _model, _model_features
    if _model is not None:
        return True
    if not os.path.exists(path):
        print(f"  [LGBM] model.pkl not found at {path}, skipping ML signal")
        return False
    try:
        with open(path, "rb") as f:
            _model = pickle.load(f)
        _model_features = _model.feature_name_
        print(f"  [LGBM] Model loaded: {len(_model_features)} features")
        return True
    except Exception as e:
        print(f"  [LGBM] Failed to load model: {e}")
        return False


def _fetch_raw_trades(count=1000):
    """Fetch recent trades from Kraken REST API. Returns DataFrame or None."""
    try:
        resp = requests.get(
            f"https://api.kraken.com/0/public/Trades?pair=XBTUSD&count={count}",
            timeout=10,
        )
        data = resp.json()
        trades_raw = data.get("result", {}).get("XXBTZUSD", [])
        if not trades_raw:
            return None

        rows = []
        for t in trades_raw:
            price, vol, ts, side = float(t[0]), float(t[1]), float(t[2]), t[3]
            rows.append({
                "timestamp": int(ts * 1000),
                "price": price,
                "qty": vol,
                "side": "buy" if side == "b" else "sell",
            })

        df = pd.DataFrame(rows)
        df = label_trade_side(df)
        df = detect_large_trades(df, threshold_btc=1.0)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("datetime").sort_index()
        return df
    except Exception as e:
        print(f"  [LGBM] Failed to fetch trades: {e}")
        return None


def _fetch_ohlc_candles(interval=1, count=65):
    """Fetch 1-min OHLC candles from Kraken. Returns DataFrame or None."""
    try:
        resp = requests.get(
            f"https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval={interval}",
            timeout=10,
        )
        data = resp.json()
        candles_raw = data.get("result", {}).get("XXBTZUSD", [])
        if not candles_raw:
            return None

        candles = pd.DataFrame(
            candles_raw[-count:],
            columns=["timestamp", "open", "high", "low", "close", "vwap", "volume", "trades"],
        )
        for col in ["open", "high", "low", "close", "volume"]:
            candles[col] = candles[col].astype(float)
        candles["datetime"] = pd.to_datetime(candles["timestamp"].astype(int), unit="s", utc=True)
        candles = candles.set_index("datetime")
        return candles
    except Exception as e:
        print(f"  [LGBM] Failed to fetch OHLC: {e}")
        return None


def _compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_features(indicators, market_data):
    """
    Build the 26-feature vector matching training format.

    Uses:
      - Fresh Kraken REST trades for CVD, delta, large trade features
      - Fresh Kraken OHLC for RSI, BB, VWAP, ROC computation
      - market_data for order book imbalance (converted to 0-1 range)

    Returns:
        pd.DataFrame (1 row) or None if data is insufficient.
    """
    try:
        trades = _fetch_raw_trades(count=1000)
        if trades is None or len(trades) < 50:
            return None

        now = trades.index[-1]

        def _cvd_window(minutes):
            cutoff = now - pd.Timedelta(minutes=minutes)
            window = trades[trades.index >= cutoff]
            return float((window["buy_vol"] - window["sell_vol"]).sum())

        cvd_5m = _cvd_window(5)
        cvd_15m = _cvd_window(15)
        cvd_1h = _cvd_window(60)

        # Last 1-minute window
        cutoff_1m = now - pd.Timedelta(minutes=1)
        last_1m = trades[trades.index >= cutoff_1m]
        vol_1m = float(last_1m["qty"].sum()) if not last_1m.empty else 0.0
        delta_1m = float((last_1m["buy_vol"] - last_1m["sell_vol"]).sum()) if not last_1m.empty else 0.0
        delta_pct_1m = delta_1m / vol_1m if vol_1m > 0 else 0.0

        # Large trade imbalance (5m window)
        cutoff_5m = now - pd.Timedelta(minutes=5)
        last_5m = trades[trades.index >= cutoff_5m]
        large_buy_5m = float(last_5m["large_buy_vol"].sum()) if not last_5m.empty else 0.0
        large_sell_5m = float(last_5m["large_sell_vol"].sum()) if not last_5m.empty else 0.0
        large_total = large_buy_5m + large_sell_5m
        large_trade_imbalance = large_buy_5m / large_total if large_total > 0 else 0.5

        # Absorption score (current 1-min candle)
        current_price_val = float(trades["price"].iloc[-1])
        if not last_1m.empty and len(last_1m) > 1:
            price_change = abs(float(last_1m["price"].iloc[-1]) - float(last_1m["price"].iloc[0]))
            price_pct = price_change / float(last_1m["price"].iloc[0])
            absorption_score = vol_1m / (price_change + 1e-8) if price_pct < 0.001 else 0.0
        else:
            absorption_score = 0.0

        # CVD divergence
        cutoff_6m = now - pd.Timedelta(minutes=6)
        older_trades = trades[trades.index <= cutoff_6m]
        price_6m_ago = float(older_trades["price"].iloc[-1]) if not older_trades.empty else current_price_val
        price_roc_5m = (current_price_val - price_6m_ago) / price_6m_ago if price_6m_ago > 0 else 0.0
        cvd_divergence = (
            1.0
            if (price_roc_5m > 0) != (cvd_5m > 0) and abs(price_roc_5m) > 0.0005
            else 0.0
        )

        # Order book imbalance: market_data uses (bid-ask)/(bid+ask) ∈ [-1,1]
        # Model expects bid/(bid+ask) ∈ [0,1], so convert
        raw_imbalance = market_data.get("orderbook_imbalance", 0.0) if market_data else 0.0
        ob_imbalance = (raw_imbalance + 1) / 2

        # Technical features from OHLC
        candles = _fetch_ohlc_candles(interval=1, count=65)
        if candles is None or len(candles) < 30:
            return None

        close = candles["close"]
        high = candles["high"].astype(float)
        low = candles["low"].astype(float)

        # RSI(14)
        rsi_series = _compute_rsi(close, 14)
        rsi_14 = float(rsi_series.iloc[-1]) if not rsi_series.isna().all() else 50.0

        # Bollinger Bands (20, 2)
        bb_mid = close.rolling(20, min_periods=1).mean()
        bb_std = close.rolling(20, min_periods=1).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        band_width = (bb_upper - bb_lower).replace(0, np.nan)
        bb_position = float((close - bb_lower).iloc[-1] / band_width.iloc[-1]) if band_width.iloc[-1] > 0 else 0.5

        # Rolling VWAP deviation (20-period)
        typical = (high + low + close) / 3
        tp_vol = typical * candles["volume"].astype(float)
        vwap_roll = (
            tp_vol.rolling(20, min_periods=1).sum()
            / candles["volume"].astype(float).rolling(20, min_periods=1).sum()
        )
        vwap_dev = float((close.iloc[-1] - vwap_roll.iloc[-1]) / vwap_roll.iloc[-1]) if vwap_roll.iloc[-1] > 0 else 0.0

        # Rate of change
        roc_5m = float(close.pct_change(5).iloc[-1]) if len(close) >= 6 else 0.0
        roc_10m = float(close.pct_change(10).iloc[-1]) if len(close) >= 11 else 0.0

        # Price vs 20m range
        high_20m = high.rolling(20, min_periods=1).max()
        low_20m = low.rolling(20, min_periods=1).min()
        range_20m = (high_20m - low_20m).replace(0, np.nan)
        price_vs_high_20m = (
            float((close.iloc[-1] - high_20m.iloc[-1]) / range_20m.iloc[-1])
            if range_20m.iloc[-1] > 0
            else 0.0
        )
        price_vs_low_20m = (
            float((close.iloc[-1] - low_20m.iloc[-1]) / range_20m.iloc[-1])
            if range_20m.iloc[-1] > 0
            else 0.0
        )

        # Microstructure
        prices = last_1m["price"].values if not last_1m.empty else np.array([current_price_val])
        upticks = np.sum(np.diff(prices) > 0) if len(prices) > 1 else 0
        uptick_ratio = upticks / max(len(prices) - 1, 1)
        spread_proxy = float(prices.max() - prices.min()) if len(prices) > 0 else 0.0
        trade_count_1m = len(last_1m)

        now_utc = datetime.now(timezone.utc)

        # === NEW FEATURES (12 additional indicators) ===
        vol = candles["volume"].astype(float)

        # MACD histogram
        ema_12 = close.ewm(span=12, min_periods=1).mean()
        ema_26 = close.ewm(span=26, min_periods=1).mean()
        macd_line = ema_12 - ema_26
        macd_signal_line = macd_line.ewm(span=9, min_periods=1).mean()
        macd_histogram = float((macd_line - macd_signal_line).iloc[-1])

        # StochRSI K
        rsi_min = rsi_series.rolling(14, min_periods=1).min()
        rsi_max = rsi_series.rolling(14, min_periods=1).max()
        rsi_range = (rsi_max - rsi_min).replace(0, np.nan)
        stoch_rsi = ((rsi_series - rsi_min) / rsi_range).rolling(3, min_periods=1).mean()
        stoch_rsi_k = float(stoch_rsi.iloc[-1]) if not stoch_rsi.isna().all() else 0.5

        # EMA crossover
        ema_9 = close.ewm(span=9, min_periods=1).mean()
        ema_21 = close.ewm(span=21, min_periods=1).mean()
        ema_cross = float((ema_9.iloc[-1] - ema_21.iloc[-1]) / close.iloc[-1]) if close.iloc[-1] > 0 else 0.0

        # Momentum 11
        momentum_11 = float(close.diff(11).iloc[-1]) if len(close) >= 12 else 0.0

        # OBV slope
        obv_dir = np.where(close.diff() > 0, 1, np.where(close.diff() < 0, -1, 0))
        obv = pd.Series((obv_dir * vol.values).cumsum(), index=close.index)
        obv_mean = obv.rolling(10, min_periods=1).mean().replace(0, np.nan)
        obv_slope = float(obv.diff(10).iloc[-1] / obv_mean.iloc[-1]) if len(obv) >= 11 and obv_mean.iloc[-1] > 0 else 0.0

        # ATR(14) normalized
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=1).mean()
        atr_14 = float(atr.iloc[-1] / close.iloc[-1]) if close.iloc[-1] > 0 else 0.0

        # ROC 15m
        roc_15m = float(close.pct_change(15).iloc[-1]) if len(close) >= 16 else 0.0

        # Volume ratio 5m
        vol_5m = vol.rolling(5, min_periods=1).sum()
        vol_20m_avg = vol.rolling(20, min_periods=1).mean() * 5
        volume_ratio_5m = float(vol_5m.iloc[-1] / vol_20m_avg.iloc[-1]) if vol_20m_avg.iloc[-1] > 0 else 1.0

        # Delta ratio 5m (using CVD as proxy)
        delta_ratio_5m = cvd_5m / vol_1m if vol_1m > 0 else 0.0

        # Large trade ratio
        large_total_vol = large_buy_5m + large_sell_5m
        vol_5m_val = float(vol.iloc[-5:].sum()) if len(vol) >= 5 else vol_1m
        large_trade_ratio = large_total_vol / vol_5m_val if vol_5m_val > 0 else 0.0

        # Price momentum normalized by ATR
        price_diff_5 = float(close.diff(5).iloc[-1]) if len(close) >= 6 else 0.0
        price_momentum_5m = price_diff_5 / float(atr.iloc[-1]) if float(atr.iloc[-1]) > 0 else 0.0

        # Candle body ratio
        last_open = float(candles["open"].astype(float).iloc[-1])
        last_high = float(high.iloc[-1])
        last_low = float(low.iloc[-1])
        last_close = float(close.iloc[-1])
        candle_range = last_high - last_low
        candle_body_ratio = abs(last_close - last_open) / candle_range if candle_range > 0 else 0.5

        features = {
            "cvd_5m": cvd_5m,
            "cvd_15m": cvd_15m,
            "cvd_1h": cvd_1h,
            "delta_1m": delta_1m,
            "delta_pct_1m": delta_pct_1m,
            "large_buy_vol_5m": large_buy_5m,
            "large_sell_vol_5m": large_sell_5m,
            "large_trade_imbalance": large_trade_imbalance,
            "absorption_score": absorption_score,
            "ob_imbalance": ob_imbalance,
            "cvd_divergence": cvd_divergence,
            "rsi_14": rsi_14,
            "bb_position": bb_position,
            "vwap_dev": vwap_dev,
            "roc_5m": roc_5m,
            "roc_10m": roc_10m,
            "price_vs_high_20m": price_vs_high_20m,
            "price_vs_low_20m": price_vs_low_20m,
            "uptick_ratio": uptick_ratio,
            "spread_proxy": spread_proxy,
            "trade_count_1m": trade_count_1m,
            "hour_utc": now_utc.hour,
            "minute_utc": float(now_utc.minute),
            "volume_1m": vol_1m,
            "buy_vol_1m": float(last_1m["buy_vol"].sum()) if not last_1m.empty else 0.0,
            "sell_vol_1m": float(last_1m["sell_vol"].sum()) if not last_1m.empty else 0.0,
            # New features
            "macd_histogram": macd_histogram,
            "stoch_rsi_k": stoch_rsi_k,
            "ema_cross": ema_cross,
            "momentum_11": momentum_11,
            "obv_slope": obv_slope,
            "atr_14": atr_14,
            "roc_15m": roc_15m,
            "volume_ratio_5m": volume_ratio_5m,
            "delta_ratio_5m": delta_ratio_5m,
            "large_trade_ratio": large_trade_ratio,
            "price_momentum_5m": price_momentum_5m,
            "candle_body_ratio": candle_body_ratio,
        }

        # Add rule-based signals (stacked model expects these)
        features["rule_bb_signal"] = (
            -(bb_position - 0.85) * 5 if bb_position > 0.85
            else (0.15 - bb_position) * 5 if bb_position < 0.15
            else 0.0
        )
        features["rule_rsi_signal"] = (
            -(rsi_14 - 75) / 25 if rsi_14 > 75
            else (25 - rsi_14) / 25 if rsi_14 < 25
            else 0.0
        )
        features["rule_anti_mom"] = -roc_5m * 100
        features["rule_vwap_signal"] = -vwap_dev * 100
        features["rule_cvd_signal"] = float(np.sign(cvd_5m))
        features["rule_stoch_signal"] = (
            -1.0 if stoch_rsi_k > 0.8
            else 1.0 if stoch_rsi_k < 0.2
            else 0.0
        )
        features["rule_combined_score"] = (
            features["rule_bb_signal"] + features["rule_rsi_signal"] +
            features["rule_anti_mom"] + features["rule_vwap_signal"]
        )

        return pd.DataFrame([features])

    except Exception as e:
        print(f"  [LGBM] Feature computation failed: {e}")
        return None


# Confidence threshold from sweep: 0.10 = 55.9% accuracy sweet spot
CONFIDENCE_THRESHOLD = 0.10

# Hours (UTC) where model accuracy drops below 40% — skip these
BAD_HOURS_UTC = {1, 6, 9}


def predict(features_df):
    """
    Run stacked model prediction with confidence threshold and bad-hour filter.

    Returns:
        (signal: "UP"/"DOWN"/"SKIP", confidence: float 0-1)
        "SKIP" when confidence is below threshold or during bad hours.
    """
    global _model, _model_features

    if _model is None:
        if not load_model():
            return "SKIP", 0.0

    if features_df is None:
        return "SKIP", 0.0

    try:
        if _model_features is not None:
            for col in _model_features:
                if col not in features_df.columns:
                    features_df[col] = 0.0
            X = features_df[_model_features]
        else:
            X = features_df

        prob = _model.predict_proba(X)[0]  # [prob_down, prob_up]
        prob_up = float(prob[1])
        signal = "UP" if prob_up >= 0.5 else "DOWN"
        confidence = abs(prob_up - 0.5) * 2  # 0 at 50/50, 1 at 100% certain

        # Bad-hour filter
        hour = int(features_df["hour_utc"].iloc[0]) if "hour_utc" in features_df.columns else -1
        if hour in BAD_HOURS_UTC:
            print(f"  [LGBM] Skipping — bad hour ({hour:02d}:00 UTC)")
            return "SKIP", 0.0

        # Confidence threshold filter
        if confidence < CONFIDENCE_THRESHOLD:
            print(f"  [LGBM] Low confidence ({confidence:.3f} < {CONFIDENCE_THRESHOLD}) — signal logged but weak")
            return signal, round(confidence, 3)

        print(f"  [LGBM] HIGH CONFIDENCE: {signal} ({confidence:.3f})")
        return signal, round(confidence, 3)

    except Exception as e:
        print(f"  [LGBM] Prediction failed: {e}")
        return "SKIP", 0.0
