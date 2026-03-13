"""
BTC Oracle - Weighted Scoring Model (Production Grade)
Weights based on actual price movement, not bot accuracy.
Only uses recent 100 signals. Handles None and 0.0 values correctly.
"""

import db
import numpy as np


def calculate_indicator_weights():
    """Calculate how predictive each indicator is for ACTUAL price direction."""
    signals = db.select("signals", "outcome=not.is.null&order=created_at.desc&limit=100")
    if not signals or len(signals) < 15:
        return get_default_weights()

    weights = {}

    for s in signals:
        if not s.get("btc_price_at_signal") or not s.get("btc_price_at_close"):
            continue
        s["actual_up"] = s["btc_price_at_close"] > s["btc_price_at_signal"]

    valid = [s for s in signals if "actual_up" in s]
    if len(valid) < 15:
        return get_default_weights()

    # RSI - use "is not None" not truthiness (0.0 is valid!)
    rsi_oversold = [s for s in valid if s.get("rsi") is not None and s["rsi"] < 30]
    rsi_overbought = [s for s in valid if s.get("rsi") is not None and s["rsi"] > 70]
    rsi_low = [s for s in valid if s.get("rsi") is not None and 30 <= s["rsi"] < 45]
    rsi_high = [s for s in valid if s.get("rsi") is not None and 55 < s["rsi"] <= 70]

    if len(rsi_oversold) >= 3:
        weights["rsi_oversold_predicts_up"] = sum(1 for s in rsi_oversold if s["actual_up"]) / len(rsi_oversold)
    if len(rsi_overbought) >= 3:
        weights["rsi_overbought_predicts_down"] = sum(1 for s in rsi_overbought if not s["actual_up"]) / len(rsi_overbought)
    if len(rsi_low) >= 3:
        weights["rsi_low_predicts_up"] = sum(1 for s in rsi_low if s["actual_up"]) / len(rsi_low)
    if len(rsi_high) >= 3:
        weights["rsi_high_predicts_down"] = sum(1 for s in rsi_high if not s["actual_up"]) / len(rsi_high)

    # MACD
    macd_pos = [s for s in valid if s.get("macd") is not None and s["macd"] > 0]
    macd_neg = [s for s in valid if s.get("macd") is not None and s["macd"] < 0]

    if len(macd_pos) >= 3:
        weights["macd_pos_predicts_up"] = sum(1 for s in macd_pos if s["actual_up"]) / len(macd_pos)
    if len(macd_neg) >= 3:
        weights["macd_neg_predicts_down"] = sum(1 for s in macd_neg if not s["actual_up"]) / len(macd_neg)

    # MACD histogram
    hist_pos = [s for s in valid if s.get("macd_histogram") is not None and s["macd_histogram"] > 0]
    hist_neg = [s for s in valid if s.get("macd_histogram") is not None and s["macd_histogram"] < 0]

    if len(hist_pos) >= 3:
        weights["hist_pos_predicts_up"] = sum(1 for s in hist_pos if s["actual_up"]) / len(hist_pos)
    if len(hist_neg) >= 3:
        weights["hist_neg_predicts_down"] = sum(1 for s in hist_neg if not s["actual_up"]) / len(hist_neg)

    # Momentum
    mom_pos = [s for s in valid if s.get("momentum") is not None and s["momentum"] > 0]
    mom_neg = [s for s in valid if s.get("momentum") is not None and s["momentum"] < 0]

    if len(mom_pos) >= 3:
        weights["mom_pos_predicts_up"] = sum(1 for s in mom_pos if s["actual_up"]) / len(mom_pos)
    if len(mom_neg) >= 3:
        weights["mom_neg_predicts_down"] = sum(1 for s in mom_neg if not s["actual_up"]) / len(mom_neg)

    # EMA crossover
    ema_bull = [s for s in valid if s.get("ema_9") is not None and s.get("ema_21") is not None and s["ema_9"] > s["ema_21"]]
    ema_bear = [s for s in valid if s.get("ema_9") is not None and s.get("ema_21") is not None and s["ema_9"] <= s["ema_21"]]

    if len(ema_bull) >= 3:
        weights["ema_bull_predicts_up"] = sum(1 for s in ema_bull if s["actual_up"]) / len(ema_bull)
    if len(ema_bear) >= 3:
        weights["ema_bear_predicts_down"] = sum(1 for s in ema_bear if not s["actual_up"]) / len(ema_bear)

    weights["base_up_rate"] = sum(1 for s in valid if s["actual_up"]) / len(valid)

    print(f"  Learned weights ({len(valid)} signals):")
    for k, v in weights.items():
        print(f"    {k}: {v:.1%}")

    return weights


def get_default_weights():
    return {
        "rsi_oversold_predicts_up": 0.55, "rsi_overbought_predicts_down": 0.55,
        "rsi_low_predicts_up": 0.52, "rsi_high_predicts_down": 0.52,
        "macd_pos_predicts_up": 0.52, "macd_neg_predicts_down": 0.52,
        "hist_pos_predicts_up": 0.53, "hist_neg_predicts_down": 0.53,
        "mom_pos_predicts_up": 0.55, "mom_neg_predicts_down": 0.55,
        "ema_bull_predicts_up": 0.53, "ema_bear_predicts_down": 0.53,
        "base_up_rate": 0.50,
    }


def score_signal(indicators, market_data=None):
    weights = calculate_indicator_weights()
    votes = []
    price = indicators.get("current_price", 0)

    # TREND - highest weight
    trend_1m = indicators.get("trend_1m")
    trend_5m = indicators.get("trend_5m")
    trend_map = {"STRONG_UPTREND": (+1, 1.5), "UPTREND": (+1, 1.0), "STRONG_DOWNTREND": (-1, 1.5), "DOWNTREND": (-1, 1.0)}
    if trend_1m in trend_map:
        votes.append(trend_map[trend_1m])
    trend_5m_map = {"STRONG_UPTREND": (+1, 1.8), "UPTREND": (+1, 1.2), "STRONG_DOWNTREND": (-1, 1.8), "DOWNTREND": (-1, 1.2)}
    if trend_5m in trend_5m_map:
        votes.append(trend_5m_map[trend_5m])

    # RSI
    rsi = indicators.get("rsi")
    if rsi is not None:
        if rsi < 30: votes.append((+1, weights.get("rsi_oversold_predicts_up", 0.55) * 1.5))
        elif rsi > 70: votes.append((-1, weights.get("rsi_overbought_predicts_down", 0.55) * 1.5))
        elif rsi < 45: votes.append((+1, weights.get("rsi_low_predicts_up", 0.52) * 0.7))
        elif rsi > 55: votes.append((-1, weights.get("rsi_high_predicts_down", 0.52) * 0.7))

    # MACD
    macd = indicators.get("macd")
    if macd is not None:
        if macd > 0: votes.append((+1, weights.get("macd_pos_predicts_up", 0.52)))
        else: votes.append((-1, weights.get("macd_neg_predicts_down", 0.52)))

    macd_hist = indicators.get("macd_histogram")
    if macd_hist is not None:
        if macd_hist > 0: votes.append((+1, weights.get("hist_pos_predicts_up", 0.53)))
        else: votes.append((-1, weights.get("hist_neg_predicts_down", 0.53)))

    # Momentum
    momentum = indicators.get("momentum")
    if momentum is not None:
        if momentum > 0: votes.append((+1, weights.get("mom_pos_predicts_up", 0.55)))
        else: votes.append((-1, weights.get("mom_neg_predicts_down", 0.55)))

    # EMA crossover
    ema_9, ema_21 = indicators.get("ema_9"), indicators.get("ema_21")
    if ema_9 is not None and ema_21 is not None:
        if ema_9 > ema_21: votes.append((+1, weights.get("ema_bull_predicts_up", 0.53)))
        else: votes.append((-1, weights.get("ema_bear_predicts_down", 0.53)))

    # Bollinger
    bb_pos = indicators.get("bollinger_position")
    if bb_pos is not None:
        if bb_pos < 0.2: votes.append((+1, 0.6))
        elif bb_pos > 0.8: votes.append((-1, 0.6))
        elif bb_pos < 0.4: votes.append((+1, 0.3))
        elif bb_pos > 0.6: votes.append((-1, 0.3))

    # VWAP
    vwap = indicators.get("vwap")
    if vwap is not None and price:
        votes.append((+1 if price > vwap else -1, 0.4))

    # OBV
    obv = indicators.get("obv_trend")
    if obv == "RISING": votes.append((+1, 0.5))
    elif obv == "FALLING": votes.append((-1, 0.5))

    # StochRSI
    stoch_k = indicators.get("stoch_rsi_k")
    if stoch_k is not None:
        if stoch_k < 20: votes.append((+1, 0.6))
        elif stoch_k > 80: votes.append((-1, 0.6))

    # ROC
    roc = indicators.get("rate_of_change")
    if roc is not None:
        if roc > 0.1: votes.append((+1, 0.4))
        elif roc < -0.1: votes.append((-1, 0.4))

    # Market data
    if market_data:
        ob = market_data.get("orderbook_imbalance_signal")
        if ob == "BUY_PRESSURE": votes.append((+1, 0.7))
        elif ob == "SELL_PRESSURE": votes.append((-1, 0.7))

        tf = market_data.get("trade_flow_signal")
        if tf == "BUYING": votes.append((+1, 0.7))
        elif tf == "SELLING": votes.append((-1, 0.7))

        whale = market_data.get("trade_flow_whale_signal")
        if whale == "WHALE_BUYING": votes.append((+1, 0.8))
        elif whale == "WHALE_SELLING": votes.append((-1, 0.8))

        align = market_data.get("momentum_alignment")
        if align == "STRONG_BULL": votes.append((+1, 0.9))
        elif align == "STRONG_BEAR": votes.append((-1, 0.9))
        elif align == "MIXED_BULL": votes.append((+1, 0.4))
        elif align == "MIXED_BEAR": votes.append((-1, 0.4))

    if not votes:
        return 0.0, 0.5, "NO_DATA"

    total_weight = sum(abs(w) for _, w in votes)
    weighted_sum = sum(d * w for d, w in votes)
    score = weighted_sum / total_weight if total_weight > 0 else 0

    signal = "UP" if score > 0 else "DOWN"
    confidence = min(0.95, 0.5 + abs(score) * 0.45)

    if market_data and market_data.get("volatility_regime") == "HIGH":
        confidence *= 0.85

    return score, round(confidence, 3), signal
