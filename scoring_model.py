"""
BTC Oracle - Scoring Model V5
Mean reversion + order flow.
ONLY learns from V5 signals (analysis_notes starts with [TRADE] or [WAIT]).
Tracks which tier drives wins vs losses and auto-adjusts weights.
"""

import db
import numpy as np
from datetime import datetime, timezone

# V5 epoch - only learn from signals after this point
V5_EPOCH = "2026-03-13T00:00:00Z"

# Tier weights - these get adjusted by learning
DEFAULT_TIER_WEIGHTS = {
    "bb_extreme": 3.0,
    "bb_moderate": 2.0,
    "bb_mild": 1.0,
    "anti_momentum": 1.5,
    "anti_roc": 1.0,
    "rsi_extreme": 2.0,
    "rsi_moderate": 0.8,
    "stoch_extreme": 1.5,
    "orderbook_imbalance": 2.5,
    "orderbook_strong": 1.5,
    "trade_flow": 2.0,
    "trade_flow_strong": 1.0,
    "whale": 2.5,
    "bid_wall": 1.5,
    "ask_wall": 1.5,
    "vwap_revert": 0.8,
    "obv_divergence": 1.5,
}


def get_v5_signals():
    """Only get signals from V5 epoch onwards."""
    params = f"outcome=not.is.null&created_at=gte.{V5_EPOCH}&order=created_at.desc&limit=200"
    return db.select("signals", params)


def learn_tier_weights():
    """
    Analyze V5 signals to see which tiers predict correctly.
    Returns adjusted weights dict.
    """
    signals = get_v5_signals()
    if not signals or len(signals) < 10:
        print(f"  Learning: only {len(signals) if signals else 0} V5 signals, using defaults")
        return DEFAULT_TIER_WEIGHTS.copy()

    # For each signal, determine what conditions were present and if price went up
    tier_stats = {}  # tier_name -> {"correct": N, "total": N}

    for s in signals:
        if not s.get("btc_price_at_signal") or not s.get("btc_price_at_close"):
            continue

        actual_up = s["btc_price_at_close"] > s["btc_price_at_signal"]
        price = s["btc_price_at_signal"]

        # Check Bollinger position
        if s.get("bollinger_upper") is not None and s.get("bollinger_lower") is not None:
            bu, bl = s["bollinger_upper"], s["bollinger_lower"]
            if bu != bl:
                bb_pos = (price - bl) / (bu - bl)
                if bb_pos > 0.8:
                    # BB said go DOWN (mean revert)
                    _track(tier_stats, "bb_extreme" if bb_pos > 0.9 else "bb_moderate", not actual_up)
                elif bb_pos < 0.2:
                    # BB said go UP (mean revert)
                    _track(tier_stats, "bb_extreme" if bb_pos < 0.1 else "bb_moderate", actual_up)

        # Check momentum (anti-momentum: positive momentum = predict DOWN)
        if s.get("momentum") is not None:
            if s["momentum"] > 0:
                _track(tier_stats, "anti_momentum", not actual_up)
            elif s["momentum"] < 0:
                _track(tier_stats, "anti_momentum", actual_up)

        # Check RSI
        if s.get("rsi") is not None:
            if s["rsi"] > 75:
                _track(tier_stats, "rsi_extreme", not actual_up)
            elif s["rsi"] < 25:
                _track(tier_stats, "rsi_extreme", actual_up)

    # Adjust weights based on accuracy
    adjusted = DEFAULT_TIER_WEIGHTS.copy()
    print(f"  Learning from {len(signals)} V5 signals:")

    for tier, stats in tier_stats.items():
        if stats["total"] >= 3:
            accuracy = stats["correct"] / stats["total"]
            # Scale weight: if accuracy > 50%, increase weight. If < 50%, decrease.
            multiplier = 0.5 + accuracy  # ranges from 0.5 (0% accuracy) to 1.5 (100% accuracy)
            if tier in adjusted:
                old = adjusted[tier]
                adjusted[tier] = round(old * multiplier, 2)
                print(f"    {tier}: {accuracy:.0%} accurate ({stats['correct']}/{stats['total']}) -> weight {old} -> {adjusted[tier]}")

    return adjusted


def _track(stats, tier, was_correct):
    """Track whether a tier's prediction was correct."""
    if tier not in stats:
        stats[tier] = {"correct": 0, "total": 0}
    stats[tier]["total"] += 1
    if was_correct:
        stats[tier]["correct"] += 1


def score_signal(indicators, market_data=None):
    """
    Mean Reversion + Order Flow scoring with learned weights.
    """
    weights = learn_tier_weights()
    votes = []
    current = indicators.get("current_price", 0)

    # ============================================
    # TIER 1: BOLLINGER EXTREMES
    # ============================================
    bb_pos = indicators.get("bollinger_position")
    if bb_pos is not None:
        if bb_pos > 0.9:
            votes.append((-1, weights["bb_extreme"]))
        elif bb_pos > 0.8:
            votes.append((-1, weights["bb_moderate"]))
        elif bb_pos > 0.7:
            votes.append((-1, weights["bb_mild"]))
        elif bb_pos < 0.1:
            votes.append((+1, weights["bb_extreme"]))
        elif bb_pos < 0.2:
            votes.append((+1, weights["bb_moderate"]))
        elif bb_pos < 0.3:
            votes.append((+1, weights["bb_mild"]))

    # ============================================
    # TIER 2: ANTI-MOMENTUM
    # ============================================
    momentum = indicators.get("momentum")
    if momentum is not None:
        if momentum > 0:
            votes.append((-1, weights["anti_momentum"]))
        elif momentum < 0:
            votes.append((+1, weights["anti_momentum"]))

    roc = indicators.get("rate_of_change")
    if roc is not None:
        if roc > 0.15:
            votes.append((-1, weights["anti_roc"]))
        elif roc < -0.15:
            votes.append((+1, weights["anti_roc"]))

    # ============================================
    # TIER 3: RSI EXTREMES
    # ============================================
    rsi = indicators.get("rsi")
    if rsi is not None:
        if rsi > 75:
            votes.append((-1, weights["rsi_extreme"]))
        elif rsi > 65:
            votes.append((-1, weights["rsi_moderate"]))
        elif rsi < 25:
            votes.append((+1, weights["rsi_extreme"]))
        elif rsi < 35:
            votes.append((+1, weights["rsi_moderate"]))

    stoch_k = indicators.get("stoch_rsi_k")
    if stoch_k is not None:
        if stoch_k > 80:
            votes.append((-1, weights["stoch_extreme"]))
        elif stoch_k < 20:
            votes.append((+1, weights["stoch_extreme"]))

    # ============================================
    # TIER 4: ORDER FLOW (live edge)
    # ============================================
    if market_data:
        ob_signal = market_data.get("orderbook_imbalance_signal")
        ob_imbalance = market_data.get("orderbook_imbalance")

        if ob_signal == "BUY_PRESSURE":
            votes.append((+1, weights["orderbook_imbalance"]))
        elif ob_signal == "SELL_PRESSURE":
            votes.append((-1, weights["orderbook_imbalance"]))

        if ob_imbalance is not None:
            if ob_imbalance > 0.3:
                votes.append((+1, weights["orderbook_strong"]))
            elif ob_imbalance < -0.3:
                votes.append((-1, weights["orderbook_strong"]))

        tf_signal = market_data.get("trade_flow_signal")
        tf_buy_pct = market_data.get("trade_flow_buy_pct")

        if tf_signal == "BUYING":
            votes.append((+1, weights["trade_flow"]))
        elif tf_signal == "SELLING":
            votes.append((-1, weights["trade_flow"]))

        if tf_buy_pct is not None:
            if tf_buy_pct > 60:
                votes.append((+1, weights["trade_flow_strong"]))
            elif tf_buy_pct < 40:
                votes.append((-1, weights["trade_flow_strong"]))

        whale = market_data.get("trade_flow_whale_signal")
        if whale == "WHALE_BUYING":
            votes.append((+1, weights["whale"]))
        elif whale == "WHALE_SELLING":
            votes.append((-1, weights["whale"]))

        if market_data.get("orderbook_bid_wall_detected"):
            votes.append((+1, weights["bid_wall"]))
        if market_data.get("orderbook_ask_wall_detected"):
            votes.append((-1, weights["ask_wall"]))

    # ============================================
    # TIER 5: VWAP MEAN REVERSION
    # ============================================
    vwap = indicators.get("vwap")
    if vwap is not None and current:
        vwap_dist = (current - vwap) / vwap * 100
        if vwap_dist > 0.3:
            votes.append((-1, weights["vwap_revert"]))
        elif vwap_dist < -0.3:
            votes.append((+1, weights["vwap_revert"]))

    # ============================================
    # TIER 6: OBV DIVERGENCE
    # ============================================
    obv = indicators.get("obv_trend")
    trend_1m = indicators.get("trend_1m", "")
    if obv == "FALLING" and "UPTREND" in trend_1m:
        votes.append((-1, weights["obv_divergence"]))
    elif obv == "RISING" and "DOWNTREND" in trend_1m:
        votes.append((+1, weights["obv_divergence"]))

    # ============================================
    # CALCULATE
    # ============================================
    if not votes:
        return 0.0, 0.5, "NO_DATA"

    total_weight = sum(abs(w) for _, w in votes)
    weighted_sum = sum(d * w for d, w in votes)
    score = weighted_sum / total_weight if total_weight > 0 else 0

    signal = "UP" if score > 0 else "DOWN"
    confidence = min(0.95, 0.5 + abs(score) * 0.45)

    if market_data and market_data.get("volatility_regime") == "HIGH":
        confidence *= 0.85

    # Boost when order flow confirms mean reversion
    if market_data:
        ob_s = market_data.get("orderbook_imbalance_signal")
        tf_s = market_data.get("trade_flow_signal")
        if signal == "UP" and ob_s == "BUY_PRESSURE" and tf_s == "BUYING":
            confidence = min(0.95, confidence + 0.08)
        elif signal == "DOWN" and ob_s == "SELL_PRESSURE" and tf_s == "SELLING":
            confidence = min(0.95, confidence + 0.08)

    return score, round(confidence, 3), signal
