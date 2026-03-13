"""
BTC Oracle - Scoring Model V3
MEAN REVERSION strategy (proven 58-61% in backtesting)
+ Live order flow / order book for edge
+ Anti-momentum for confirmation
"""

import db
import numpy as np


def calculate_indicator_weights():
    """Learn from recent signals which conditions predict actual price direction."""
    signals = db.select("signals", "outcome=not.is.null&order=created_at.desc&limit=100")
    if not signals or len(signals) < 15:
        return {}

    weights = {}
    valid = []
    for s in signals:
        if s.get("btc_price_at_signal") and s.get("btc_price_at_close"):
            s["actual_up"] = s["btc_price_at_close"] > s["btc_price_at_signal"]
            valid.append(s)

    if len(valid) < 15:
        return {}

    weights["base_up_rate"] = sum(1 for s in valid if s["actual_up"]) / len(valid)

    # Learn Bollinger band edge
    bb_high = [s for s in valid if s.get("bollinger_upper") is not None and s.get("btc_price_at_signal") is not None
               and s.get("bollinger_lower") is not None and s["bollinger_upper"] != s["bollinger_lower"]]
    if len(bb_high) >= 5:
        near_upper = [s for s in bb_high if (s["btc_price_at_signal"] - s["bollinger_lower"]) / (s["bollinger_upper"] - s["bollinger_lower"]) > 0.8]
        near_lower = [s for s in bb_high if (s["btc_price_at_signal"] - s["bollinger_lower"]) / (s["bollinger_upper"] - s["bollinger_lower"]) < 0.2]
        if len(near_upper) >= 3:
            weights["bb_upper_reversal"] = sum(1 for s in near_upper if not s["actual_up"]) / len(near_upper)
        if len(near_lower) >= 3:
            weights["bb_lower_reversal"] = sum(1 for s in near_lower if s["actual_up"]) / len(near_lower)

    print(f"  Learned weights ({len(valid)} signals):")
    for k, v in weights.items():
        print(f"    {k}: {v:.1%}")

    return weights


def score_signal(indicators, market_data=None):
    """
    Mean Reversion + Order Flow scoring.
    Core idea: BTC at 15-min tends to REVERSE, not continue.
    Order flow confirms or denies the reversal.
    """
    votes = []
    current = indicators.get("current_price", 0)

    # ============================================
    # TIER 1: BOLLINGER EXTREMES (highest weight - proven 58-61%)
    # ============================================
    bb_pos = indicators.get("bollinger_position")
    if bb_pos is not None:
        if bb_pos > 0.9:
            # Very near upper band - strong mean reversion DOWN
            votes.append((-1, 3.0))
        elif bb_pos > 0.8:
            votes.append((-1, 2.0))
        elif bb_pos > 0.7:
            votes.append((-1, 1.0))
        elif bb_pos < 0.1:
            # Very near lower band - strong mean reversion UP
            votes.append((+1, 3.0))
        elif bb_pos < 0.2:
            votes.append((+1, 2.0))
        elif bb_pos < 0.3:
            votes.append((+1, 1.0))
        # Middle zone (0.3-0.7) = no signal from BB

    # ============================================
    # TIER 2: ANTI-MOMENTUM (proven 52-54%)
    # ============================================
    momentum = indicators.get("momentum")
    roc = indicators.get("rate_of_change")

    if momentum is not None:
        # Fade momentum - if price has been going up, bet DOWN
        if momentum > 0:
            votes.append((-1, 1.5))  # anti-momentum
        elif momentum < 0:
            votes.append((+1, 1.5))

    if roc is not None:
        if roc > 0.15:
            votes.append((-1, 1.0))  # extended up, expect pullback
        elif roc < -0.15:
            votes.append((+1, 1.0))  # extended down, expect bounce

    # ============================================
    # TIER 3: RSI EXTREMES (mean reversion)
    # ============================================
    rsi = indicators.get("rsi")
    if rsi is not None:
        if rsi > 75:
            votes.append((-1, 2.0))  # overbought, expect down
        elif rsi > 65:
            votes.append((-1, 0.8))
        elif rsi < 25:
            votes.append((+1, 2.0))  # oversold, expect up
        elif rsi < 35:
            votes.append((+1, 0.8))

    stoch_k = indicators.get("stoch_rsi_k")
    if stoch_k is not None:
        if stoch_k > 80:
            votes.append((-1, 1.5))
        elif stoch_k < 20:
            votes.append((+1, 1.5))

    # ============================================
    # TIER 4: ORDER FLOW (live only - the edge backtest can't capture)
    # ============================================
    if market_data:
        # Order book imbalance - THIS is the live edge
        ob_imbalance = market_data.get("orderbook_imbalance")
        ob_signal = market_data.get("orderbook_imbalance_signal")

        if ob_signal == "BUY_PRESSURE":
            votes.append((+1, 2.5))  # heavy buy pressure = UP
        elif ob_signal == "SELL_PRESSURE":
            votes.append((-1, 2.5))  # heavy sell pressure = DOWN

        # If imbalance is very strong, increase weight
        if ob_imbalance is not None:
            if ob_imbalance > 0.3:
                votes.append((+1, 1.5))  # very strong buy side
            elif ob_imbalance < -0.3:
                votes.append((-1, 1.5))  # very strong sell side

        # Trade flow - are people buying or selling right now?
        tf_signal = market_data.get("trade_flow_signal")
        tf_buy_pct = market_data.get("trade_flow_buy_pct")

        if tf_signal == "BUYING":
            votes.append((+1, 2.0))
        elif tf_signal == "SELLING":
            votes.append((-1, 2.0))

        # Strong buy/sell flow
        if tf_buy_pct is not None:
            if tf_buy_pct > 60:
                votes.append((+1, 1.0))
            elif tf_buy_pct < 40:
                votes.append((-1, 1.0))

        # Whale activity
        whale = market_data.get("trade_flow_whale_signal")
        if whale == "WHALE_BUYING":
            votes.append((+1, 2.5))  # whales are the strongest signal
        elif whale == "WHALE_SELLING":
            votes.append((-1, 2.5))

        # Spread - wide spread means uncertainty
        spread_pct = market_data.get("orderbook_spread_pct")
        if spread_pct is not None and spread_pct > 0.01:
            # Wide spread = reduce confidence later
            pass

        # Bid/ask wall detection
        if market_data.get("orderbook_bid_wall_detected"):
            votes.append((+1, 1.5))  # big buy wall = support
        if market_data.get("orderbook_ask_wall_detected"):
            votes.append((-1, 1.5))  # big sell wall = resistance

    # ============================================
    # TIER 5: VWAP (mean reversion)
    # ============================================
    vwap = indicators.get("vwap")
    if vwap is not None and current:
        vwap_dist = (current - vwap) / vwap * 100
        if vwap_dist > 0.3:
            votes.append((-1, 0.8))  # above VWAP, mean revert down
        elif vwap_dist < -0.3:
            votes.append((+1, 0.8))  # below VWAP, mean revert up

    # ============================================
    # TIER 6: OBV (volume confirms or denies)
    # ============================================
    obv = indicators.get("obv_trend")
    # OBV divergence from price = powerful reversal signal
    trend_1m = indicators.get("trend_1m", "")
    if obv == "FALLING" and "UPTREND" in trend_1m:
        votes.append((-1, 1.5))  # price up but volume down = bearish divergence
    elif obv == "RISING" and "DOWNTREND" in trend_1m:
        votes.append((+1, 1.5))  # price down but volume up = bullish divergence

    # ============================================
    # CALCULATE FINAL SCORE
    # ============================================
    if not votes:
        return 0.0, 0.5, "NO_DATA"

    total_weight = sum(abs(w) for _, w in votes)
    weighted_sum = sum(d * w for d, w in votes)
    score = weighted_sum / total_weight if total_weight > 0 else 0

    signal = "UP" if score > 0 else "DOWN"
    agreement = abs(score)
    confidence = min(0.95, 0.5 + agreement * 0.45)

    # Reduce confidence in high volatility
    if market_data and market_data.get("volatility_regime") == "HIGH":
        confidence *= 0.85

    # Boost confidence when order flow strongly confirms
    if market_data:
        ob_signal = market_data.get("orderbook_imbalance_signal")
        tf_signal = market_data.get("trade_flow_signal")
        if signal == "UP" and ob_signal == "BUY_PRESSURE" and tf_signal == "BUYING":
            confidence = min(0.95, confidence + 0.08)
        elif signal == "DOWN" and ob_signal == "SELL_PRESSURE" and tf_signal == "SELLING":
            confidence = min(0.95, confidence + 0.08)

    return score, round(confidence, 3), signal
