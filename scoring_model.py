"""
BTC Oracle - Weighted Scoring Model
Each indicator gets a weight based on historical predictive accuracy.
Combines all scores into a single numerical signal.
Updates weights automatically as more data comes in.
"""

import db
import numpy as np


def calculate_indicator_weights():
    """Analyze past signals to determine how predictive each indicator is."""
    signals = db.select("signals", "outcome=not.is.null&order=created_at.desc&limit=500")
    if not signals or len(signals) < 20:
        return get_default_weights()

    weights = {}

    # RSI accuracy by condition
    rsi_oversold = [s for s in signals if s.get("rsi") and s["rsi"] < 30]
    rsi_overbought = [s for s in signals if s.get("rsi") and s["rsi"] > 70]
    rsi_neutral = [s for s in signals if s.get("rsi") and 30 <= s["rsi"] <= 70]

    if rsi_oversold:
        # When RSI < 30, how often does UP win?
        up_wins = len([s for s in rsi_oversold if s["signal"] == "UP" and s["outcome"] == "WIN"])
        weights["rsi_oversold_up"] = up_wins / len(rsi_oversold) if rsi_oversold else 0.5
    if rsi_overbought:
        down_wins = len([s for s in rsi_overbought if s["signal"] == "DOWN" and s["outcome"] == "WIN"])
        weights["rsi_overbought_down"] = down_wins / len(rsi_overbought) if rsi_overbought else 0.5

    # MACD accuracy
    macd_pos = [s for s in signals if s.get("macd") and s["macd"] > 0]
    macd_neg = [s for s in signals if s.get("macd") and s["macd"] < 0]

    if macd_pos:
        up_wins = len([s for s in macd_pos if s["signal"] == "UP" and s["outcome"] == "WIN"])
        weights["macd_positive_up"] = up_wins / len(macd_pos)
    if macd_neg:
        down_wins = len([s for s in macd_neg if s["signal"] == "DOWN" and s["outcome"] == "WIN"])
        weights["macd_negative_down"] = down_wins / len(macd_neg)

    # Momentum accuracy
    mom_pos = [s for s in signals if s.get("momentum") and s["momentum"] > 0]
    mom_neg = [s for s in signals if s.get("momentum") and s["momentum"] < 0]

    if mom_pos:
        up_wins = len([s for s in mom_pos if s["signal"] == "UP" and s["outcome"] == "WIN"])
        weights["momentum_positive_up"] = up_wins / len(mom_pos)
    if mom_neg:
        down_wins = len([s for s in mom_neg if s["signal"] == "DOWN" and s["outcome"] == "WIN"])
        weights["momentum_negative_down"] = down_wins / len(mom_neg)

    # EMA crossover accuracy
    ema_signals = [s for s in signals if s.get("ema_9") and s.get("ema_21")]
    if ema_signals:
        bullish_cross = [s for s in ema_signals if s["ema_9"] > s["ema_21"]]
        bearish_cross = [s for s in ema_signals if s["ema_9"] <= s["ema_21"]]
        if bullish_cross:
            up_wins = len([s for s in bullish_cross if s["signal"] == "UP" and s["outcome"] == "WIN"])
            weights["ema_bullish_up"] = up_wins / len(bullish_cross)
        if bearish_cross:
            down_wins = len([s for s in bearish_cross if s["signal"] == "DOWN" and s["outcome"] == "WIN"])
            weights["ema_bearish_down"] = down_wins / len(bearish_cross)

    # Overall direction accuracy
    up_signals = [s for s in signals if s["signal"] == "UP"]
    down_signals = [s for s in signals if s["signal"] == "DOWN"]
    if up_signals:
        weights["overall_up_accuracy"] = len([s for s in up_signals if s["outcome"] == "WIN"]) / len(up_signals)
    if down_signals:
        weights["overall_down_accuracy"] = len([s for s in down_signals if s["outcome"] == "WIN"]) / len(down_signals)

    return weights


def get_default_weights():
    """Default weights before we have enough data."""
    return {
        "rsi_oversold_up": 0.55,
        "rsi_overbought_down": 0.55,
        "macd_positive_up": 0.52,
        "macd_negative_down": 0.52,
        "momentum_positive_up": 0.53,
        "momentum_negative_down": 0.53,
        "ema_bullish_up": 0.52,
        "ema_bearish_down": 0.52,
        "overall_up_accuracy": 0.50,
        "overall_down_accuracy": 0.50,
    }


def score_signal(indicators, market_data=None):
    """
    Generate a numerical score from -1.0 (strong DOWN) to +1.0 (strong UP).
    Each indicator contributes a weighted vote.
    """
    weights = calculate_indicator_weights()
    votes = []  # list of (direction_score, weight) where direction is -1 or +1

    price = indicators.get("current_price", 0)

    # RSI signal
    rsi = indicators.get("rsi")
    if rsi is not None:
        if rsi < 30:
            votes.append((+1, weights.get("rsi_oversold_up", 0.55) * 1.5))  # oversold = likely bounce
        elif rsi > 70:
            votes.append((-1, weights.get("rsi_overbought_down", 0.55) * 1.5))
        elif rsi < 45:
            votes.append((+1, 0.3))
        elif rsi > 55:
            votes.append((-1, 0.3))

    # MACD signal
    macd = indicators.get("macd")
    macd_hist = indicators.get("macd_histogram")
    if macd is not None:
        if macd > 0:
            votes.append((+1, weights.get("macd_positive_up", 0.52)))
        else:
            votes.append((-1, weights.get("macd_negative_down", 0.52)))
    if macd_hist is not None:
        if macd_hist > 0:
            votes.append((+1, 0.4))
        else:
            votes.append((-1, 0.4))

    # Momentum
    momentum = indicators.get("momentum")
    if momentum is not None:
        if momentum > 0:
            votes.append((+1, weights.get("momentum_positive_up", 0.53)))
        else:
            votes.append((-1, weights.get("momentum_negative_down", 0.53)))

    # EMA crossover
    ema_9 = indicators.get("ema_9")
    ema_21 = indicators.get("ema_21")
    if ema_9 and ema_21:
        if ema_9 > ema_21:
            votes.append((+1, weights.get("ema_bullish_up", 0.52)))
        else:
            votes.append((-1, weights.get("ema_bearish_down", 0.52)))

    # Bollinger Band position
    bb_pos = indicators.get("bollinger_position")
    if bb_pos is not None:
        if bb_pos < 0.2:
            votes.append((+1, 0.6))  # near lower band = bounce likely
        elif bb_pos > 0.8:
            votes.append((-1, 0.6))  # near upper band = pullback likely
        elif bb_pos < 0.4:
            votes.append((+1, 0.3))
        elif bb_pos > 0.6:
            votes.append((-1, 0.3))

    # VWAP
    vwap = indicators.get("vwap")
    if vwap and price:
        if price > vwap:
            votes.append((+1, 0.4))
        else:
            votes.append((-1, 0.4))

    # OBV trend
    obv = indicators.get("obv_trend")
    if obv == "RISING":
        votes.append((+1, 0.5))
    elif obv == "FALLING":
        votes.append((-1, 0.5))

    # Stochastic RSI
    stoch_k = indicators.get("stoch_rsi_k")
    if stoch_k is not None:
        if stoch_k < 20:
            votes.append((+1, 0.6))
        elif stoch_k > 80:
            votes.append((-1, 0.6))

    # Rate of change
    roc = indicators.get("rate_of_change")
    if roc is not None:
        if roc > 0.1:
            votes.append((+1, 0.4))
        elif roc < -0.1:
            votes.append((-1, 0.4))

    # Market data signals
    if market_data:
        # Order book imbalance
        ob_signal = market_data.get("orderbook_imbalance_signal")
        if ob_signal == "BUY_PRESSURE":
            votes.append((+1, 0.7))
        elif ob_signal == "SELL_PRESSURE":
            votes.append((-1, 0.7))

        # Trade flow
        tf_signal = market_data.get("trade_flow_signal")
        if tf_signal == "BUYING":
            votes.append((+1, 0.7))
        elif tf_signal == "SELLING":
            votes.append((-1, 0.7))

        # Whale activity
        whale = market_data.get("trade_flow_whale_signal")
        if whale == "WHALE_BUYING":
            votes.append((+1, 0.8))
        elif whale == "WHALE_SELLING":
            votes.append((-1, 0.8))

        # Momentum alignment
        alignment = market_data.get("momentum_alignment")
        if alignment == "STRONG_BULL":
            votes.append((+1, 0.9))
        elif alignment == "STRONG_BEAR":
            votes.append((-1, 0.9))
        elif alignment == "MIXED_BULL":
            votes.append((+1, 0.4))
        elif alignment == "MIXED_BEAR":
            votes.append((-1, 0.4))

        # Volatility regime
        vol_regime = market_data.get("volatility_regime")
        if vol_regime == "HIGH":
            # In high vol, signals are less reliable - reduce all weights
            pass  # handled by confidence later

    # Calculate weighted score
    if not votes:
        return 0.0, 0.5, "NO_DATA"

    total_weight = sum(abs(w) for _, w in votes)
    if total_weight == 0:
        return 0.0, 0.5, "NO_DATA"

    weighted_sum = sum(direction * weight for direction, weight in votes)
    score = weighted_sum / total_weight  # -1 to +1

    # Convert to signal
    signal = "UP" if score > 0 else "DOWN"

    # Confidence based on agreement strength
    agreement = abs(score)
    confidence = min(0.95, 0.5 + agreement * 0.45)

    # Reduce confidence in high volatility
    if market_data and market_data.get("volatility_regime") == "HIGH":
        confidence *= 0.85

    return score, round(confidence, 3), signal


def get_scoring_summary(indicators, market_data=None):
    """Get a readable summary of the scoring model output."""
    score, confidence, signal = score_signal(indicators, market_data)
    weights = calculate_indicator_weights()

    summary = f"Score: {score:+.3f} | Signal: {signal} | Confidence: {confidence:.0%}\n"
    summary += "Indicator weights (from history):\n"
    for k, v in weights.items():
        summary += f"  {k}: {v:.1%}\n"

    return summary


if __name__ == "__main__":
    print("Scoring model - needs indicators to run.")
    weights = calculate_indicator_weights()
    print("Current weights:")
    for k, v in weights.items():
        print(f"  {k}: {v:.1%}")
