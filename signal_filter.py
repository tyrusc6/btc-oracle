"""
BTC Oracle - Signal Filter & Shadow Mode
Decides whether to TRADE or WAIT based on signal quality.
Tracks scoring model vs Claude agreement.
Calculates feature importance from historical data.
"""

import db
import numpy as np


def should_trade(score, score_confidence, claude_signal, claude_confidence, 
                 regime_data, indicators, market_data=None):
    """
    STRICT filter - only trade high-probability setups.
    Returns (should_trade: bool, reason: str, adjusted_confidence: float)
    """
    reasons_to_trade = []
    reasons_to_wait = []
    
    # Rule 1: Scoring model and Claude MUST agree (hard requirement)
    score_signal = "UP" if score > 0 else "DOWN"
    if score_signal == claude_signal:
        reasons_to_trade.append("Model and Claude AGREE")
    else:
        reasons_to_wait.append(f"DISAGREE: model={score_signal} claude={claude_signal}")
        # Disagreement is an automatic WAIT
        adjusted = min(0.4, (score_confidence + claude_confidence) / 2)
        return False, f"WAIT (DISAGREE): {reasons_to_wait[0]}", round(adjusted, 3)
    
    # Rule 2: High confidence required (raised from 0.65 to 0.75)
    min_confidence = 0.75
    if score_confidence >= min_confidence and claude_confidence >= 0.70:
        reasons_to_trade.append(f"High confidence: model {score_confidence:.0%}, Claude {claude_confidence:.0%}")
    else:
        reasons_to_wait.append(f"Low confidence: model {score_confidence:.0%}, Claude {claude_confidence:.0%}")
    
    # Rule 3: Score magnitude must be strong
    if abs(score) > 0.5:
        reasons_to_trade.append(f"Strong score: {score:+.3f}")
    elif abs(score) > 0.35:
        pass  # neutral, neither helps nor hurts
    else:
        reasons_to_wait.append(f"Weak score: {score:+.3f}")
    
    # Rule 4: Regime must support the trade
    regime = regime_data.get("regime", "UNKNOWN")
    if regime in ("TRENDING_UP", "BREAKOUT_UP") and claude_signal == "UP":
        reasons_to_trade.append(f"Regime {regime} supports UP")
    elif regime in ("TRENDING_DOWN", "BREAKOUT_DOWN") and claude_signal == "DOWN":
        reasons_to_trade.append(f"Regime {regime} supports DOWN")
    elif regime == "HIGH_VOLATILITY":
        reasons_to_wait.append("HIGH_VOLATILITY - skip")
    elif regime == "CHOPPY":
        reasons_to_wait.append("CHOPPY - skip")
    elif regime == "RANGING":
        bb_pos = indicators.get("bollinger_position")
        if bb_pos is not None and (bb_pos < 0.1 or bb_pos > 0.9):
            reasons_to_trade.append("RANGING at extreme")
        else:
            reasons_to_wait.append("RANGING not at extreme")
    elif regime in ("WEAK_UPTREND", "WEAK_DOWNTREND"):
        reasons_to_wait.append(f"Weak trend: {regime}")
    
    # Rule 5: Both timeframe trends must align
    trend_1m = indicators.get("trend_1m", "UNKNOWN")
    trend_5m = indicators.get("trend_5m", "UNKNOWN")
    
    if claude_signal == "UP" and "UPTREND" in trend_1m and "UPTREND" in trend_5m:
        reasons_to_trade.append("Both trends align UP")
    elif claude_signal == "DOWN" and "DOWNTREND" in trend_1m and "DOWNTREND" in trend_5m:
        reasons_to_trade.append("Both trends align DOWN")
    elif "SIDEWAYS" in trend_1m or "SIDEWAYS" in trend_5m or "UNKNOWN" in trend_1m:
        reasons_to_wait.append("Unclear trend")
    
    # Rule 6: Order book and trade flow (live data only)
    if market_data:
        ob = market_data.get("orderbook_imbalance_signal", "BALANCED")
        tf = market_data.get("trade_flow_signal", "NEUTRAL")
        
        if claude_signal == "UP" and ob == "BUY_PRESSURE" and tf == "BUYING":
            reasons_to_trade.append("Order flow confirms UP")
        elif claude_signal == "DOWN" and ob == "SELL_PRESSURE" and tf == "SELLING":
            reasons_to_trade.append("Order flow confirms DOWN")
        elif (ob == "BUY_PRESSURE" and claude_signal == "DOWN") or (ob == "SELL_PRESSURE" and claude_signal == "UP"):
            reasons_to_wait.append("Order flow CONTRADICTS signal")
    
    # Decision: need 4+ reasons to trade AND more trade than wait reasons
    trade_score = len(reasons_to_trade)
    wait_score = len(reasons_to_wait)
    
    should = trade_score >= 4 and trade_score > wait_score
    
    if should:
        adjusted = min(0.95, (score_confidence + claude_confidence) / 2 + 0.05 * trade_score)
    else:
        adjusted = min(0.5, (score_confidence + claude_confidence) / 2 - 0.05 * wait_score)
    
    reason = f"TRADE ({trade_score} for, {wait_score} against)" if should else f"WAIT ({wait_score} against, {trade_score} for)"
    detail = " | ".join(reasons_to_trade + reasons_to_wait)
    
    return should, f"{reason}: {detail}", round(adjusted, 3)
    
    # Rule 6: Strong score magnitude
    if abs(score) > 0.5:
        reasons_to_trade.append(f"Strong score magnitude: {score:+.3f}")
    elif abs(score) < 0.2:
        reasons_to_wait.append(f"Weak score: {score:+.3f}")
    
    # Decision
    trade_score = len(reasons_to_trade)
    wait_score = len(reasons_to_wait)
    
    # Need at least 3 reasons to trade and more reasons to trade than wait
    should = trade_score >= 3 and trade_score > wait_score
    
    # Adjust confidence based on filter
    if should:
        # Boost confidence when everything aligns
        adjusted = min(0.95, (score_confidence + claude_confidence) / 2 + 0.05 * trade_score)
    else:
        adjusted = min(0.5, (score_confidence + claude_confidence) / 2 - 0.05 * wait_score)
    
    reason = f"TRADE ({trade_score} for, {wait_score} against)" if should else f"WAIT ({wait_score} against, {trade_score} for)"
    detail = " | ".join(reasons_to_trade + reasons_to_wait)
    
    return should, f"{reason}: {detail}", round(adjusted, 3)


def track_shadow_mode(score_signal, claude_signal, actual_outcome):
    """Track when scoring model vs Claude agree/disagree and who was right."""
    agreed = score_signal == claude_signal
    
    record = {
        "score_signal": score_signal,
        "claude_signal": claude_signal,
        "agreed": agreed,
        "outcome": actual_outcome,
        "score_correct": score_signal == ("UP" if actual_outcome == "WIN" else "DOWN") if actual_outcome else None,
        "claude_correct": claude_signal == ("UP" if actual_outcome == "WIN" else "DOWN") if actual_outcome else None,
    }
    
    return record


def get_shadow_stats():
    """Calculate shadow mode statistics - who's more accurate?"""
    signals = db.select("signals", "outcome=not.is.null&order=created_at.desc&limit=100")
    if not signals or len(signals) < 10:
        return "Not enough data for shadow analysis."
    
    # We don't have separate score_signal stored, so approximate from indicators
    # In future, we should store the scoring model's signal separately
    return "Shadow mode tracking active. Need separate score storage for full stats."


def calculate_feature_importance():
    """Calculate which features actually correlate with price direction."""
    signals = db.select("signals", "outcome=not.is.null&order=created_at.desc&limit=100")
    if not signals or len(signals) < 20:
        return {}
    
    # For each feature, calculate correlation with actual price direction
    features = {}
    
    for s in signals:
        if not s.get("btc_price_at_signal") or not s.get("btc_price_at_close"):
            continue
        went_up = 1 if s["btc_price_at_close"] > s["btc_price_at_signal"] else 0
        
        # RSI correlation
        if s.get("rsi") is not None:
            if "rsi" not in features:
                features["rsi"] = {"values": [], "outcomes": []}
            features["rsi"]["values"].append(s["rsi"])
            features["rsi"]["outcomes"].append(went_up)
        
        # MACD correlation
        if s.get("macd") is not None:
            if "macd" not in features:
                features["macd"] = {"values": [], "outcomes": []}
            features["macd"]["values"].append(s["macd"])
            features["macd"]["outcomes"].append(went_up)
        
        # MACD histogram
        if s.get("macd_histogram") is not None:
            if "macd_histogram" not in features:
                features["macd_histogram"] = {"values": [], "outcomes": []}
            features["macd_histogram"]["values"].append(s["macd_histogram"])
            features["macd_histogram"]["outcomes"].append(went_up)
        
        # Momentum
        if s.get("momentum") is not None:
            if "momentum" not in features:
                features["momentum"] = {"values": [], "outcomes": []}
            features["momentum"]["values"].append(s["momentum"])
            features["momentum"]["outcomes"].append(went_up)
        
        # EMA difference
        if s.get("ema_9") is not None and s.get("ema_21") is not None:
            if "ema_diff" not in features:
                features["ema_diff"] = {"values": [], "outcomes": []}
            features["ema_diff"]["values"].append(s["ema_9"] - s["ema_21"])
            features["ema_diff"]["outcomes"].append(went_up)
    
    # Calculate correlation for each feature
    importance = {}
    for name, data in features.items():
        if len(data["values"]) >= 10:
            vals = np.array(data["values"])
            outs = np.array(data["outcomes"])
            # Pearson correlation
            if np.std(vals) > 0 and np.std(outs) > 0:
                corr = np.corrcoef(vals, outs)[0, 1]
                importance[name] = {
                    "correlation": round(float(corr), 4),
                    "abs_correlation": round(abs(float(corr)), 4),
                    "predictive": abs(float(corr)) > 0.1,
                    "direction": "POSITIVE" if corr > 0 else "NEGATIVE",
                    "samples": len(data["values"])
                }
    
    # Sort by absolute correlation
    importance = dict(sorted(importance.items(), key=lambda x: x[1]["abs_correlation"], reverse=True))
    
    return importance


def get_feature_importance_summary():
    """Get readable summary for Claude."""
    importance = calculate_feature_importance()
    if not importance:
        return "Not enough data for feature importance."
    
    lines = ["Feature Importance (correlation with price direction):"]
    for name, data in importance.items():
        star = " ***" if data["predictive"] else ""
        lines.append(f"  {name}: {data['correlation']:+.4f} ({data['direction']}, {data['samples']} samples){star}")
    
    predictive = [n for n, d in importance.items() if d["predictive"]]
    noise = [n for n, d in importance.items() if not d["predictive"]]
    
    if predictive:
        lines.append(f"\nPREDICTIVE features: {', '.join(predictive)}")
    if noise:
        lines.append(f"NOISE features (low correlation): {', '.join(noise)}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    print(get_feature_importance_summary())
