"""
BTC Oracle - Pattern Analyzer
Analyzes the bot's own trading history to find when it performs best/worst.
Feeds statistical insights back into Claude's decision making.
"""

import db


def analyze_patterns():
    """Analyze all past signals to find performance patterns."""
    signals = db.select("signals", "outcome=not.is.null&order=created_at.desc&limit=200")
    if not signals or len(signals) < 10:
        return {}

    results = {
        "total_analyzed": len(signals),
    }

    # Win rate by signal direction
    up_signals = [s for s in signals if s["signal"] == "UP"]
    down_signals = [s for s in signals if s["signal"] == "DOWN"]
    up_wins = len([s for s in up_signals if s["outcome"] == "WIN"])
    down_wins = len([s for s in down_signals if s["outcome"] == "WIN"])

    results["up_signal_win_rate"] = round(up_wins / len(up_signals) * 100, 1) if up_signals else 0
    results["down_signal_win_rate"] = round(down_wins / len(down_signals) * 100, 1) if down_signals else 0
    results["better_direction"] = "UP" if results["up_signal_win_rate"] > results["down_signal_win_rate"] else "DOWN"

    # Win rate by confidence level
    high_conf = [s for s in signals if s.get("confidence") and s["confidence"] >= 0.7]
    mid_conf = [s for s in signals if s.get("confidence") and 0.5 <= s["confidence"] < 0.7]
    low_conf = [s for s in signals if s.get("confidence") and s["confidence"] < 0.5]

    results["high_confidence_win_rate"] = round(len([s for s in high_conf if s["outcome"] == "WIN"]) / len(high_conf) * 100, 1) if high_conf else 0
    results["mid_confidence_win_rate"] = round(len([s for s in mid_conf if s["outcome"] == "WIN"]) / len(mid_conf) * 100, 1) if mid_conf else 0
    results["low_confidence_win_rate"] = round(len([s for s in low_conf if s["outcome"] == "WIN"]) / len(low_conf) * 100, 1) if low_conf else 0

    # Win rate by RSI range
    rsi_low = [s for s in signals if s.get("rsi") and s["rsi"] < 30]
    rsi_mid = [s for s in signals if s.get("rsi") and 30 <= s["rsi"] <= 70]
    rsi_high = [s for s in signals if s.get("rsi") and s["rsi"] > 70]

    results["rsi_oversold_win_rate"] = round(len([s for s in rsi_low if s["outcome"] == "WIN"]) / len(rsi_low) * 100, 1) if rsi_low else 0
    results["rsi_neutral_win_rate"] = round(len([s for s in rsi_mid if s["outcome"] == "WIN"]) / len(rsi_mid) * 100, 1) if rsi_mid else 0
    results["rsi_overbought_win_rate"] = round(len([s for s in rsi_high if s["outcome"] == "WIN"]) / len(rsi_high) * 100, 1) if rsi_high else 0

    # Win rate by MACD direction
    macd_pos = [s for s in signals if s.get("macd") and s["macd"] > 0]
    macd_neg = [s for s in signals if s.get("macd") and s["macd"] < 0]

    results["macd_positive_win_rate"] = round(len([s for s in macd_pos if s["outcome"] == "WIN"]) / len(macd_pos) * 100, 1) if macd_pos else 0
    results["macd_negative_win_rate"] = round(len([s for s in macd_neg if s["outcome"] == "WIN"]) / len(macd_neg) * 100, 1) if macd_neg else 0

    # Win rate by momentum direction
    mom_pos = [s for s in signals if s.get("momentum") and s["momentum"] > 0]
    mom_neg = [s for s in signals if s.get("momentum") and s["momentum"] < 0]

    results["momentum_positive_win_rate"] = round(len([s for s in mom_pos if s["outcome"] == "WIN"]) / len(mom_pos) * 100, 1) if mom_pos else 0
    results["momentum_negative_win_rate"] = round(len([s for s in mom_neg if s["outcome"] == "WIN"]) / len(mom_neg) * 100, 1) if mom_neg else 0

    # Price change analysis - how much does BTC typically move in 15 min?
    changes = []
    for s in signals:
        if s.get("btc_price_at_signal") and s.get("btc_price_at_close"):
            change = abs(s["btc_price_at_close"] - s["btc_price_at_signal"])
            changes.append(change)

    if changes:
        results["avg_15m_move"] = round(sum(changes) / len(changes), 2)
        results["max_15m_move"] = round(max(changes), 2)
        results["min_15m_move"] = round(min(changes), 2)

    # Recent trend - last 10 vs overall
    recent_10 = signals[:10]
    recent_wins = len([s for s in recent_10 if s["outcome"] == "WIN"])
    results["recent_10_win_rate"] = round(recent_wins / len(recent_10) * 100, 1) if recent_10 else 0
    results["trending_better"] = results["recent_10_win_rate"] > results.get("up_signal_win_rate", 50)

    # Streak analysis
    streaks = []
    current_streak = 0
    current_type = None
    for s in reversed(signals):
        if current_type is None:
            current_type = s["outcome"]
            current_streak = 1
        elif s["outcome"] == current_type:
            current_streak += 1
        else:
            streaks.append((current_type, current_streak))
            current_type = s["outcome"]
            current_streak = 1
    if current_type:
        streaks.append((current_type, current_streak))

    win_streaks = [s[1] for s in streaks if s[0] == "WIN"]
    loss_streaks = [s[1] for s in streaks if s[0] == "LOSS"]
    results["avg_win_streak"] = round(sum(win_streaks) / len(win_streaks), 1) if win_streaks else 0
    results["avg_loss_streak"] = round(sum(loss_streaks) / len(loss_streaks), 1) if loss_streaks else 0
    results["longest_win_streak"] = max(win_streaks) if win_streaks else 0
    results["longest_loss_streak"] = max(loss_streaks) if loss_streaks else 0

    return results


def get_pattern_summary():
    """Get a human-readable summary of patterns for Claude."""
    patterns = analyze_patterns()
    if not patterns:
        return "Not enough data for pattern analysis yet."

    lines = []
    lines.append(f"Signals analyzed: {patterns.get('total_analyzed', 0)}")
    lines.append(f"UP signals win rate: {patterns.get('up_signal_win_rate', 0)}% | DOWN signals win rate: {patterns.get('down_signal_win_rate', 0)}%")
    lines.append(f"Better direction: {patterns.get('better_direction', 'N/A')}")
    lines.append(f"High confidence (>70%) win rate: {patterns.get('high_confidence_win_rate', 0)}%")
    lines.append(f"Mid confidence (50-70%) win rate: {patterns.get('mid_confidence_win_rate', 0)}%")
    lines.append(f"RSI oversold win rate: {patterns.get('rsi_oversold_win_rate', 0)}% | Neutral: {patterns.get('rsi_neutral_win_rate', 0)}% | Overbought: {patterns.get('rsi_overbought_win_rate', 0)}%")
    lines.append(f"MACD positive win rate: {patterns.get('macd_positive_win_rate', 0)}% | Negative: {patterns.get('macd_negative_win_rate', 0)}%")
    lines.append(f"Recent 10 win rate: {patterns.get('recent_10_win_rate', 0)}% | Trending {'BETTER' if patterns.get('trending_better') else 'WORSE'}")
    lines.append(f"Avg 15m move: ${patterns.get('avg_15m_move', 0)} | Max: ${patterns.get('max_15m_move', 0)}")
    lines.append(f"Avg win streak: {patterns.get('avg_win_streak', 0)} | Avg loss streak: {patterns.get('avg_loss_streak', 0)}")

    return "\n".join(lines)


if __name__ == "__main__":
    print(get_pattern_summary())
