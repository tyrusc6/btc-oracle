"""
BTC Oracle - Pattern Analyzer (Production Grade)
"""

import db


def analyze_patterns():
    signals = db.select("signals", "outcome=not.is.null&order=created_at.desc&limit=200")
    if not signals or len(signals) < 10:
        return {}

    results = {"total_analyzed": len(signals)}

    up_signals = [s for s in signals if s["signal"] == "UP"]
    down_signals = [s for s in signals if s["signal"] == "DOWN"]
    up_wins = len([s for s in up_signals if s["outcome"] == "WIN"])
    down_wins = len([s for s in down_signals if s["outcome"] == "WIN"])

    results["up_signal_win_rate"] = round(up_wins / len(up_signals) * 100, 1) if up_signals else 0
    results["down_signal_win_rate"] = round(down_wins / len(down_signals) * 100, 1) if down_signals else 0
    results["better_direction"] = "UP" if results["up_signal_win_rate"] > results["down_signal_win_rate"] else "DOWN"

    # Use "is not None" for numeric checks (0.0 is valid)
    high_conf = [s for s in signals if s.get("confidence") is not None and s["confidence"] >= 0.7]
    mid_conf = [s for s in signals if s.get("confidence") is not None and 0.5 <= s["confidence"] < 0.7]
    low_conf = [s for s in signals if s.get("confidence") is not None and s["confidence"] < 0.5]

    results["high_confidence_win_rate"] = round(len([s for s in high_conf if s["outcome"] == "WIN"]) / len(high_conf) * 100, 1) if high_conf else 0
    results["mid_confidence_win_rate"] = round(len([s for s in mid_conf if s["outcome"] == "WIN"]) / len(mid_conf) * 100, 1) if mid_conf else 0
    results["low_confidence_win_rate"] = round(len([s for s in low_conf if s["outcome"] == "WIN"]) / len(low_conf) * 100, 1) if low_conf else 0

    rsi_low = [s for s in signals if s.get("rsi") is not None and s["rsi"] < 30]
    rsi_mid = [s for s in signals if s.get("rsi") is not None and 30 <= s["rsi"] <= 70]
    rsi_high = [s for s in signals if s.get("rsi") is not None and s["rsi"] > 70]

    results["rsi_oversold_win_rate"] = round(len([s for s in rsi_low if s["outcome"] == "WIN"]) / len(rsi_low) * 100, 1) if rsi_low else 0
    results["rsi_neutral_win_rate"] = round(len([s for s in rsi_mid if s["outcome"] == "WIN"]) / len(rsi_mid) * 100, 1) if rsi_mid else 0
    results["rsi_overbought_win_rate"] = round(len([s for s in rsi_high if s["outcome"] == "WIN"]) / len(rsi_high) * 100, 1) if rsi_high else 0

    macd_pos = [s for s in signals if s.get("macd") is not None and s["macd"] > 0]
    macd_neg = [s for s in signals if s.get("macd") is not None and s["macd"] < 0]

    results["macd_positive_win_rate"] = round(len([s for s in macd_pos if s["outcome"] == "WIN"]) / len(macd_pos) * 100, 1) if macd_pos else 0
    results["macd_negative_win_rate"] = round(len([s for s in macd_neg if s["outcome"] == "WIN"]) / len(macd_neg) * 100, 1) if macd_neg else 0

    mom_pos = [s for s in signals if s.get("momentum") is not None and s["momentum"] > 0]
    mom_neg = [s for s in signals if s.get("momentum") is not None and s["momentum"] < 0]

    results["momentum_positive_win_rate"] = round(len([s for s in mom_pos if s["outcome"] == "WIN"]) / len(mom_pos) * 100, 1) if mom_pos else 0
    results["momentum_negative_win_rate"] = round(len([s for s in mom_neg if s["outcome"] == "WIN"]) / len(mom_neg) * 100, 1) if mom_neg else 0

    changes = []
    for s in signals:
        if s.get("btc_price_at_signal") is not None and s.get("btc_price_at_close") is not None:
            changes.append(abs(s["btc_price_at_close"] - s["btc_price_at_signal"]))

    if changes:
        results["avg_15m_move"] = round(sum(changes) / len(changes), 2)
        results["max_15m_move"] = round(max(changes), 2)
        results["min_15m_move"] = round(min(changes), 2)

    recent_10 = signals[:10]
    recent_wins = len([s for s in recent_10 if s["outcome"] == "WIN"])
    results["recent_10_win_rate"] = round(recent_wins / len(recent_10) * 100, 1) if recent_10 else 0

    return results


def get_pattern_summary():
    patterns = analyze_patterns()
    if not patterns:
        return "Not enough data for pattern analysis yet."

    lines = []
    lines.append(f"Analyzed: {patterns.get('total_analyzed', 0)} signals")
    lines.append(f"UP WR: {patterns.get('up_signal_win_rate', 0)}% | DOWN WR: {patterns.get('down_signal_win_rate', 0)}% | Better: {patterns.get('better_direction', 'N/A')}")
    lines.append(f"High conf WR: {patterns.get('high_confidence_win_rate', 0)}% | Mid: {patterns.get('mid_confidence_win_rate', 0)}% | Low: {patterns.get('low_confidence_win_rate', 0)}%")
    lines.append(f"RSI <30 WR: {patterns.get('rsi_oversold_win_rate', 0)}% | 30-70: {patterns.get('rsi_neutral_win_rate', 0)}% | >70: {patterns.get('rsi_overbought_win_rate', 0)}%")
    lines.append(f"MACD+ WR: {patterns.get('macd_positive_win_rate', 0)}% | MACD- WR: {patterns.get('macd_negative_win_rate', 0)}%")
    lines.append(f"Recent 10 WR: {patterns.get('recent_10_win_rate', 0)}%")
    if patterns.get("avg_15m_move"):
        lines.append(f"Avg 15m move: ${patterns['avg_15m_move']} | Max: ${patterns.get('max_15m_move', 0)}")

    return "\n".join(lines)


if __name__ == "__main__":
    print(get_pattern_summary())
