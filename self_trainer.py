"""
BTC Oracle - Self Training Engine (V2)
- Reviews EACH trade exactly once (tracks reviewed IDs)
- Every 20 trades: deep strategy review
- Feeds structured rules back, not just text
"""

import os
import json
from datetime import datetime, timezone
from dotenv import load_dotenv
import anthropic
import db
from pattern_analyzer import get_pattern_summary

load_dotenv()

claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Track which signals we've already reviewed (in memory, resets on deploy)
reviewed_signal_ids = set()


def get_unreviewed_signals():
    """Get resolved signals that haven't been reviewed yet."""
    signals = db.select("signals", "outcome=not.is.null&order=created_at.desc&limit=10")
    if not signals:
        return []
    return [s for s in signals if s["id"] not in reviewed_signal_ids]


def get_strategy_document():
    data = db.select("journal", "entry_type=eq.STRATEGY&order=created_at.desc&limit=1")
    return (data[0].get("content") or "No strategy document yet.") if data else "No strategy document yet. Build one from scratch."


def analyze_single_trade(signal):
    """Quick analysis after each trade resolves. Only called once per trade."""
    if not signal or not signal.get("outcome"):
        return

    price_change = signal.get('btc_price_at_close', 0) - signal.get('btc_price_at_signal', 0)

    prompt = f"""Analyze this completed BTC trade in 1-2 sentences:

Signal: {signal['signal']} | Outcome: {signal['outcome']}
Price: ${signal.get('btc_price_at_signal', 0):,.2f} -> ${signal.get('btc_price_at_close', 0):,.2f} (${price_change:+,.2f})
Confidence: {signal.get('confidence', 'N/A')}
RSI: {signal.get('rsi', 'N/A')} | MACD: {signal.get('macd', 'N/A')} | Momentum: {signal.get('momentum', 'N/A')}
Bot's reasoning: {(signal.get('analysis_notes', '') or '')[:200]}

What specifically caused this {'win' if signal['outcome'] == 'WIN' else 'loss'}? One actionable takeaway.

JSON: {{"entry_type": "TRADE_REVIEW", "content": "your 1-2 sentence analysis"}}"""

    try:
        resp = claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        try:
            entry = json.loads(text)
        except json.JSONDecodeError:
            entry = {"entry_type": "TRADE_REVIEW", "content": text[:500]}

        db.insert("journal", {
            "entry_type": "TRADE_REVIEW",
            "content": entry.get("content", ""),
            "win_rate_at_time": 0,
            "total_signals_at_time": signal.get("id", 0)
        })
        
        # Mark as reviewed
        reviewed_signal_ids.add(signal["id"])
        print(f"  Trade #{signal['id']} Review: {entry.get('content', '')[:80]}...")
        return entry
    except Exception as e:
        print(f"  Error reviewing trade #{signal.get('id', '?')}: {e}")
        reviewed_signal_ids.add(signal.get("id", 0))
        return None


def deep_strategy_review():
    """Every 20 trades: comprehensive strategy update."""
    all_signals = db.select("signals", "outcome=not.is.null&order=created_at.desc&limit=200")
    if not all_signals:
        return

    total = len(all_signals)
    if total < 20 or total % 20 != 0:
        return

    print(f"\n  === DEEP STRATEGY REVIEW === Signal #{total}")

    wins = [s for s in all_signals if s["outcome"] == "WIN"]
    losses = [s for s in all_signals if s["outcome"] == "LOSS"]
    win_rate = len(wins) / total * 100

    # Only analyze recent 50 signals for patterns (avoid old broken data)
    recent = all_signals[:50]
    recent_wins = [s for s in recent if s["outcome"] == "WIN"]
    recent_wr = len(recent_wins) / len(recent) * 100 if recent else 0

    # Find what conditions correlate with actual price going up
    up_moves = [s for s in recent if s.get("btc_price_at_close") and s.get("btc_price_at_signal") and s["btc_price_at_close"] > s["btc_price_at_signal"]]
    down_moves = [s for s in recent if s.get("btc_price_at_close") and s.get("btc_price_at_signal") and s["btc_price_at_close"] <= s["btc_price_at_signal"]]

    up_conditions = ""
    if up_moves:
        avg_rsi_up = sum(s.get("rsi", 50) or 50 for s in up_moves) / len(up_moves)
        avg_macd_up = sum(s.get("macd", 0) or 0 for s in up_moves) / len(up_moves)
        avg_mom_up = sum(s.get("momentum", 0) or 0 for s in up_moves) / len(up_moves)
        up_conditions = f"When price went UP ({len(up_moves)} times): avg RSI={avg_rsi_up:.1f}, avg MACD={avg_macd_up:.2f}, avg Momentum={avg_mom_up:.2f}"

    down_conditions = ""
    if down_moves:
        avg_rsi_dn = sum(s.get("rsi", 50) or 50 for s in down_moves) / len(down_moves)
        avg_macd_dn = sum(s.get("macd", 0) or 0 for s in down_moves) / len(down_moves)
        avg_mom_dn = sum(s.get("momentum", 0) or 0 for s in down_moves) / len(down_moves)
        down_conditions = f"When price went DOWN ({len(down_moves)} times): avg RSI={avg_rsi_dn:.1f}, avg MACD={avg_macd_dn:.2f}, avg Momentum={avg_mom_dn:.2f}"

    # Get recent trade reviews
    reviews = db.select("journal", "entry_type=eq.TRADE_REVIEW&order=created_at.desc&limit=20")
    review_text = "\n".join(f"  - {r['content']}" for r in reviews) if reviews else "None"

    # Time analysis
    hour_stats = {}
    for s in recent:
        try:
            hour = datetime.fromisoformat(s["created_at"].replace("Z", "+00:00")).hour
            if hour not in hour_stats:
                hour_stats[hour] = {"total": 0, "up": 0}
            hour_stats[hour]["total"] += 1
            if s.get("btc_price_at_close") and s.get("btc_price_at_signal") and s["btc_price_at_close"] > s["btc_price_at_signal"]:
                hour_stats[hour]["up"] += 1
        except:
            pass

    time_text = ""
    for h in sorted(hour_stats.keys()):
        if hour_stats[h]["total"] >= 3:
            up_pct = hour_stats[h]["up"] / hour_stats[h]["total"] * 100
            time_text += f"  {h}:00 UTC: {up_pct:.0f}% up ({hour_stats[h]['total']} signals)\n"

    prompt = f"""You are BTC Oracle's strategic brain. Write a NEW strategy document after {total} total trades.

RECENT PERFORMANCE (last 50): {recent_wr:.1f}% win rate
OVERALL: {win_rate:.1f}% ({len(wins)}W/{len(losses)}L)

WHAT ACTUALLY PREDICTS PRICE MOVEMENT:
{up_conditions}
{down_conditions}

TIME OF DAY PATTERNS:
{time_text}

RECENT TRADE REVIEWS:
{review_text}

Write a CONCISE strategy (5-7 bullet points max). Focus ONLY on:
1. Which indicators actually predict price direction (based on the data above)
2. When to signal UP vs DOWN
3. What conditions to avoid

DO NOT reference old rules. Only use the data provided above.
Keep it under 500 characters total.

JSON: {{"entry_type": "STRATEGY", "content": "your strategy"}}"""

    try:
        resp = claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        try:
            entry = json.loads(text)
        except json.JSONDecodeError:
            content = text
            if '"content"' in text:
                start = text.find('"content"')
                start = text.find(':', start) + 1
                content = text[start:].strip().strip('"').rstrip('"}')
            entry = {"entry_type": "STRATEGY", "content": content[:2000]}

        db.insert("journal", {
            "entry_type": "STRATEGY",
            "content": entry.get("content", "")[:2000],
            "win_rate_at_time": win_rate / 100,
            "total_signals_at_time": total
        })
        print(f"  STRATEGY UPDATED: {entry.get('content', '')[:150]}...")
        return entry
    except Exception as e:
        print(f"  Error in deep review: {e}")
        return None


def run_self_training():
    """Run after each signal cycle."""
    print("  Running self-training...")

    # Review any unreviewed trades (usually 1)
    unreviewed = get_unreviewed_signals()
    if unreviewed:
        for signal in unreviewed[:3]:  # max 3 per cycle
            analyze_single_trade(signal)
    else:
        print("  No new trades to review.")

    # Deep strategy review every 20 trades
    all_resolved = db.select("signals", "outcome=not.is.null&select=id")
    if all_resolved:
        total = len(all_resolved)
        if total >= 20 and total % 20 == 0:
            deep_strategy_review()


if __name__ == "__main__":
    run_self_training()
