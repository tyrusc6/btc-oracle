"""
BTC Oracle - Self Training Engine
Analyzes every single trade after outcome.
Does deep strategic review every 20 trades.
Builds a living strategy document that feeds into every prediction.
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


def get_latest_resolved_signal():
    """Get the most recently resolved signal."""
    data = db.select("signals", "outcome=not.is.null&order=created_at.desc&limit=1")
    return data[0] if data else None


def get_strategy_document():
    """Get the current strategy document from journal."""
    data = db.select("journal", "entry_type=eq.STRATEGY&order=created_at.desc&limit=1")
    return data[0]["content"] if data else "No strategy document yet. Build one from scratch."


def analyze_single_trade(signal):
    """Quick analysis after every single trade resolves."""
    if not signal or not signal.get("outcome"):
        return

    prompt = f"""You are BTC Oracle's self-training module. Analyze this completed trade:

Signal: {signal['signal']} @ ${signal.get('btc_price_at_signal', 0):,.2f}
Close Price: ${signal.get('btc_price_at_close', 0):,.2f}
Outcome: {signal['outcome']}
Confidence: {signal.get('confidence', 'N/A')}
RSI: {signal.get('rsi', 'N/A')}
MACD: {signal.get('macd', 'N/A')}
Momentum: {signal.get('momentum', 'N/A')}
Your Reasoning: {signal.get('analysis_notes', 'N/A')[:300]}

Price moved: ${(signal.get('btc_price_at_close', 0) - signal.get('btc_price_at_signal', 0)):+,.2f}

In 1-2 sentences: What specifically went right or wrong? What's the ONE takeaway?

JSON only: {{"entry_type": "TRADE_REVIEW", "content": "your analysis"}}"""

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
            "total_signals_at_time": 0
        })
        print(f"  Trade Review: {entry.get('content', '')[:80]}...")
        return entry
    except Exception as e:
        print(f"  Error in trade review: {e}")
        return None


def deep_strategy_review():
    """Every 20 trades: deep dive creating/updating the master strategy document."""
    # Get all resolved signals
    all_signals = db.select("signals", "outcome=not.is.null&order=created_at.desc&limit=200")
    if not all_signals:
        return

    total = len(all_signals)
    if total < 20 or total % 20 != 0:
        return

    print(f"\n  === DEEP STRATEGY REVIEW (every 20 trades) === Signal #{total}")

    # Get current strategy
    current_strategy = get_strategy_document()

    # Get pattern analysis
    pattern_summary = get_pattern_summary()

    # Get recent trade reviews
    reviews = db.select("journal", "entry_type=eq.TRADE_REVIEW&order=created_at.desc&limit=20")
    review_text = ""
    for r in reviews:
        review_text += f"  - {r['content']}\n"

    # Build detailed signal breakdown
    wins = [s for s in all_signals if s["outcome"] == "WIN"]
    losses = [s for s in all_signals if s["outcome"] == "LOSS"]

    # Analyze winning conditions
    win_conditions = ""
    if wins:
        avg_win_rsi = sum(s.get("rsi", 50) or 50 for s in wins) / len(wins)
        avg_win_macd = sum(s.get("macd", 0) or 0 for s in wins) / len(wins)
        avg_win_momentum = sum(s.get("momentum", 0) or 0 for s in wins) / len(wins)
        avg_win_conf = sum(s.get("confidence", 0.5) or 0.5 for s in wins) / len(wins)
        up_wins = len([s for s in wins if s["signal"] == "UP"])
        down_wins = len([s for s in wins if s["signal"] == "DOWN"])
        win_conditions = f"""
  Winning trades ({len(wins)}):
    Avg RSI: {avg_win_rsi:.1f} | Avg MACD: {avg_win_macd:.4f} | Avg Momentum: {avg_win_momentum:.2f}
    Avg Confidence: {avg_win_conf:.0%}
    UP wins: {up_wins} | DOWN wins: {down_wins}"""

    # Analyze losing conditions
    loss_conditions = ""
    if losses:
        avg_loss_rsi = sum(s.get("rsi", 50) or 50 for s in losses) / len(losses)
        avg_loss_macd = sum(s.get("macd", 0) or 0 for s in losses) / len(losses)
        avg_loss_momentum = sum(s.get("momentum", 0) or 0 for s in losses) / len(losses)
        avg_loss_conf = sum(s.get("confidence", 0.5) or 0.5 for s in losses) / len(losses)
        up_losses = len([s for s in losses if s["signal"] == "UP"])
        down_losses = len([s for s in losses if s["signal"] == "DOWN"])
        loss_conditions = f"""
  Losing trades ({len(losses)}):
    Avg RSI: {avg_loss_rsi:.1f} | Avg MACD: {avg_loss_macd:.4f} | Avg Momentum: {avg_loss_momentum:.2f}
    Avg Confidence: {avg_loss_conf:.0%}
    UP losses: {up_losses} | DOWN losses: {down_losses}"""

    # Time analysis
    hour_wins = {}
    hour_total = {}
    for s in all_signals:
        try:
            hour = datetime.fromisoformat(s["created_at"].replace("Z", "+00:00")).hour
            hour_total[hour] = hour_total.get(hour, 0) + 1
            if s["outcome"] == "WIN":
                hour_wins[hour] = hour_wins.get(hour, 0) + 1
        except:
            pass

    time_analysis = ""
    if hour_total:
        best_hour = max(hour_total.keys(), key=lambda h: hour_wins.get(h, 0) / hour_total[h] if hour_total[h] >= 3 else 0)
        worst_hour = min(hour_total.keys(), key=lambda h: hour_wins.get(h, 0) / hour_total[h] if hour_total[h] >= 3 else 1)
        time_analysis = f"""
  Best hour (UTC): {best_hour}:00 ({hour_wins.get(best_hour, 0)}/{hour_total[best_hour]} wins)
  Worst hour (UTC): {worst_hour}:00 ({hour_wins.get(worst_hour, 0)}/{hour_total[worst_hour]} wins)"""

    prompt = f"""You are BTC Oracle's strategic brain. This is your DEEP STRATEGY REVIEW after {total} trades.

=== CURRENT STRATEGY ===
{current_strategy}

=== PERFORMANCE PATTERNS ===
{pattern_summary}

=== WINNING vs LOSING CONDITIONS ===
{win_conditions}
{loss_conditions}

=== TIME OF DAY ANALYSIS ===
{time_analysis}

=== RECENT TRADE REVIEWS (individual learnings) ===
{review_text}

=== YOUR TASK ===
Write a comprehensive STRATEGY DOCUMENT that will guide all future predictions.
This replaces your previous strategy. Include:

1. CORE RULES - 3-5 specific rules based on what ACTUALLY works (backed by your data)
2. AVOID LIST - specific conditions where you consistently lose
3. SWEET SPOTS - specific conditions where you consistently win
4. CONFIDENCE CALIBRATION - when to be high vs low confidence
5. TIME FACTORS - best/worst times to trade
6. DIRECTION BIAS - any bias toward UP or DOWN based on your stats
7. KEY INSIGHT - the single most important thing you've learned

Be SPECIFIC with numbers. Reference actual win rates and conditions.
This document will be fed into every future prediction you make.

JSON only: {{"entry_type": "STRATEGY", "content": "your full strategy document"}}"""

    try:
        resp = claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        
        # Try direct JSON parse first
        try:
            entry = json.loads(text)
        except json.JSONDecodeError:
            # If JSON fails, extract content manually
            content = text
            if '"content"' in text:
                start = text.find('"content"')
                start = text.find(':', start) + 1
                # Find the content value
                content = text[start:].strip()
                if content.startswith('"'):
                    content = content[1:]
                if content.endswith('"}'):
                    content = content[:-2]
                elif content.endswith('"'):
                    content = content[:-1]
            entry = {"entry_type": "STRATEGY", "content": content}

        db.insert("journal", {
            "entry_type": "STRATEGY",
            "content": entry.get("content", "")[:5000],
            "win_rate_at_time": len(wins) / total if total > 0 else 0,
            "total_signals_at_time": total
        })
        print(f"\n  STRATEGY UPDATED:")
        print(f"  {entry.get('content', '')[:200]}...")
        return entry
    except Exception as e:
        print(f"  Error in deep review: {e}")
        return None


def run_self_training():
    """Run after each signal cycle - analyze latest trade, deep review every 20."""
    print("  Running self-training...")

    # Always analyze the latest resolved trade
    latest = get_latest_resolved_signal()
    if latest:
        analyze_single_trade(latest)

    # Deep strategy review every 20 trades
    all_resolved = db.select("signals", "outcome=not.is.null&select=id")
    if all_resolved:
        total = len(all_resolved)
        if total >= 20 and total % 20 == 0:
            deep_strategy_review()


if __name__ == "__main__":
    run_self_training()
