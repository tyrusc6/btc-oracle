"""
BTC Oracle - Self Trainer V3
Only learns from V5 signals.
Tracks which tiers caused wins vs losses.
Writes actionable strategy updates every 20 V5 trades.
"""

import os
import json
from datetime import datetime, timezone
from dotenv import load_dotenv
import anthropic
import db

load_dotenv()

claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

V5_EPOCH = "2026-03-13T00:00:00Z"
reviewed_ids = set()


def get_v5_signals(limit=200):
    params = f"outcome=not.is.null&created_at=gte.{V5_EPOCH}&order=created_at.desc&limit={limit}"
    return db.select("signals", params)


def get_unreviewed():
    signals = get_v5_signals(10)
    return [s for s in signals if s["id"] not in reviewed_ids] if signals else []


def analyze_single_trade(signal):
    """Quick review after each V5 trade."""
    if not signal or not signal.get("outcome"):
        return

    price = signal.get("btc_price_at_signal", 0)
    close = signal.get("btc_price_at_close", 0)
    change = close - price if price and close else 0

    # Determine what the mean reversion strategy would have predicted
    notes = signal.get("analysis_notes", "") or ""
    was_trade = notes.startswith("[TRADE]")

    # Analyze which tiers were active
    tier_analysis = []
    bb_upper = signal.get("bollinger_upper")
    bb_lower = signal.get("bollinger_lower")
    if bb_upper and bb_lower and bb_upper != bb_lower and price:
        bb_pos = (price - bb_lower) / (bb_upper - bb_lower)
        if bb_pos > 0.8 or bb_pos < 0.2:
            tier_analysis.append(f"BB extreme (pos={bb_pos:.2f})")

    rsi = signal.get("rsi")
    if rsi is not None and (rsi > 70 or rsi < 30):
        tier_analysis.append(f"RSI extreme ({rsi:.1f})")

    mom = signal.get("momentum")
    if mom is not None:
        tier_analysis.append(f"Momentum {'positive' if mom > 0 else 'negative'} ({mom:.2f})")

    tier_text = ", ".join(tier_analysis) if tier_analysis else "No extreme tiers active"

    prompt = f"""Analyze this BTC trade (mean reversion strategy):

Signal: {signal['signal']} | Outcome: {signal['outcome']} | {'TRADED' if was_trade else 'WAITED'}
Price: ${price:,.2f} -> ${close:,.2f} (${change:+,.2f})
RSI: {rsi} | Momentum: {mom}
Active tiers: {tier_text}

In 1 sentence: Why did this {'win' if signal['outcome'] == 'WIN' else 'lose'}? Which tier helped or hurt?

JSON: {{"entry_type": "TRADE_REVIEW", "content": "your analysis"}}"""

    try:
        resp = claude.messages.create(model="claude-sonnet-4-20250514", max_tokens=150, messages=[{"role": "user", "content": prompt}])
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        try:
            entry = json.loads(text)
        except:
            entry = {"entry_type": "TRADE_REVIEW", "content": text[:500]}

        db.insert("journal", {
            "entry_type": "TRADE_REVIEW",
            "content": entry.get("content", ""),
            "win_rate_at_time": 0,
            "total_signals_at_time": signal.get("id", 0)
        })
        reviewed_ids.add(signal["id"])
        print(f"  Trade #{signal['id']}: {entry.get('content', '')[:80]}...")
    except Exception as e:
        print(f"  Review error: {e}")
        reviewed_ids.add(signal.get("id", 0))


def deep_strategy_review():
    """Every 20 V5 trades: analyze tier performance and write strategy update."""
    signals = get_v5_signals(200)
    if not signals:
        return

    total = len(signals)
    if total < 20 or total % 20 != 0:
        return

    print(f"\n  === V5 DEEP REVIEW === ({total} V5 signals)")

    wins = [s for s in signals if s["outcome"] == "WIN"]
    losses = [s for s in signals if s["outcome"] == "LOSS"]
    wr = len(wins) / total * 100

    # Analyze tier performance
    tier_results = {
        "bb_extreme_up": {"correct": 0, "total": 0},
        "bb_extreme_down": {"correct": 0, "total": 0},
        "anti_momentum": {"correct": 0, "total": 0},
        "rsi_extreme": {"correct": 0, "total": 0},
    }

    for s in signals:
        if not s.get("btc_price_at_signal") or not s.get("btc_price_at_close"):
            continue
        actual_up = s["btc_price_at_close"] > s["btc_price_at_signal"]
        price = s["btc_price_at_signal"]

        # BB tier
        bu, bl = s.get("bollinger_upper"), s.get("bollinger_lower")
        if bu and bl and bu != bl:
            bb_pos = (price - bl) / (bu - bl)
            if bb_pos > 0.8:
                tier_results["bb_extreme_down"]["total"] += 1
                if not actual_up:
                    tier_results["bb_extreme_down"]["correct"] += 1
            elif bb_pos < 0.2:
                tier_results["bb_extreme_up"]["total"] += 1
                if actual_up:
                    tier_results["bb_extreme_up"]["correct"] += 1

        # Anti-momentum tier
        mom = s.get("momentum")
        if mom is not None and mom != 0:
            tier_results["anti_momentum"]["total"] += 1
            if (mom > 0 and not actual_up) or (mom < 0 and actual_up):
                tier_results["anti_momentum"]["correct"] += 1

        # RSI tier
        rsi = s.get("rsi")
        if rsi is not None and (rsi > 70 or rsi < 30):
            tier_results["rsi_extreme"]["total"] += 1
            if (rsi > 70 and not actual_up) or (rsi < 30 and actual_up):
                tier_results["rsi_extreme"]["correct"] += 1

    tier_text = ""
    for tier, stats in tier_results.items():
        if stats["total"] >= 3:
            acc = stats["correct"] / stats["total"] * 100
            tier_text += f"  {tier}: {acc:.0f}% ({stats['correct']}/{stats['total']})\n"

    # Traded vs waited
    traded = [s for s in signals if (s.get("analysis_notes") or "").startswith("[TRADE]")]
    waited = [s for s in signals if (s.get("analysis_notes") or "").startswith("[WAIT]")]
    trade_wins = len([s for s in traded if s["outcome"] == "WIN"])
    trade_total = len(traded)
    trade_wr = trade_wins / trade_total * 100 if trade_total > 0 else 0

    prompt = f"""V5 Strategy Review after {total} signals.

Overall WR: {wr:.1f}% ({len(wins)}W/{len(losses)}L)
TRADE WR: {trade_wr:.1f}% ({trade_wins}/{trade_total})
Waited: {len(waited)} signals

TIER PERFORMANCE:
{tier_text}

Write 3-4 bullet points: which tiers work, which don't, what to adjust.
Keep under 300 chars.

JSON: {{"entry_type": "STRATEGY", "content": "your bullets"}}"""

    try:
        resp = claude.messages.create(model="claude-sonnet-4-20250514", max_tokens=400, messages=[{"role": "user", "content": prompt}])
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        try:
            entry = json.loads(text)
        except:
            entry = {"entry_type": "STRATEGY", "content": text[:1000]}

        db.insert("journal", {
            "entry_type": "STRATEGY",
            "content": entry.get("content", "")[:2000],
            "win_rate_at_time": wr / 100,
            "total_signals_at_time": total
        })
        print(f"  Strategy updated: {entry.get('content', '')[:150]}...")
    except Exception as e:
        print(f"  Strategy review error: {e}")


def run_self_training():
    """Run after each cycle."""
    print("  Running V5 self-training...")

    # Review unreviewed V5 trades
    unreviewed = get_unreviewed()
    if unreviewed:
        for s in unreviewed[:3]:
            analyze_single_trade(s)
    else:
        print("  No new V5 trades to review.")

    # Deep review every 20 V5 trades
    v5_count = get_v5_signals(500)
    if v5_count:
        total = len(v5_count)
        if total >= 20 and total % 20 == 0:
            deep_strategy_review()


if __name__ == "__main__":
    run_self_training()
