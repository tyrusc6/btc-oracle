"""
BTC Oracle - Main Signal Bot
"""

import os
import json
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import anthropic
import db
from indicators import get_all_indicators, fetch_recent_ticks

load_dotenv()

claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def get_past_signals(limit=20):
    return db.select("signals", f"order=created_at.desc&limit={limit}")


def get_journal_entries(limit=10):
    return db.select("journal", f"order=created_at.desc&limit={limit}")


def get_performance_stats():
    data = db.select("performance", "order=recorded_at.desc&limit=1")
    return data[0] if data else None


def ask_claude_for_signal(indicators, past_signals, journal_entries, performance):
    past_text = ""
    if past_signals:
        for s in past_signals[:20]:
            outcome = s.get('outcome', 'PENDING') or 'PENDING'
            past_text += f"  {s['created_at']}: {s['signal']} @ ${s.get('btc_price_at_signal', 'N/A')} -> {outcome}\n"
    else:
        past_text = "  No previous signals yet.\n"

    journal_text = ""
    if journal_entries:
        for j in journal_entries[:10]:
            journal_text += f"  [{j['entry_type']}] {j['content']}\n"
    else:
        journal_text = "  No journal entries yet.\n"

    perf_text = "No performance data yet."
    if performance:
        perf_text = f"Win Rate: {performance.get('win_rate', 0):.1%} | Total: {performance.get('total_signals', 0)} | Wins: {performance.get('total_wins', 0)} | Losses: {performance.get('total_losses', 0)}"

    prompt = f"""You are BTC Oracle, an elite Bitcoin price prediction AI in LEARNING MODE.

Predict whether BTC will be HIGHER or LOWER than its current price in exactly 15 minutes.

CURRENT MARKET DATA:
  Price: ${indicators['current_price']:,.2f}
  RSI (14): {indicators.get('rsi', 'N/A')}
  MACD: {indicators.get('macd', 'N/A')} (Signal: {indicators.get('macd_signal', 'N/A')}, Hist: {indicators.get('macd_histogram', 'N/A')})
  Bollinger Bands: Lower={indicators.get('bollinger_lower', 'N/A')} | Mid={indicators.get('bollinger_middle', 'N/A')} | Upper={indicators.get('bollinger_upper', 'N/A')}
  EMA 9: {indicators.get('ema_9', 'N/A')} | EMA 21: {indicators.get('ema_21', 'N/A')} | SMA 50: {indicators.get('sma_50', 'N/A')}
  Momentum: {indicators.get('momentum', 'N/A')}
  VWAP: {indicators.get('vwap', 'N/A')}
  Ticks Analyzed: {indicators.get('tick_count', 'N/A')}

PAST SIGNALS:
{past_text}

PERFORMANCE: {perf_text}

JOURNAL:
{journal_text}

RULES:
- LEARNING MODE: Always output UP or DOWN. No WAIT.
- Learn from past signals and adjust.
- Consider ALL indicators together.

Respond in this exact JSON format only:
{{"signal": "UP" or "DOWN", "confidence": 0.0 to 1.0, "reasoning": "Your analysis"}}"""

    try:
        response = claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(text)
    except Exception as e:
        print(f"Error calling Claude: {e}")
        return {"signal": "UP", "confidence": 0.5, "reasoning": f"Error: {e}"}


def log_signal(signal_data, indicators):
    record = {
        "signal": signal_data["signal"],
        "confidence": signal_data["confidence"],
        "btc_price_at_signal": indicators["current_price"],
        "rsi": indicators.get("rsi"),
        "macd": indicators.get("macd"),
        "macd_signal": indicators.get("macd_signal"),
        "macd_histogram": indicators.get("macd_histogram"),
        "bollinger_upper": indicators.get("bollinger_upper"),
        "bollinger_middle": indicators.get("bollinger_middle"),
        "bollinger_lower": indicators.get("bollinger_lower"),
        "volume_24h": indicators.get("volume_24h"),
        "momentum": indicators.get("momentum"),
        "ema_9": indicators.get("ema_9"),
        "ema_21": indicators.get("ema_21"),
        "sma_50": indicators.get("sma_50"),
        "vwap": indicators.get("vwap"),
        "analysis_notes": signal_data.get("reasoning", "")
    }
    result = db.insert("signals", record)
    print(f"  Signal logged: {signal_data['signal']} ({signal_data['confidence']:.0%} confidence)")
    return result


def check_previous_signals():
    cutoff_start = (datetime.now(timezone.utc) - timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    cutoff_end = (datetime.now(timezone.utc) - timedelta(minutes=14)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    signals = db.select("signals", f"outcome=is.null&created_at=gte.{cutoff_start}&created_at=lte.{cutoff_end}")

    if not signals:
        return

    recent = fetch_recent_ticks(minutes=5)
    if recent.empty:
        return
    current_price = float(recent["price"].iloc[-1])

    for signal in signals:
        price_at = signal["btc_price_at_signal"]
        if not price_at:
            continue
        went_up = current_price > price_at
        predicted_up = signal["signal"] == "UP"
        outcome = "WIN" if went_up == predicted_up else "LOSS"

        db.update("signals", "id", signal["id"], {
            "btc_price_at_close": current_price,
            "outcome": outcome
        })
        change = current_price - price_at
        print(f"  Signal #{signal['id']}: {signal['signal']} -> {'UP' if went_up else 'DOWN'} (${change:+,.2f}) = {outcome}")


def update_performance():
    data = db.select("signals", "outcome=not.is.null&select=outcome")
    if not data:
        return
    outcomes = [r["outcome"] for r in data]
    total = len(outcomes)
    wins = outcomes.count("WIN")
    losses = outcomes.count("LOSS")
    win_rate = wins / total if total > 0 else 0

    streak = 0
    for o in reversed(outcomes):
        if streak == 0:
            streak = 1 if o == "WIN" else -1
        elif (streak > 0 and o == "WIN") or (streak < 0 and o == "LOSS"):
            streak += 1 if streak > 0 else -1
        else:
            break

    db.insert("performance", {
        "total_signals": total, "total_wins": wins,
        "total_losses": losses, "win_rate": win_rate, "streak_current": streak
    })
    print(f"  Performance: {win_rate:.1%} ({wins}W/{losses}L) | Streak: {streak}")


def write_journal_entry(performance):
    if not performance or performance.get("total_signals", 0) < 5:
        return
    if performance["total_signals"] % 10 != 0:
        return

    past = get_past_signals(20)
    recent_outcomes = [s.get("outcome") for s in past if s.get("outcome")]
    recent_wr = recent_outcomes.count("WIN") / len(recent_outcomes) if recent_outcomes else 0

    prompt = f"You are BTC Oracle. Win rate: {recent_wr:.1%}. Total: {performance['total_signals']}.\n\nRecent:\n"
    for s in past[:10]:
        prompt += f"  {s.get('signal')} @ ${s.get('btc_price_at_signal', 0):,.2f} -> {s.get('outcome', '?')} | RSI: {s.get('rsi')}\n"
    prompt += '\nWrite a 2-3 sentence journal entry. JSON: {"entry_type": "REFLECTION" or "PATTERN" or "MISTAKE" or "INSIGHT", "content": "entry"}'

    try:
        resp = claude.messages.create(model="claude-sonnet-4-20250514", max_tokens=300, messages=[{"role": "user", "content": prompt}])
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        entry = json.loads(text)
        db.insert("journal", {
            "entry_type": entry.get("entry_type", "REFLECTION"),
            "content": entry.get("content", ""),
            "win_rate_at_time": performance.get("win_rate", 0),
            "total_signals_at_time": performance.get("total_signals", 0)
        })
        print(f"  Journal: [{entry['entry_type']}] {entry['content'][:80]}...")
    except Exception as e:
        print(f"Error writing journal: {e}")


def run_signal_cycle():
    print("\n" + "=" * 60)
    print(f"BTC ORACLE | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)

    print("\n[1/5] Checking previous signals...")
    check_previous_signals()

    print("\n[2/5] Updating performance...")
    update_performance()

    print("\n[3/5] Calculating indicators...")
    indicators = get_all_indicators()
    if not indicators:
        print("  Not enough data. Run collector.py first.")
        return

    print("\n[4/5] Consulting Claude...")
    signal_data = ask_claude_for_signal(
        indicators, get_past_signals(20), get_journal_entries(10), get_performance_stats()
    )

    print(f"\n  >>> SIGNAL: {signal_data['signal']} ({signal_data['confidence']:.0%})")
    print(f"  >>> {signal_data.get('reasoning', '')[:100]}...")

    print("\n[5/5] Logging signal...")
    log_signal(signal_data, indicators)
    write_journal_entry(get_performance_stats())

    print("\nCycle complete. Next signal in 15 minutes.")


if __name__ == "__main__":
    run_signal_cycle()
