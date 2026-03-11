"""
BTC Oracle - Main Signal Bot (V2 - Full Arsenal)
"""

import os
import json
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import anthropic
import db
from indicators import get_all_indicators, fetch_recent_ticks
from market_data import get_all_market_data
from pattern_analyzer import get_pattern_summary

load_dotenv()

claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def get_past_signals(limit=20):
    return db.select("signals", f"order=created_at.desc&limit={limit}")


def get_journal_entries(limit=10):
    return db.select("journal", f"order=created_at.desc&limit={limit}")


def get_performance_stats():
    data = db.select("performance", "order=recorded_at.desc&limit=1")
    return data[0] if data else None


def ask_claude_for_signal(indicators, market_data, pattern_summary, past_signals, journal_entries, performance):
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

    # Build market data section
    market_text = ""
    for k, v in market_data.items():
        market_text += f"  {k}: {v}\n"

    prompt = f"""You are BTC Oracle, an elite Bitcoin price prediction AI in LEARNING MODE.
You have access to an extensive arsenal of market data. Use ALL of it.

Predict whether BTC will be HIGHER or LOWER than its current price in exactly 15 minutes.

=== TECHNICAL INDICATORS ===
  Price: ${indicators['current_price']:,.2f}
  RSI (14): {indicators.get('rsi', 'N/A')}
  Stochastic RSI: K={indicators.get('stoch_rsi_k', 'N/A')} D={indicators.get('stoch_rsi_d', 'N/A')}
  MACD: {indicators.get('macd', 'N/A')} (Signal: {indicators.get('macd_signal', 'N/A')}, Hist: {indicators.get('macd_histogram', 'N/A')})
  Bollinger: Lower={indicators.get('bollinger_lower', 'N/A')} | Mid={indicators.get('bollinger_middle', 'N/A')} | Upper={indicators.get('bollinger_upper', 'N/A')}
  Bollinger Position: {indicators.get('bollinger_position', 'N/A')} (0=lower band, 1=upper band)
  EMA 9: {indicators.get('ema_9', 'N/A')} | EMA 21: {indicators.get('ema_21', 'N/A')} | SMA 50: {indicators.get('sma_50', 'N/A')}
  EMA Crossover: {indicators.get('ema_crossover', 'N/A')}
  Momentum: {indicators.get('momentum', 'N/A')}
  Rate of Change: {indicators.get('rate_of_change', 'N/A')}%
  VWAP: {indicators.get('vwap', 'N/A')} | Price vs VWAP: {indicators.get('price_vs_vwap', 'N/A')}
  ATR (volatility): {indicators.get('atr', 'N/A')}
  OBV Trend: {indicators.get('obv_trend', 'N/A')}
  Ticks Analyzed: {indicators.get('tick_count', 'N/A')}

=== MARKET SENTIMENT & EXTERNAL DATA ===
{market_text}

=== YOUR PATTERN ANALYSIS (your own win/loss stats) ===
{pattern_summary}

=== PAST SIGNALS ===
{past_text}

=== PERFORMANCE === 
{perf_text}

=== JOURNAL ===
{journal_text}

=== DECISION FRAMEWORK ===
Weight these factors in order of importance for 15-minute predictions:
1. Order book imbalance & trade flow (most immediate signal)
2. Multi-timeframe momentum alignment
3. Candlestick patterns & volatility regime
4. Technical indicators (RSI, MACD, Bollinger position, StochRSI)
5. Your own pattern analysis (which conditions you win/lose in)
6. Market sentiment (Fear & Greed, BTC dominance trends)

Key principles:
- If order flow and momentum ALIGN, follow them with high confidence
- If they CONFLICT, lower your confidence and lean toward mean reversion
- Check your pattern stats: if you historically lose in these conditions, consider the opposite
- Volatility regime matters: in LOW volatility, price tends to continue; in HIGH, reversals are more likely
- OBV trend confirms or denies price moves

RULES:
- LEARNING MODE: Always output UP or DOWN. No WAIT.
- Be brutally honest in your reasoning about conflicting signals
- Reference your pattern analysis when relevant

Respond in this exact JSON format only:
{{"signal": "UP" or "DOWN", "confidence": 0.0 to 1.0, "reasoning": "Your detailed analysis"}}"""

    try:
        response = claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(text)
    except Exception as e:
        print(f"Error calling Claude: {e}")
        return {"signal": "DOWN", "confidence": 0.5, "reasoning": f"Error: {e}"}


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
    if performance["total_signals"] % 5 != 0:  # Journal every 5 signals now
        return

    past = get_past_signals(20)
    recent_outcomes = [s.get("outcome") for s in past if s.get("outcome")]
    recent_wr = recent_outcomes.count("WIN") / len(recent_outcomes) if recent_outcomes else 0
    pattern_summary = get_pattern_summary()

    prompt = f"""You are BTC Oracle reviewing performance.

Win rate: {recent_wr:.1%} | Total: {performance['total_signals']}

Pattern Analysis:
{pattern_summary}

Recent signals:
"""
    for s in past[:10]:
        prompt += f"  {s.get('signal')} @ ${s.get('btc_price_at_signal', 0):,.2f} -> {s.get('outcome', '?')} | RSI: {s.get('rsi')} | MACD: {s.get('macd')}\n"

    prompt += """
Write a 2-3 sentence journal entry analyzing patterns. Focus on ACTIONABLE insights.
JSON: {"entry_type": "REFLECTION" or "PATTERN" or "MISTAKE" or "INSIGHT", "content": "entry"}"""

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
    print(f"BTC ORACLE V2 | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)

    print("\n[1/6] Checking previous signals...")
    check_previous_signals()

    print("\n[2/6] Updating performance...")
    update_performance()

    print("\n[3/6] Calculating technical indicators...")
    indicators = get_all_indicators()
    if not indicators:
        print("  Not enough data. Run collector.py first.")
        return

    print("\n[4/6] Fetching market data...")
    market_data = get_all_market_data()

    print("\n[5/6] Analyzing patterns...")
    pattern_summary = get_pattern_summary()
    print(f"  {pattern_summary[:100]}...")

    print("\n[6/6] Consulting Claude with full arsenal...")
    signal_data = ask_claude_for_signal(
        indicators, market_data, pattern_summary,
        get_past_signals(20), get_journal_entries(10), get_performance_stats()
    )

    print(f"\n  >>> SIGNAL: {signal_data['signal']} ({signal_data['confidence']:.0%})")
    print(f"  >>> {signal_data.get('reasoning', '')[:150]}...")

    print("\n  Logging signal...")
    log_signal(signal_data, indicators)
    write_journal_entry(get_performance_stats())

    print("\nCycle complete.")


if __name__ == "__main__":
    run_signal_cycle()
