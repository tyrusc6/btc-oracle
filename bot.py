"""
BTC Oracle V3 - Full Professional Arsenal
Scoring model + Claude hybrid, advanced metrics, Kalshi odds, combos
"""

import os
import json
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import anthropic
import db
from indicators import get_all_indicators, fetch_recent_ticks
from market_data import get_all_market_data
from news_scanner import analyze_news_sentiment
from correlated_assets import get_all_correlated_data
from pattern_analyzer import get_pattern_summary
from self_trainer import run_self_training
from scoring_model import score_signal, get_scoring_summary
from advanced_metrics import get_metrics_summary
from kalshi_odds import get_kalshi_btc_contracts, analyze_edge

load_dotenv()

claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def get_past_signals(limit=20):
    return db.select("signals", f"order=created_at.desc&limit={limit}")

def get_journal_entries(limit=10):
    return db.select("journal", f"order=created_at.desc&limit={limit}")

def get_performance_stats():
    data = db.select("performance", "order=recorded_at.desc&limit=1")
    return data[0] if data else None

def get_strategy_document():
    data = db.select("journal", "entry_type=eq.STRATEGY&order=created_at.desc&limit=1")
    return data[0]["content"] if data else "No strategy document yet."

def get_trade_reviews(limit=5):
    data = db.select("journal", f"entry_type=eq.TRADE_REVIEW&order=created_at.desc&limit={limit}")
    return "\n".join(f"  - {r['content']}" for r in data) if data else "No trade reviews yet."


def ask_claude_for_signal(indicators, market_data, news_data, correlated_data,
                          pattern_summary, strategy_doc, trade_reviews, metrics_summary,
                          scoring_output, kalshi_data, edge_analysis,
                          past_signals, journal_entries, performance):
    past_text = ""
    if past_signals:
        for s in past_signals[:15]:
            outcome = s.get('outcome', 'PENDING') or 'PENDING'
            past_text += f"  {s['created_at']}: {s['signal']} @ ${s.get('btc_price_at_signal', 'N/A')} -> {outcome}\n"
    else:
        past_text = "  No previous signals yet.\n"

    journal_text = "\n".join(f"  [{j['entry_type']}] {j['content'][:120]}" for j in (journal_entries or [])[:8]) or "  None yet."

    perf_text = "No data yet."
    if performance:
        perf_text = f"WR: {performance.get('win_rate', 0):.1%} | Total: {performance.get('total_signals', 0)} | {performance.get('total_wins', 0)}W/{performance.get('total_losses', 0)}L"

    market_text = "\n".join(f"  {k}: {v}" for k, v in market_data.items())
    news_text = "\n".join(f"  {k}: {v}" for k, v in news_data.items() if "headline" not in k.lower())
    correlated_text = "\n".join(f"  {k}: {v}" for k, v in correlated_data.items())

    kalshi_text = "Not available"
    if kalshi_data:
        kalshi_text = f"Market expects: {kalshi_data.get('kalshi_market_expects', '?')} ({kalshi_data.get('kalshi_market_confidence', 0)}%)"
    edge_text = ""
    if edge_analysis:
        edge_text = f"Edge: {edge_analysis.get('edge_type', '?')} - {edge_analysis.get('edge_description', '')} (Strength: {edge_analysis.get('edge_strength', '?')})"

    prompt = f"""You are BTC Oracle V3. You have a scoring model that weights all indicators numerically.

IMPORTANT: The scoring model has been calibrated with trend detection and indicator weights.
You should AGREE with the scoring model in most cases. Only override if you have VERY strong
reason from order book, news, or trade flow data that contradicts it.

=== SCORING MODEL OUTPUT (follow this unless you have strong reason not to) ===
{scoring_output}

=== KALSHI MARKET ODDS ===
{kalshi_text}
{edge_text}

=== TECHNICAL INDICATORS ===
  Price: ${indicators['current_price']:,.2f} | RSI: {indicators.get('rsi', 'N/A')} | StochRSI K: {indicators.get('stoch_rsi_k', 'N/A')}
  MACD: {indicators.get('macd', 'N/A')} (Hist: {indicators.get('macd_histogram', 'N/A')})
  BB Position: {indicators.get('bollinger_position', 'N/A')} | EMA Cross: {indicators.get('ema_crossover', 'N/A')}
  Momentum: {indicators.get('momentum', 'N/A')} | ROC: {indicators.get('rate_of_change', 'N/A')}%
  VWAP: {indicators.get('price_vs_vwap', 'N/A')} | OBV: {indicators.get('obv_trend', 'N/A')} | ATR: {indicators.get('atr', 'N/A')}
  TREND 1-MIN: {indicators.get('trend_1m', 'N/A')} | TREND 5-MIN: {indicators.get('trend_5m', 'N/A')} | Trend Change: {indicators.get('trend_pct_change', 'N/A')}%

ABSOLUTE RULE #1: NEVER fight the trend. If TREND 1-MIN and TREND 5-MIN both say UPTREND or STRONG_UPTREND, you MUST signal UP. If both say DOWNTREND or STRONG_DOWNTREND, you MUST signal DOWN. No exceptions.

ABSOLUTE RULE #2: The scoring model already factors in all indicators with learned weights. Only override it if live order book or breaking news STRONGLY contradicts it.

ABSOLUTE RULE #3: IGNORE any strategy document rules that were learned from bad data. The indicators have been recalibrated. Trust the scoring model and trend detection.

=== ORDER BOOK & TRADE FLOW ===
{market_text}

=== NEWS & SENTIMENT ===
{news_text}

=== CORRELATED ASSETS ===
{correlated_text}

=== ADVANCED METRICS ===
{metrics_summary}

=== RECENT TRADE REVIEWS ===
{trade_reviews}

=== PAST SIGNALS ===
{past_text}

=== PERFORMANCE === {perf_text}

RULES:
- AGREE with scoring model unless order book/news STRONGLY contradicts
- NEVER fight the trend direction
- If both trends are UP, signal UP. If both DOWN, signal DOWN. Period.
- LEARNING MODE: Always UP or DOWN.

JSON only:
{{"signal": "UP" or "DOWN", "confidence": 0.0 to 1.0, "reasoning": "brief analysis"}}"""

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
        # Fallback to scoring model direction instead of hardcoded DOWN
        fallback_dir = "DOWN"
        try:
            from scoring_model import score_signal as fallback_score
            _, _, fallback_dir = fallback_score(indicators)
        except:
            pass
        return {"signal": fallback_dir, "confidence": 0.5, "reasoning": f"Error, using scoring model fallback: {e}"}


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
        db.update("signals", "id", signal["id"], {"btc_price_at_close": current_price, "outcome": outcome})
        change = current_price - price_at
        print(f"  #{signal['id']}: {signal['signal']} -> {'UP' if went_up else 'DOWN'} (${change:+,.2f}) = {outcome}")


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
    db.insert("performance", {"total_signals": total, "total_wins": wins, "total_losses": losses, "win_rate": win_rate, "streak_current": streak})
    print(f"  {win_rate:.1%} ({wins}W/{losses}L) | Streak: {streak}")


def run_signal_cycle():
    print("\n" + "=" * 60)
    print(f"BTC ORACLE V3 | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)

    print("\n[1/10] Checking previous signals...")
    check_previous_signals()

    print("\n[2/10] Updating performance...")
    update_performance()

    print("\n[3/10] Self-training...")
    run_self_training()

    print("\n[4/10] Technical indicators...")
    indicators = get_all_indicators()
    if not indicators:
        print("  Not enough data yet.")
        return

    print("\n[5/10] Market data (order book, trade flow)...")
    market_data = get_all_market_data()

    print("\n[6/10] News & sentiment...")
    news_data = analyze_news_sentiment()

    print("\n[7/10] Correlated assets...")
    correlated_data = get_all_correlated_data()

    print("\n[8/10] Scoring model...")
    score, score_conf, score_signal_dir = score_signal(indicators, market_data)
    scoring_output = f"Score: {score:+.3f} | Model says: {score_signal_dir} ({score_conf:.0%} confidence)"
    print(f"  {scoring_output}")

    print("\n[9/10] Kalshi odds...")
    kalshi_data = get_kalshi_btc_contracts()
    edge_analysis = analyze_edge(score_signal_dir, score_conf, kalshi_data) if kalshi_data else {}
    if kalshi_data:
        print(f"  Market expects: {kalshi_data.get('kalshi_market_expects', '?')} ({kalshi_data.get('kalshi_market_confidence', 0)}%)")
    else:
        print("  No Kalshi data available")

    print("\n[10/10] Consulting Claude V3 (full arsenal)...")
    signal_data = ask_claude_for_signal(
        indicators, market_data, news_data, correlated_data,
        get_pattern_summary(), get_strategy_document(), get_trade_reviews(5),
        get_metrics_summary(), scoring_output, kalshi_data, edge_analysis,
        get_past_signals(15), get_journal_entries(8), get_performance_stats()
    )

    print(f"\n  >>> SIGNAL: {signal_data['signal']} ({signal_data['confidence']:.0%})")
    print(f"  >>> Model: {score_signal_dir} | Claude: {signal_data['signal']} | {'AGREE' if score_signal_dir == signal_data['signal'] else 'OVERRIDE'}")
    print(f"  >>> {signal_data.get('reasoning', '')[:200]}...")

    log_signal(signal_data, indicators)
    print("\nCycle complete.")


if __name__ == "__main__":
    run_signal_cycle()
