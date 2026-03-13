"""
BTC Oracle V4 - Professional Grade
Regime detection, WAIT signal filter, shadow mode, feature importance
"""

import os
import json
import requests
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import anthropic
import db
from indicators import get_all_indicators
from market_data import get_all_market_data
from news_scanner import analyze_news_sentiment
from correlated_assets import get_all_correlated_data
from pattern_analyzer import get_pattern_summary
from self_trainer import run_self_training
from scoring_model import score_signal
from advanced_metrics import get_metrics_summary
from kalshi_odds import get_kalshi_btc_contracts, analyze_edge
from regime_detector import detect_regime
from signal_filter import should_trade, get_feature_importance_summary

load_dotenv()

claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def get_past_signals(limit=20):
    return db.select("signals", f"order=created_at.desc&limit={limit}")

def get_journal_entries(limit=10):
    return db.select("journal", f"order=created_at.desc&limit={limit}")

def get_performance_stats():
    data = db.select("performance", "order=recorded_at.desc&limit=1")
    return data[0] if data else None

def get_trade_reviews(limit=5):
    data = db.select("journal", f"entry_type=eq.TRADE_REVIEW&order=created_at.desc&limit={limit}")
    return "\n".join(f"  - {r.get('content', '')}" for r in data) if data else "No trade reviews yet."

def get_current_btc_price():
    try:
        resp = requests.get("https://api.kraken.com/0/public/Ticker?pair=XBTUSD", timeout=10)
        data = resp.json()
        if data.get("result"):
            return float(data["result"]["XXBTZUSD"]["c"][0])
    except:
        pass
    return None


def ask_claude_for_signal(indicators, market_data, news_data, correlated_data,
                          metrics_summary, scoring_output, trade_reviews,
                          kalshi_data, edge_analysis, regime_data, feature_summary,
                          past_signals, performance):

    past_text = ""
    if past_signals:
        for s in past_signals[:15]:
            outcome = s.get('outcome', 'PENDING') or 'PENDING'
            past_text += f"  {s['created_at']}: {s['signal']} @ ${s.get('btc_price_at_signal', 'N/A')} -> {outcome}\n"
    else:
        past_text = "  No previous signals yet.\n"

    perf_text = "No data yet."
    if performance:
        perf_text = f"WR: {performance.get('win_rate', 0):.1%} | {performance.get('total_wins', 0)}W/{performance.get('total_losses', 0)}L"

    market_text = "\n".join(f"  {k}: {v}" for k, v in market_data.items()) if market_data else "  No data"
    news_text = "\n".join(f"  {k}: {v}" for k, v in news_data.items() if "headline" not in k.lower()) if news_data else "  No data"
    correlated_text = "\n".join(f"  {k}: {v}" for k, v in correlated_data.items()) if correlated_data else "  No data"
    regime_text = "\n".join(f"  {k}: {v}" for k, v in regime_data.items()) if regime_data else "  Unknown"

    kalshi_text = "Not available"
    if kalshi_data:
        kalshi_text = f"Market expects: {kalshi_data.get('kalshi_market_expects', '?')} ({kalshi_data.get('kalshi_market_confidence', 0)}%)"
    edge_text = f"Edge: {edge_analysis.get('edge_type', '?')} ({edge_analysis.get('edge_strength', '?')})" if edge_analysis else ""

    prompt = f"""You are BTC Oracle V4. The scoring model gives a calibrated signal.
You should AGREE unless order book or news STRONGLY contradicts.

=== SCORING MODEL (follow this) ===
{scoring_output}

=== MARKET REGIME ===
{regime_text}

=== KALSHI ODDS ===
{kalshi_text}
{edge_text}

=== FEATURE IMPORTANCE (which indicators actually predict price) ===
{feature_summary}

=== INDICATORS ===
  Price: ${indicators['current_price']:,.2f} | RSI: {indicators.get('rsi', 'N/A')} | StochRSI: {indicators.get('stoch_rsi_k', 'N/A')}
  MACD: {indicators.get('macd', 'N/A')} (Hist: {indicators.get('macd_histogram', 'N/A')})
  BB Pos: {indicators.get('bollinger_position', 'N/A')} | EMA Cross: {indicators.get('ema_crossover', 'N/A')}
  Mom: {indicators.get('momentum', 'N/A')} | ROC: {indicators.get('rate_of_change', 'N/A')}%
  VWAP: {indicators.get('price_vs_vwap', 'N/A')} | OBV: {indicators.get('obv_trend', 'N/A')} | ATR: {indicators.get('atr', 'N/A')}
  TREND 1m: {indicators.get('trend_1m', 'N/A')} | TREND 5m: {indicators.get('trend_5m', 'N/A')}

ABSOLUTE RULES:
1. NEVER fight the trend. Both UP = signal UP. Both DOWN = signal DOWN.
2. Trust the scoring model. Only override with STRONG evidence.
3. Adapt to the REGIME. Trending = follow trend. Ranging = fade extremes.
4. Weight features by their IMPORTANCE scores above.

=== ORDER BOOK & TRADE FLOW ===
{market_text}

=== NEWS ===
{news_text}

=== CORRELATED ASSETS ===
{correlated_text}

=== METRICS ===
{metrics_summary}

=== TRADE REVIEWS ===
{trade_reviews}

=== PAST SIGNALS ===
{past_text}

=== PERFORMANCE === {perf_text}

JSON only: {{"signal": "UP" or "DOWN", "confidence": 0.0 to 1.0, "reasoning": "brief"}}"""

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
        try:
            _, _, fallback = score_signal(indicators)
            return {"signal": fallback, "confidence": 0.5, "reasoning": f"Scoring model fallback: {e}"}
        except:
            return {"signal": "UP", "confidence": 0.5, "reasoning": f"Error: {e}"}


def log_signal(signal_data, indicators, traded):
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
        "analysis_notes": f"[{'TRADE' if traded else 'WAIT'}] {signal_data.get('reasoning', '')}"
    }
    db.insert("signals", record)
    status = "TRADE" if traded else "WAIT (logged but not traded)"
    print(f"  {status}: {signal_data['signal']} ({signal_data['confidence']:.0%})")


def check_previous_signals():
    cutoff_start = (datetime.now(timezone.utc) - timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    cutoff_end = (datetime.now(timezone.utc) - timedelta(minutes=14)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    signals = db.select("signals", f"outcome=is.null&created_at=gte.{cutoff_start}&created_at=lte.{cutoff_end}")
    if not signals:
        return

    current_price = get_current_btc_price()
    if not current_price:
        print("  Could not get current price")
        return

    for signal in signals:
        price_at = signal.get("btc_price_at_signal")
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
    print(f"BTC ORACLE V4 | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)

    print("\n[1/12] Checking previous signals...")
    check_previous_signals()

    print("\n[2/12] Updating performance...")
    update_performance()

    print("\n[3/12] Self-training...")
    run_self_training()

    print("\n[4/12] Technical indicators...")
    indicators = get_all_indicators()
    if not indicators:
        print("  Not enough data.")
        return

    print("\n[5/12] Market regime...")
    regime_data = detect_regime()
    print(f"  Regime: {regime_data.get('regime', '?')} | Trend strength: {regime_data.get('trend_strength', '?')} | Vol: {regime_data.get('volatility_pct', '?')}%")

    print("\n[6/12] Market data...")
    market_data = get_all_market_data()

    print("\n[7/12] News & sentiment...")
    news_data = analyze_news_sentiment()

    print("\n[8/12] Correlated assets...")
    correlated_data = get_all_correlated_data()

    print("\n[9/12] Scoring model...")
    score, score_conf, score_dir = score_signal(indicators, market_data)
    scoring_output = f"Score: {score:+.3f} | Signal: {score_dir} | Confidence: {score_conf:.0%}"
    print(f"  {scoring_output}")

    print("\n[10/12] Kalshi odds...")
    kalshi_data = get_kalshi_btc_contracts()
    edge = analyze_edge(score_dir, score_conf, kalshi_data) if kalshi_data else {}
    print(f"  {kalshi_data.get('kalshi_market_expects', 'No data')}" if kalshi_data else "  No Kalshi data")

    print("\n[11/12] Feature importance...")
    feature_summary = get_feature_importance_summary()
    print(f"  {feature_summary[:100]}...")

    print("\n[12/12] Claude V4...")
    signal_data = ask_claude_for_signal(
        indicators, market_data, news_data, correlated_data,
        get_metrics_summary(), scoring_output, get_trade_reviews(5),
        kalshi_data, edge, regime_data, feature_summary,
        get_past_signals(15), get_performance_stats()
    )

    # SIGNAL FILTER - should we trade or wait?
    trade, filter_reason, adjusted_conf = should_trade(
        score, score_conf, signal_data["signal"], signal_data["confidence"],
        regime_data, indicators, market_data
    )

    print(f"\n  >>> SIGNAL: {signal_data['signal']} ({signal_data['confidence']:.0%})")
    print(f"  >>> Model: {score_dir} | Claude: {signal_data['signal']} | {'AGREE' if score_dir == signal_data['signal'] else 'OVERRIDE'}")
    print(f"  >>> Regime: {regime_data.get('regime', '?')}")
    print(f"  >>> Filter: {'TRADE' if trade else 'WAIT'} | {filter_reason}")
    print(f"  >>> {signal_data.get('reasoning', '')[:150]}...")

    # Log signal (mark whether it was traded or waited)
    if trade:
        signal_data["confidence"] = adjusted_conf
    log_signal(signal_data, indicators, trade)

    print("\nCycle complete.")


if __name__ == "__main__":
    run_signal_cycle()
