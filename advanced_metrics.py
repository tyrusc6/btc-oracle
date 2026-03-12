"""
BTC Oracle - Advanced Metrics
Tracks expected value, Sharpe ratio, max drawdown, win rate by condition,
and combination analysis.
"""

import db
import numpy as np
from datetime import datetime, timezone


def get_all_resolved_signals(limit=500):
    return db.select("signals", f"outcome=not.is.null&order=created_at.desc&limit={limit}")


def calculate_advanced_metrics():
    """Calculate professional-grade trading metrics."""
    signals = get_all_resolved_signals()
    if not signals or len(signals) < 10:
        return {"status": "Need at least 10 resolved signals"}

    results = {"total_signals": len(signals)}

    # Basic win/loss
    wins = [s for s in signals if s["outcome"] == "WIN"]
    losses = [s for s in signals if s["outcome"] == "LOSS"]
    results["wins"] = len(wins)
    results["losses"] = len(losses)
    results["win_rate"] = round(len(wins) / len(signals) * 100, 2)

    # Price changes
    changes = []
    win_magnitudes = []
    loss_magnitudes = []
    for s in signals:
        if s.get("btc_price_at_signal") and s.get("btc_price_at_close"):
            change = s["btc_price_at_close"] - s["btc_price_at_signal"]
            predicted_up = s["signal"] == "UP"
            # Profit/loss from perspective of the prediction
            pnl = change if predicted_up else -change
            changes.append(pnl)
            if s["outcome"] == "WIN":
                win_magnitudes.append(abs(change))
            else:
                loss_magnitudes.append(abs(change))

    if changes:
        # Expected Value per trade
        results["expected_value"] = round(np.mean(changes), 4)
        results["ev_positive"] = results["expected_value"] > 0

        # Average win size vs average loss size
        results["avg_win_magnitude"] = round(np.mean(win_magnitudes), 2) if win_magnitudes else 0
        results["avg_loss_magnitude"] = round(np.mean(loss_magnitudes), 2) if loss_magnitudes else 0
        results["win_loss_ratio"] = round(results["avg_win_magnitude"] / results["avg_loss_magnitude"], 3) if results["avg_loss_magnitude"] > 0 else 999

        # Sharpe Ratio (annualized, assuming 15-min intervals)
        if len(changes) > 1:
            mean_return = np.mean(changes)
            std_return = np.std(changes)
            if std_return > 0:
                # ~35,000 15-min periods per year
                results["sharpe_ratio"] = round((mean_return / std_return) * np.sqrt(35040), 3)
            else:
                results["sharpe_ratio"] = 0

        # Max drawdown (consecutive losses)
        cumulative = np.cumsum(changes)
        peak = np.maximum.accumulate(cumulative)
        drawdown = peak - cumulative
        results["max_drawdown"] = round(float(np.max(drawdown)), 2) if len(drawdown) > 0 else 0

        # Profit factor
        gross_profit = sum(c for c in changes if c > 0)
        gross_loss = abs(sum(c for c in changes if c < 0))
        results["profit_factor"] = round(gross_profit / gross_loss, 3) if gross_loss > 0 else 999

    # Win rate by market condition
    # High volatility (ATR-based, using price change as proxy)
    big_moves = [s for s in signals if s.get("btc_price_at_signal") and s.get("btc_price_at_close") and abs(s["btc_price_at_close"] - s["btc_price_at_signal"]) > 100]
    small_moves = [s for s in signals if s.get("btc_price_at_signal") and s.get("btc_price_at_close") and abs(s["btc_price_at_close"] - s["btc_price_at_signal"]) <= 100]

    if big_moves:
        bm_wins = len([s for s in big_moves if s["outcome"] == "WIN"])
        results["high_vol_win_rate"] = round(bm_wins / len(big_moves) * 100, 1)
    if small_moves:
        sm_wins = len([s for s in small_moves if s["outcome"] == "WIN"])
        results["low_vol_win_rate"] = round(sm_wins / len(small_moves) * 100, 1)

    # Win rate by time of day
    hour_stats = {}
    for s in signals:
        try:
            hour = datetime.fromisoformat(s["created_at"].replace("Z", "+00:00")).hour
            if hour not in hour_stats:
                hour_stats[hour] = {"total": 0, "wins": 0}
            hour_stats[hour]["total"] += 1
            if s["outcome"] == "WIN":
                hour_stats[hour]["wins"] += 1
        except:
            pass

    results["best_hours"] = []
    results["worst_hours"] = []
    for hour, stats in sorted(hour_stats.items()):
        if stats["total"] >= 3:
            wr = round(stats["wins"] / stats["total"] * 100, 1)
            entry = {"hour_utc": hour, "win_rate": wr, "total": stats["total"]}
            if wr >= 55:
                results["best_hours"].append(entry)
            elif wr <= 45:
                results["worst_hours"].append(entry)

    return results


def find_winning_combinations():
    """Find which specific indicator combinations produce wins."""
    signals = get_all_resolved_signals(500)
    if not signals or len(signals) < 20:
        return {"status": "Need more data"}

    combos = {}

    for s in signals:
        conditions = []

        # RSI condition
        rsi = s.get("rsi")
        if rsi is not None:
            if rsi < 30:
                conditions.append("RSI_OVERSOLD")
            elif rsi > 70:
                conditions.append("RSI_OVERBOUGHT")
            elif rsi < 45:
                conditions.append("RSI_LOW")
            elif rsi > 55:
                conditions.append("RSI_HIGH")
            else:
                conditions.append("RSI_NEUTRAL")

        # MACD condition
        macd = s.get("macd")
        if macd is not None:
            conditions.append("MACD_POS" if macd > 0 else "MACD_NEG")

        # MACD histogram
        hist = s.get("macd_histogram")
        if hist is not None:
            conditions.append("HIST_POS" if hist > 0 else "HIST_NEG")

        # Momentum
        mom = s.get("momentum")
        if mom is not None:
            conditions.append("MOM_POS" if mom > 0 else "MOM_NEG")

        # EMA cross
        ema9 = s.get("ema_9")
        ema21 = s.get("ema_21")
        if ema9 and ema21:
            conditions.append("EMA_BULL" if ema9 > ema21 else "EMA_BEAR")

        # VWAP
        vwap = s.get("vwap")
        price = s.get("btc_price_at_signal")
        if vwap and price:
            conditions.append("ABOVE_VWAP" if price > vwap else "BELOW_VWAP")

        # Test all pairs and triples of conditions
        for i in range(len(conditions)):
            for j in range(i + 1, len(conditions)):
                pair = tuple(sorted([conditions[i], conditions[j]]))
                if pair not in combos:
                    combos[pair] = {"wins": 0, "total": 0}
                combos[pair]["total"] += 1
                if s["outcome"] == "WIN":
                    combos[pair]["wins"] += 1

                # Triples
                for k in range(j + 1, len(conditions)):
                    triple = tuple(sorted([conditions[i], conditions[j], conditions[k]]))
                    if triple not in combos:
                        combos[triple] = {"wins": 0, "total": 0}
                    combos[triple]["total"] += 1
                    if s["outcome"] == "WIN":
                        combos[triple]["wins"] += 1

    # Filter to combos with enough data and sort by win rate
    strong_combos = []
    for combo, stats in combos.items():
        if stats["total"] >= 5:
            wr = round(stats["wins"] / stats["total"] * 100, 1)
            strong_combos.append({
                "conditions": " + ".join(combo),
                "win_rate": wr,
                "total": stats["total"],
                "wins": stats["wins"]
            })

    strong_combos.sort(key=lambda x: x["win_rate"], reverse=True)

    return {
        "best_combos": strong_combos[:10],
        "worst_combos": strong_combos[-10:] if len(strong_combos) >= 10 else [],
        "total_combos_analyzed": len(strong_combos)
    }


def get_metrics_summary():
    """Get a readable summary for Claude."""
    metrics = calculate_advanced_metrics()
    combos = find_winning_combinations()

    lines = []
    lines.append(f"=== ADVANCED METRICS ({metrics.get('total_signals', 0)} signals) ===")
    lines.append(f"Win Rate: {metrics.get('win_rate', 0)}%")
    lines.append(f"Expected Value: ${metrics.get('expected_value', 0):.2f} per trade {'(PROFITABLE)' if metrics.get('ev_positive') else '(LOSING)'}")
    lines.append(f"Avg Win: ${metrics.get('avg_win_magnitude', 0):.2f} | Avg Loss: ${metrics.get('avg_loss_magnitude', 0):.2f} | Ratio: {metrics.get('win_loss_ratio', 0)}")
    lines.append(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0)}")
    lines.append(f"Max Drawdown: ${metrics.get('max_drawdown', 0):.2f}")
    lines.append(f"Profit Factor: {metrics.get('profit_factor', 0)}")

    if metrics.get("high_vol_win_rate"):
        lines.append(f"High Volatility WR: {metrics['high_vol_win_rate']}% | Low Volatility WR: {metrics.get('low_vol_win_rate', 'N/A')}%")

    if metrics.get("best_hours"):
        best = metrics["best_hours"][:3]
        lines.append(f"Best Hours (UTC): {', '.join(f'{h[\"hour_utc\"]}:00 ({h[\"win_rate\"]}%)' for h in best)}")
    if metrics.get("worst_hours"):
        worst = metrics["worst_hours"][:3]
        lines.append(f"Worst Hours (UTC): {', '.join(f'{h[\"hour_utc\"]}:00 ({h[\"win_rate\"]}%)' for h in worst)}")

    if combos.get("best_combos"):
        lines.append("\n=== BEST INDICATOR COMBOS ===")
        for c in combos["best_combos"][:5]:
            lines.append(f"  {c['conditions']}: {c['win_rate']}% ({c['wins']}/{c['total']})")

    if combos.get("worst_combos"):
        lines.append("\n=== WORST INDICATOR COMBOS (AVOID) ===")
        for c in combos["worst_combos"][:5]:
            lines.append(f"  {c['conditions']}: {c['win_rate']}% ({c['wins']}/{c['total']})")

    return "\n".join(lines)


if __name__ == "__main__":
    print(get_metrics_summary())
