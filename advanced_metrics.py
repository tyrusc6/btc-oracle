"""
BTC Oracle - Advanced Metrics (Production Grade)
"""

import db
import numpy as np
from datetime import datetime, timezone


def get_all_resolved_signals(limit=500):
    return db.select("signals", f"outcome=not.is.null&order=created_at.desc&limit={limit}")


def calculate_advanced_metrics():
    signals = get_all_resolved_signals()
    if not signals or len(signals) < 10:
        return {"status": "Need at least 10 resolved signals"}

    results = {"total_signals": len(signals)}
    wins = [s for s in signals if s["outcome"] == "WIN"]
    losses = [s for s in signals if s["outcome"] == "LOSS"]
    results["wins"] = len(wins)
    results["losses"] = len(losses)
    results["win_rate"] = round(len(wins) / len(signals) * 100, 2)

    changes = []
    win_mags = []
    loss_mags = []
    for s in signals:
        if s.get("btc_price_at_signal") is not None and s.get("btc_price_at_close") is not None:
            change = s["btc_price_at_close"] - s["btc_price_at_signal"]
            pnl = change if s["signal"] == "UP" else -change
            changes.append(pnl)
            if s["outcome"] == "WIN":
                win_mags.append(abs(change))
            else:
                loss_mags.append(abs(change))

    if changes:
        results["expected_value"] = round(np.mean(changes), 4)
        results["ev_positive"] = results["expected_value"] > 0
        results["avg_win_magnitude"] = round(np.mean(win_mags), 2) if win_mags else 0
        results["avg_loss_magnitude"] = round(np.mean(loss_mags), 2) if loss_mags else 0
        results["win_loss_ratio"] = round(results["avg_win_magnitude"] / results["avg_loss_magnitude"], 3) if results["avg_loss_magnitude"] > 0 else 999

        if len(changes) > 1:
            std = np.std(changes)
            results["sharpe_ratio"] = round((np.mean(changes) / std) * np.sqrt(35040), 3) if std > 0 else 0

        cumulative = np.cumsum(changes)
        peak = np.maximum.accumulate(cumulative)
        results["max_drawdown"] = round(float(np.max(peak - cumulative)), 2)

        gross_profit = sum(c for c in changes if c > 0)
        gross_loss = abs(sum(c for c in changes if c < 0))
        results["profit_factor"] = round(gross_profit / gross_loss, 3) if gross_loss > 0 else 999

    # Win rate by volatility
    big_moves = [s for s in signals if s.get("btc_price_at_signal") is not None and s.get("btc_price_at_close") is not None and abs(s["btc_price_at_close"] - s["btc_price_at_signal"]) > 100]
    small_moves = [s for s in signals if s.get("btc_price_at_signal") is not None and s.get("btc_price_at_close") is not None and abs(s["btc_price_at_close"] - s["btc_price_at_signal"]) <= 100]

    if big_moves:
        results["high_vol_win_rate"] = round(len([s for s in big_moves if s["outcome"] == "WIN"]) / len(big_moves) * 100, 1)
    if small_moves:
        results["low_vol_win_rate"] = round(len([s for s in small_moves if s["outcome"] == "WIN"]) / len(small_moves) * 100, 1)

    # Win rate by hour
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
    signals = get_all_resolved_signals(500)
    if not signals or len(signals) < 20:
        return {"status": "Need more data"}

    combos = {}
    for s in signals:
        conditions = []

        rsi = s.get("rsi")
        if rsi is not None:
            if rsi < 30: conditions.append("RSI_OVERSOLD")
            elif rsi > 70: conditions.append("RSI_OVERBOUGHT")
            elif rsi < 45: conditions.append("RSI_LOW")
            elif rsi > 55: conditions.append("RSI_HIGH")
            else: conditions.append("RSI_NEUTRAL")

        macd = s.get("macd")
        if macd is not None:
            conditions.append("MACD_POS" if macd > 0 else "MACD_NEG")

        hist = s.get("macd_histogram")
        if hist is not None:
            conditions.append("HIST_POS" if hist > 0 else "HIST_NEG")

        mom = s.get("momentum")
        if mom is not None:
            conditions.append("MOM_POS" if mom > 0 else "MOM_NEG")

        if s.get("ema_9") is not None and s.get("ema_21") is not None:
            conditions.append("EMA_BULL" if s["ema_9"] > s["ema_21"] else "EMA_BEAR")

        if s.get("vwap") is not None and s.get("btc_price_at_signal") is not None:
            conditions.append("ABOVE_VWAP" if s["btc_price_at_signal"] > s["vwap"] else "BELOW_VWAP")

        for i in range(len(conditions)):
            for j in range(i + 1, len(conditions)):
                pair = tuple(sorted([conditions[i], conditions[j]]))
                if pair not in combos:
                    combos[pair] = {"wins": 0, "total": 0}
                combos[pair]["total"] += 1
                if s["outcome"] == "WIN":
                    combos[pair]["wins"] += 1

                for k in range(j + 1, len(conditions)):
                    triple = tuple(sorted([conditions[i], conditions[j], conditions[k]]))
                    if triple not in combos:
                        combos[triple] = {"wins": 0, "total": 0}
                    combos[triple]["total"] += 1
                    if s["outcome"] == "WIN":
                        combos[triple]["wins"] += 1

    strong = []
    for combo, stats in combos.items():
        if stats["total"] >= 5:
            wr = round(stats["wins"] / stats["total"] * 100, 1)
            strong.append({"conditions": " + ".join(combo), "win_rate": wr, "total": stats["total"], "wins": stats["wins"]})

    strong.sort(key=lambda x: x["win_rate"], reverse=True)
    return {"best_combos": strong[:10], "worst_combos": strong[-10:] if len(strong) >= 10 else [], "total_combos": len(strong)}


def get_metrics_summary():
    metrics = calculate_advanced_metrics()
    combos = find_winning_combinations()

    lines = []
    lines.append(f"=== METRICS ({metrics.get('total_signals', 0)} signals) ===")
    lines.append(f"WR: {metrics.get('win_rate', 0)}% | EV: ${metrics.get('expected_value', 0):.2f}/trade {'(PROFITABLE)' if metrics.get('ev_positive') else '(LOSING)'}")
    lines.append(f"Avg Win: ${metrics.get('avg_win_magnitude', 0):.2f} | Avg Loss: ${metrics.get('avg_loss_magnitude', 0):.2f} | Ratio: {metrics.get('win_loss_ratio', 0)}")
    lines.append(f"Sharpe: {metrics.get('sharpe_ratio', 0)} | Drawdown: ${metrics.get('max_drawdown', 0):.2f} | PF: {metrics.get('profit_factor', 0)}")

    if metrics.get("high_vol_win_rate") is not None:
        lines.append(f"High Vol WR: {metrics['high_vol_win_rate']}% | Low Vol WR: {metrics.get('low_vol_win_rate', 'N/A')}%")

    if metrics.get("best_hours"):
        best = metrics["best_hours"][:3]
        best_parts = [str(h["hour_utc"]) + ":00 (" + str(h["win_rate"]) + "%)" for h in best]
        lines.append("Best Hours: " + ", ".join(best_parts))
    if metrics.get("worst_hours"):
        worst = metrics["worst_hours"][:3]
        worst_parts = [str(h["hour_utc"]) + ":00 (" + str(h["win_rate"]) + "%)" for h in worst]
        lines.append("Worst Hours: " + ", ".join(worst_parts))

    if combos.get("best_combos"):
        lines.append("=== BEST COMBOS ===")
        for c in combos["best_combos"][:5]:
            lines.append(f"  {c['conditions']}: {c['win_rate']}% ({c['wins']}/{c['total']})")

    if combos.get("worst_combos"):
        lines.append("=== WORST COMBOS ===")
        for c in combos["worst_combos"][:5]:
            lines.append(f"  {c['conditions']}: {c['win_rate']}% ({c['wins']}/{c['total']})")

    return "\n".join(lines)


if __name__ == "__main__":
    print(get_metrics_summary())
