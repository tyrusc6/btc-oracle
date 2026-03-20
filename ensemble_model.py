"""
ensemble_model.py — Combine LightGBM ML model with rule-based mean reversion strategies.

The ML model alone gets ~51% on unseen data.
The rule-based Bollinger + anti-momentum strategy gets ~58%.
The ensemble combines both — when they agree, we have high confidence.
When they disagree, we skip (reducing losing trades).

This replaces the standalone ML backtest with a smarter hybrid approach.

Usage:
    python ensemble_model.py
    python ensemble_model.py --start 2024-09-01 --end 2025-03-01
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

FEATURE_COLS = [
    "cvd_5m", "cvd_15m", "cvd_1h", "delta_1m", "delta_pct_1m",
    "large_buy_vol_5m", "large_sell_vol_5m", "large_trade_imbalance",
    "absorption_score", "ob_imbalance", "cvd_divergence",
    "rsi_14", "bb_position", "vwap_dev", "roc_5m", "roc_10m",
    "price_vs_high_20m", "price_vs_low_20m",
    "uptick_ratio", "spread_proxy", "trade_count_1m",
    "hour_utc", "minute_utc",
    "volume_1m", "buy_vol_1m", "sell_vol_1m",
    "macd_histogram", "stoch_rsi_k", "ema_cross", "momentum_11",
    "obv_slope", "atr_14", "roc_15m", "volume_ratio_5m",
    "delta_ratio_5m", "large_trade_ratio", "price_momentum_5m",
    "candle_body_ratio",
]


def bollinger_signal(row):
    """
    Mean reversion at Bollinger Band extremes.
    Returns (direction: 1=UP/0=DOWN, confidence: 0-1, should_trade: bool)
    """
    bb_pos = row.get("bb_position", 0.5)
    if pd.isna(bb_pos):
        return 0, 0.0, False

    if bb_pos > 0.85:
        confidence = min(0.95, 0.6 + (bb_pos - 0.85) * 2.0)
        return 0, confidence, True  # Near upper band → DOWN
    elif bb_pos < 0.15:
        confidence = min(0.95, 0.6 + (0.15 - bb_pos) * 2.0)
        return 1, confidence, True  # Near lower band → UP
    return 0, 0.0, False


def anti_momentum_signal(row):
    """
    Fade recent momentum — bet against recent move.
    Returns (direction, confidence, should_trade)
    """
    roc_5m = row.get("roc_5m", 0)
    momentum = row.get("momentum_11", 0)
    if pd.isna(roc_5m) or pd.isna(momentum):
        return 0, 0.0, False

    if roc_5m > 0 and momentum > 0:
        strength = abs(roc_5m) * 100
        confidence = min(0.85, 0.55 + strength * 5)
        return 0, confidence, True  # Been going up → bet DOWN
    elif roc_5m < 0 and momentum < 0:
        strength = abs(roc_5m) * 100
        confidence = min(0.85, 0.55 + strength * 5)
        return 1, confidence, True  # Been going down → bet UP
    return 0, 0.0, False


def rsi_extreme_signal(row):
    """RSI at extreme levels — mean reversion."""
    rsi = row.get("rsi_14", 50)
    if pd.isna(rsi):
        return 0, 0.0, False

    if rsi > 75:
        conf = min(0.90, 0.55 + (rsi - 75) * 0.02)
        return 0, conf, True  # Overbought → DOWN
    elif rsi < 25:
        conf = min(0.90, 0.55 + (25 - rsi) * 0.02)
        return 1, conf, True  # Oversold → UP
    return 0, 0.0, False


def cvd_divergence_signal(row):
    """CVD divergence — price and volume disagree."""
    cvd_div = row.get("cvd_divergence", 0)
    cvd_5m = row.get("cvd_5m", 0)
    roc_5m = row.get("roc_5m", 0)
    if cvd_div != 1.0:
        return 0, 0.0, False

    # Price up + CVD down = distribution → DOWN
    # Price down + CVD up = accumulation → UP
    if roc_5m > 0 and cvd_5m < 0:
        return 0, 0.65, True
    elif roc_5m < 0 and cvd_5m > 0:
        return 1, 0.65, True
    return 0, 0.0, False


def vwap_reversion_signal(row):
    """VWAP mean reversion — price deviating from VWAP."""
    vwap_dev = row.get("vwap_dev", 0)
    if pd.isna(vwap_dev):
        return 0, 0.0, False

    if vwap_dev > 0.002:  # Price well above VWAP
        conf = min(0.80, 0.55 + abs(vwap_dev) * 50)
        return 0, conf, True  # Expect reversion DOWN
    elif vwap_dev < -0.002:  # Price well below VWAP
        conf = min(0.80, 0.55 + abs(vwap_dev) * 50)
        return 1, conf, True  # Expect reversion UP
    return 0, 0.0, False


def ensemble_predict(row, ml_prob):
    """
    Combine ML model probability with rule-based strategies.

    Voting system:
    - ML model gets 1 vote (direction based on probability)
    - Each rule-based strategy that fires gets 1 vote
    - Final direction = majority vote
    - Confidence = agreement level (how many signals align)
    - Only trade when 2+ signals agree

    Returns (direction: 1/0, confidence: float, should_trade: bool, n_signals: int)
    """
    votes_up = 0
    votes_down = 0
    total_confidence = 0.0
    n_signals = 0

    # ML model vote
    ml_direction = 1 if ml_prob >= 0.5 else 0
    ml_confidence = abs(ml_prob - 0.5) * 2
    if ml_confidence > 0.05:  # Only count if model has some opinion
        if ml_direction == 1:
            votes_up += 1
        else:
            votes_down += 1
        total_confidence += ml_confidence
        n_signals += 1

    # Rule-based strategy votes
    strategies = [
        bollinger_signal(row),
        anti_momentum_signal(row),
        rsi_extreme_signal(row),
        cvd_divergence_signal(row),
        vwap_reversion_signal(row),
    ]

    for direction, confidence, should_trade in strategies:
        if should_trade:
            if direction == 1:
                votes_up += 1
            else:
                votes_down += 1
            total_confidence += confidence
            n_signals += 1

    if n_signals == 0:
        return 0, 0.0, False, 0

    # Majority vote
    direction = 1 if votes_up > votes_down else 0

    # Agreement ratio = how aligned the signals are
    max_votes = max(votes_up, votes_down)
    agreement = max_votes / n_signals  # 1.0 = unanimous, 0.5 = split

    # Average confidence weighted by agreement
    avg_confidence = total_confidence / n_signals
    final_confidence = avg_confidence * agreement

    # Only trade when 2+ signals agree in same direction
    should_trade = max_votes >= 2 and agreement >= 0.6

    return direction, round(final_confidence, 4), should_trade, n_signals


def run_ensemble_backtest(features_path, model_path="model.pkl", test_size=0.2):
    """Run the ensemble backtest on unseen data."""
    if not os.path.exists(features_path):
        print(f"ERROR: Features not found: {features_path}")
        sys.exit(1)

    df = pd.read_parquet(features_path)
    df = df.dropna(subset=["price_up_15m"])

    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_cols].fillna(0)
    y = df["price_up_15m"].astype(int)

    # Split: only evaluate on test set
    n = len(X)
    split_idx = int(n * (1 - test_size))
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    print(f"Loaded {len(df)} total samples, evaluating on {len(X_test)} test samples")
    print(f"Features: {len(available_cols)}")

    # Load ML model
    model = None
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print(f"ML model loaded from {model_path}")
    else:
        print("No ML model found — using rule-based strategies only")

    # Get ML probabilities
    if model is not None:
        try:
            ml_probs = model.predict_proba(X_test)[:, 1]
        except Exception as e:
            print(f"ML prediction failed: {e}, using 0.5 for all")
            ml_probs = np.full(len(X_test), 0.5)
    else:
        ml_probs = np.full(len(X_test), 0.5)

    # Run ensemble on each sample
    results = []
    for i, (idx, row) in enumerate(X_test.iterrows()):
        direction, confidence, should_trade, n_signals = ensemble_predict(row, ml_probs[i])
        results.append({
            "true_label": y_test.iloc[i],
            "pred_direction": direction,
            "confidence": confidence,
            "should_trade": should_trade,
            "n_signals": n_signals,
        })

    results_df = pd.DataFrame(results)

    # Overall accuracy (all predictions)
    all_correct = (results_df["true_label"] == results_df["pred_direction"]).sum()
    all_accuracy = all_correct / len(results_df)

    # Filtered accuracy (only when ensemble says trade)
    traded = results_df[results_df["should_trade"]]
    if len(traded) > 0:
        trade_correct = (traded["true_label"] == traded["pred_direction"]).sum()
        trade_accuracy = trade_correct / len(traded)
        trade_coverage = len(traded) / len(results_df)
    else:
        trade_accuracy = 0.0
        trade_coverage = 0.0

    # ML-only accuracy for comparison
    ml_pred = (ml_probs >= 0.5).astype(int)
    ml_correct = (y_test.values == ml_pred).sum()
    ml_accuracy = ml_correct / len(y_test)

    # Kalshi P&L simulation
    bet_size = 10.0
    yes_price = 0.50
    contracts = int(bet_size / yes_price)
    win_pnl = contracts * (1 - yes_price)
    loss_pnl = -contracts * yes_price

    if len(traded) > 0:
        pnl = trade_correct * win_pnl + (len(traded) - trade_correct) * loss_pnl
    else:
        pnl = 0.0

    # Breakdown by signal count
    print("\n" + "=" * 60)
    print("  ENSEMBLE BACKTEST RESULTS (test set only)")
    print("=" * 60)
    print(f"\n  ML model alone:        {ml_accuracy:.1%}")
    print(f"  Ensemble (all):        {all_accuracy:.1%}")
    print(f"  Ensemble (traded):     {trade_accuracy:.1%}  <-- this is the real number")
    print(f"  Coverage:              {trade_coverage:.1%} ({len(traded)} of {len(results_df)} signals)")
    print(f"  Simulated P&L:         ${pnl:.2f}")

    # Breakdown by agreement level
    print(f"\n  --- Accuracy by # of agreeing signals ---")
    for n_sig in sorted(results_df["n_signals"].unique()):
        subset = results_df[results_df["n_signals"] == n_sig]
        traded_subset = subset[subset["should_trade"]]
        if len(traded_subset) > 0:
            acc = (traded_subset["true_label"] == traded_subset["pred_direction"]).sum() / len(traded_subset)
            print(f"  {n_sig} signals agreeing:  {acc:.1%}  ({len(traded_subset)} trades)")
        elif len(subset) > 0:
            acc = (subset["true_label"] == subset["pred_direction"]).sum() / len(subset)
            print(f"  {n_sig} signals (no trade): {acc:.1%}  ({len(subset)} samples)")

    # High confidence trades
    print(f"\n  --- Accuracy by confidence level ---")
    for threshold in [0.3, 0.4, 0.5, 0.6]:
        high_conf = traded[traded["confidence"] >= threshold] if len(traded) > 0 else pd.DataFrame()
        if len(high_conf) > 0:
            acc = (high_conf["true_label"] == high_conf["pred_direction"]).sum() / len(high_conf)
            print(f"  Confidence >= {threshold:.1f}:  {acc:.1%}  ({len(high_conf)} trades)")

    print("=" * 60)

    if trade_accuracy > 0.55:
        print(f"\n  EDGE FOUND: {trade_accuracy:.1%} on traded signals.")
        print(f"  This ensemble is ready for live testing.")
    elif trade_accuracy > 0.52:
        print(f"\n  MARGINAL EDGE: {trade_accuracy:.1%}. Consider tightening filters.")
    else:
        print(f"\n  WEAK: {trade_accuracy:.1%}. Need better signals or more data.")

    # Save results
    os.makedirs(".tmp", exist_ok=True)
    results_df.to_csv(".tmp/ensemble_results.csv", index=False)
    print(f"\nDetailed results saved to: .tmp/ensemble_results.csv")

    return {
        "ml_accuracy": ml_accuracy,
        "ensemble_all_accuracy": all_accuracy,
        "ensemble_traded_accuracy": trade_accuracy,
        "coverage": trade_coverage,
        "n_traded": len(traded),
        "pnl": pnl,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble backtest: ML + rule-based strategies")
    parser.add_argument("--start", default=os.getenv("BACKTEST_START_DATE", "2024-09-01"))
    parser.add_argument("--end", default=os.getenv("BACKTEST_END_DATE", "2025-03-01"))
    parser.add_argument("--symbol", default="XBTUSD")
    args = parser.parse_args()

    features_path = os.path.join(".tmp", f"{args.symbol.lower()}_features_{args.start}_{args.end}.parquet")
    run_ensemble_backtest(features_path)
