"""
ml_backtest.py — Evaluate ML model and rule-based signals on historical features.

Two modes:
  --mode rules   → Rule-based signals (CVD + RSI + OB imbalance). Use before training.
  --mode model   → LightGBM model predictions. Use after training.

Usage:
    python ml_backtest.py --mode rules
    python ml_backtest.py --mode model
    python ml_backtest.py --mode model --threshold 0.70
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
    # New features
    "macd_histogram", "stoch_rsi_k", "ema_cross", "momentum_11",
    "obv_slope", "atr_14", "roc_15m", "volume_ratio_5m",
    "delta_ratio_5m", "large_trade_ratio", "price_momentum_5m",
    "candle_body_ratio",
]


def load_features(features_path):
    if not os.path.exists(features_path):
        print(f"ERROR: Features file not found: {features_path}")
        print("Run: python feature_builder.py first")
        sys.exit(1)

    df = pd.read_parquet(features_path)
    df = df.dropna(subset=["price_up_15m"])

    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_cols].fillna(0)
    y = df["price_up_15m"].astype(int)

    print(f"Loaded {len(df)} samples, {len(available_cols)} features")
    print(f"Label balance: {y.value_counts().to_dict()}")
    return X, y, df


def rule_based_signal(row):
    """
    Rule-based signal engine (no model needed).
    Combines CVD, RSI, and orderbook imbalance into a directional score.
    Returns (direction: int, confidence: float).
    """
    score = 0.0
    max_score = 5.0

    cvd_5m = row.get("cvd_5m", 0)
    if cvd_5m > 0:
        score += 1.5
    elif cvd_5m < 0:
        score -= 1.5

    if row.get("cvd_divergence", 0) == 1.0:
        score = -score

    rsi = row.get("rsi_14", 50)
    if rsi < 35:
        score += 1.0
    elif rsi > 65:
        score -= 1.0

    ob = row.get("ob_imbalance", 0.5)
    if ob > 0.62:
        score += 0.8
    elif ob < 0.38:
        score -= 0.8

    vwap_dev = row.get("vwap_dev", 0)
    if vwap_dev < -0.001:
        score += 0.7
    elif vwap_dev > 0.001:
        score -= 0.7

    direction = 1 if score > 0 else 0
    confidence = min(abs(score) / max_score, 1.0)
    return direction, confidence


def evaluate_predictions(y_true, y_pred, y_prob, confidence, confidence_threshold,
                         bet_size_usd=10.0, yes_price_avg=0.50):
    """Evaluate prediction quality at a given confidence threshold."""
    mask = confidence >= confidence_threshold
    n_total = len(y_true)
    n_filtered = mask.sum()

    if n_filtered == 0:
        return {
            "total_signals": n_total, "filtered_signals": 0, "coverage": 0.0,
            "accuracy": 0.0, "edge_vs_random": 0.0, "simulated_pnl_usd": 0.0,
        }

    yt = y_true[mask]
    yp = y_pred[mask]

    correct = (yt == yp).sum()
    accuracy = correct / n_filtered

    # Simulated P&L on Kalshi-style contracts
    contracts = int(bet_size_usd / yes_price_avg)
    win_pnl = contracts * (1 - yes_price_avg)
    loss_pnl = -contracts * yes_price_avg
    pnl = correct * win_pnl + (n_filtered - correct) * loss_pnl

    return {
        "total_signals": int(n_total),
        "filtered_signals": int(n_filtered),
        "coverage": round(n_filtered / n_total, 4),
        "accuracy": round(accuracy, 4),
        "edge_vs_random": round(accuracy - 0.5, 4),
        "simulated_pnl_usd": round(float(pnl), 2),
    }


def print_backtest_report(metrics, mode, threshold):
    edge = metrics["edge_vs_random"]
    edge_str = f"+{edge:.1%}" if edge > 0 else f"{edge:.1%}"

    print("\n" + "=" * 55)
    print(f"  BACKTEST REPORT  |  mode={mode}  |  threshold={threshold:.2f}")
    print("=" * 55)
    print(f"  Total signals:       {metrics['total_signals']:>10,}")
    print(f"  Signals at threshold:{metrics['filtered_signals']:>10,}")
    print(f"  Coverage:            {metrics['coverage']:>10.1%}")
    print("-" * 55)
    print(f"  Accuracy:            {metrics['accuracy']:>10.1%}")
    print(f"  Edge vs random:      {edge_str:>10}")
    print("-" * 55)
    print(f"  Simulated P&L:       ${metrics['simulated_pnl_usd']:>9.2f}")
    print("=" * 55)

    if edge < 0.03:
        print("  WARNING: WEAK SIGNAL: edge < 3%. Investigate features.")
    elif edge < 0.05:
        print("  OK: Marginal edge. Consider more data or features.")
    else:
        print("  SOLID EDGE: Proceed to live trading.")

    if metrics["coverage"] < 0.10:
        print("  WARNING: LOW COVERAGE: < 10% of signals pass threshold.")
        print("     Consider lowering CONFIDENCE_THRESHOLD.")
    print()


def run_backtest(features_path, mode="rules", confidence_threshold=0.65,
                 model_path="model.pkl", output_path=".tmp/backtest_results.csv",
                 bet_size_usd=10.0, test_size=0.2):
    """
    Run backtest on UNSEEN data only (last test_size portion).
    This prevents inflated accuracy from evaluating on training data.
    """
    X, y, _ = load_features(features_path)

    # Only evaluate on the test portion (data the model never saw during training)
    n = len(X)
    split_idx = int(n * (1 - test_size))

    if mode == "model":
        # Evaluate only on the test set (last 20%)
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]
        print(f"Evaluating on TEST SET only: {len(X_test)} samples (last {test_size:.0%} of data)")
    else:
        # Rules have no training, so evaluate on everything
        X_test = X
        y_test = y
        print(f"Rule-based: evaluating on all {len(X_test)} samples")

    if mode == "model":
        if not os.path.exists(model_path):
            print(f"ERROR: Model not found at {model_path}")
            print("Run: python model_trainer.py first")
            sys.exit(1)

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        print(f"Running model predictions on {len(X_test)} samples...")
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        confidence = np.abs(y_prob - 0.5) * 2

    else:
        print(f"Running rule-based signals on {len(X_test)} samples...")
        predictions = [rule_based_signal(row) for _, row in X_test.iterrows()]
        y_pred = np.array([p[0] for p in predictions])
        confidence = np.array([p[1] for p in predictions])
        y_prob = y_pred.astype(float)

    metrics = evaluate_predictions(y_test.values, y_pred, y_prob, confidence, confidence_threshold, bet_size_usd)
    print_backtest_report(metrics, mode, confidence_threshold)

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df = pd.DataFrame([{"mode": mode, "threshold": confidence_threshold, **metrics}])
    suffix = f"_{mode}"
    out = output_path.replace(".csv", f"{suffix}.csv")
    results_df.to_csv(out, index=False)
    print(f"Results saved to: {out}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest BTC prediction signals")
    parser.add_argument("--mode", choices=["rules", "model"], default="rules")
    parser.add_argument("--threshold", type=float, default=float(os.getenv("CONFIDENCE_THRESHOLD", "0.65")))
    parser.add_argument("--start", default=os.getenv("BACKTEST_START_DATE", "2025-01-01"))
    parser.add_argument("--end", default=os.getenv("BACKTEST_END_DATE", "2025-03-01"))
    parser.add_argument("--symbol", default="XBTUSD")
    parser.add_argument("--bet-size", type=float, default=float(os.getenv("BET_SIZE_USD", "10")))
    args = parser.parse_args()

    features_path = os.path.join(".tmp", f"{args.symbol.lower()}_features_{args.start}_{args.end}.parquet")
    run_backtest(
        features_path=features_path,
        mode=args.mode,
        confidence_threshold=args.threshold,
        bet_size_usd=args.bet_size,
    )
