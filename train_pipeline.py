"""
train_pipeline.py — One-command pipeline to download data, build features, train model, and backtest.

This is the entry point for the full ML training workflow.
Run this ONCE to build the model, then the live bot (bot.py) uses the trained model.pkl automatically.

Usage:
    python train_pipeline.py
    python train_pipeline.py --start 2025-01-01 --end 2025-03-14
    python train_pipeline.py --skip-download   # If data is already cached
"""

import argparse
import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Full ML training pipeline for BTC Oracle")
    parser.add_argument("--start", default=os.getenv("BACKTEST_START_DATE", "2025-01-01"))
    parser.add_argument("--end", default=os.getenv("BACKTEST_END_DATE", "2025-03-01"))
    parser.add_argument("--symbol", default="XBTUSD")
    parser.add_argument("--skip-download", action="store_true", help="Skip data download (use cached)")
    parser.add_argument("--threshold", type=float, default=0.65, help="Confidence threshold for backtest")
    args = parser.parse_args()

    os.makedirs(".tmp", exist_ok=True)

    trades_path = os.path.join(".tmp", f"{args.symbol.lower()}_trades_{args.start}_{args.end}.parquet")
    features_path = os.path.join(".tmp", f"{args.symbol.lower()}_features_{args.start}_{args.end}.parquet")
    model_path = "model.pkl"

    total_start = time.time()

    # ===== STEP 1: Download historical trades =====
    print("\n" + "=" * 60)
    print("  STEP 1/4: DOWNLOAD HISTORICAL TRADES")
    print("=" * 60)

    if args.skip_download and os.path.exists(trades_path):
        print(f"Skipping download — using cached: {trades_path}")
    else:
        from historical_data import load_or_download
        load_or_download(args.symbol, args.start, args.end)

    if not os.path.exists(trades_path):
        print(f"ERROR: Trade data not found at {trades_path}")
        sys.exit(1)

    # ===== STEP 2: Build features =====
    print("\n" + "=" * 60)
    print("  STEP 2/4: BUILD FEATURES")
    print("=" * 60)

    import pandas as pd
    from feature_builder import build_historical_features

    trades_df = pd.read_parquet(trades_path)
    print(f"Loaded {len(trades_df):,} trades")
    build_historical_features(trades_df, features_path)

    # ===== STEP 3: Train model =====
    print("\n" + "=" * 60)
    print("  STEP 3/4: TRAIN MODEL")
    print("=" * 60)

    from model_trainer import load_features, train_lightgbm

    X, y, _ = load_features(features_path)
    results = train_lightgbm(X, y, output_model_path=model_path)

    # ===== STEP 4: Backtest =====
    print("\n" + "=" * 60)
    print("  STEP 4/4: BACKTEST")
    print("=" * 60)

    from ml_backtest import run_backtest

    # Run both rule-based and model-based backtests
    print("\n--- Rule-based backtest ---")
    rules_metrics = run_backtest(features_path, mode="rules", confidence_threshold=args.threshold)

    print("\n--- Model backtest ---")
    model_metrics = run_backtest(features_path, mode="model", confidence_threshold=args.threshold,
                                 model_path=model_path)

    # ===== SUMMARY =====
    elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total time:     {elapsed / 60:.1f} minutes")
    print(f"  Model AUC:      {results['auc']:.4f}")
    print(f"  Model accuracy: {results['accuracy']:.4f}")
    print(f"  Rules accuracy: {rules_metrics['accuracy']:.4f}")
    print(f"  Model accuracy: {model_metrics['accuracy']:.4f}")
    print(f"  Model edge:     {model_metrics['edge_vs_random']:+.4f}")
    print(f"  Model P&L sim:  ${model_metrics['simulated_pnl_usd']:.2f}")
    print(f"\n  Model saved to: {model_path}")
    print(f"  The live bot (bot.py) will automatically load this model.")
    print("=" * 60)

    if model_metrics["edge_vs_random"] < 0.03:
        print("\n  WARNING: Model edge is weak (<3%). Consider:")
        print("    - Downloading more data (extend date range)")
        print("    - Adding more features")
        print("    - Checking data quality")
    elif model_metrics["edge_vs_random"] >= 0.05:
        print("\n  Model shows a solid edge. Ready for live testing.")


if __name__ == "__main__":
    main()
