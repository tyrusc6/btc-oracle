"""
model_trainer.py — Train LightGBM binary classifier to predict BTC 15-min direction.

CRITICAL: Uses time-based train/test split (NOT random shuffle).
Financial time series has temporal autocorrelation — random splits leak
future information into training and produce deceptively high accuracy.

Output:
    model.pkl               — trained LightGBM model (root dir, used by lgbm_signal.py)
    .tmp/feature_importance.csv — feature importances

Usage:
    python model_trainer.py
    python model_trainer.py --start 2025-01-01 --end 2025-03-01
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

LGBM_PARAMS = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "num_leaves": 63,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "n_jobs": -1,
}


def load_features(features_path: str) -> tuple:
    if not os.path.exists(features_path):
        print(f"ERROR: Features file not found: {features_path}")
        print("Run: python feature_builder.py first")
        sys.exit(1)

    df = pd.read_parquet(features_path)
    df = df.dropna(subset=["price_up_15m"])

    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_cols].fillna(0).astype("float32")
    y = df["price_up_15m"].astype(int)

    print(f"Loaded {len(df)} samples with {len(available_cols)} features")
    print(f"Label balance: {y.value_counts().to_dict()}")
    return X, y, df


def time_based_split(X, y, test_size=0.2):
    """
    Split preserving temporal order.
    Train = first (1-test_size). Test = last test_size.
    Last 10% of train used as validation for early stopping.
    """
    n = len(X)
    split_idx = int(n * (1 - test_size))

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    val_split_idx = int(len(X_train) * 0.9)
    X_val = X_train.iloc[val_split_idx:]
    y_val = y_train.iloc[val_split_idx:]
    X_train = X_train.iloc[:val_split_idx]
    y_train = y_train.iloc[:val_split_idx]

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_lightgbm(X, y, test_size=0.2, output_model_path="model.pkl",
                    output_importance_path=".tmp/feature_importance.csv"):
    """
    Train LightGBM binary classifier with time-based split.
    Saves model to root dir (model.pkl) so lgbm_signal.py can load it.
    """
    try:
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score, accuracy_score
    except ImportError:
        print("ERROR: lightgbm and scikit-learn required. Run: pip install -r requirements.txt")
        sys.exit(1)

    X_train, X_val, X_test, y_train, y_val, y_test = time_based_split(X, y, test_size)

    # Handle class imbalance
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"Class balance — pos: {pos_count} | neg: {neg_count} | scale_pos_weight: {scale_pos_weight:.3f}")

    params = {**LGBM_PARAMS, "scale_pos_weight": scale_pos_weight}

    print("\nTraining LightGBM model...")
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )

    print(f"\nBest iteration: {model.best_iteration_}")

    # Evaluate on held-out test set
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'='*45}")
    print(f"  TEST SET RESULTS")
    print(f"{'='*45}")
    print(f"  AUC:      {auc:.4f}   (target: > 0.55)")
    print(f"  Accuracy: {accuracy:.4f}  (target: > 0.55)")

    if auc < 0.52:
        print("\n  WARNING: AUC < 0.52: Model has no real edge.")
        print("     Collect more data, add features, or check for data issues.")
    elif auc < 0.55:
        print("\n  WARNING: Marginal edge. Consider more data or better features.")
    else:
        print("\n  SOLID AUC. Proceed to backtest.")
    print(f"{'='*45}\n")

    # Feature importances
    importances = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("Top 10 features by importance:")
    print(importances.head(10).to_string(index=False))
    print()

    # Save model to root dir (where lgbm_signal.py expects it)
    with open(output_model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to: {output_model_path}")

    os.makedirs(os.path.dirname(output_importance_path), exist_ok=True)
    importances.to_csv(output_importance_path, index=False)
    print(f"Feature importances saved to: {output_importance_path}")

    return {
        "auc": float(auc),
        "accuracy": float(accuracy),
        "best_iteration": int(model.best_iteration_),
        "feature_importances": dict(zip(importances["feature"], importances["importance"])),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LightGBM model for BTC direction prediction")
    parser.add_argument("--start", default=os.getenv("BACKTEST_START_DATE", "2025-01-01"))
    parser.add_argument("--end", default=os.getenv("BACKTEST_END_DATE", "2025-03-01"))
    parser.add_argument("--symbol", default="XBTUSD")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    features_path = os.path.join(".tmp", f"{args.symbol.lower()}_features_{args.start}_{args.end}.parquet")
    X, y, _ = load_features(features_path)
    train_lightgbm(X, y, test_size=args.test_size)
