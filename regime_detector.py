"""
BTC Oracle - Market Regime Detector
Classifies current market into: TRENDING_UP, TRENDING_DOWN, RANGING, HIGH_VOLATILITY, BREAKOUT
Uses different strategy per regime.
"""

import numpy as np
import requests


def fetch_candles(interval=5, count=100):
    """Fetch candles from Kraken."""
    try:
        resp = requests.get(f"https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval={interval}", timeout=10)
        data = resp.json()
        if data.get("result"):
            candles = data["result"].get("XXBTZUSD", [])
            if candles:
                closes = [float(c[4]) for c in candles[-count:]]
                highs = [float(c[2]) for c in candles[-count:]]
                lows = [float(c[3]) for c in candles[-count:]]
                volumes = [float(c[6]) for c in candles[-count:]]
                return closes, highs, lows, volumes
    except:
        pass
    return [], [], [], []


def detect_regime():
    """Classify current market regime using multiple timeframes."""
    # Get 5-min candles (last ~8 hours)
    closes, highs, lows, volumes = fetch_candles(interval=5, count=100)
    if len(closes) < 30:
        return {"regime": "UNKNOWN", "confidence": 0, "details": "Not enough data"}

    prices = np.array(closes)
    
    # 1. Trend strength via ADX-like calculation
    price_changes = np.diff(prices)
    up_moves = np.where(price_changes > 0, price_changes, 0)
    down_moves = np.where(price_changes < 0, -price_changes, 0)
    
    avg_up = np.mean(up_moves[-14:])
    avg_down = np.mean(down_moves[-14:])
    
    if avg_up + avg_down > 0:
        trend_strength = abs(avg_up - avg_down) / (avg_up + avg_down)
    else:
        trend_strength = 0
    
    # 2. Volatility measurement
    returns = np.diff(prices) / prices[:-1]
    volatility = np.std(returns[-20:]) * 100  # as percentage
    avg_volatility = np.std(returns) * 100
    vol_expanding = volatility > avg_volatility * 1.3
    
    # 3. Range detection - are we bouncing between support/resistance?
    recent_30 = prices[-30:]
    price_range = max(recent_30) - min(recent_30)
    range_pct = (price_range / np.mean(recent_30)) * 100
    
    # Check if price keeps touching similar highs and lows
    upper_touches = sum(1 for p in recent_30 if p > max(recent_30) * 0.998)
    lower_touches = sum(1 for p in recent_30 if p < min(recent_30) * 1.002)
    is_ranging = upper_touches >= 2 and lower_touches >= 2 and range_pct < 1.5
    
    # 4. Breakout detection
    prev_high = max(prices[-40:-10]) if len(prices) >= 40 else max(prices[:-10])
    prev_low = min(prices[-40:-10]) if len(prices) >= 40 else min(prices[:-10])
    current = prices[-1]
    is_breakout_up = current > prev_high and volatility > avg_volatility
    is_breakout_down = current < prev_low and volatility > avg_volatility
    
    # 5. Trend direction
    sma_10 = np.mean(prices[-10:])
    sma_30 = np.mean(prices[-30:])
    slope = np.polyfit(np.arange(20), prices[-20:], 1)[0]
    trending_up = sma_10 > sma_30 and slope > 0
    trending_down = sma_10 < sma_30 and slope < 0
    
    # 6. Volume analysis
    recent_vol = np.mean(volumes[-5:]) if volumes else 0
    avg_vol = np.mean(volumes[-30:]) if volumes else 0
    high_volume = recent_vol > avg_vol * 1.5 if avg_vol > 0 else False
    
    # Classify regime
    confidence = 0
    details = {}
    
    if is_breakout_up:
        regime = "BREAKOUT_UP"
        confidence = min(0.9, 0.6 + trend_strength * 0.3)
        details["strategy"] = "Follow breakout direction. High confidence UP signals."
    elif is_breakout_down:
        regime = "BREAKOUT_DOWN"
        confidence = min(0.9, 0.6 + trend_strength * 0.3)
        details["strategy"] = "Follow breakout direction. High confidence DOWN signals."
    elif volatility > avg_volatility * 2:
        regime = "HIGH_VOLATILITY"
        confidence = 0.5
        details["strategy"] = "Reduce confidence on all signals. Expect reversals. Mean reversion works."
    elif is_ranging:
        regime = "RANGING"
        confidence = 0.6
        details["strategy"] = "Buy near support, sell near resistance. Mean reversion. Fade extremes."
    elif trending_up and trend_strength > 0.3:
        regime = "TRENDING_UP"
        confidence = min(0.85, 0.5 + trend_strength * 0.5)
        details["strategy"] = "Follow trend. Favor UP signals. Buy dips within trend."
    elif trending_down and trend_strength > 0.3:
        regime = "TRENDING_DOWN"
        confidence = min(0.85, 0.5 + trend_strength * 0.5)
        details["strategy"] = "Follow trend. Favor DOWN signals. Sell rallies within trend."
    elif trending_up:
        regime = "WEAK_UPTREND"
        confidence = 0.55
        details["strategy"] = "Slight UP bias but be cautious. Low confidence."
    elif trending_down:
        regime = "WEAK_DOWNTREND"
        confidence = 0.55
        details["strategy"] = "Slight DOWN bias but be cautious. Low confidence."
    else:
        regime = "CHOPPY"
        confidence = 0.4
        details["strategy"] = "No clear direction. Avoid trading if possible. Low confidence."
    
    details.update({
        "trend_strength": round(trend_strength, 3),
        "volatility_pct": round(volatility, 4),
        "avg_volatility_pct": round(avg_volatility, 4),
        "vol_expanding": vol_expanding,
        "range_pct": round(range_pct, 3),
        "is_ranging": is_ranging,
        "high_volume": high_volume,
        "slope": round(slope, 4),
    })
    
    return {
        "regime": regime,
        "regime_confidence": round(confidence, 3),
        "regime_strategy": details.get("strategy", ""),
        "trend_strength": details["trend_strength"],
        "volatility_pct": details["volatility_pct"],
        "vol_expanding": details["vol_expanding"],
        "range_pct": details["range_pct"],
        "high_volume": details["high_volume"],
    }


if __name__ == "__main__":
    result = detect_regime()
    for k, v in result.items():
        print(f"  {k}: {v}")
