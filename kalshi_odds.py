"""
BTC Oracle - Kalshi Odds Fetcher
Pulls live contract prices from Kalshi to see market expectations.
Compares bot signal against market consensus for edge detection.
"""

import requests
import time
from datetime import datetime, timezone


def safe_get(url, timeout=10, headers=None):
    try:
        resp = requests.get(url, timeout=timeout, headers=headers or {})
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return None


def get_kalshi_btc_contracts():
    """
    Fetch current BTC prediction market data from Kalshi.
    Kalshi's public API for market data.
    """
    try:
        # Kalshi public API endpoint for BTC markets
        url = "https://api.elections.kalshi.com/trade-api/v2/markets"
        params = {
            "status": "open",
            "series_ticker": "KXBTC",
            "limit": 10
        }
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            markets = data.get("markets", [])
            if markets:
                # Find the nearest 15-min BTC contract
                btc_markets = []
                for m in markets:
                    ticker = m.get("ticker", "")
                    if "BTC" in ticker.upper() or "KXBTC" in ticker.upper():
                        yes_price = m.get("yes_price", 0)  # cents, so 65 = 65% chance
                        no_price = m.get("no_price", 0)
                        volume = m.get("volume", 0)
                        btc_markets.append({
                            "ticker": ticker,
                            "title": m.get("title", ""),
                            "yes_price": yes_price,
                            "no_price": no_price,
                            "yes_pct": yes_price,  # already in percentage
                            "no_pct": no_price,
                            "volume": volume,
                            "close_time": m.get("close_time", "")
                        })

                if btc_markets:
                    # Sort by close time to get nearest expiry
                    btc_markets.sort(key=lambda x: x["close_time"])
                    nearest = btc_markets[0]
                    return {
                        "kalshi_ticker": nearest["ticker"],
                        "kalshi_title": nearest["title"],
                        "kalshi_yes_pct": nearest["yes_pct"],
                        "kalshi_no_pct": nearest["no_pct"],
                        "kalshi_volume": nearest["volume"],
                        "kalshi_market_expects": "UP" if nearest["yes_pct"] > 50 else "DOWN" if nearest["yes_pct"] < 50 else "NEUTRAL",
                        "kalshi_market_confidence": max(nearest["yes_pct"], nearest["no_pct"])
                    }
    except Exception as e:
        pass

    return {}


def analyze_edge(bot_signal, bot_confidence, kalshi_data):
    """
    Compare bot's prediction against Kalshi market odds.
    Identifies contrarian opportunities and confirms consensus plays.
    """
    if not kalshi_data or not kalshi_data.get("kalshi_market_expects"):
        return {
            "edge_type": "NO_KALSHI_DATA",
            "edge_description": "No Kalshi data available for comparison"
        }

    market_expects = kalshi_data["kalshi_market_expects"]
    market_conf = kalshi_data.get("kalshi_market_confidence", 50)

    result = {}

    if bot_signal == market_expects:
        # Bot agrees with market
        result["edge_type"] = "CONSENSUS"
        result["edge_description"] = f"Bot and market both expect {bot_signal}"
        result["edge_strength"] = "WEAK"  # less edge when agreeing with market
        # Higher confidence if both strongly agree
        if bot_confidence > 0.7 and market_conf > 60:
            result["edge_strength"] = "MODERATE"
    else:
        # Bot disagrees with market - potential contrarian edge
        result["edge_type"] = "CONTRARIAN"
        result["edge_description"] = f"Bot says {bot_signal} but market expects {market_expects}"
        if bot_confidence > 0.7:
            result["edge_strength"] = "STRONG"  # high confidence contrarian = best edge
        else:
            result["edge_strength"] = "MODERATE"

    # Calculate implied edge
    if market_expects == "UP":
        market_up_prob = market_conf / 100
    else:
        market_up_prob = (100 - market_conf) / 100

    if bot_signal == "UP":
        bot_up_prob = bot_confidence
    else:
        bot_up_prob = 1 - bot_confidence

    result["market_up_probability"] = round(market_up_prob * 100, 1)
    result["bot_up_probability"] = round(bot_up_prob * 100, 1)
    result["probability_gap"] = round(abs(bot_up_prob - market_up_prob) * 100, 1)

    return result


def get_kalshi_summary():
    """Get Kalshi data and format for Claude."""
    data = get_kalshi_btc_contracts()
    if not data:
        return "Kalshi data not available.", {}

    lines = []
    lines.append(f"Market: {data.get('kalshi_title', 'N/A')}")
    lines.append(f"Market expects: {data.get('kalshi_market_expects', 'N/A')} ({data.get('kalshi_market_confidence', 0)}% confidence)")
    lines.append(f"Yes: {data.get('kalshi_yes_pct', 0)}% | No: {data.get('kalshi_no_pct', 0)}%")
    lines.append(f"Volume: {data.get('kalshi_volume', 0)}")

    return "\n".join(lines), data


if __name__ == "__main__":
    summary, data = get_kalshi_summary()
    print(summary)
