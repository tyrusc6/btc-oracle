"""
BTC Oracle - Market Data Fetcher
Pulls external data: Fear & Greed, funding rates, open interest,
liquidations, ETH correlation, BTC dominance, order book depth,
global macro indicators, and more.
"""

import requests
import time


def safe_get(url, timeout=10):
    """Safe HTTP GET with error handling."""
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return None


def get_fear_greed():
    """Crypto Fear & Greed Index (0=extreme fear, 100=extreme greed)."""
    data = safe_get("https://api.alternative.me/fng/?limit=2")
    if data and data.get("data"):
        current = data["data"][0]
        previous = data["data"][1] if len(data["data"]) > 1 else None
        return {
            "fear_greed_value": int(current["value"]),
            "fear_greed_label": current["value_classification"],
            "fear_greed_previous": int(previous["value"]) if previous else None,
            "fear_greed_change": int(current["value"]) - int(previous["value"]) if previous else None
        }
    return {}


def get_btc_dominance():
    """BTC market dominance percentage."""
    data = safe_get("https://api.coingecko.com/api/v3/global")
    if data and data.get("data"):
        market = data["data"]
        return {
            "btc_dominance": round(market.get("market_cap_percentage", {}).get("btc", 0), 2),
            "eth_dominance": round(market.get("market_cap_percentage", {}).get("eth", 0), 2),
            "total_market_cap_usd": market.get("total_market_cap", {}).get("usd", 0),
            "total_volume_24h": market.get("total_volume", {}).get("usd", 0),
            "market_cap_change_24h": round(market.get("market_cap_change_percentage_24h_usd", 0), 2)
        }
    return {}


def get_eth_price():
    """ETH price for correlation analysis."""
    data = safe_get("https://api.kraken.com/0/public/Ticker?pair=ETHUSD")
    if data and data.get("result"):
        result = data["result"].get("XETHZUSD", {})
        if result:
            return {
                "eth_price": float(result["c"][0]),
                "eth_volume_24h": float(result["v"][1]),
                "eth_high_24h": float(result["h"][1]),
                "eth_low_24h": float(result["l"][1])
            }
    return {}


def get_btc_ohlc_recent():
    """Recent BTC OHLC candles from Kraken for pattern analysis."""
    data = safe_get("https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=5")
    if data and data.get("result"):
        candles = data["result"].get("XXBTZUSD", [])
        if candles and len(candles) > 5:
            recent = candles[-6:]  # last 30 min of 5-min candles
            bodies = []
            wicks_upper = []
            wicks_lower = []
            for c in recent:
                o, h, l, close = float(c[1]), float(c[2]), float(c[3]), float(c[4])
                body = close - o
                bodies.append(body)
                wicks_upper.append(h - max(o, close))
                wicks_lower.append(min(o, close) - l)

            consecutive_green = 0
            consecutive_red = 0
            for b in reversed(bodies):
                if b > 0:
                    consecutive_green += 1
                else:
                    break
            for b in reversed(bodies):
                if b < 0:
                    consecutive_red += 1
                else:
                    break

            avg_body = sum(abs(b) for b in bodies) / len(bodies)
            latest_body = bodies[-1]
            is_doji = abs(latest_body) < avg_body * 0.1

            return {
                "candle_consecutive_green": consecutive_green,
                "candle_consecutive_red": consecutive_red,
                "candle_avg_body_size": round(avg_body, 2),
                "candle_latest_body": round(latest_body, 2),
                "candle_is_doji": is_doji,
                "candle_upper_wick_avg": round(sum(wicks_upper) / len(wicks_upper), 2),
                "candle_lower_wick_avg": round(sum(wicks_lower) / len(wicks_lower), 2),
                "candle_trend_30m": "BULLISH" if sum(bodies) > 0 else "BEARISH"
            }
    return {}


def get_kraken_orderbook():
    """Order book depth analysis - detect buy/sell walls."""
    data = safe_get("https://api.kraken.com/0/public/Depth?pair=XBTUSD&count=20")
    if data and data.get("result"):
        book = data["result"].get("XXBTZUSD", {})
        bids = book.get("bids", [])
        asks = book.get("asks", [])
        if bids and asks:
            bid_volume = sum(float(b[1]) for b in bids)
            ask_volume = sum(float(a[1]) for a in asks)
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0

            # Detect walls (single order > 20% of side)
            max_bid = max(float(b[1]) for b in bids)
            max_ask = max(float(a[1]) for a in asks)
            bid_wall = max_bid > bid_volume * 0.2
            ask_wall = max_ask > ask_volume * 0.2

            spread = float(asks[0][0]) - float(bids[0][0])
            mid_price = (float(asks[0][0]) + float(bids[0][0])) / 2

            return {
                "orderbook_bid_volume": round(bid_volume, 4),
                "orderbook_ask_volume": round(ask_volume, 4),
                "orderbook_imbalance": round(imbalance, 4),
                "orderbook_imbalance_signal": "BUY_PRESSURE" if imbalance > 0.1 else "SELL_PRESSURE" if imbalance < -0.1 else "BALANCED",
                "orderbook_bid_wall_detected": bid_wall,
                "orderbook_ask_wall_detected": ask_wall,
                "orderbook_spread": round(spread, 2),
                "orderbook_spread_pct": round((spread / mid_price) * 100, 6)
            }
    return {}


def get_kraken_recent_trades():
    """Analyze recent trade flow - are big trades buying or selling?"""
    data = safe_get("https://api.kraken.com/0/public/Trades?pair=XBTUSD&count=200")
    if data and data.get("result"):
        trades = data["result"].get("XXBTZUSD", [])
        if trades:
            buy_volume = 0
            sell_volume = 0
            buy_count = 0
            sell_count = 0
            large_buys = 0
            large_sells = 0

            for t in trades[-200:]:
                price, vol, ts, side = float(t[0]), float(t[1]), float(t[2]), t[3]
                if side == "b":
                    buy_volume += vol
                    buy_count += 1
                    if vol > 0.1:
                        large_buys += 1
                else:
                    sell_volume += vol
                    sell_count += 1
                    if vol > 0.1:
                        large_sells += 1

            total_vol = buy_volume + sell_volume
            buy_pct = (buy_volume / total_vol * 100) if total_vol > 0 else 50

            return {
                "trade_flow_buy_pct": round(buy_pct, 1),
                "trade_flow_sell_pct": round(100 - buy_pct, 1),
                "trade_flow_signal": "BUYING" if buy_pct > 55 else "SELLING" if buy_pct < 45 else "NEUTRAL",
                "trade_flow_buy_volume": round(buy_volume, 4),
                "trade_flow_sell_volume": round(sell_volume, 4),
                "trade_flow_large_buys": large_buys,
                "trade_flow_large_sells": large_sells,
                "trade_flow_whale_signal": "WHALE_BUYING" if large_buys > large_sells * 1.5 else "WHALE_SELLING" if large_sells > large_buys * 1.5 else "NEUTRAL"
            }
    return {}


def get_multi_timeframe_momentum():
    """Check price change across multiple timeframes using Kraken OHLC."""
    result = {}
    for interval, label in [(1, "1m"), (5, "5m"), (15, "15m"), (60, "1h")]:
        data = safe_get(f"https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval={interval}")
        if data and data.get("result"):
            candles = data["result"].get("XXBTZUSD", [])
            if candles and len(candles) >= 2:
                current_close = float(candles[-1][4])
                prev_close = float(candles[-2][4])
                change_pct = ((current_close - prev_close) / prev_close) * 100
                result[f"momentum_{label}_pct"] = round(change_pct, 4)
                result[f"momentum_{label}_direction"] = "UP" if change_pct > 0 else "DOWN"
        time.sleep(0.3)  # rate limit

    # Overall momentum alignment
    directions = [v for k, v in result.items() if k.endswith("_direction")]
    if directions:
        up_count = directions.count("UP")
        result["momentum_alignment"] = "STRONG_BULL" if up_count == len(directions) else \
                                        "STRONG_BEAR" if up_count == 0 else \
                                        "MIXED_BULL" if up_count > len(directions) / 2 else "MIXED_BEAR"

    return result


def get_volatility_analysis():
    """Analyze current volatility regime."""
    data = safe_get("https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=15")
    if data and data.get("result"):
        candles = data["result"].get("XXBTZUSD", [])
        if candles and len(candles) > 10:
            recent = candles[-10:]
            ranges = [float(c[2]) - float(c[3]) for c in recent]
            avg_range = sum(ranges) / len(ranges)
            latest_range = ranges[-1]
            closes = [float(c[4]) for c in recent]
            avg_price = sum(closes) / len(closes)
            volatility_pct = (avg_range / avg_price) * 100

            return {
                "volatility_15m_avg_range": round(avg_range, 2),
                "volatility_15m_latest_range": round(latest_range, 2),
                "volatility_pct": round(volatility_pct, 4),
                "volatility_expanding": latest_range > avg_range * 1.2,
                "volatility_contracting": latest_range < avg_range * 0.8,
                "volatility_regime": "HIGH" if volatility_pct > 0.5 else "LOW" if volatility_pct < 0.15 else "NORMAL"
            }
    return {}


def get_all_market_data():
    """Fetch all external market data."""
    print("  Fetching external market data...")
    all_data = {}

    sources = [
        ("Fear & Greed", get_fear_greed),
        ("BTC Dominance", get_btc_dominance),
        ("ETH Price", get_eth_price),
        ("Candlestick Patterns", get_btc_ohlc_recent),
        ("Order Book", get_kraken_orderbook),
        ("Trade Flow", get_kraken_recent_trades),
        ("Multi-TF Momentum", get_multi_timeframe_momentum),
        ("Volatility", get_volatility_analysis),
    ]

    for name, func in sources:
        try:
            data = func()
            if data:
                all_data.update(data)
                print(f"    {name}: OK")
            else:
                print(f"    {name}: No data")
        except Exception as e:
            print(f"    {name}: Error - {e}")
        time.sleep(0.2)

    return all_data


if __name__ == "__main__":
    data = get_all_market_data()
    print(f"\nCollected {len(data)} data points:")
    for k, v in data.items():
        print(f"  {k}: {v}")
