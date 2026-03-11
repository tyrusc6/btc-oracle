"""
BTC Oracle - Data Collector
Pulls live BTC tick data from Kraken and stores in Supabase.
"""

import json
import time
import threading
from datetime import datetime, timezone
from dotenv import load_dotenv
import requests
import websocket
import db

load_dotenv()

tick_buffer = []
buffer_lock = threading.Lock()


def get_kraken_ticker():
    try:
        url = "https://api.kraken.com/0/public/Ticker?pair=XBTUSD"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data.get("error") and len(data["error"]) > 0:
            print(f"Kraken API error: {data['error']}")
            return None
        result = data["result"]["XXBTZUSD"]
        return {
            "price": float(result["c"][0]),
            "volume": float(result["v"][1]),
            "bid": float(result["b"][0]),
            "ask": float(result["a"][0]),
            "spread": float(result["a"][0]) - float(result["b"][0])
        }
    except Exception as e:
        print(f"Error fetching ticker: {e}")
        return None


def store_ticks(ticks):
    try:
        if not ticks:
            return
        for tick in ticks:
            db.insert("tick_data", tick)
        print(f"  Stored {len(ticks)} ticks | Latest: ${ticks[-1]['price']:,.2f}")
    except Exception as e:
        print(f"Error storing ticks: {e}")


def on_ws_message(ws, message):
    global tick_buffer
    try:
        data = json.loads(message)
        if isinstance(data, list) and len(data) > 1:
            channel = data[-2]
            if channel == "trade":
                trades = data[1]
                for trade in trades:
                    tick = {
                        "price": float(trade[0]),
                        "volume": float(trade[1]),
                    }
                    with buffer_lock:
                        tick_buffer.append(tick)
    except:
        pass


def on_ws_error(ws, error):
    print(f"WebSocket error: {error}")


def on_ws_close(ws, close_status_code, close_msg):
    print("WebSocket closed. Reconnecting in 5s...")
    time.sleep(5)
    start_websocket()


def on_ws_open(ws):
    print("WebSocket connected to Kraken!")
    subscribe = {
        "event": "subscribe",
        "pair": ["XBT/USD"],
        "subscription": {"name": "trade"}
    }
    ws.send(json.dumps(subscribe))
    print("Subscribed to BTC/USD trades")


def start_websocket():
    ws = websocket.WebSocketApp(
        "wss://ws.kraken.com",
        on_open=on_ws_open,
        on_message=on_ws_message,
        on_error=on_ws_error,
        on_close=on_ws_close
    )
    ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
    ws_thread.start()
    return ws


def flush_buffer():
    global tick_buffer
    ticks_to_store = []
    with buffer_lock:
        if tick_buffer:
            ticks_to_store = tick_buffer.copy()
            tick_buffer = []
    if ticks_to_store:
        store_ticks(ticks_to_store)


def run_rest_collector():
    print("Running REST API collector (1s intervals)...")
    batch = []
    while True:
        try:
            ticker = get_kraken_ticker()
            if ticker:
                batch.append(ticker)
                print(f"  BTC: ${ticker['price']:,.2f} | Vol: {ticker['volume']:,.0f} | Spread: ${ticker['spread']:.2f}")
                if len(batch) >= 50:
                    store_ticks(batch)
                    batch = []
            time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping collector...")
            if batch:
                store_ticks(batch)
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)


def run_collector():
    print("=" * 50)
    print("BTC ORACLE - DATA COLLECTOR")
    print("=" * 50)
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print()

    try:
        print("Attempting WebSocket connection...")
        ws = start_websocket()
        time.sleep(3)
        print("Flushing tick buffer every 2 seconds...")
        while True:
            time.sleep(2)
            flush_buffer()
    except Exception as e:
        print(f"WebSocket failed: {e}")
        print("Falling back to REST API...")
        run_rest_collector()


if __name__ == "__main__":
    run_collector()
