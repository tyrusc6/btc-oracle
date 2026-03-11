"""
BTC Oracle - Railway Runner
Runs both the collector and scheduler in one process for Railway deployment.
"""

import threading
import time
from datetime import datetime, timezone
from collector import run_collector
from bot import run_signal_cycle


def scheduler_loop():
    """Run signal cycles every 15 minutes."""
    print("[SCHEDULER] Starting... waiting 60s for collector to gather data...")
    time.sleep(60)  # Give collector time to get initial data

    while True:
        try:
            print(f"\n[SCHEDULER] Running signal cycle...")
            run_signal_cycle()
        except Exception as e:
            print(f"[SCHEDULER ERROR] {e}")
        print(f"[SCHEDULER] Sleeping 15 minutes...")
        time.sleep(900)


def main():
    print("=" * 60)
    print("BTC ORACLE - RAILWAY DEPLOYMENT")
    print("=" * 60)
    print(f"Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("Running collector + scheduler in one process")
    print("=" * 60)

    # Start scheduler in background thread
    sched_thread = threading.Thread(target=scheduler_loop, daemon=True)
    sched_thread.start()

    # Run collector in main thread (keeps process alive)
    run_collector()


if __name__ == "__main__":
    main()
