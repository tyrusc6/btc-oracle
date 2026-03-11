"""
BTC Oracle - Scheduler
Runs the signal bot every 15 minutes.
"""

import time
from datetime import datetime, timezone
from bot import run_signal_cycle


def main():
    print("=" * 60)
    print("BTC ORACLE - SCHEDULER")
    print("=" * 60)
    print(f"Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("Running every 15 minutes. Press Ctrl+C to stop.")
    print("=" * 60)

    print("\nRunning first cycle now...")
    try:
        run_signal_cycle()
    except Exception as e:
        print(f"[ERROR] {e}")

    while True:
        print(f"\nSleeping 15 minutes...")
        time.sleep(900)
        try:
            run_signal_cycle()
        except Exception as e:
            print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()
