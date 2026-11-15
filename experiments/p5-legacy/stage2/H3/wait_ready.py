#!/usr/bin/env python3
"""Poll the stage2 data-ready marker without touching the data tree."""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/Users/nilan/Documents/ire-p5-minimal-evidence/stage2/data"))
    parser.add_argument("--interval", type=int, default=120)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    marker = args.data_root / "READY.json"
    start = time.monotonic()
    while True:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if marker.is_file() and marker.stat().st_size > 0:
            print(f"{now} READY {marker} bytes={marker.stat().st_size}", flush=True)
            return 0
        elapsed = int(time.monotonic() - start)
        print(f"{now} WAIT elapsed_seconds={elapsed} marker={marker}", flush=True)
        if elapsed >= args.timeout:
            print(f"{now} TIMEOUT max_seconds={args.timeout}", flush=True)
            return 2
        time.sleep(min(args.interval, max(1, args.timeout - elapsed)))


if __name__ == "__main__":
    raise SystemExit(main())
