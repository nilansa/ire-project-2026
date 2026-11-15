from __future__ import annotations

import os
import signal
import subprocess
import sys
import time


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: run_phase.py seconds command [args...]")
    deadline = time.monotonic() + float(sys.argv[1])
    proc = subprocess.Popen(sys.argv[2:], start_new_session=True)
    while proc.poll() is None:
        left = deadline - time.monotonic()
        if left <= 0:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise SystemExit(124)
        time.sleep(min(2, left))
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
