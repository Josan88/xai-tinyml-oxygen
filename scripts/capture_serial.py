"""Capture a live image's USB serial output to a text file (docs/edge-kan-label0-banner-amendment.md).

  python scripts/capture_serial.py <port> <seconds> <out.txt> [--method cap|dtr-toggle]

--method cap         as build/v2/cap.py: open the port (pyserial raises DTR, then purges the receive buffer)
                     and read. A line the board prints the moment DTR rises can fall into that purge.
--method dtr-toggle  open as above, wait 0.5 s, drop DTR for 1.0 s, raise it again, then read. Images that
                     re-announce on each DTR rise (the ref0 and P5 live images) print their banner after the
                     purge. At 115200 baud a DTR drop does not reset the board (only the 1200-baud touch does).

Output format is cap.py's: a header line, then "HH:MM:SS.mmm > <line>" per received line. Never writes to the
board except the DTR line state.
"""
import argparse
import datetime
from pathlib import Path
import time

import serial


def capture(port, seconds, path, method="cap", settle=0.5, low=1.0):
    t0 = time.time()
    while True:
        try:
            s = serial.Serial(port, 115200, timeout=0.2)
            break
        except serial.SerialException:
            if time.time() - t0 > 15:
                raise
            time.sleep(0.2)
    try:
        if method == "dtr-toggle":
            time.sleep(settle)
            s.dtr = False
            time.sleep(low)
            s.reset_input_buffer()
            s.dtr = True
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="") as f:
            f.write(f"--- capture {port} 115200 start {datetime.datetime.now().isoformat(timespec='seconds')} "
                    f"duration {seconds:.0f}s method {method} (scripts/capture_serial.py)\n")
            buf, end = b"", time.time() + seconds
            while time.time() < end:
                buf += s.read(4096)
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    f.write(f"{ts} > {line.decode('utf-8', 'replace').rstrip()}\n")
                    f.flush()
    finally:
        s.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("port")
    p.add_argument("seconds", type=float)
    p.add_argument("out", type=Path)
    p.add_argument("--method", choices=("cap", "dtr-toggle"), default="cap")
    a = p.parse_args()
    capture(a.port, a.seconds, a.out, a.method)


if __name__ == "__main__":
    main()
