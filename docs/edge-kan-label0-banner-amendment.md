# Edge KAN label-0 live image: why the banner was lost, and an amended capture (pre-registration)

Written 15 Nov 2026 before any of the board runs below. Jason asked on 15 Nov 2026: "Fix the banner check".

This file amends the live idle check of `docs/edge-kan-label0-board-preregistration.md` (`9685e4d`). That file is not edited. Its result stays as reported in `docs/edge-kan-label0-board-results.md`: **fail as written**, on the banner clause only.

## The suspected cause (a hypothesis until the diagnostic below)

**The serial-library side.** pyserial 3.5's Windows `Serial.open()` (`serialwin32.py`) does two things in order:

1. It calls `_reconfigure_port()`, which applies a DCB with DTR enabled. The host then raises DTR.
2. It calls `PurgeComm(... PURGE_RXCLEAR | PURGE_RXABORT)`, which discards any received bytes.

**The firmware side.** The ref0 live image (`src/main.cpp`, `EKAN_XAI_REF_LABEL0`) prints its banner on the first loop pass that sees `Serial.dtr()` high. mbed sends nothing over USB until DTR is high.

So the banner is likely sent between steps 1 and 2 and thrown away. The status lines come about every 0.5 s and survive. The same mechanism would explain ADR-0008's P5 banner (1 of 4 opens) and the 15 Nov result (0 of 4).

## The amended capture method (no firmware change)

`scripts/capture_serial.py --method dtr-toggle`:

1. Open the port as `build/v2/cap.py` does.
2. Wait 0.5 s after `open()` returns, then drop DTR for 1.0 s and raise it again. 1.0 s is longer than the 640 ms worst case before the loop runs again.
3. Capture from the moment DTR is raised again.

The firmware's `ekanRefAnnounced = ekanSerialConnected` then sees DTR fall, re-arms, and prints the banner on the rise, after the purge. At 115200 baud a DTR drop does not reset the board; only the 1200-baud touch does.

`--method cap` reproduces `build/v2/cap.py`: open, then read. The output format is `cap.py`'s, with the method added to the header line.

## Diagnostic (on the archived `nano33ble_ekan_f1a_ref0`, `df389020…`, flashed with bossac and a receipt; no finger)

Twenty opens of 8 s each, alternating `cap` and `dtr-toggle` (10 of each), with 1 s between opens. For each open, record whether the banner (`XAI_REF=LABEL0: edge KAN`) appears and how many `Hz:` lines appear.

- **Hypothesis supported** if `dtr-toggle` shows the banner on all 10 opens.
- **Otherwise** the hypothesis is incomplete. That is reported, the amended idle check below is not run, and the board is restored.

The `cap` count is reported for comparison, with no prediction.

## Amended live idle check (only if the diagnostic supports the hypothesis)

One 60 s capture with `--method dtr-toggle`. Pass = both clauses of the original check:

- every `Hz:` line is ≥ 49.5;
- the banner is present.

It is reported as the amended check, next to the original "fail as written". It does not replace that result.

## Afterwards

Reflash the image found on the board (the public repo's additive r3 image, `1f63b52f…`, as after the 15 Nov session), with a receipt written by hand. Do not change the firmware, the paper, or any archived image. The paper does not mention the live idle check.
