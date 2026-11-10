# v2 flash runbook (ADR-0006)

Steps for the v2 r2 board session. **Status, 22 Oct 2026.** The v2 images were rebuilt as **r2** on the fixed numeric source: the two-pass `window_mean_two_pass` for the raw IR/red means, plus the v2-only Kahan m2/m3/m4 sums in `extract_features_from_filtered`. All three r2 images are built and archived, not flashed. The board evidence from 15 Oct (board parity, timing) and from 16–22 Oct (live sits) was taken on the old images. It does not carry over to r2. Flash the archived files below. Do not rebuild on the day, and do not use `pio run -t upload`, which rebuilds and flashes an image nobody hashed.

| Job | Archive (`$ARCH`) | Env (`$JOB`) | `firmware.bin` SHA-256 | Identity |
|---|---|---|---|---|
| (a) Board parity | `outputs/firmware-v2-prepared-r2` | `nano33ble_v2_context_parity` | `8bcf8e8fcf3326cd414afbbf06968ddc846659cf9ecef019d1f500ca22894fd6` | `CONTEXT_PARITY_V2 WINDOW=500 CONTEXT=250 STRIDE=250 N=2 THRESHOLD=0.50 INPUTS=16 PAIRS=IR,RED` |
| (b) Timing | `outputs/timing-v2-prepared-r2` | `nano33ble_v2_context_timing` | `9d08702e33075dd49723b31394b9fd7338dec2eb5a29b6d0af916e634cbbf93d` | `TIMING_REPLAY_V2 WINDOW=500 CONTEXT=250 STRIDE=250 N=2 THRESHOLD=0.50 CLOCK_HZ=64000000 GPIO=D2 ATTR16=0 INPUTS=16 PAIRS=IR,RED` |
| (c), (d) Live | `outputs/firmware-v2-f1a-r2` | `nano33ble_v2_f1a` | `cdefac1dc9e04fe3f9e4df9e7466008166e9b6bb59f341e8a0ee85c1c89a50d2` | Serial banner `Model: TinyKAN v2 (16 inputs …)`; `Feat:` line with 16 names; BLE name `OxygenRiskDemo` |
| Rollback | `outputs/firmware-v1-rollback` | `nano33ble` | `1f2f95dbd287a862e4086327f55ef3db213a31de6b70891c9deb2dcc140c07e0` | v1 live: `Feat:` line with 14 names, no v2 banner |

Each r2 archive has a `README.md` covering what changed against the old image, its size diff and its symbol diff. The old archives (`outputs/firmware-v2-prepared/`, `outputs/firmware-v2-f1a/`, `outputs/timing-v2-prepared/`) and the old prepared set (`outputs/parity/v2-board-prepared/`) are left unchanged and are not used below.

**Host side prepared for r2 (22 Oct 2026).**
- `outputs/parity/v2-board-prepared-r2/`, `prepared_not_run`. Same fixtures and same 611 locked cases as the 15 Oct set (identical `cases.npz`/`fixtures.npz` arrays and keys, same seed and criteria). Host expectations come from the r2 probe `9f8311ed…` (header `8c2816cf…`, zig 0.16.0 via uv). Host check: 21 fixture and 1,222 locked decisions, 0 failures, no locked case failing host-vs-Python.
- `outputs/timing-v2-prepared-r2-host/report.json`, `prepared_not_run`, `host_pass` true. A no-port run of the timing driver on the r2 image and the r2 prepared set.

**Drivers.** Without the new flags, the scripts behave as before (15 Oct images, `outputs/parity/v2-board-prepared/`).
- `scripts/record_v2_upload_receipt.py --archive <r2 dir>` checks the image against that archive's `SHA256SUMS`.
- `scripts/replay_context_board_v2.py run --prepared <dir>` takes the image archive from the prepared manifest.
- `scripts/benchmark_context_board_v2.py --archive <r2 dir> --prepared <dir>` refuses to mix an r2 image with 15 Oct host rows.

Run the jobs in the order (a), (b), (c), (d). Jobs (a) and (b) check the arithmetic and the time budget on the board before any person sits at it. Commands are bash (Git Bash), run from the repo root.

## 0. Pre-flight

```bash
cd /d/fyrp-xai-tinyml
PY="python"   # numpy, scipy, pandas, pyserial
BOSSAC=~/.platformio/packages/tool-bossac-nordicnrf52/bossac.exe
DAY=$(date +%Y%m%d)
"$PY" -c "import numpy, scipy, pandas, serial"
"$PY" -m serial.tools.list_ports -v          # application port COM8, USB VID:PID=2341:805A; COM3/COM4 are Bluetooth
for ARCH in outputs/firmware-v2-prepared-r2 outputs/timing-v2-prepared-r2 outputs/firmware-v2-f1a-r2 outputs/firmware-v1-rollback; do (cd "$ARCH" && tr -d '\r' < SHA256SUMS | sha256sum -c -) || echo "STOP: $ARCH"; done   # tr: some SHA256SUMS have CRLF
```

1. Close every serial monitor, PlatformIO monitor, Chrome dashboard tab and Arduino IDE that may hold the port.
2. Every `sha256sum -c` line must say `OK`. If one does not, stop.
3. Keep the rollback image at hand. It is the v1 source build, and a clean rebuild on 22 Oct 2026 was byte-identical.

## Flashing an archived image (same for every job)

This is the bossac procedure that wrote every earlier image (`build/v2/*-bossac-upload.log`). Set `JOB` and `ARCH` from the table, then:

```bash
"$PY" -u build/v2/touch1200.py COM8          # 1200-baud touch: board enters the bootloader (or double-tap reset)
"$PY" -m serial.tools.list_ports -v          # the bootloader appears as VID:PID=2341:005A, recently COM7
BOOT=COM7                                    # set to the port just listed
LOG=build/v2/$JOB-r2-$DAY-bossac-upload.log
"$BOSSAC" --port $BOOT --write --erase --verify -U --reset "$ARCH/firmware.bin" 2>&1 | tee "$LOG"
"$PY" -u scripts/record_v2_upload_receipt.py --env $JOB --archive $ARCH --bossac-log "$LOG" --bootloader-port $BOOT \
    --output outputs/firmware-v2-uploads/$JOB-r2-$DAY-upload-receipt.json
```

- The log must end with `Verify successful`, and `Write N bytes to flash` must equal the BIN size. If it does not, stop. Do not retry with other tools.
- The receipt step reads the log and the archive only, and opens no port. It refuses an image that does not match the archive's `SHA256SUMS`, and it refuses to overwrite an existing receipt.
- `build/v2/touch1200.py` and `build/v2/cap.py` are the local, untracked helpers used for the 22 Oct captures.
- The application port comes back as COM8 after reset.

## (a) Board parity (`CONTEXT_PARITY_V2`)

```bash
JOB=nano33ble_v2_context_parity; ARCH=outputs/firmware-v2-prepared-r2
# flash as above, then:
"$PY" -u scripts/replay_context_board_v2.py run --port COM8 \
    --upload-receipt outputs/firmware-v2-uploads/$JOB-r2-$DAY-upload-receipt.json \
    --prepared outputs/parity/v2-board-prepared-r2 \
    --output outputs/parity/v2-board-$DAY-r2
```

- **Handshake.** The driver sends `H` and stops unless it gets the `CONTEXT_PARITY_V2` identity.
- **Verification before the port opens.** It checks the prepared hashes and that the archive's `SHA256SUMS` is unchanged since preparation. It also checks that the upload receipt names image `8bcf8e8f…`.
- **Protocol checks.** It checks mid-warmup reset and malformed-block rejection.
- **Replay.** 3 synthetic paired streams (21 decisions), then all 611 locked recorded contexts (1,222 decisions). After every decision it requests all 16 deltas with `A`.
- **Duration.** About 5 s per context, so roughly an hour for the full set. `--locked 50` gives a first look; the reported result is the full set.
- **Gate.** Every decision must pass `parity_criteria_v2.check_window` against both the Python reference and the C++ host. Metadata, Candidate/Confirmed/streak, top-3 payload and `two_wavelength_valid` must match exactly. The budgets are frozen.
- **Expected failures.** None on the host side. The r2 host check has no locked case failing host-vs-Python; cases 379 and 380 failed only under the old probe. Report every board failure; do not exclude any.
- **Context.** The complete host run on the r2 source has **7** frozen-criteria failures out of 71,462 decisions (`outputs/parity/v2-host/20261021T163919Z/`, `parity_complete` false). None of the 611 locked cases is among them. A board pass does not close the host residual.

## (b) Timing (`TIMING_REPLAY_V2`)

```bash
JOB=nano33ble_v2_context_timing; ARCH=outputs/timing-v2-prepared-r2
# flash as above, then:
"$PY" -u scripts/benchmark_context_board_v2.py --port COM8 \
    --upload-receipt outputs/firmware-v2-uploads/$JOB-r2-$DAY-upload-receipt.json \
    --archive outputs/timing-v2-prepared-r2 --prepared outputs/parity/v2-board-prepared-r2 \
    --repetitions 10 --output outputs/timing-v2-board-$DAY-r2
```

- 10 × 21 timed decisions, plus warmup blocks, 8 empty-bracket calibrations and memory snapshots, as in v1 (`docs/on-device-timing.md`). Each timed decision is checked against the r2 C++ host rows.
- Read `latency_summary.subsequent_decisions.max_us`, `max_block_ms` and `fifo_headroom_ratio` from the report.
- The r2 source adds work inside the timed block: a second 500-sample pass per channel for the raw means, and the compensated moment sums. The 15 Oct result (`outputs/timing-v2-board-20261015/`) is for the old image and is not this image's number.
- The sensor FIFO holds 640 ms (32 records at 50 Hz). The live image does not detect overflow (register 0x05 is not used), so a decision block near 640 ms would drop samples silently.
- **Stop rule.** If any timed block reaches 640 ms, do not run (c). Report the number. A slower image is not a live candidate.
- The D2/GPIO oscilloscope cross-check is optional on the day. It is still outstanding from v1.

## (c) Live `nano33ble_v2_f1a` r2: 60 s idle

```bash
JOB=nano33ble_v2_f1a; ARCH=outputs/firmware-v2-f1a-r2
# flash as above (the receipt must exist in outputs/firmware-v2-uploads/), then, with no finger on the sensor:
CAP=outputs/live-v2/$(date +%Y-%m-%d)_f1a-r2_idle.txt
"$PY" -u build/v2/cap.py COM8 60 "$CAP"
"$PY" -u -c "import re,sys; v=[float(x) for x in re.findall(r'Hz: ([0-9.]+)', open(sys.argv[1]).read())]; print(len(v), 'Hz lines', (min(v), max(v)) if v else None, 'no rate collapse' if v and min(v) >= 49.5 else 'RATE COLLAPSE OR NO Hz LINES')" "$CAP"
```

- **Sensor.** The second MAX30102 module (installed 16 Oct 2026, commit `fc6041f`).
- **Idle.** Every status line `NO FINGER`, IR about 150.
- **Identity.** `cap.py` opens the port after the flash reset, so the boot banner may be missed. The upload receipt binds the image; the 16-name `Feat:` line appears only with a finger, in (d).

**Rate criterion (decided 22 Oct 2026, before any r2 flash).** Report the shipped image's own `Hz:` range, minimum to maximum over every `Hz:` line of the capture. The run counts as **no rate collapse** if every `Hz:` line is ≥ 49.5. There is no narrower pass band.
- Context, not the r2 result: the pre-fix f1a image (`96b7f60b…`) read `Hz:` 50.04–50.10 on 22 Oct 2026 (`outputs/live-v2/2026-10-22_f1a_sit.sidecar.md`).
- 49.93 Hz is the `looptiming3` diagnostic's figure. It is not a criterion for this image.

## (d) Live sit, 180 s: needs Jason

**Not run by the orchestrator.** Jason must be seated with his finger on the sensor, and he confirms that before capture starts.

```bash
CAP=outputs/live-v2/$(date +%Y-%m-%d)_f1a-r2_sit.txt
"$PY" -u build/v2/cap.py COM8 180 "$CAP"
"$PY" -u -c "import re,sys; t=open(sys.argv[1]).read(); v=[float(x) for x in re.findall(r'Hz: ([0-9.]+)', t)]; print(len(v), 'Hz lines', (min(v), max(v)) if v else None, 'no rate collapse' if v and min(v) >= 49.5 else 'RATE COLLAPSE OR NO Hz LINES', '| NO FINGER lines:', t.count('NO FINGER'))" "$CAP"
```

- Same rate criterion as (c).
- The first decision arrives at 15 s, then one every 5 s. Each `Feat:` line lists 16 names ending `ratio_of_ratios=… ir_perfusion_index=… two_wavelength_valid=1`. `two_wavelength_valid=0` with a finger on the sensor is a fault: record it and stop.
- Write a sidecar note per capture: image hash, receipt, module, ambient light and enclosure state, as for v1.

**Sensor 1's score.** On sensor 1, at rest, v2 scored P 0.89 and Confirmed. `ratio_of_ratios` (R ≈ 2.2) drives that score (`outputs/score-investigation-v2-20261016/VERDICT.md`). Why sensor 1 reads that R is still unexplained: the LED1 identity check did not settle it. The model and threshold were not retuned for it. A normal score on the second module does not explain sensor 1.

What a sit can and cannot show: on-device behaviour, rate, stability and alert behaviour. It cannot show detection of low oxygen. A healthy seated subject gives a normal-saturation recording only. Any Candidate or Confirmed flag in it is a false positive or domain shift, not a Low-Oxygen Risk finding.

## Rollback to v1

Flash the archived v1 image by the same procedure, with the 15 Oct receipt path (no `--archive`):

```bash
JOB=nano33ble; ARCH=outputs/firmware-v1-rollback
"$BOSSAC" --port $BOOT --write --erase --verify -U --reset "$ARCH/nano33ble/firmware.bin" 2>&1 | tee build/v2/$JOB-rollback-$DAY-bossac-upload.log
"$PY" -u scripts/record_v2_upload_receipt.py --env $JOB --bossac-log build/v2/$JOB-rollback-$DAY-bossac-upload.log \
    --bootloader-port $BOOT --output outputs/firmware-v2-uploads/$JOB-rollback-$DAY-upload-receipt.json
```

Confirm the rollback with a capture:

- `Feat:` has 14 names.
- There is no `Model: TinyKAN v2` banner.
- `Hz:` reads about 49.97.

`pio run -e nano33ble` rebuilds the identical BIN (`1f2f95db…`) if the archive is ever lost. Last checked by a clean rebuild on 22 Oct 2026, after the r2 edit.

## Do not

- Do not edit firmware, weights, the threshold, N, the window/context/stride, LED settings or the finger gate between jobs. Each receipt binds one archived image.
- Do not move a tolerance or the rate criterion after a run. Report the residual.
- Do not write into existing `outputs/` directories. Every run takes a fresh `--output`, and the scripts refuse an existing one.
- Do not use MAX30105 register 0x05.
- Do not use `pio run -t upload`.
