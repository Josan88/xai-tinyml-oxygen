# Label-0 explainer reference: board-session runbook

**Status, 30 Oct 2026.** Images built and archived, not flashed. Host parity done. Board prepared set ready, not run. The pass criteria below were fixed on 30 Oct 2026, **before any flash**. They are not changed after a run. Report every residual.

The orchestrator runs this session. Jason named the job (F1). Commands are bash (Git Bash), run from the repo root. Flash only the archived files. Do not rebuild on the day, and do not use `pio run -t upload`.

What the images are: `outputs/firmware-v2-ref0/README.md`. The Mean-Perturbation reference is the label-0 median, `KAN_XAI_REF_Z` in `firmware/Arduino Nano 33/include/kan_xai_reference_v2.h`. That is the median standardized input over windows whose label is 0 (window-median ScalcO2 >= 90). It is not the training mean. The shipped v2 image (`outputs/firmware-v2-f1a-r2/`) stays the shipped image, and the session ends with it restored.

| Job | Archive (`$ARCH`) | Env (`$JOB`) | `firmware.bin` SHA-256 | Identity |
|---|---|---|---|---|
| (a) Board parity | `outputs/firmware-v2-ref0/nano33ble_v2_context_parity_ref0` | `nano33ble_v2_context_parity_ref0` | `352447664045946c521b78903f80c339a904dff5ac48a124c5686897f598f347` | `H` → `CONTEXT_PARITY_V2 WINDOW=500 CONTEXT=250 STRIDE=250 N=2 THRESHOLD=0.50 INPUTS=16 PAIRS=IR,RED XAI_REF=LABEL0` |
| (b) Live idle | `outputs/firmware-v2-ref0/nano33ble_v2_f1a_ref0` | `nano33ble_v2_f1a_ref0` | `368e2233d29bacb0fef71dff5a04dc0b590e819c778c481638d4882e91f6cfd7` | serial line `XAI_REF=LABEL0: Mean-Perturbation reference is the label-0 median of each standardized input, not the training mean` |
| (r) Restore (shipped) | `outputs/firmware-v2-f1a-r2` | `nano33ble_v2_f1a` | `cdefac1dc9e04fe3f9e4df9e7466008166e9b6bb59f341e8a0ee85c1c89a50d2` | no `XAI_REF=LABEL0` line; `Hz:` lines |
| P5 images | *reserved: see the end of this file* | | | |

There is no timing job. The decision's op count is unchanged: the same 1 + 16 forward passes, each reading a flash constant instead of writing 0 (`outputs/firmware-v2-ref0/README.md`).

## Pass criteria (fixed 30 Oct 2026, before any flash)

1. **Board parity (a).** Every decision passes the unchanged v2 budgets (`scripts/parity_criteria_v2.py`) against both the Python reference and the C++ host. Metadata, Candidate/Confirmed/streak, the top-3 payload and `two_wavelength_valid` must match exactly. That is, `summary.failed == 0` in the report over all 21 fixture and 1,222 locked decisions.
   - Report every failure. Exclude nothing.
   - `--locked 50` is a first look, not the result.
   - Host context: the full host run on this source has 9 frozen-criteria failures of 71,462 (`outputs/parity/v2-host-ref0-20261030/`). None is in the locked set: the host check of the prepared set is 0 of 1,243. A board pass does not close that host residual.
2. **Live idle 60 s (b).** Both must hold:
   - every `Hz:` line of the capture is >= 49.5. There is no narrower band, the same rule as r2;
   - the capture contains the boot banner line naming the reference, `XAI_REF=LABEL0: Mean-Perturbation reference is the label-0 median …`.
   - The image prints that line at boot and again on each DTR rise. mbed drops USB serial output until the host raises DTR, so a capture with any `Hz:` line was opened with DTR high, and the line was printed into it.
3. **Restore (r).** All three must hold:
   - the bootloader log shows a verified write of `outputs/firmware-v2-f1a-r2/firmware.bin`;
   - the receipt records image `cdefac1d…`;
   - a capture of at least 30 s re-identifies the shipped image: `Hz:` lines present and **no** `XAI_REF=LABEL0` line. The shipped image never prints it.
   - Restore runs whatever happens in (a) or (b).

## 0. Pre-flight

```bash
cd /d/fyrp-xai-tinyml
PY=/c/Python314/python.exe                     # numpy, scipy, pandas, pyserial
BOSSAC="$USERPROFILE/.platformio/packages/tool-bossac-nordicnrf52/bossac.exe"
DAY=$(date +%Y%m%d)
"$PY" -c "import numpy, scipy, pandas, serial"
ls -l "$BOSSAC"
for ARCH in outputs/firmware-v2-ref0/nano33ble_v2_context_parity_ref0 outputs/firmware-v2-ref0/nano33ble_v2_f1a_ref0 outputs/firmware-v2-f1a-r2; do
  (cd "$ARCH" && tr -d '\r' < SHA256SUMS | sha256sum -c -) || echo "STOP: $ARCH"
done
"$PY" -u scripts/replay_context_board_v2.py run --dry-run --prepared outputs/parity/v2-board-prepared-ref0
```

- Every `sha256sum -c` line must say `OK`. Otherwise stop.
- The dry run must print `"archive_and_image_ok": true`, the ref0 identity above and 611 locked cases. It opens no port.
- Close every serial monitor, PlatformIO monitor, Chrome dashboard tab and Arduino IDE that may hold the port.

Ports, found by VID:PID and never assumed:

```bash
app_port()  { "$PY" -c "import serial.tools.list_ports as L; p=[x.device for x in L.comports() if (x.vid,x.pid)==(0x2341,0x805A)]; print(p[0] if len(p)==1 else 'NONE')"; }
boot_port() { "$PY" -c "import serial.tools.list_ports as L; p=[x.device for x in L.comports() if (x.vid,x.pid)==(0x2341,0x005A)]; print(p[0] if len(p)==1 else 'NONE')"; }
APP=$(app_port); echo "application port: $APP"   # recently COM8; NONE or several -> stop
```

## Flashing an archived image (same for every job)

Set `JOB` and `ARCH` from the table, then:

```bash
"$PY" -u build/v2/touch1200.py $APP            # 1200-baud touch into the bootloader (or double-tap reset), as in docs/v2-flash-runbook.md
BOOT=$(boot_port); echo "bootloader: $BOOT"    # VID:PID 2341:005A; NONE -> stop
LOG=build/v2/$JOB-$DAY-bossac-upload.log
"$BOSSAC" --port $BOOT --write --erase --verify -U --reset "$ARCH/firmware.bin" 2>&1 | tee "$LOG"
"$PY" -u scripts/record_v2_upload_receipt.py --env $JOB --archive $ARCH --bossac-log "$LOG" --bootloader-port $BOOT \
    --output outputs/firmware-v2-uploads/$JOB-$DAY-upload-receipt.json
APP=$(app_port); echo "application port: $APP"
```

- The log must end `Verify successful`, and `Write N bytes to flash` must equal the BIN size: 336,900 for (a), 344,204 for (b), 343,620 for (r). Otherwise stop, and go to (r) if a different image may now be on the board.
- The receipt step opens no port. It refuses an image that does not match the archive's `SHA256SUMS`, an env the archive does not hold, and an existing receipt.

## (a) Board parity, `nano33ble_v2_context_parity_ref0`

```bash
JOB=nano33ble_v2_context_parity_ref0; ARCH=outputs/firmware-v2-ref0/nano33ble_v2_context_parity_ref0
# flash as above, then:
RCPT=outputs/firmware-v2-uploads/$JOB-$DAY-upload-receipt.json
"$PY" -u scripts/replay_context_board_v2.py run --dry-run --prepared outputs/parity/v2-board-prepared-ref0 --upload-receipt $RCPT
"$PY" -u scripts/replay_context_board_v2.py run --port $APP --upload-receipt $RCPT \
    --prepared outputs/parity/v2-board-prepared-ref0 --output outputs/parity/v2-board-$DAY-ref0
```

- The driver takes the image archive, env and expected `H` identity from the prepared manifest. It stops before sending samples unless the board answers exactly the ref0 identity. The r2 image's identity lacks ` XAI_REF=LABEL0` and is refused.
- It then runs the protocol checks: mid-warmup reset and malformed-block rejection.
- The replay is 3 synthetic paired streams (21 decisions), then 611 locked contexts (1,222 decisions), with `A` (`ATTR16`, all 16 label-0-reference deltas) after every decision. It takes about an hour.
- Criterion 1 applies. Read `summary` in `outputs/parity/v2-board-$DAY-ref0/report.json`. List every failing decision with its check.

## (b) Live idle 60 s, `nano33ble_v2_f1a_ref0`

```bash
JOB=nano33ble_v2_f1a_ref0; ARCH=outputs/firmware-v2-ref0/nano33ble_v2_f1a_ref0
# flash as above (receipt written), then, with no finger on the sensor:
CAP=outputs/live-v2/$(date +%Y-%m-%d)_f1a-ref0_idle.txt
"$PY" -u build/v2/cap.py $APP 60 "$CAP"
"$PY" -c "import re,sys; t=open(sys.argv[1]).read(); v=[float(x) for x in re.findall(r'Hz: ([0-9.]+)', t)]; b='XAI_REF=LABEL0: Mean-Perturbation reference is the label-0 median' in t; print(len(v), 'Hz lines', (min(v), max(v)) if v else None, '| banner', b, '| PASS' if v and min(v) >= 49.5 and b else '| FAIL')" "$CAP"
```

- Criterion 2 applies. Report the `Hz:` range and whether the banner line is present.
- Idle: every status line reads `NO FINGER`, with IR about 150. The second MAX30102 module is installed.
- BLE is not part of the criterion. With a dashboard connected, the packet has 12 fields ending `|-|label0`. Dashboard commit `0605a1a` parses fields 10–11.

## (c) Sit, needs Jason, not part of this session's criteria

Run only if Jason is seated with a finger on the sensor and confirms it. Use the same capture with 180 s and the same rate rule. The `Feat:` lines list 16 names ending `two_wavelength_valid=1`. A sit shows on-device behaviour only, not detection of low oxygen. A flag on a healthy seated subject is a false positive or domain shift.

## (r) Restore the shipped image (always last)

```bash
JOB=nano33ble_v2_f1a; ARCH=outputs/firmware-v2-f1a-r2
# flash as above: the receipt step checks outputs/firmware-v2-f1a-r2/SHA256SUMS (cdefac1d...), then:
CAP=outputs/live-v2/$(date +%Y-%m-%d)_f1a-r2_restore-idle.txt
"$PY" -u build/v2/cap.py $APP 30 "$CAP"
"$PY" -c "import re,sys; t=open(sys.argv[1]).read(); v=[float(x) for x in re.findall(r'Hz: ([0-9.]+)', t)]; r='XAI_REF=LABEL0' in t; print(len(v), 'Hz lines', (min(v), max(v)) if v else None, '| ref0 line present', r, '| RESTORED' if v and not r else '| NOT RE-IDENTIFIED')" "$CAP"
```

- Criterion 3 applies.
- If the restore cannot be re-identified, do not leave the board on a ref0 image: report and stop.

## Do not

- Do not edit firmware, the reference header, weights, the threshold, N, the window/context/stride, LED settings or the finger gate between jobs. Each receipt binds one archived image.
- Do not move a budget or the rate criterion after a run.
- Do not write into existing `outputs/` directories. The scripts refuse an existing `--output`.
- Do not use MAX30105 register 0x05.
- Do not use `pio run -t upload`.

## P5 images (job F2, ADR-0008)

**Status, 30 Oct 2026.** Images built and archived, not flashed. Host parity done. Board prepared set ready, not run. The pass criteria below were fixed on 30 Oct 2026, **before any flash**. They are not changed after a run. Report every residual. Nothing above this section changes.

**Run 30 Oct 2026 (evening), criteria unchanged:** (p-a) pass 0/1,243; (p-t) fail as written, stopped at the first decision on the exact-equality check; (p-t2) re-run under the amended check: pass, median 425.2 ms, max 426.8 ms; (p-b) fail as written, `MODEL=P5` banner absent, rate held; (p-s) fail as written on the banner clause, rate held, 0 Confirmed; (r) pass. Detail: `outputs/live-v2/2026-10-30_p5-session.sidecar.md`.

**What the images are.** `outputs/firmware-p5/README.md`.
- A third model beside v1 and v2, never replacing them.
- The 5 beat inputs `r_beat, pi_beat, hr_bpm, sqi_template, sqi_ibi_ratio` (`include/physio_features.h`), on the v2 paired IR/red context filter.
- TinyKAN P5, label-0-median explainer reference `kan_xai_reference_p5.h`, same 0.50 / N=2.
- `s_hat`, estimated ScalcO2 (model saturation estimate), displayed only. A P5 sit, if Jason ever runs one, shows on-device behaviour only. `s_hat` is not an oxygen reading and is uncalibrated for the MAX30102.

The mechanism is F1's, reused:
- the reference header from `scripts/make_xai_reference_header.py --check-s3-model P5`;
- `-DOXY_XAI_REF_LABEL0`;
- `scripts/archive_firmware_env.py`;
- `scripts/record_v2_upload_receipt.py --archive outputs/firmware-p5/<env>`.

The P5 board drivers are new scripts: `scripts/replay_context_board_p5.py` and `scripts/benchmark_context_board_p5.py`.

| Job | Archive (`$ARCH`) | Env (`$JOB`) | `firmware.bin` SHA-256 | bytes | Identity |
|---|---|---|---|---|---|
| (p-a) Board parity | `outputs/firmware-p5/nano33ble_p5_context_parity` | `nano33ble_p5_context_parity` | `67ebb3c73e8c84760e69d58f0b9d1cb485a9148fffafda0365dc6fbce32351e9` | 336,676 | `H` → `CONTEXT_PARITY_P5 WINDOW=500 CONTEXT=250 STRIDE=250 N=2 THRESHOLD=0.50 INPUTS=5 PAIRS=IR,RED XAI_REF=LABEL0` |
| (p-t) Timing | `outputs/firmware-p5/nano33ble_p5_context_timing` | `nano33ble_p5_context_timing` | `3473e1c230b57a07d1945d52cec6bcb90c083441cdebdcdf8efa89548b461584` | 337,500 | `H` → `TIMING_REPLAY_P5 WINDOW=500 CONTEXT=250 STRIDE=250 N=2 THRESHOLD=0.50 CLOCK_HZ=64000000 GPIO=D2 ATTR5=0 INPUTS=5 PAIRS=IR,RED XAI_REF=LABEL0` |
| (p-b) Live idle | `outputs/firmware-p5/nano33ble_p5_f1a` | `nano33ble_p5_f1a` | `9b423d842cb112c68535a9aafd117e822bc2f1ce9d741bfd6a4fd58da3eab547` | 344,516 | serial `MODEL=P5: TinyKAN P5, 5 beat inputs …` and `XAI_REF=LABEL0: …` |
| (r) Restore (shipped) | `outputs/firmware-v2-f1a-r2` | `nano33ble_v2_f1a` | `cdefac1dc9e04fe3f9e4df9e7466008166e9b6bb59f341e8a0ee85c1c89a50d2` | 343,620 | no `MODEL=P5` and no `XAI_REF=LABEL0` line; `Hz:` lines |

### P5 pass criteria (fixed 30 Oct 2026, before any flash)

1. **Board parity (p-a).** Every decision passes the pre-registered P5 classes (`docs/p5-parity-criteria.md`, board section) against **both** the Python reference and the C++ host, so `summary.failed == 0` over all 21 fixture and 1,222 locked decisions. Per decision:
   - `n_beats` and `beat_valid` are identical to both;
   - the inputs are within the feature budgets of Python, and bit-equal to the C++ host;
   - probability 1e-4, the five deltas 1e-5, ranking gap 2e-5 and exact 0.50, against both;
   - `s_hat` within 4e-4 x 8.828 + 8 FP32 spacings, against both;
   - metadata, Candidate/Confirmed/streak and the top-3 payload are exact.

   Notes:
   - Report every failure. Exclude nothing. `--locked 50` is a first look, not the result.
   - Host context: the full host run has 18 failures of 71,462 (`outputs/parity/p5-host-20261030/`). None is in the locked set: the host check of the prepared set is 0 of 1,243. A board pass does not close that host residual.
2. **Timing (p-t).** Stop rule: the run stops at the first **subsequent-decision block >= 640 ms**, the FIFO's 32 records at 50 Hz. Status `stopped_fifo_bound` is a fail.
   - Pass: status `completed` over 10 repetitions of the 3 fixtures, every timed decision bit-equal to the C++ host row, and the max subsequent-decision block < 640 ms.
   - Report the subsequent-decision median, p95 and max. The D2 pulse is the scope marker as for v2. It is not part of the criterion; the D2/GPIO item stays open.
3. **Live idle 60 s (p-b).** All three must hold:
   - every `Hz:` line of the capture is >= 49.5. There is no narrower band, the same rule as r2;
   - the capture contains the line starting `MODEL=P5: TinyKAN P5, 5 beat inputs`;
   - the capture contains the line starting `XAI_REF=LABEL0: Mean-Perturbation reference is the label-0 median`.

   Both banner lines are printed at boot and again on each DTR rise. The F1 ref0 image prints only the second, so the first identifies P5.
4. **Restore (r).** Criterion 3 of F1, unchanged, with one addition: the 30 s restore capture must contain **no** `MODEL=P5` line either. Restore runs whatever happens in (p-a), (p-t) or (p-b).

### P5 commands

Pre-flight: as section 0. Then also run these; every line must say `OK` and each dry run must print `"archive_and_image_ok": true`. No port is opened.

```bash
for ARCH in outputs/firmware-p5/nano33ble_p5_context_parity outputs/firmware-p5/nano33ble_p5_context_timing outputs/firmware-p5/nano33ble_p5_f1a; do
  (cd "$ARCH" && tr -d '\r' < SHA256SUMS | sha256sum -c -) || echo "STOP: $ARCH"
done
"$PY" -u scripts/replay_context_board_p5.py run --dry-run          # loopback 1,243/1,243 and the identity above
"$PY" -u scripts/benchmark_context_board_p5.py --dry-run
```

**Flash.** Each image is flashed with the block "Flashing an archived image" above, with `JOB` and `ARCH` from the P5 table. `Write N bytes to flash` must equal the bytes column: 336,676 (p-a), 337,500 (p-t), 344,516 (p-b), 343,620 (r).

**(p-a) Board parity.**

```bash
JOB=nano33ble_p5_context_parity; ARCH=outputs/firmware-p5/nano33ble_p5_context_parity
# flash as above, then:
RCPT=outputs/firmware-v2-uploads/$JOB-$DAY-upload-receipt.json
"$PY" -u scripts/replay_context_board_p5.py run --dry-run --upload-receipt $RCPT
"$PY" -u scripts/replay_context_board_p5.py run --port $APP --upload-receipt $RCPT --output outputs/parity/p5-board-$DAY
```

- The driver stops before sending samples unless the board answers exactly the P5 identity. It then runs the mid-warmup reset and malformed-block checks.
- The replay is 3 synthetic paired streams (21 decisions), then 611 locked contexts (1,222 decisions). Criterion 1 applies. Read `summary` in `outputs/parity/p5-board-$DAY/report.json`.

**(p-t) Timing.**

```bash
JOB=nano33ble_p5_context_timing; ARCH=outputs/firmware-p5/nano33ble_p5_context_timing
# flash as above, then:
RCPT=outputs/firmware-v2-uploads/$JOB-$DAY-upload-receipt.json
"$PY" -u scripts/benchmark_context_board_p5.py --port $APP --upload-receipt $RCPT --repetitions 10 --output outputs/timing-p5-board-$DAY
```

Criterion 2 applies. Read `status`, `max_subsequent_decision_block_ms` and `latency_summary.subsequent_decisions`.

**(p-b) Live idle 60 s.** Flash `nano33ble_p5_f1a`; the receipt is written. Then, with no finger on the sensor:

```bash
CAP=outputs/live-v2/$(date +%Y-%m-%d)_p5-f1a_idle.txt
"$PY" -u build/v2/cap.py $APP 60 "$CAP"
"$PY" -c "import re,sys; t=open(sys.argv[1]).read(); v=[float(x) for x in re.findall(r'Hz: ([0-9.]+)', t)]; m='MODEL=P5: TinyKAN P5, 5 beat inputs' in t; b='XAI_REF=LABEL0: Mean-Perturbation reference is the label-0 median' in t; print(len(v), 'Hz lines', (min(v), max(v)) if v else None, '| MODEL=P5', m, '| XAI_REF', b, '| PASS' if v and min(v) >= 49.5 and m and b else '| FAIL')" "$CAP"
```

- Idle: every status line reads `NO FINGER`.
- BLE is not part of the criterion. With a dashboard connected, the packet has 12 fields ending `|-|label0`. Field 10 is `-` with no finger.

**(p-s) Sit, 180 s. Amendment of 30 Oct 2026, written before any P5 flash.** Jason named the sit for this session after (p-b) and before (r), on `nano33ble_p5_f1a` with no re-flash. Conditions: second MAX30102 module, seated, lid on, no dashboard connected, finger on at the cue, no lift until cued. Capture `outputs/live-v2/$(date +%Y-%m-%d)_p5-f1a_sit.txt` with `build/v2/cap.py $APP 180`.

- Pass criterion: criterion 3 unchanged. Every `Hz:` line is >= 49.5, and both banner lines are in the capture (printed on the DTR rise).
- Recorded expectation, not part of the criterion: 0 Confirmed at rest. Oxygen is not varied and there is no reference reading. A Confirmed on a seated subject is reported as a false positive or domain shift, not as detection.
- Reported with no band: decisions keyed on `(Win, Samples)` with `Samples > 0`, KAN_Prob median and range, Candidate and Confirmed counts, `beat_valid` count, the `s_hat` range (estimated ScalcO2, display only, not an oxygen reading), TopFeature counts, and `NO FINGER` lines after the first decision.
- (r) runs after the sit whatever its outcome.

**(p-t2) Timing re-run. Amendment of 30 Oct 2026, written after the (p-t) fail and before its re-flash.** Jason approved re-running timing with the per-decision check aligned to the parity budgets. This changes criterion 2 after a run. The (p-t) result stays on file as a fail under the original criterion (`outputs/timing-p5-board-20261030/`).

- Reason: in the (p-a) run the board probability is 1 ulp off the x86 C++ host on 2 of 21 fixture decisions (7.5e-9), inside the parity budgets. The exact check could not pass on this board (`build/v2/p5_timing_check_modes.py`: exact 19/21, parity-budgets 21/21 on the (p-a) fixture values).
- Changed: `--functional-check parity-budgets`, against the same C++ host row. Probability within 1e-4 and on the same side of 0.50. Each top-3 value within 1e-5 of the host value for that index. `s_hat` within 4e-4 x 8.828 + 8 FP32 spacings. These are the pre-registered P5 parity budgets (`scripts/parity_criteria_p5.py`).
- Unchanged, exact: features (FP32 bit-equal), states, top-3 indices, `beat_valid`, `n_beats`, decision metadata.
- Unchanged: 10 repetitions of the 3 fixtures, the stop at the first subsequent-decision block >= 640 ms (`stopped_fifo_bound` is a fail), and reporting median, p95 and max.
- Pass: status `completed`, every timed decision passes the amended check, and the max subsequent-decision block is < 640 ms. The result is a timing result plus a functional check within the parity budgets, not bit-equality.
- Output `outputs/timing-p5-board-20261030-budgets`, a new receipt for the re-flash, then (r) again with its own receipt and 30 s capture.

**(r) Restore.** As F1's (r), then this check on the restore capture:

```bash
"$PY" -c "import re,sys; t=open(sys.argv[1]).read(); v=re.findall(r'Hz: ([0-9.]+)', t); bad=('XAI_REF=LABEL0' in t) or ('MODEL=P5' in t); print(len(v), 'Hz lines | ref0/P5 line present', bad, '| RESTORED' if v and not bad else '| NOT RE-IDENTIFIED')" "$CAP"
```

**Not dry-run in preparation.** The receipt script's flash-log path and the flash command could not be dry-run: a permission rule denies any command containing the flasher's name. Their first use is on the day, with the checks of "Flashing an archived image".

**Closed by first use, 30 Oct.** Both paths were first used in the P5 session. There were six flashes through COM7: (p-a), (p-t), the (p-t) re-run, (p-b), and two (r) restores. Each has a receipt at `outputs/firmware-v2-uploads/*-20261030*-upload-receipt.json`. Each log ends `Verify successful`, its `Write N bytes` equals the image size, and its SHA-256 matches the receipt. Afterwards, the receipt script's `--dry-run --bossac-log` was run on each of the six logs (no port, nothing written): all six report `bossac_log_ok`, and the (p-b) log checked against the (p-a) archive is refused. The flash command still has no dry run of its own. Its evidence is these six verified writes.
