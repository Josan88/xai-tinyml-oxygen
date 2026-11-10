# SpO2 est (Maxim reference algorithm) board runbook (ADR-0010)

**Run only when Jason names this board session.** Jason sits at the sensor. Written 5 Nov 2026, before any flash of this image. Recommended after the paper (22 Nov). The paper is unaffected either way.

Image: `outputs/firmware-v2-f1a-spo2maxim/firmware.bin`, env `nano33ble_v2_f1a_spo2`, SHA-256 `f4a78cf6ce9afd78c61d26824686cccfcc70408d75104f837e047565973d6af0`, 346,692 bytes. **Left on the board, 6 Nov 2026** (keep receipt `outputs/firmware-v2-uploads/nano33ble_v2_f1a_spo2-keep-20261106-upload-receipt.json`). The sit is `outputs/live-v2/2026-11-06_f1a-spo2_sit.sidecar.md`.
Restore target, only if Jason names that flash: r2 `outputs/firmware-v2-f1a-r2/firmware.bin`, `cdefac1dc9e04fe3f9e4df9e7466008166e9b6bb59f341e8a0ee85c1c89a50d2`.

SpO2 est on this image is a display value from Maxim's reference algorithm. It is uncalibrated for this module and enclosure, not a TinyKAN input, not used by any decision or explanation, and not a blood-oxygen reading. Fraczkiewicz's algorithm runs beside it on the `SPO2CMP:` serial line only. **What the comparison may conclude (D10):** which algorithm gives a value more often and moves less at rest on this sensor; not which is more accurate.

## Pre-registered criteria (5 Nov 2026)

These criteria are fixed before the flash. **Do not edit them after any capture of this image has been read.** A change needs a dated amendment below this table, with the reason. The table itself stays as written.

| # | Criterion | Pass rule |
|---|---|---|
| 1 | Identity | The receipt SHA equals the archive SHA. Every finger-on `SpO2_est:` field is `---` or a value in 70.0–100.0 ending `.0`. |
| 2 | Rate | Every `Hz:` line ≥ 49.5, idle and sit. Decision intervals: median ≈ 5 s, none > 5.5 s. |
| 3 | TinyKAN behaviour | **Report only:** P range, Candidate and Confirmed counts, top-feature counts. Context is the r2 sit (P 0.07–0.10, 0/35 Candidate); no pass band. |
| 4 | SpO2 est | **Report only:** share of finger-on lines with a value, median and range, and the time to first value. No accuracy claim. A mostly-dash result is a finding, not a failure. |
| 5 | Algorithm comparison | **Report only**, from `scripts/analyze_spo2_compare.py` on the sit capture: per algorithm, share shown under the same rule, median and range, median step change, time to first value; their difference when both are shown. The conclusion is limited to D10. Line counts: one `SPO2CMP:` line per second of finger contact after the first 4 s. A large shortfall is reported, not a fail. |
| 6 | Restore | Flash r2 `cdefac1d…` back and re-identify, unless Jason says keep the new image live (D6). Write a receipt for every flash. |

A fail on criterion 1 or 2 means the image is not used. Write the sidecar. Do not tune the image in the same session.

Amendments:

- 6 Nov 2026, after the sit and the r2 restore: Jason said leave the new image on. The criteria table above is unchanged. The image was flashed again (keep receipt `nano33ble_v2_f1a_spo2-keep-20261106-upload-receipt.json`, SHA `f4a78cf6…`, 346,692 bytes, `Verify successful`) and left on the board. A no-finger re-identify shows a live `Red:`/`IR:` stream at 50.09–50.11 Hz and cannot show `SPO2CMP:` (that line needs a finger). Do not restore r2 unless he names that flash.

## B1. Pre-flight (no port opened)

Use the pre-flight of `docs/v2-flash-runbook.md` §0 (`PY`, `BOSSAC`, `DAY`; close every monitor, dashboard tab and IDE). Then:

```bash
for ARCH in outputs/firmware-v2-f1a-spo2maxim outputs/firmware-v2-f1a-r2; do (cd "$ARCH" && tr -d '\r' < SHA256SUMS | sha256sum -c -) || echo "STOP: $ARCH"; done
"$PY" -m serial.tools.list_ports -v          # lists ports without opening them; app port 2341:805A (COM8)
"$PY" -u scripts/record_v2_upload_receipt.py --dry-run --env nano33ble_v2_f1a_spo2 --archive outputs/firmware-v2-f1a-spo2maxim
```

- Every `sha256sum -c` line says `OK`, and the dry-run prints `"archive_and_image_ok": true`. Otherwise stop.
- Identify the current image. No `H` answer plus a `Red:/IR:` stream means a live image. The newest receipt in `outputs/firmware-v2-uploads/` says which one.
- Sensor 2 fitted, lid on.

## B2. Flash and capture

Flash with the procedure in `docs/v2-flash-runbook.md` ("Flashing an archived image"), with:

```bash
JOB=nano33ble_v2_f1a_spo2
ARCH=outputs/firmware-v2-f1a-spo2maxim
LOG=build/v2/$JOB-$DAY-bossac-upload.log      # this image has no "-r2-" in its names
# touch1200 -> bootloader 2341:005A (usually COM7) -> BOOT=<port>
"$BOSSAC" --port $BOOT --write --erase --verify -U --reset "$ARCH/firmware.bin" 2>&1 | tee "$LOG"
"$PY" -u scripts/record_v2_upload_receipt.py --env $JOB --archive $ARCH --bossac-log "$LOG" --bootloader-port $BOOT \
    --output outputs/firmware-v2-uploads/$JOB-$DAY-upload-receipt.json
```

- The log ends `Verify successful`, and `Write N bytes to flash` equals 346,692. Otherwise stop. Do not retry with other tools.
- **60 s idle, no finger:** `"$PY" -u build/v2/cap.py COM8 60 outputs/live-v2/<date>_f1a-spo2_idle.txt`.
- **180 s sit:** sensor 2, seated, lid on, no dashboard. Give the clarify cue ("Ready, finger is on" / "Not now"), then run the capture alone in the foreground: `"$PY" -u build/v2/cap.py COM8 180 outputs/live-v2/<date>_f1a-spo2_sit.txt`. Tell Jason when to lift. Write the marks file (`HH:MM:SS LABEL`, e.g. `SIT` at finger-on and `END` at lift) for `build/v2/session_summary.py`.
- **Optional, only if Jason asks:** a Bluetooth check with the dashboard connected, to see the tile show values or a dash.
- **Optional, Jason's choice only:** a certified pulse oximeter on another of Jason's fingers, with readings and times recorded in the sidecar. It is not required and gives no accuracy claim (one person, at rest). It is not for anyone else (ADR-0009 scope).

## B3. Checking the criteria

```bash
CAP=outputs/live-v2/<date>_f1a-spo2_sit.txt
"$PY" build/v2/sit_summary.py "$CAP"                        # 2: Hz range; 3: P, Candidate, Confirmed; 4: SpO2_est median/range
"$PY" build/v2/session_summary.py "$CAP" <marks.txt>        # 2: hz_below_49_5, decision intervals (median, over_5_5); 3: top features
"$PY" - "$CAP" <<'EOF'                                      # 1 and 4: finger-on SpO2_est fields
import re, sys
on = [l for l in open(sys.argv[1], encoding="utf-8") if "KAN_Prob:" in l]
bad = [l for l in on if not re.search(r"SpO2_est: (---|(?:[7-9]\d|100)\.0%)", l)]
val = [l for l in on if not "SpO2_est: ---" in l]
print(f"finger-on lines {len(on)}, with a value {len(val)}, bad fields {len(bad)}"); print(*bad[:5], sep="")
EOF
"$PY" scripts/analyze_spo2_compare.py "$CAP" --json "${CAP%.txt}.spo2cmp.json" --md "${CAP%.txt}.spo2cmp.md"   # 5
```

- Criterion 1 needs `bad fields 0` on both captures and the receipt SHA equal to `f4a78cf6…`.
- Criterion 2 is read on the idle capture as well (Hz only).
- Criterion 5 line count: about one `SPO2CMP:` line per second of contact after the first 4 s. `missing_n` in the analyser output counts gaps.
- The analyser's Maxim display is recomputed from the raw fields; `firmware_shown_mismatches` should be 0. If it is not, report it.

## B4. Restore and record

- **Restore r2** (criterion 6) unless Jason says keep this image live. Use the same procedure with `JOB=nano33ble_v2_f1a`, `ARCH=outputs/firmware-v2-f1a-r2`, a new log name, and a new receipt. Then re-identify (a `Red:/IR:` stream without `SPO2CMP:` lines).
- **Sidecar** `outputs/live-v2/<date>_f1a-spo2_sit.sidecar.md`: image SHA, receipt, conditions (sensor, lid, light, posture), each criterion's result, capture SHA-256s, and anything missed.
- `git add -f` the exact new captures, analyser outputs, receipts and sidecar. Commit.
- Queue the result for the final report (vault `Final report - queued results`), **not** the IEEE paper.
- If the comparison favours Fraczkiewicz's algorithm, switching the tile is Jason's decision and a new image. Do not do it in this session.

## Do not

- Do not tune the image, thresholds or display rule in the session.
- Do not edit the criteria table after reading a capture.
- Do not write "SpO2" alone or "oxygen level" for SpO2 est, and do not present any number from this session as accuracy.
