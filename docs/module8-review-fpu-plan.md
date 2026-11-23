# Module 8 review M11: edge KAN with the hardware FPU (plan and runbook)

Written 19 Nov 2026, before any FPU image was built or run. Reviewer finding M11 asks whether the reported
latency reflects the Cortex-M4F. It does not. PlatformIO's mbed builder (`arduino-core-mbed.py`, `configure_fpu_flags`)
appends `-mfloat-abi=soft` after the framework's `-mfloat-abi=softfp -mfpu=fpv4-sp-d16`, so every image in the
paper ran FP32 in software. This job measures the same three edge-KAN images built with the FPU. It changes
nothing else.

## What changes

- `firmware/Arduino Nano 33/fpu_flags.py` (post-script): it replaces `-mfloat-abi=soft` with `-mfloat-abi=softfp -mfpu=fpv4-sp-d16`
  in the main, project and library build environments. softfp keeps the soft calling convention, so the
  precompiled mbed core still links.
- `platformio.ini`: `nano33ble_ekan_context_parity_fpu`, `nano33ble_ekan_pipeline_timing_fpu` and
  `nano33ble_ekan_context_timing_fpu`. Each one is `extends =` of the archived env plus `extra_scripts = post:fpu_flags.py`.
- `scripts/archive_ekan_fpu_build.py` (new) archives a verbose build to `outputs/firmware-ekan-fpu/<env>/` after these checks:
  - the log carries the banner;
  - no compile line carries `-mfloat-abi=soft`;
  - every compile line carries softfp and fpv4-sp-d16;
  - the image differs from the soft-float archive;
  - VFP f32 arithmetic appears in the project functions (`arm-none-eabi-objdump`).
- `scripts/record_v2_upload_receipt.py` knows the three FPU archives.
- `scripts/replay_edge_kan_board.py --fpu` uses the same identities, prepared set, host probe and gates. It writes under
  `outputs/edge-kan-fpu/`. The `pipeline` summary adds a decision-by-decision comparison with the soft-float board
  run `outputs/edge-kan-v1/board-pipeline-20261112`.

## Soft-float reference (12 Nov 2026, COM8)

| Quantity | Soft float | Source |
|---|---|---|
| Integrated pipeline, subsequent decision block, median / p95 | 433.9 / 435.4 ms | board-pipeline-20261112 |
| First decision after reset, median | 350.2 ms | board-pipeline-20261112 |
| Functional decisions passing / candidates / confirmed | 1243 of 1243 / 642 / 308 | board-pipeline-20261112 |
| Edge forward, median of per-decision minimum | 4.32 ms | board-timing-20261112 |
| Edge incremental XAI (17 evaluations), median | 24.52 ms | board-timing-20261112 |

## Fixed before running

1. **Control:** a clean rebuild of `nano33ble_ekan_pipeline_timing` (soft float) must give `firmware.bin` SHA-256
   `d1cccebe74901fb6ccfc2f1845c9fde8ffc404ac1e8c31d28f3c111044ebe69c`. If it does not, the tree has moved since 12 Nov, the FPU-vs-soft comparison is confounded, and we stop.
2. **Build gate:** `archive_ekan_fpu_build.py` must accept the build.
3. **Functional gate (primary):** `replay_edge_kan_board.py pipeline --fpu` must pass all 1,243 decisions under the same contract
   as the soft-float run (meta, features, probability ≤ contract and same side of 0.50, states, top-3 order and deltas,
   valid), with 0 timed decisions failed. Bit identity with soft float is **not** required. With the FPU, GCC may fuse
   multiply-adds (`-ffp-contract=fast` is the GNU default), so last-bit differences are expected. They are reported:
   bit-identical count, maximum |ΔP|, flag, confirmed-state and top-3 differences.
4. **Timing (primary quantity, no pass/fail threshold):** median and p95 of the subsequent-decision block, and the
   first decision after reset, over 10 fixture repetitions, as in the soft-float run. Expectation (not a gate): lower than 433.9 ms. The
   feature extraction (zero-phase filter over 1,000 samples, moments) dominates the block and is FP32-heavy.
5. **Optional:** `timing --fpu` on `nano33ble_ekan_context_timing_fpu` gives the edge forward / XAI kernel split. Run
   `parity --fpu` on `nano33ble_ekan_context_parity_fpu` only if the pipeline gate fails, as a diagnostic.
6. **Reporting:** the paper's deployed numbers stay the soft-float image that every other board result used. The FPU
   result is added as a measured alternative build: one sentence in III-B/IV, a column or row in the device table,
   and the M11 response. If gate 3 fails, the paper says the FPU build was attempted and did not reproduce the
   host arithmetic, and keeps soft float.

## Runbook (Git Bash, from the repo root; same flashing procedure as docs/v2-flash-runbook.md)

```bash
cd /d/fyrp-xai-tinyml
export PLATFORMIO_CORE_DIR=~/.platformio           # avoids a platform reinstall
PIO=~/.platformio/penv/Scripts/pio.exe
PY="python"   # Python >= 3.11 with numpy, scipy, pandas, pyserial
"$PY" -c "import numpy, scipy, pandas, serial; print('ok')"
BOSSAC=~/.platformio/packages/tool-bossac-nordicnrf52/bossac.exe
DAY=$(date +%Y%m%d); mkdir -p build/ekan-fpu
FWD="firmware/Arduino Nano 33"

# 1. Control: soft-float rebuild must be byte-identical to the 12 Nov archive
(cd "$FWD" && "$PIO" run -e nano33ble_ekan_pipeline_timing -t clean && "$PIO" run -e nano33ble_ekan_pipeline_timing) \
    > build/ekan-fpu/control-soft-$DAY.log 2>&1
sha256sum "$FWD/.pio/build/nano33ble_ekan_pipeline_timing/firmware.bin"   # must be d1cccebe74901fb6...

# 2. FPU builds (verbose, clean) and archives
for ENV in nano33ble_ekan_pipeline_timing_fpu nano33ble_ekan_context_timing_fpu; do
  (cd "$FWD" && "$PIO" run -e $ENV -t clean && "$PIO" run -v -e $ENV) > build/ekan-fpu/$ENV.log 2>&1
  "$PY" -u scripts/archive_ekan_fpu_build.py --env $ENV --build-log build/ekan-fpu/$ENV.log
done

# 3. Flash the pipeline FPU image (archived file, never `pio run -t upload`) and record the receipt
JOB=nano33ble_ekan_pipeline_timing_fpu; ARCH=outputs/firmware-ekan-fpu/$JOB
"$PY" -u build/v2/touch1200.py COM8; "$PY" -m serial.tools.list_ports -v; BOOT=COM7   # set to the bootloader port
LOG=build/ekan-fpu/$JOB-$DAY-bossac-upload.log
"$BOSSAC" --port $BOOT --write --erase --verify -U --reset "$ARCH/firmware.bin" 2>&1 | tee "$LOG"
"$PY" -u scripts/record_v2_upload_receipt.py --env $JOB --archive $ARCH --bossac-log "$LOG" --bootloader-port $BOOT \
    --output outputs/firmware-v2-uploads/$JOB-$DAY-upload-receipt.json

# 4. Primary run: functional gate on 1,243 decisions + 10 timing repetitions (about the same duration as 12 Nov)
"$PY" -u scripts/replay_edge_kan_board.py pipeline --fpu --port COM8 --repetitions 10 \
    --upload-receipt outputs/firmware-v2-uploads/$JOB-$DAY-upload-receipt.json \
    --output outputs/edge-kan-fpu/board-pipeline-$DAY

# 5. Optional kernel split: flash nano33ble_ekan_context_timing_fpu as in step 3, then
"$PY" -u scripts/replay_edge_kan_board.py timing --fpu --port COM8 --repetitions 10 \
    --upload-receipt outputs/firmware-v2-uploads/nano33ble_ekan_context_timing_fpu-$DAY-upload-receipt.json \
    --output outputs/edge-kan-fpu/board-timing-$DAY

# 6. Restore the image you normally run (e.g. outputs/firmware-ekan/nano33ble_ekan_f1a) with the same bossac steps.
```

Stop rules: if the step 1 hash differs, if `archive_ekan_fpu_build.py` refuses, or if bossac does not end in
`Verify successful` with the byte count equal to the BIN size, stop and report. Do not retry with other flags or tools.
