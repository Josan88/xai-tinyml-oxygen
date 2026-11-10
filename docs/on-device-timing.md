# On-device timing

11 September 2026 (local time). The exact archived `nano33ble_context_timing` image was authorized, uploaded and flash-verified. **All 210 synthetic decision blocks passed the observed functional checks. Cycle-derived median latency for subsequent decisions was 249.702 ms, with a maximum of 250.797 ms. Physical GPIO agreement and complete runtime memory qualification remain outstanding. Parity remains incomplete.**

The prepared image and build receipt are under `outputs/parity/qualification/timing-image-v1/`. The receipt records the exact BIN/ELF hashes, source hashes, frozen candidate, build log and test evidence. The existing all-14 diagnostic archive and its recorded-development results remain separate.

## Authorized synthetic measurement, 11 September

The authorized BIN SHA-256 is `7db771c18ef99fb087933b638ab399bef6f49e74425937e19311374209bbe633`. Upload completed at **2026-09-10 16:47:52 UTC**, and the timing application identified itself on COM8. The flash tool reported successful verification. The build receipt retains its historical preparation status; the separate verified-upload receipt is `outputs/parity/qualification/timing-board-upload-v1/receipt.json`.

Ten repetitions of the three built-in synthetic streams produced **270 blocks: 60 warmup blocks and 210 decision blocks**. Each repetition resets each stream. There are 21 distinct synthetic decisions repeated ten times, not 210 independent input windows. Every observed decision passes the frozen local-scale feature check and the host probability, top-three attribution, metadata and N=2 state comparisons.

Cycle-derived timings, normalized by the declared 64 MHz core clock, retain instrumentation and interrupt overhead:

- **Warmup, 60 blocks:** minimum/median **0.164594 ms**, p95 **0.191798 ms**, maximum **0.214000 ms**.
- **First decision after reset, 30 blocks:** minimum **203.236 ms**, median **204.176 ms**, p95 **204.670 ms**, maximum **204.790 ms**.
- **Subsequent decisions, 180 blocks:** minimum **248.170 ms**, median **249.702 ms**, p95 **250.535 ms**, maximum **250.797 ms**.
- All eight empty brackets returned **6 cycles**, or **0.09375 microseconds** at 64 MHz. This was not subtracted from the block measurements.

Percentiles use linear interpolation. First decisions use only 750 available samples for filtering context; later decisions have the full 1,000 samples. These are whole-block compute timings, not isolated inference/XAI latency or live sensor/BLE latency.

The coarse `millis()` interval was **1.612–4.060 ms shorter** than the cycle-derived interval for the 210 decision blocks. The installed Arduino core implements `millis()` with a separate `LowPowerTimer`; the discrepancy exceeds simple one-millisecond rounding on some blocks and is unresolved. Keep both recorded values. Do not rescale either clock or call the cycle-derived milliseconds physically calibrated until the D2 pulse is measured with an independent instrument.

All **31 memory snapshots** reported a main-thread stack reservation of **32,768 bytes**, watermark peak used of **31,308 bytes**, and newlib heap arena/allocated snapshots of **1,040 bytes**, with **0 bytes free inside that currently allocated arena**. The stack value was already present before the first benchmark block and never changed. It cannot be attributed to benchmark execution, and the heap snapshot is not all available SRAM. Whole-device peak RAM and the reason for the high pre-replay stack watermark remain unqualified.

Board/host maximum probability error is **zero**; maximum transmitted top-three signed-attribution error is **5.960464478e-8**. Original uniform absolute 1e-4 raw-feature checks still fail on **210/210** timed decisions. Every original error and adopted budget remains in the raw report. Maximum absolute errors are zero for features 0–10, **0.001953125** for lower-band power, **0.0009765625** for upper-band power, and **0.0001220703125** for their ratio. All 14 signed attributions were not re-observed in this timing image.

Evidence: `outputs/parity/timing-board-v1/report.json`, SHA-256 `a7c1861064dd0053b0d415bb5bf0d210aef52d97dae7ae9bd7dd907d9f41cb42`. An independent read-only count/summary check verified all 270 blocks, 210 functional passes, group statistics, calibration, memory counts and retained raw errors. No restricted recorded input or locked validation window was replayed.

## What is timed

One D2 pulse brackets ingestion of a complete, already-parsed 250-sample block through the unchanged `OxygenRiskPipeline`. A decision block includes all 250 pushes, filtering, 14 features, one baseline inference, the normal 14 Mean-Perturbation passes, top-three sorting, and debounce/payload construction. It does not add the all-14 diagnostic recomputation.

The first two blocks after reset only fill the buffer. The third emits the first decision at sample 750; later decisions follow every 250 samples. The client reports warmup blocks, first decisions after reset, and subsequent decisions separately. This follows the frozen five-second stride/context contract; it does not use the older two-second stride wording in `CONTEXT.md`.

The ARM DWT cycle counter is sampled inside the D2 pulse. Counter reads and barriers add overhead; the GPIO pulse also contains the counter-read bracket. Eight empty brackets are retained separately as calibration, with no subtraction from measured blocks. Unsigned subtraction handles one counter wrap. A coarse elapsed-time guard rejects intervals of 60 seconds or more, below the 32-bit wrap period at 64 MHz. Firmware checks that the counter advances and the core clock is 64 MHz before accepting timing commands. See the [Cortex-M4 reference manual](https://documentation-service.arm.com/static/5f19da2a20b7cf4bc524d99a).

USB parsing, output formatting, memory queries and reset are outside the timed interval. Interrupts remain enabled, so observed execution includes interrupt/preemption effects. These accelerated synthetic compute timings do not establish sensor acquisition, BLE transmission, the five-second future-context wait, or live end-to-end latency.

## Memory and flash scope

The initial linker estimates are **336,588 bytes flash** and **85,304 bytes static RAM**. These are build sizes, not runtime memory measurements or the normal sensor/BLE image's footprint.

The `M` command samples two separate quantities:

- Current Arduino loop-thread stack reservation and peak used space from the RTX watermark, since thread creation. The built RTX configuration has watermark support enabled; firmware also checks the flag at runtime. Unsupported watermark results are reported as unavailable. Resetting the pipeline does not reset this lifetime watermark. Startup, USB and reporting can contribute. The API's meaning is documented in [CMSIS thread management](https://arm-software.github.io/CMSIS_6/main/RTOS2/group__CMSIS__RTOS__ThreadMgmt.html).
- Newlib heap arena, allocated bytes and free bytes inside that arena at query time, using [mallinfo](https://sourceware.org/newlib/libc.html). These are snapshots, not transient heap peaks or all free device RAM.

The client samples memory before replay and after each case. It does not add heap and stack values into a total: thread stacks may already be represented in allocator usage. Other-thread stacks, ISR-stack peaks and whole-device peak RAM remain unmeasured.

## Protocol and reproduction

Build only, from `firmware/Arduino Nano 33`:

```powershell
pio run -e nano33ble_context_timing
```

The default PlatformIO environment remains the existing sensor image. Always name the timing environment when building this diagnostic. No upload command is part of the replay client.

Firmware identity:

```text
TIMING_REPLAY_V1 WINDOW=500 CONTEXT=250 STRIDE=250 N=2 THRESHOLD=0.50 CLOCK_HZ=64000000 GPIO=D2 ATTR14=0
```

`H` identifies, `R` resets, `C` measures an empty bracket, `M` reports memory, and `B` accepts exactly 250 finite FP32 values as eight-digit hexadecimal words. A full block is staged before ingestion. `A` and malformed blocks are rejected without changing pipeline state. Responses are `TIMED cycles elapsed_ms BUFFER total` during warmup, or `TIMED cycles elapsed_ms RESULT ...` using the existing 29-field decision payload. `CAL cycles elapsed_ms` and `MEMORY stack_reserved stack_peak_used heap_arena heap_used heap_free` are separate responses; `stack_peak_used=-1` means unavailable.

Host-only preparation, from the repository root with the configured NumPy/SciPy Python:

```powershell
python scripts/benchmark_context_board.py --probe build/parity/audit-release/context_probe.exe --build-receipt outputs/parity/qualification/timing-image-v1/build-receipt.json --output outputs/parity/timing-prepared-v1
```

Outputs must be fresh directories under ignored `outputs/`. Preparation checks the frozen candidate/contract, source and BIN hashes, then independently checks 21 synthetic decisions against SciPy. Each feature's original absolute error, original 1e-4 result and frozen local budget remain in the report. It records no measured latency without a board run.

After authorization for this exact archived image, upload and flash verification must produce a separate receipt with `uploaded_and_verified: true`, `image_sha256` matching the build receipt and `candidate_sha256` matching the frozen candidate. A subsequent replay adds `--port <current-port> --upload-receipt <receipt-path>` and uses a fresh output directory. The client checks that receipt and the firmware identity before sending any reset or sample commands. The handshake identifies the protocol; the separate flash-verification receipt establishes the image provenance.

Default replay is ten repetitions of the three synthetic streams, or 210 decision blocks. It records all raw cycles, calibration, memory snapshots and functional comparisons. Original absolute feature errors remain visible even when adopted checks pass. Only the transmitted top-three attributions are observed; this timing image does not repeat the full all-14 qualification. The report leaves `parity_complete` and `physical_gpio_verified` false.

For the physical cross-check, use a high-impedance oscilloscope or logic-analyzer input on **D2**, with its ground connected to **GND**. D2 maps to nRF52840 P1.11 in the installed Nano variant. Record corresponding pulse widths and DWT results, including the empty bracket. DWT/GPIO agreement has not yet been measured; instrument availability was requested from the user.

## Verification before upload

All **13/13 CTests pass**. The added native test exercises the real frozen pipeline through a simulated clock/GPIO boundary, checking exact decisions over ring wrap and reset plus counter wrap. Six Python protocol tests cover decoding, invalid/stale timing, memory scope, wrong firmware, complete replay/partial failure evidence, and rejection of a build receipt that attempts to override a frozen source digest. These are host tests, not board measurements.

The bounded review found one provenance defect: merging frozen and build hash maps allowed a build entry to replace a frozen expected hash. The runner now verifies both maps independently; the reproducer failed before that fix and passes after it. No other actionable issue was reported. A dedicated Bugbot API was unavailable, so the review used a general review subagent.

The locked validation set was not replayed or used to select a change. No weights, thresholds, numerical budgets or frozen pipeline sources were changed. Physical GPIO/clock verification, complete runtime memory qualification, live still/wiggle/walk captures, AOPC, PTT-PPG and the unmet recall/specificity gate remain outstanding; Android remains on hold. The board now runs the timing diagnostic, with no live sensor/BLE.
