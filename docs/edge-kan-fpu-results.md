# Edge KAN with the hardware FPU: build checks and board results (19 Nov 2026)

Results of `docs/module8-review-fpu-plan.md` (Module 8 review M11). All measurements were taken on the Arduino Nano 33 BLE Sense Rev2 (nRF52840, Cortex-M4F at nominal 64 MHz, COM8) using the hardware-FPU firmware builds. Aggregate numbers, build-check metadata and disassembly counts are stored in `outputs/edge-kan-fpu/summary.json`.

## Why an opt-in build was needed

PlatformIO's mbed builder (`arduino-core-mbed.py`, `configure_fpu_flags`) appends `-mfloat-abi=soft` after the Arduino Nano 33 BLE variant's `-mfloat-abi=softfp -mfpu=fpv4-sp-d16`, so every default and archived image in the repository ran single-precision floating point in software (`__aeabi_f*` helper calls).

- `firmware/Arduino Nano 33/fpu_flags.py` strips `-mfloat-abi=soft` and appends `-mfloat-abi=softfp -mfpu=fpv4-sp-d16` across the main, project and library build environments (`softfp` retains the software calling convention so precompiled mbed core libraries still link).
- `firmware/Arduino Nano 33/platformio.ini` exposes three opt-in environments (`nano33ble_ekan_context_parity_fpu`, `nano33ble_ekan_pipeline_timing_fpu`, `nano33ble_ekan_context_timing_fpu`), each extending its soft-float archive counterpart with `extra_scripts = post:fpu_flags.py`. Default and archived soft-float environments remain untouched.

## Build and disassembly verification (`scripts/archive_ekan_fpu_build.py`)

Both timing images passed every build-log and ELF disassembly check before flashing (`outputs/edge-kan-fpu/summary.json`):

| Image env | `firmware.bin` SHA-256 | Flash (vs soft float) | Static RAM | Compile/link lines checked (`-mfloat-abi=soft`) | Project VFP f32 instrs / soft calls | Soft-float archive VFP / soft calls |
|---|---|---:|---:|---:|---:|---:|
| `nano33ble_ekan_pipeline_timing_fpu` | `b4d5dc6e269def3a57ec2fa1dc221fc018a14123c0ee4f257806a0cf1a645a21` | 340,660 B (−2,112 B vs 342,772 B) | 92,320 B (±0) | 69 / 69 (0 soft) | **125** / **0** (8 fns) | 0 / 116 (`d1cccebe…`) |
| `nano33ble_ekan_context_timing_fpu` | `1e5eec76551367226e74305d34ebaee9a6e3a9ee9c3510aa7b81b9b065038809` | 343,084 B (−2,136 B vs 345,220 B) | 92,320 B (±0) | 69 / 69 (0 soft) | **121** / **0** (9 fns) | 0 / 113 (`cc5e724d…`) |

## Integrated pipeline replay (`replay_edge_kan_board.py pipeline --fpu`, COM8)

Pre-registration SHA-256 recorded in the run: `c989aeedad475aa58957215dbc7c0f54d2d65aecd72b4aa42f39a2a595c55449` (for public release, `docs/module8-review-fpu-plan.md` replaces the author's local Python interpreter path with `PY="python"`).

- **Functional gate: pass, 1,243 of 1,243 decisions.** Zero failures across all seven checks (`meta`, `features`, `probability`, `states`, `top3_order`, `top3_deltas`, `valid`), with `0` timed decisions failed. Board counts: `642` candidates and `308` confirmed; maximum host probability error `5.66e-7` (`5.662e-7`); top-3 set and order identical to the host on every decision.
- **Comparison against the 12 Nov soft-float board run (`board-pipeline-20261112`):** `506` of `1,243` probabilities are bit-identical; maximum `|ΔP|` is `5.96e-7` (`5.960e-7`); **0** candidate-flag differences at 0.50; **0** confirmed-state differences; **0** top-3 differences.

**Pipeline block timing** (10 fixture repetitions, nominal 64 MHz DWT cycle counter, empty-bracket calibration max 6 cycles):

| Decision-block phase | Count | Hardware FPU | Soft float (`2026-11-12`) | Notes |
|---|---:|---:|---:|---|
| Subsequent decisions, median | 180 | **19.71 ms** (`19,709.46 µs`) | 433.9 ms | **22.0× speedup**; 0.39% of 5 s stride |
| Subsequent decisions, p95 | 180 | **19.77 ms** (`19,770.43 µs`) | 435.4 ms | FIFO headroom **32.33×** (vs 640 ms) |
| Subsequent decisions, min / max | 180 | `19.55 ms` / `19.80 ms` (`19,795.41 µs`) | — / 435.7 ms | |
| First decision after reset, median | 30 | **16.81 ms** (`16,806.02 µs`) | 350.2 ms | p95 `16.90 ms`, max `16.95 ms` |
| Warmup blocks, median / p95 / max | 60 | `203.84 µs` / `231.52 µs` / `247.48 µs` | — | |

## Kernel split (`replay_edge_kan_board.py timing --fpu`, COM8)

210 timed fixture decisions (10 repetitions, 5 board repetitions per kernel, 0 checks failed, calibration max 6 cycles):

| Kernel / derived block (min over 5 reps) | Hardware FPU median (max) | Soft float median (`2026-11-12`) | Speedup |
|---|---:|---:|---:|
| Edge KAN forward | **0.49 ms** (`488.05 µs`, max `506.58 µs`) | 4.32 ms | 8.85× |
| Edge KAN incremental XAI (`P` + 16 deltas) | **3.00 ms** (`3,001.43 µs`, max `3,005.88 µs`) | 24.52 ms | 8.17× |
| TinyKAN forward | `148.53 µs` (max `150.34 µs`) | 1.37 ms | — |
| TinyKAN 17-pass XAI | `2.55 ms` (`2,553.20 µs`, max `2,605.14 µs`) | 22.60 ms | — |
| Derived edge decision block | `19.62 ms` (`19,618.36 µs`, max `19,743.81 µs`) | 451.3 ms | — |

## Scope and limits

- Both runs replayed the prepared OpenOximetry set over USB with BLE disabled and the training-mean explainer reference, using cycle-derived timing at the nominal 64 MHz clock.
- Live-sensor FPU capture, FPU peak RAM, battery energy draw, and external GPIO timing remain unmeasured.
- Per-window board replay reports are withheld under the PhysioNet Restricted Health Data License; aggregate metrics and build checks are in `outputs/edge-kan-fpu/summary.json`.
