# Technical remaining review

Corrected 8 October 2026 from disk: the live-capture and FIFO rows below were updated from `outputs/live/` and `main.cpp`; the 11 September COM8 identity lines were not re-read and are not a claim about the current board.

Re-verified 11 September 2026, 13:14 MYT (locked-set **board** replay). Do not `git reset --hard` (or checkout/delete) the dirty parity/timing worktree. Host AOPC is write-once (`openox-aopc-v1`, SHA `3524d43ae47ed82216f3b8cefbd5f8dca37dabdc01722deb52d5a067dd7bf74c`); PTT subject-1 is write-once (`outputs/ptt-ppg/subject-1-v1/`, `report.json` SHA `fed427028bef91002d512453c9b198e700c2911f32ef7abba913c970f18767b8`); locked-board replay is write-once (`outputs/parity/qualification/locked-board-replay-v1/`, `report.json` SHA `7634173107357b9040684ed5b85f9d777e451ae9b25918ab085c1a9f94d34d9a`). Do not retune rankings, weights, 0.50, or N=2; firmware XAI is not newly faithful. Frozen weights, threshold 0.50, ADR-0005 N=2, native FP32, and the adopted numerical contract were not changed. `parity_complete` remains false. `physical_gpio_verified` remains false.

**Still incomplete:** physical D2/GPIO vs DWT. That needs explicit authorization plus COM8/flash/instrument. A baseline-model walk capture now exists (29 Oct, second module, [sidecar](../outputs/live-v2/2026-10-29_v1_walk-sensor2.sidecar.md)); still and wiggle logs are on disk at `outputs/live/2026-10-07_still.txt` and `outputs/live/2026-10-07_wiggle.txt`. Do not label numerical parity complete. Host AOPC is not firmware or clinical faithfulness. PTT subject-1 is not live still/wiggle/walk and not a 22-subject cohort. Locked-set board replay is sparse USB, not consecutive-stream or live.

Last documented COM8 identity (read-only `H` on 11 Sep, after the locked replay finished; not re-read on 8 Oct):

```text
CONTEXT_PARITY_V1 WINDOW=500 CONTEXT=250 STRIDE=250 N=2 THRESHOLD=0.50
```

The 11 Sep read, not repeated on 8 Oct, showed the frozen all-14 diagnostic (image SHA `30d390c7ad19cc335f8453124e14fbeef7986bd9bb258c5a60dde27fd76b9193`) and said the timing BIN was no longer installed and the sensor/BLE image was not restored. That is not a current identity. The 7 Oct still and wiggle logs required a sensor image that evening. Timing receipts remain at `outputs/parity/qualification/timing-board-upload-v1/` and `outputs/parity/timing-board-v1/`.

| Item | Status | Blocker |
|---|---|---|
| Host numerical checks (adopted local-scale rule) | Pass on 2,557 development + 2,464 locked validation windows | Original absolute 1e-4 still fails; `parity_complete` remains false |
| Locked-set **board** replay | Complete: [§1](#1-numerical-parity) | Sparse USB ≠ consecutive stream; live/GPIO/RAM still outside this gate |
| 5 s context vs full-recording | Documented; max ΔP ≈ 0.01238; 0 threshold flips in that sample | Sparse contexts ≠ consecutive-stream agreement |
| Physical GPIO / DWT | Cycle-derived USB timings exist | No oscilloscope/logic analyzer; `physical_gpio_verified` false; timing image not currently on the board |
| Runtime memory | **Sampler probe run, 2 Nov (finger on, dashboard connected; [SESSION.md](../outputs/peak-ram-sampler-board-20261102/SESSION.md)):** peak RAM between 80,552 B and 81,392 B (v1) and between 86,576 B and 87,348 B (v2 f1a) on the probe image; both runs pass every pre-registered check (v1 on its one allowed rerun). Earlier: for the 25 Oct probe run (probe image, no finger, forced decisions, no central) a derived bracket puts it between 79,156 B and 80,660 B (v1) and between 85,220 B and 86,644 B (v2 f1a) ([bracket](../outputs/peak-ram-bracket-v1/README.md), [SESSION.md](../outputs/peak-ram-board-20261025/SESSION.md)). Timing-image stack watermark + mallinfo snapshots are separate | Probe image, not the unmodified live image (the sampler image adds 1,440 B static RAM; the live image's own peak needs an SWD debug probe); instants between 250 µs sampler ticks and inside interrupt handlers not seen; one wearer, one module, one session; timing-image snapshots are not whole-device peak RAM |
| Live still / wiggle / walk | Still and wiggle logs on disk: `outputs/live/2026-10-07_still.txt`, `outputs/live/2026-10-07_wiggle.txt`. Walk (29 Oct, v1 image, second module): rate 49.98–50.09 Hz; walking 22 decisions, after the first four 15/17 below 0.50, 0 Confirmed ([sidecar](../outputs/live-v2/2026-10-29_v1_walk-sensor2.sidecar.md)) | Walk cue timing not followed, finger-off reset, dashboard connected mid-walk (all recorded). FIFO defect fixed: `main.cpp` sampleRate 200 / sampleAverage 4; the 7 Oct logs measured 49.97 Hz. Author only, oxygen not varied; not live validation; `parity_complete` remains false |
| Bluetooth central connected | 29 Oct, shipped v2 f1a-r2, second module, dashboard connected 180 s: 0 disconnects, 50.07–50.10 Hz, 34/35 decisions 0.08–0.10, 0 Candidate ([sidecar](../outputs/live-v2/2026-10-29_f1a-r2_dash-sensor2.sidecar.md)) | Payload not logged or compared: delivery and rate only, not numerical agreement over Bluetooth |
| AOPC (ADR-0003) | Host run complete: [xai-faithfulness-aopc.md](xai-faithfulness-aopc.md) | In-sample masking only; firmware \|Δ\| ranking ≠ signed ranking; not board XAI |
| PTT-PPG sit/walk/run | Host subject-1 spike complete: [ptt-ppg-motion-spike.md](ptt-ppg-motion-spike.md) | One subject; not board P; not live; no window SpO2 labels |

Details and numbers live in the linked docs. This file maps remaining work; it does not replace those reports.

### This session (disk only)

| Check | Result |
|---|---|
| `outputs/parity/qualification/locked-board-upload-v1/receipt.json` | `uploaded_and_verified: true`, image SHA `30d390c7…9193`, receipt SHA `8a64cee3187234690d32af50399f861e5491d9edc3d6caf31c2880d393e97f6a` |
| `outputs/parity/qualification/locked-board-replay-v1/report.json` | `status: completed`, `board_run_completed: true`, `parity_complete: false`, **2464 / 2464** adopted-check pass, **0** candidate disagreements, original absolute 1e-4 failed **2464 / 2464**, max \|ΔP\| **8.252e-7**, max \|Δattr\| **1.580e-6** (matches host locked-validation maxima) |
| `outputs/parity/qualification/locked-board-prepared-v1/report.json` | Historical host preflight; leave write-once. Not this board run. |
| `outputs/parity/timing-board-v1/report.json` | `status: completed`, `parity_complete: false`, `physical_gpio_verified: false`, `all_14_attributions_observed: false` |
| Subsequent-decision DWT median | **249.702 ms** (180 blocks). Do not substitute pooled `all_decisions_ms.median` **249.542 ms** (210 first+subsequent). |
| `outputs/live/` | Absent on 11 Sep. Checked 8 Oct 2026: `2026-10-07_still.txt` and `2026-10-07_wiggle.txt` present; no walk log |
| `outputs/ptt_ppg_motion_spike.csv` | Present; SHA-256 `27a91e01ede1c675e8a4db7268ba094fb6c646dc3c189cf976c2b7b8f95d4734` (same as `outputs/ptt-ppg/subject-1-v1/activity_summary.csv`) |
| `data/physionet.org/files/pulse-transit-time-ppg/1.1.0/` | Subject 1 sit/walk/run `.hea`+`.dat` present and checksum-matched; subjects 2–22 not fetched |
| `outputs/openox_operating_point_sweep.csv` | **0 / 364** rows `feasible=True` |
| TinyKAN window @ **0.50** | recall **0.7668**, spec **0.3941** |
| LogReg window @ **0.50** (sweep table) | recall **0.8589**, spec **0.2637** |

Close any serial monitor on COM8 before a later authorized upload. Do not retune TinyKAN to beat LogReg. Do not overwrite the locked-board replay directory.

## 1. Numerical parity

**Evidence.** Adopted rule: [feature-parity-criterion-proposal.md](feature-parity-criterion-proposal.md). Lock and first host use: [numerical-validation-lock.md](numerical-validation-lock.md). Repair and earlier board-scope results: [fp32-filter-repair-validation.md](fp32-filter-repair-validation.md).

The frozen candidate (`1cfd4ed28b5aad114d1064a1a7323673b8725e434700e7b6303b5009ace74d0b`) passed all 2,464 locked windows on the host after freeze. Authorized Nano work: 21-decision synthetic all-14 replay, sparse recorded **development** replay (2,557 windows), then sparse locked **validation** board replay (2,464 windows) on 11 September 2026, 13:14 MYT. Original uniform absolute 1e-4 feature comparisons still fail on essentially every window and remain recorded. Timing-image replay observed only top-three attributions (`all_14_attributions_observed: false`).

Locked-set board replay: `outputs/parity/qualification/locked-board-replay-v1/report.json` SHA-256 `7634173107357b9040684ed5b85f9d777e451ae9b25918ab085c1a9f94d34d9a`. Upload receipt `outputs/parity/qualification/locked-board-upload-v1/receipt.json` SHA `8a64cee3187234690d32af50399f861e5491d9edc3d6caf31c2880d393e97f6a`. Frozen image SHA `30d390c7ad19cc335f8453124e14fbeef7986bd9bb258c5a60dde27fd76b9193`. Runner `scripts/qualify_validation_board.py`. **2464 / 2464** windows pass the adopted checks; candidate disagreements **0**; timing-state failures **0**; original absolute 1e-4 failed **2464 / 2464**. Max probability absolute error **8.251690626e-7**; max signed attribution absolute error **1.579872946e-6**. `parity_complete` is **false**. Host preflight `locked-board-prepared-v1` remains `prepared_not_run` and was not overwritten.

**v2 board parity (15 Oct 2026, ADR-0006).** Image `nano33ble_v2_context_parity`, report `outputs/parity/v2-board-20261015/report.json` (SHA-256 `f732137d654917a690fcb2976fe568895d69cd30848e6b38a21f0be89523e1bb`). **1,241 / 1,243** decisions pass: 21/21 synthetic, and 1,220/1,222 from the 611 locked recorded contexts. The 2 failures are locked cases 379 and 380 on board-vs-Python attribution (1.28e-5 and 1.09e-5 against 1e-5), as the host run predicted. Both pass board-vs-host, with the top-3 unchanged. Max board-vs-host error: probability 8.94e-7, delta 9.54e-7. `parity_complete` is false. The v2 live sit has not been run.

**v2 board timing (15 Oct 2026, ADR-0006).** Report `outputs/timing-v2-board-20261015/report.json` (SHA-256 `7cb61465532e9d49187af3b9a32d2ec473aadad58789cdd0f1846a63b917a5a1`). Subsequent-decision whole 250-sample block, cycle-derived at 64 MHz: median **433.4 ms**, max **435.4 ms** (n = 180; v1 249.7 / 250.8). The first decision after a reset has a median of 346.5 ms. FIFO headroom is **1.47x** against 640 ms, so the stop rule is not triggered. All 210 functional checks pass. This is synthetic USB replay; `physical_gpio_verified` is false.

**v2 live sit (15 Oct 2026, ADR-0006): stopped on the rate rule.** In `outputs/live-v2/2026-10-15_v2_still.txt`, `Hz:` falls to a median of 47.56 and decisions arrive every 5.31 s, an effective rate of about 47.1 Hz. The image does not detect overflow, so samples are assumed lost, and the recover capture was not run. 54 of 64 decisions are Confirmed at 0.89 on a healthy seated subject, with `ratio_of_ratios` as the top driver. These are false positives or a shift between the training data and this sensor, and the windows are not clean 50 Hz windows. The board is on the v2 live image.

**Blocker.** Sparse per-window replay cannot establish consecutive-state agreement over a full stream. Passing the locked set does not complete parity: live sensor/BLE, physical GPIO, and RAM remain outside this gate. Do not raise frozen budgets.

**Safe next action.** Do not rerun into `locked-board-replay-v1`. Do not use the original-absolute 1e-4 failures to choose another repair. Do not restore the sensor image as a substitute for remaining hardware work.

## 2. Five-second context vs full-recording filtering

**Evidence.** [training-context-decision.md](training-context-decision.md). Expanded host audit: max probability difference **0.01238**, one window ≥ 0.01, **zero** 0.50 candidate flips. Validation-set finite-context vs full-recording max is **0.00854544** ([fp32-filter-repair-validation.md](fp32-filter-repair-validation.md)). This is filtfilt context, not native vs SciPy arithmetic.

**Blocker.** Sparse recorded contexts do not prove consecutive-state agreement on a continuous stream. The gap must not be absorbed into numerical tolerances.

**Safe next action.** No firmware or budget change. If a later decision wants stream-level evidence, that is a separately authorized consecutive-recording replay, not a retune.

## 3. Normal sensor/BLE image restoration

**Evidence.** Default PlatformIO env is still `nano33ble`. The 11 Sep read showed `nano33ble_context_parity`. That identity was not re-read on 8 Oct, so it is not the current board. The 7 Oct still and wiggle logs required a sensor image that evening; they do not identify what is installed now. Timing receipts are historical ([on-device-timing.md](on-device-timing.md)). `outputs/live/` holds `2026-10-07_still.txt` and `2026-10-07_wiggle.txt`; there is no walk log.

**Audit finding (11 Sep; since fixed).** On 11 Sep, `firmware/Arduino Nano 33/src/main.cpp` set `sampleAverage = 4` and `sampleRate = 50` while commenting “50 Hz — matches ML model training parameters”. On MAX30105, FIFO output rate is sample rate / sample average, so that configuration was **~12.5 FIFO samples/s**. `main.cpp` now sets `sampleRate = 200` with `sampleAverage = 4`, and the 7 Oct logs measured **49.97 Hz**. On 11 Sep the loop used `getRed()` / `getIR()` / `getGreen()`, and those getters each blocked on a new record. The current `main.cpp` does not call those getters in the ingest loop; it drains FIFO records over Wire (`readFifoValue`). The 7 Oct logs measured 49.97 Hz on that path.

**Blocker.** The FIFO-rate blocker is closed in source (sampleRate 200 / sampleAverage 4) and was measured at 49.97 Hz on 7 Oct. Do not treat the 11 Sep diagnostic image as the image that produced those logs, and do not treat either as the current COM8 identity.

**Safe next action.** Do not flash until explicitly authorized. The FIFO-rate repair is already in `main.cpp` and was measured at 49.97 Hz on 7 Oct. Current COM8 identity was not re-read on 8 Oct. Close any serial monitor on COM8 before a later authorized upload.

## 4. GPIO / DWT timing

**Evidence.** [on-device-timing.md](on-device-timing.md); report `outputs/parity/timing-board-v1/report.json` (`status: completed`, `physical_gpio_verified: false`). Ten repetitions × three synthetic streams = 210 decision blocks (21 distinct decisions repeated). Subsequent-decision DWT-derived median **249.702 ms** (max 250.797 ms) at declared 64 MHz. `millis()` is 1.612–4.060 ms shorter. Empty-bracket calibration was **not** subtracted. Accelerated USB synthetic input; interrupts enabled; not sensor/BLE or end-to-end latency.

**Blocker.** No oscilloscope or logic analyzer is available. D2 is nRF52840 P1.11 vs GND. Until that pulse is measured independently, do not call the cycle-derived milliseconds physically calibrated, and do not rescale either clock. Whether the archived timing BIN (`7db771c18ef99fb087933b638ab399bef6f49e74425937e19311374209bbe633`) is installed now was not re-read on 8 Oct. The 11 Sep read is not a current identity.

**Safe next action.** Requires an instrument and explicit authorization. Re-flash the archived timing BIN only if that job is named, then record D2 pulse widths against the same DWT brackets, including the empty calibration pulse.

## 5. Runtime memory

**Evidence.** Same timing report / [on-device-timing.md](on-device-timing.md). All 31 snapshots: stack reservation 32,768 B, watermark used **31,308 B**, newlib heap arena/used **1,040 B**, 0 B free in that arena. Watermark was already 31,308 **before** the first benchmark block and did not change. Linker sizes for the timing image (flash 336,588 B, static RAM 85,304 B) are not runtime peaks. No fixed ~30 kB pipeline stack frame was found that would explain the pre-replay watermark.

**Blocker.** RTX watermark is lifetime/thread-scope; mallinfo is allocator-arena snapshot. Do not add heap + stack. These snapshots do not cover ISR or other-thread stacks (the probe images below do), and the pre-replay watermark is still not diagnosed.

**25 Oct probe run ([SESSION.md](../outputs/peak-ram-board-20261025/SESSION.md)); bracket derived 30 Oct ([outputs/peak-ram-bracket-v1/](../outputs/peak-ram-bracket-v1/README.md)).** Probe images from `outputs/firmware-peakram-prepared/` flashed per [docs/peak-ram-runbook.md](peak-ram-runbook.md); live v2 f1a-r2 restored after. That run measured `whole_hw`, the RAM high-water (a union of touched addresses over the run), not the peak. Wording: *"whole-device peak RAM is not measured; for the 25 Oct probe run (probe image, no finger, forced decisions, no central) a derived bracket puts it between 79,156 B and 80,660 B (v1, 30.2–30.8% of 256 KB) and between 85,220 B and 86,644 B (v2 f1a, 32.5–33.1%); the upper end is the RAM high-water"*. The lower end assumes no heap block was freed and re-allocated between two 5 s reports; the reports support that but do not prove it (method: [peak-ram-design.md §7](peak-ram-design.md#7-peak-versus-high-water-added-30-oct-2026)). Main thread peak about 4.6 kB of 32,768 B on both; the 31,308 B watermark above came from the USB timing image, not a live image. Do not sum region peaks; no `KAN_Prob`/`Candidate` value from those captures. Do not treat `M` command output as peak RAM in the paper/report.

**Sampler run, 2 Nov 2026 ([SESSION.md](../outputs/peak-ram-sampler-board-20261102/SESSION.md)).** r2 sampler images (`outputs/firmware-peakram-sampler-prepared-r2/`, v1 `90894011…`, v2 `bd3a494d…`; the 31 Oct images never started their sampler on 1 Nov) flashed per [peak-ram-runbook.md](peak-ram-runbook.md), sampler sections; f1a-r2 restored after. Sensor module 2, lid on, finger on and dashboard connected through each 180 s capture. Validity checks pre-registered before the firmware (`f782745`), unchanged. v2 f1a valid on its first capture; v1's first capture failed `central` (no central connected) and is kept; its one allowed rerun is valid. Wording: *"peak RAM between 80,552 B and 81,392 B (v1, 30.7–31.0% of 256 KB) and between 86,576 B and 87,348 B (v2 f1a, 33.0–33.3%) on the probe image, finger on, dashboard connected"*. Lower end: the largest instantaneous total the 4 kHz sampler saw in the capture; upper end: `whole_hw`, the RAM high-water. **Not covered:** the unmodified live image (the sampler image adds 1,440 B static RAM plus its own frames; its peak needs an SWD debug probe), other workloads (connect/disconnect, finger placement), and instants between ticks or inside interrupt handlers. Do not quote `KAN_Prob`/`Candidate` from those captures.

## 6. Live still / wiggle / walk

**Evidence.** Skill/repo convention: `outputs/live/YYYY-MM-DD_<condition>.txt` from the **sensor** image on COM8 @ 115200. That folder now holds `2026-10-07_still.txt` and `2026-10-07_wiggle.txt`; there is no walk log. Those two logs are author only, oxygen was not varied, and they are not live validation. A 2026-09-01 still-finger serial demonstration (P≈0.39, kurtosis, Candidate NO) is not a substitute for `outputs/live/` and is not live validation. Accelerated USB replay is not still/wiggle/walk.

**Blocker.** Walk is not captured. Current COM8 identity was not re-read on 8 Oct, so do not assume a sensor image is installed now. The FIFO-rate defect is fixed in source and was measured at 49.97 Hz on 7 Oct. Finger-gate 3000 / LED 0x3F are the live firmware settings; an old 50000-gate binary is a different failure mode.

**Safe next action.** Under a named, authorized live-capture job, capture the missing walk file with serial throttling still in place. `NO FINGER` / `CLIPPED` voids the run. Outcome columns are `KAN_Prob`, `Candidate`, `Confirmed` — not `SpO2_est`. Close the serial monitor before any later upload.

## 7. AOPC (ADR-0003)

**Evidence.** Host evaluation 11 September 2026, 11:03 MYT: [xai-faithfulness-aopc.md](xai-faithfulness-aopc.md). Output `outputs/xai-faithfulness/openox-aopc-v1/` (`summary.json` SHA-256 `3524d43ae47ed82216f3b8cefbd5f8dca37dabdc01722deb52d5a067dd7bf74c`). Tests `tests/test_xai_faithfulness.py` **7/7**. Decision record: [0003-xai-faithfulness-evaluation-via-deletion-auc.md](adr/0003-xai-faithfulness-evaluation-via-deletion-auc.md).

All **52,090** windows. Deletion AOPC / insertion AOPC / deletion AUC / insertion AUC kept separate. Window means:

| Ranking | Deletion AOPC | Insertion AOPC | Deletion AUC | Insertion AUC |
|---|---:|---:|---:|---:|
| signed_risk | 0.1246 | 0.1283 | 0.3796 | 0.5788 |
| firmware_absolute | 0.0443 | 0.0444 | 0.4599 | 0.4949 |
| random_mean | 0.0283 | 0.0254 | 0.4759 | 0.4759 |

Firmware-absolute (on-device sort) is only modestly above random on all windows (77.6% better than own random mean). On exported-model candidates (P≥0.50, n=34,127) it is clearly above random (deletion AOPC **0.0958** vs **0.0576**; 98.3% better). Signed ranking is strongly above random. Native probe on 211 windows: max P error **1.53e-7**.

**Blocker.** None for this host job. Remaining limits are interpretive, not missing files: in-sample exported model; full 14-feature order not the top-3 BLE payload; not board replay.

**Safe next action.** Do not rerun into the filled directory. Do not cite firmware XAI as newly faithful. Do not retune rankings or weights from these curves.

## 8. PTT-PPG sit/walk/run

**Evidence.** 11 September 2026, 11:29 MYT: [ptt-ppg-motion-spike.md](ptt-ppg-motion-spike.md). Output `outputs/ptt-ppg/subject-1-v1/` (`report.json` SHA-256 `fed427028bef91002d512453c9b198e700c2911f32ef7abba913c970f18767b8`). Success CSV `outputs/ptt_ppg_motion_spike.csv`. Tests `tests/test_ptt_ppg.py` **7/7**. Plan: `.hermes/plans/2026-09-04_231701-ptt-ppg-motion-spike.md`.

Subject 1 only (`s1_sit` / `s1_walk` / `s1_run`), channel `pleth_2`, frozen TinyKAN @ **0.50**, no training. **294** windows. Sit/walk/run mean P **0.5522 / 0.5759 / 0.5378**; candidate fractions **0.990 / 0.979 / 0.990**. Header SpO2 stays ≥96%. Top-1 is mostly kurtosis; walk also shows `bandpower_ratio_0_5_2_to_2_5` (24/97). IMU dynamic RMS sit **0.023** vs walk **2.816** vs run **2.323** (context only).

**Blocker.** None for this host spike. Remaining limits are interpretive: one subject; NumPy twin not board; no window labels; not live still/wiggle/walk; 1.1.0 is not fully fetched.

**Safe next action.** Do not rerun into `subject-1-v1` or overwrite `outputs/ptt_ppg_motion_spike.csv`. Do not treat candidate rate as hypoxia or motion robustness. A 2–22 fetch, native `probability_probe`, or another channel is a separately named job. Hardware work is still authorization-gated.

## Out of this remaining list (do not reopen)

- Operating point stays **0.50**; do not retune. For **v1** (frozen 14-IR `(14, 12, 1)`, notebook-10 labels), ETPB recall≥0.80 AND spec≥0.50 is not achievable ([operating-point-calibration.md](operating-point-calibration.md)). The hunt is closed: 0 of 364 v1 sweep cuts meet both gates. **v2** (header-aligned labels, 16 inputs, ADR-0006) meets it offline at the same 0.50: recall 0.893 / spec 0.868 recording-grouped, 0.899 / 0.864 patient-grouped, all 5 folds in both (`outputs/analysis-lamaakal-style-v1/c_bootstrap_ci.csv`, `c_per_fold.csv`). Limits: window level, not N = 2 confirmed; offline header-aligned ScalcO2 labels, not exact; v2 was built after v1 failed; live recall unmeasured.
- Native FP32 only (no quantise).
- Android pager remains ON HOLD.
- Do not retune TinyKAN to beat LogReg unless Sheena asks. FYRP A already compared them at 0.50 (LogReg recall 85.9% vs TinyKAN 76.7%). TinyKAN remains the on-device KAN + Mean-Perturbation object.
- Do not rerun locked-set board replay into `locked-board-replay-v1`.

## Documentation that is not current evidence

Do not edit these unless asked. They are not remaining experimental jobs.

- [training-context-decision.md](training-context-decision.md) still says COM8 “now runs” `nano33ble_context_parity` from 10 September, then timing superseded it. The 11 Sep text said COM8 ran the all-14 identity after the locked replay. That line was not re-read on 8 Oct and is not a current identity claim.
- [fp32-filter-repair-validation.md](fp32-filter-repair-validation.md) § timing image said the board ran the timing diagnostic; that was true until the locked-replay flash. Locked-validation board numbers live here and in `locked-board-replay-v1/`, not in that older write-up.
- [on-device-timing.md](on-device-timing.md) describes the timing image and receipts; those receipts still stand. Whether that BIN is installed now was not re-read on 8 Oct.
- `CONTEXT.md` Sliding Stride is already 5.0 s (250 samples at 50 Hz), with 5 s future context. Do not reopen that as remaining hardware work.
- `github-pages-dashboard/` was rebuilt 16 Sep 2026 for the assumed field list (score / candidate / confirmed / glossed top-3; toy SpO₂ hidden). Saved examples are gone. Do not assume which image is installed, and do not flash unless that job is named.
- `outputs/openox_tinykan_summary.csv` matches the **0.30** TinyKAN counts (recall 0.971), not the frozen 0.50 window cut.
- ADR-0003 originally said “true low-oxygen causal drivers”. The 11:03 host run does not make that true; the ADR now points at [xai-faithfulness-aopc.md](xai-faithfulness-aopc.md).

## Worktree and session drafts (do not treat as results)

The tree was already dirty with parity/timing work. **Reset is locked:** do not `git reset --hard`, checkout, or delete those files.

Untracked drafts that are **not** completed evidence:

- `outputs/parity/qualification/locked-board-prepared-v1/` (`prepared_not_run` — host preflight only)

Completed (still gitignored outputs, not a git result): `outputs/xai-faithfulness/openox-aopc-v1/` plus [xai-faithfulness-aopc.md](xai-faithfulness-aopc.md); `outputs/ptt-ppg/subject-1-v1/` plus [ptt-ppg-motion-spike.md](ptt-ppg-motion-spike.md) and `outputs/ptt_ppg_motion_spike.csv`; `outputs/parity/qualification/locked-board-replay-v1/` plus `locked-board-upload-v1/`.

Remove session-created files only if explicitly asked. Preserve pre-existing uncommitted parity/timing edits.

## Authorization checklist for any later implementation session

1. Name the exact job (FIFO repair / sensor restore / GPIO / RAM method / live capture). Host AOPC, PTT subject-1, and locked-set board replay are done; do not overwrite `openox-aopc-v1`, `outputs/ptt-ppg/subject-1-v1/`, or `outputs/parity/qualification/locked-board-replay-v1/`.
2. Do not upload unless the user authorizes that image on the current port. Do not open COM8, flash, GPIO, RAM method, or live capture without that named job. GPIO needs the archived timing BIN back on the board.
3. Close the COM8 serial monitor before `pio` upload.
4. Do not change weights, 0.50, N=2, rankings, or frozen numerical budgets to make a check pass. Do not cite firmware XAI as newly faithful.
5. Write new outputs to a fresh ignored directory; leave write-once locks and receipts intact.
6. Do not `git reset --hard`. The FIFO rate is fixed (`main.cpp` sampleRate 200 / sampleAverage 4; 49.97 Hz in the 7 Oct logs); do not cite ~12.5 Hz as the installed rate.
