# 0006 - Header-aligned labels and a red channel: retrain TinyKAN as a 16-input model

**Status:** Accepted by Jason, 15 Oct 2026. Supervisor not yet informed.

## Context

The OpenOximetry labels behind every v1 TinyKAN result are misaligned in time. Notebook 10 shifts each PPG window by `signed_delta = ppg_start - hz2_first_row`, which is measured from the first 2 Hz *row*. It then measures the reference elapsed time from the first *non-null* ScalcO2 row. Each window therefore reads its reference later by that recording's leading-null gap. The median gap is 233.8 s and the maximum 1,519.5 s, and 128 of 162 recordings have a gap above 10 s.

Evidence:

- `outputs/label-alignment-v1/`: `README.md`, `report.json`, `per_recording.csv`, `lag_scan.csv`; script `scripts/audit_openox_label_alignment.py`.
- As coded, the exported `median_spo2` reproduces on 162/162 recordings. Header-aligned, it reproduces on 17/162.
- The red/IR ratio of ratios separates `< 90` windows at AUC 0.619 with the exported labels and 0.860 with header-aligned labels. The median within-recording Spearman is -0.266 exported and -0.840 header-aligned. The lag scan peaks at -20 s, not at the hundreds of seconds the defect implies.
- Header-aligned export: `scripts/export_openox_windows_v2.py`, writing `outputs/openox-v2/`. It has 48,662 windows from 154 recordings and 87 patients, prevalence 0.592. On the 37,505 windows shared with the frozen export, label agreement is 0.686. Windows where a Red/IR Drive channel changes are dropped (2,899).
- Recording-grouped `GroupKFold(5)` results are in `scripts/train_tinykan_v2.py` and `outputs/openox-v2/summary.json`:

| Variant | AUC | Recall | Specificity |
|---|---|---|---|
| Frozen 14-input recipe, frozen labels | 0.609 | 0.732 | 0.429 |
| Frozen 14-input recipe, v2 labels | 0.712 | 0.610 | 0.681 |
| 16-input TinyKAN, v2 labels | **0.908** | 0.893 | 0.868 |
| Same 16 inputs, logistic regression | 0.863 | 0.821 | 0.870 |
| 16-input TinyKAN, patient-grouped folds | 0.907 | 0.899 | 0.864 |

The v1 IR-only model cannot use the information that header alignment exposes. That information sits in the red/IR amplitude relation, not in the IR waveform shape alone.

## Decision

Retrain on header-aligned labels and deploy a 16-input TinyKAN. The 16 inputs are the 14 unchanged IR features plus two red/IR inputs computed on the scored 500-sample window:

- `ratio_of_ratios = (std(filtered red) / mean(raw red)) / (std(filtered IR) / mean(raw IR))`
- `ir_perfusion_index = std(filtered IR) / mean(raw IR)`

The std is population std (ddof 0). Red takes exactly the IR path: the same 50 Hz FIFO records, the same 1,000-sample context buffer, the same zero-phase filter. The architecture and recipe are unchanged except for the input count: 16 → 12 → 1, 321 parameters, weights in `kan_model_weights_v2.h` (SHA-256 `238db15a…0c574`). Threshold 0.50, N = 2 Temporal Debounce, window 500 / context 250 / stride 250, and all sensor settings are unchanged. Mean-Perturbation Attribution now runs over 16 inputs (17 forward passes).

Neither new input is an oxygen saturation value. The device still reports a Low-Oxygen Risk flag, not SpO2.

The firmware port sits behind `-DOXY_MODEL_V2` in the new envs `nano33ble_v2`, `nano33ble_v2_context_parity` (`CONTEXT_PARITY_V2`) and `nano33ble_v2_context_timing` (`TIMING_REPLAY_V2`). `nano33ble` remains the default and remains v1. All four v1 `firmware.bin` files are byte-identical before and after the change. Where the export would give NaN (a raw mean ≤ 0, or IR perfusion ≤ 0), the firmware substitutes the training means for both inputs. They then standardize to 0, carry zero attribution, and no NaN reaches the KAN.

## Consequences

**v1 evidence stays v1.** The v1 parity and validation lock, the board replays, the timing measurement, AOPC, the PTT-PPG subject-1 scoring and the 7 Oct live logs all describe the v1 IR-only model. They stay as they are. None of them transfers to v2.

**v2 evidence so far is host-only.** The host run is in `outputs/parity/v2-host/20261015T071822Z/`. It covers 154 recordings and 71,462 continuous-stream decisions, comparing compiled C++ against the Python deployed-path reference under the frozen budgets.

- **114 decisions fail.** 113 exceed the 1e-5 attribution budget (max 6.16e-5). One exceeds the `bandpower_2_5_hz` local-scale budget.
- No probability failures (max 5.75e-5), no candidate disagreements, and no failures of streaming state, metadata, top-3 payload or the validity flag.
- The diagnosis is in `…-diagnosis/report.json`. The KAN arithmetic contributes ≤ 1.28e-6. In every failing window the standardized `ir_perfusion_index` is about 13.5. The frozen v1 model on the same IR features of those windows has 0 failures.
- The budgets were not changed. Whether to accept this residual, or to change the arithmetic, is a separate decision.

**Deployed vs offline.** On the 48,625 export windows the deployed path can decide, the Candidate flag flips on 8 (0.016 %, Wilson 95 % 0.008–0.032 %). The median |ΔP| is 1.7e-6 and the maximum 0.82. This is the buffer-edge gap v1 also has: whole-recording vs 20 s buffer filtering, now on two channels.

**v2 still needs, before any v2 claim:**

1. Board parity: `CONTEXT_PARITY_V2`, 21 synthetic decisions plus the frozen 611-context recorded set in `outputs/parity/v2-board-prepared/`.
2. Timing: `TIMING_REPLAY_V2`. The v1 filter accounted for about 182 ms of the 250 ms decision block, because newlib `fmaf` is software double on this core. A second filtered channel projects to about 430 ms against the 640 ms the sensor FIFO holds. This is a projection, not a measurement.
3. Peak RAM on the live image. Static RAM is 94,032 B for v2 vs 88,008 B for v1. The deepest decision-path stack frame grows by 56 B.
4. AOPC on the v2 model.
5. A live sit.

The steps are in [v2-flash-runbook.md](../v2-flash-runbook.md).

**Training-set caveat.** The shipped v2 header is fitted on all v2 rows. Any metric computed on those rows with that header is resubstitution. Generalisation numbers come only from the grouped cross-validation above. The PTT-PPG subject-1 v2 scoring (`outputs/openox-v2/ptt-subject-1/`) remains a domain-shift check, not a hypoxia result.

## Correction, 15 Oct 2026 (host diagnosis)

The Consequences line "In every failing window the standardized `ir_perfusion_index` is about 13.5" is wrong. It holds for 96 of the 114. Those 96 are all in recording `9c01688d…`, windows 264–487, where perfusion standardizes to 13.5–13.9. The other 18 are in 18 recordings, one each. Eleven of the 18 are window 0, the first decision after a reset; there kurtosis standardizes to about 14.4 and the band-power ratio to 46–775. The rest are single windows with an artefact-scale input (for example `bandpower_ratio_0_5_2_to_2_5` at 36.7, or `std` at 29.8). In all 114, at least one standardized input lies beyond 5 training SDs. Source: `windows[*].standardized_inputs` in `outputs/parity/v2-host/20261015T071822Z-diagnosis/report.json`.

## Decision on the host residual, 15 Oct 2026

**Accept the 114 as a recorded residual. Keep the frozen budgets, and do not change the arithmetic.** The v2 host parity result stays a FAIL with 114 of 71,462 decisions named. It is not relabelled as a pass. No window is excluded.

Reasons:

1. **It is not a port defect.** KAN arithmetic contributes ≤ 1.28e-6. The frozen v1 model passes on the same IR features of the same windows. In 111 of 114, swapping one C++ feature value into the Python model reproduces the error. Feature-level FP32 differences on inputs far outside the training range are what the v2 model amplifies.
2. **Nothing a user sees changes.** Probability failures are 0 (max error 5.75e-5), Candidate disagreements are 0, and top-3 payload failures are 0. The failing window nearest the cutoff has P = 0.541. That is 0.041 from 0.50, about 700 times the largest probability error.
3. **The repair costs more than it fixes.** Double-precision feature extraction would run in software on the nRF52840, whose FPU is single-precision only (ADR-0004). The decision block is already projected at about 430 ms of the 640 ms FIFO, and the live image cannot detect overflow. Spending that headroom to close a residual in out-of-range attributions is the wrong trade.
4. **Loosening or excluding is not allowed.** The budgets were frozen in `4e543b5` before any v2 run.

Consequences: the board replay reports its own failures against the same budgets, including locked cases 379 and 380, and does not close this residual. The paper reports v2 host parity as 71,348 of 71,462 decisions within budget. It states that every failure sits on an input beyond 5 training SDs and that none changes a probability past budget, a Candidate flag or the top-3 payload. `parity_complete` stays false. An input-range guard that withholds the score or explanation on out-of-range windows would change device behaviour and is a separate decision. It is not part of this one.

## Board parity result, 15 Oct 2026

Runbook job 1 was run. Archived image `nano33ble_v2_context_parity` (SHA `52bdf791…a9a0`) was written by bossac via bootloader COM7, and the log shows `Verify successful` with 336,660 bytes. Upload receipt: `outputs/firmware-v2-uploads/nano33ble_v2_context_parity-upload-receipt.json`. The board answered `H` with `CONTEXT_PARITY_V2 WINDOW=500 CONTEXT=250 STRIDE=250 N=2 THRESHOLD=0.50 INPUTS=16 PAIRS=IR,RED`. Report: `outputs/parity/v2-board-20261015/report.json` (SHA-256 `f732137d654917a690fcb2976fe568895d69cd30848e6b38a21f0be89523e1bb`), `status: completed`.

- The mid-warmup reset and malformed-block protocol checks passed; the driver aborts on either failure.
- All 21 synthetic decisions passed.
- 1,220 of the 1,222 decisions from the 611 locked recorded contexts passed against both the Python reference and the C++ host. Overall, 1,241 of 1,243 decisions passed.
- The 2 failures are locked cases 379 and 380, at the second decision of each. They fail board-vs-Python on attribution only (1.28e-5 and 1.09e-5 against the 1e-5 budget), as predicted from the host-vs-Python run. They pass board-vs-host (attribution error 1.19e-7 and 6.0e-8), and the top-3 order and set are unchanged.
- Largest board-vs-host differences: probability 8.94e-7, delta 9.54e-7.
- The legacy original-absolute 1e-4 field fails on 1,243 of 1,243 decisions vs Python, as it did in v1. It is the superseded criterion, not the active gate.

The board computes the same function as the C++ host export to within about 1e-6. The remaining Python-side failures are the host residual already accepted above, not a board defect. Budgets are unchanged and no case is excluded. This is sparse USB replay of recorded contexts: it does not establish consecutive-stream agreement, the live sensor path, BLE, timing or RAM, and `parity_complete` stays false. Runbook jobs 2 (timing) and 3 (live sit) have not been run. The board was left on the `CONTEXT_PARITY_V2` image.

## Board timing result, 15 Oct 2026

Runbook job 2 was run. Archived image `nano33ble_v2_context_timing` (SHA `1091aadc…4623`) was written by bossac via COM7, and the log shows `Verify successful` with 337,396 bytes. Upload receipt: `outputs/firmware-v2-uploads/nano33ble_v2_context_timing-upload-receipt.json`. The board's identity is `TIMING_REPLAY_V2 … CLOCK_HZ=64000000 GPIO=D2 ATTR16=0 INPUTS=16 PAIRS=IR,RED`. Report: `outputs/timing-v2-board-20261015/report.json` (SHA-256 `7cb61465532e9d49187af3b9a32d2ec473aadad58789cdd0f1846a63b917a5a1`), `status: completed`.

DWT cycle-derived times at 64 MHz, for the whole 250-sample block:

- **Subsequent decision:** n = 180, median 433.4 ms, p95 435.0 ms, max 435.4 ms. `millis()` reads a median of 428 ms and a max of 430 ms on the separate timer. v1 was 249.7 ms median and 250.8 ms max.
- **First decision after a reset:** n = 30, median 346.5 ms, max 347.7 ms. v1 was 204.2 ms.
- **Warmup block:** max 0.28 ms.

Each block includes all 250 paired pushes, filtering on both channels, the 16 features, one baseline inference, the 16 perturbation passes, the top-three sort, and the debounce and payload step. The file does not split the block into inference time and explanation time.

All 210 functional checks against the host passed: probability, top-3, states, and features vs Python. The stop rule is not triggered. The worst block (435.4 ms) is under the 640 ms FIFO, a headroom ratio of 1.47 against v1's roughly 2.5. The pre-run projection of about 430 ms held.

Scope: this is accelerated synthetic USB replay. It measures no sensor or BLE latency, and the clock is not physically calibrated: `physical_gpio_verified` is false and the D2 check is outstanding. Main-thread stack peak was 4,600 of 32,768 bytes in this image. That is not whole-device peak RAM, and not the live image's figure. `parity_complete` stays false. The board was left on `TIMING_REPLAY_V2`. Job 3 (live sit) has not been run.

## Live sit, stopped on the rate rule, 15 Oct 2026

Runbook job 3 was started. Archived image `nano33ble_v2` (SHA `94725049…1635`) was written by bossac via COM7, and the log shows `Verify successful` with 343,164 bytes. Upload receipt: `outputs/firmware-v2-uploads/nano33ble_v2-upload-receipt.json`. A 25 s pre-check is at `build/v2/v2_boot_check.txt`. The still capture is `outputs/live-v2/2026-10-15_v2_still.txt`: 360 s, 17:10:39–17:16:39, one seated subject, with one brief finger lift at 17:10:48 that reset the pipeline. The banner was not captured because the port reopened after reset. v2 identity rests on the verified image hash and the 16-name `Feat:` line, and `two_wavelength_valid=1` on every decision.

**Rate.** In the 25 s pre-check `Hz:` read 49.92–50.05, but during the sit it fell to 47.2–48.9, median 47.56. Decisions arrived every 5.31 s on average instead of 5.00 s, which is an effective drain of about 47.1 Hz. That meets the runbook stop rule ("a value well below 50 with v2 means decision blocks are stalling ingestion"), so the recover capture was not run. The live image does not detect FIFO overflow, so this capture is assumed to have lost samples, about 6 % at that rate. The v2 timing image measured a 435.4 ms worst-case block. The live loop adds work that that image does not time, and how long it takes was not measured. The cause of the shortfall is not established.

**Scores** (recorded, not interpreted as detection). 64 decisions in total. 57 of 64 score 0.89, 4 score 0.07, and 1 each score 0.39 and 0.61. 59 were Candidate and 54 were Confirmed. The top driver was `ratio_of_ratios` on 60 decisions. Live `ratio_of_ratios` was 1.83–2.41 on the 0.89 decisions and 0.91–1.06 on the four 0.07 decisions. `ir_perfusion_index` was 0.0007–0.0056 (training mean 0.0296, SD 0.157), so neither input was beyond 5 training SDs. The subject was healthy and seated, and oxygen was not varied, so the flags are false positives or a shift between the training data and this sensor, not a low-oxygen finding. The samples lost to the rate shortfall also mean these windows are not clean 50 Hz windows.

The board was left on the v2 live image. Rollback to v1 is the runbook's archived image, and Jason has to name it.

### Correction (16 Oct 2026)

The **Scores** paragraph above is left as written. Its counts for `outputs/live-v2/2026-10-15_v2_still.txt` are wrong, judged by its own definition. "64 decisions" is the run after the 17:10:48 finger lift, windows 0–63. The one decision logged before the lift (log line 11, window 9, P 0.89) is excluded. Recounting that same 64-decision run from the log gives: **59** score 0.89, **3** score 0.07 (windows 0, 20, 21), 1 each score 0.39 (window 4) and 0.61 (window 5). **60** were Candidate and **57** Confirmed. The top driver was `ratio_of_ratios` on 60. The paragraph's own breakdown (57 + 4 + 1 + 1 = 63) does not add up to 64. The three 0.07 decisions have `ratio_of_ratios` 0.909, 1.057 and 1.041. With the pre-lift decision included, the log gives 65 / 61 Candidate / 58 Confirmed / 61. Source: `outputs/score-investigation-v2-20261016/tables/live_v2_decisions.csv`. The verdict on why rest scores 0.89 is in `outputs/score-investigation-v2-20261016/VERDICT.md`.

### Correction (16 Oct 2026, supervisor)

The status line above, "Supervisor not yet informed," is left standing as what was true when this ADR was accepted on 15 Oct 2026. On 16 Oct 2026 Jason stated that he had told Sheena about the notebook-10 label defect and that she gave no direction. That statement is not in the Week 6 minutes, which record a dashboard comment and no decision. Hearing it is not a direction to replace the reported system with v2. The IEEE draft keeps the v1 numbers and must disclose the misalignment. v2 stays unreported until she directs a replacement.

### Correction (16 Oct 2026, whose choice)

The sentence above, "v2 stays unreported until she directs a replacement," overstates her role. It is left standing as what was written when her silence was treated as a veto. She gave no direction. That is not a prohibition and not an order. Jason chooses whether the paper compares v1 and v2. A comparison must not call v2 better, and it must not drop v2's unfinished axes.

### Correction (17 Oct 2026, host residual)

Reasons 1 and 3 in "Decision on the host residual, 15 Oct 2026" are left standing as written that day.

- **Reason 1 is not supported for 96 of the 114.** `window_mean` at `firmware/Arduino Nano 33/include/oxygen_risk_pipeline.h:163` subtracts `raw_500[PPG_WINDOW_SIZE / 2]` and sums the residuals in FP32. On recording `9c01688d2219df17bec437beeb36229b4f26370fd04e717486deb9996a9ba40c` that centre sample is a spike: about 70,872 counts against a window median of about 485, in all 96 failing windows (264–487). That function is the defect. That the KAN arithmetic contributes at most 1.28e-6 does not stop it being a port defect.
- **Reason 3 is not supported.** The repair for those 96 is a two-pass FP32 mean, not double precision. Double precision was the wrong trade because it was the wrong repair.
- **"Outside the training range" is wrong for those 96.** Their inputs are beyond 5 training SDs, but they are rows in `outputs/openox-v2/window_features.csv`, the training table.
- **The 15 Oct accept stands as the decision made that day.** This correction does not relabel that run as a pass. `parity_complete` stays false until a new run, in a new directory, passes under the frozen budgets. No budget moves.
- **The paper still reports v1.** This correction does not put v2 or v3 on the paper.

### Reported system changed (5 Nov 2026, Jason)

Jason chose to lead the IEEE paper's title, abstract and results with v2, named the Header-aligned model, and to keep v1, the Baseline model, only where a comparison is needed. The earlier lines that say the paper reports v1 are left standing as what was true when they were written. v2 was built after the notebook-10 label defect was found, with the recipe, 0.50 and N=2 unchanged; the choice to report it was made after both models were scored, and the paper says so. The two models are still scored on different labels and tables, so no figure ranks them, and the paper still must not call v2 better or drop its unfinished axes (host parity 7 of 71,462 over budget, GPIO timing, sensor-image peak RAM, the live path). Repo IDs stay v1/v2.

### Baseline removed from the paper (6 Nov 2026, Jason)

Jason decided that the IEEE paper reports one model and does not mention v1. There is no Baseline column, no Baseline number, and neither reader-facing name: the paper calls the reported model TinyKAN. This supersedes the 5 Nov "keep v1 only where a comparison is needed". The 5 Nov section above is left standing as what was decided that day.

- **What stays.** The label defect stays in Methods as a correction to an earlier export: the gap (median 233.8 s; over 10 s on 128 of 162), label agreement 0.686, and the ablation step 0.712. It also says that the header alignment and the two red/infrared inputs were adopted after the label audit, while 0.50 and N = 2 were fixed before it.
- **v1-sourced claims under the reported model are replaced with v2 sources, not dropped.** The host-vs-device filter check becomes 8 of 48,625 Candidate flips (0.016 %), median |ΔP| 1.7e-6, max 0.82 (`outputs/parity/v2-host/20261021T163919Z/report.json`, `deployed_vs_offline`). It replaces the v1 640-window "no alert differences". The ingest rate becomes the r2 sit's printed 49.94–50.09 Hz instead of the 7 Oct v1 capture's 49.97 Hz. The original 1e-4 check is cited as failing on 1243 of 1243 r2 board decisions.
- **v1 evidence is unchanged.** It stays in the repo and goes to the final report, where both models can still sit side by side as different exams. Supervisor sheets are not affected.
- **Rules that still hold.** The paper must not call this model better than anything it no longer shows, and it must not drop the unfinished axes (host parity 7 of 71,462 over budget, GPIO timing, sensor-image peak RAM, the live path).

Not yet applied to the docx. Cut list: vault `paper/Paper redraft - drop Baseline - 6 Nov.md`.
