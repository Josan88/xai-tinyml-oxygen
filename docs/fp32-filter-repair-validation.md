# FP32 filter repair and locked host validation

10 September 2026; timing continuation 11 September local time. **The frozen host candidate passes the adopted numerical checks on 2,557 development windows and 2,464 validation windows locked before repair. The authorized Nano replay passes all observed checks on 21 synthetic decisions, including all 14 signed deltas. The same frozen diagnostic then passed sparse recorded-data target qualification on all 2,557 development windows. Parity is not complete.** The separately authorized timing image now has cycle-derived measurements for 210 synthetic decision blocks; physical GPIO/clock verification, complete runtime memory qualification and live captures remain outstanding. The separate context gap remains. The locked validation set was not replayed on the board.

The contract in [feature-parity-criterion-proposal.md](feature-parity-criterion-proposal.md) is unchanged: local-scale factor 1e-4, eight FP32 output spacings, probability absolute error 1e-4, each of 14 signed attribution errors 1e-5, ranking gap 2e-5, exact dominant-frequency bin and candidate agreement. Threshold stays 0.50, debounce stays N=2, future context stays 250 samples at 50 Hz, and model/scaler constants are unchanged. No retraining or retuning was performed.

## Diagnosis of every original failure

The new qualification runner reproduced the original development baseline exactly: 19 feature, one probability and 52 attribution failures, with a union of **66 / 2,557**. Candidate disagreements were zero. The original uniform absolute 1e-4 feature comparison failed on all 2,557 windows.

`scripts/diagnose_parity_failures.py` then evaluated **all 66 failures individually**, extending the earlier review's 11 selected cases. Every case contains all 14 absolute feature errors and budgets, probability error, all signed attribution errors, and separate lower-band power, upper-band power and ratio components.

- Python features/inference on native filtered samples retained 64 failing windows: 18 feature, one probability and 51 attribution failures.
- Python inference on native feature vectors retained 65: 19 feature, one probability and 51 attribution failures.
- Complete native processing retained the original 66: 19 feature, one probability and 52 attribution failures.
- FP64 filtering with the original rounded firmware coefficients, used only as an offline control, retained one skewness feature failure. Probability and attribution checks passed.
- Comparing native feature math with Python feature math on the same native filtered samples produced zero feature failures under the adopted rule.

Thus filtering dominates the observed failures, including the problematic bandpower ratios, but stage errors can add or cancel. The per-window results preserve those distinctions; aggregate counts do not assign a single universal cause. All original absolute errors remain in the diagnosis report.

## Native FP32 repair

The shared `ppg_bandpass_filter()` now removes the target window's center-sample DC value before filtering and adjusts both odd extensions consistently. It still uses the same input context, 27-sample padding, forward/backward passes and intended Butterworth bandpass.

The two low-pass sections use fused single-precision multiply-add. The two high-pass sections use an equivalent integrated-state realization with compensated FP32 updates. For an original denominator `1 + a1*z^-1 + a2*z^-2`, its coordinates are `k = 1 + a1 + a2` and `h = 1 - a2`. These small coefficients are computed from the canonical SciPy coefficients before rounding once to FP32, avoiding cancellation from subtracting already-rounded values near -2 and +1. The equivalent transfer function is `(1-z^-1)^2 / (1 + (k+h-2)*z^-1 + (1-h)*z^-2)`.

All production constants, states, residuals and operations remain FP32. There is no FP64 firmware path. The compensation requires arithmetic without reassociation; builds defining `__FAST_MATH__` are rejected. Feature definitions, FFT/Welch, inference and attribution ranking were not changed.

Six isolated development experiments reduced the original failing subset to 30, 5, 2, 2, 1 and finally zero failures. The final experiment then passed all 2,557 development windows before integration. The integrated native output matches that successful experiment exactly for every emitted value. No validation outcomes were used to select the repair.

## Lock and candidate chronology

[numerical-validation-lock.md](numerical-validation-lock.md) records the pre-repair selection: **2,464 windows from 155 recordings**. Their filter-input intervals do not overlap the documented development exposures or one another. Selection used metadata and a fixed hash ordering, without labels, feature values or native outcomes. They are previously unevaluated windows within already-seen recordings, not independent subjects or a clinical test set. Whole-recording Fourier resampling is shared preprocessing.

- Contract SHA-256: `9e616224232854c9dcf3a42289ad0d27560f065462e9b4c41ee0a007b4914c49`.
- Pre-repair lock SHA-256: `341909bffbcaa396beb174b5e2403ea6aa864a8dd0aee0eb85207edbca5b2339`.
- Complete integrated development pass: **11:26:54 UTC**.
- Candidate frozen: **11:27:14 UTC**, including source, probe, runner, criteria, contract, lock, development report and passing test-log hashes.
- Candidate SHA-256: `1cfd4ed28b5aad114d1064a1a7323673b8725e434700e7b6303b5009ace74d0b`.
- First native use of validation: **11:28:08 UTC**; report completed **11:28:12 UTC**.

The validation manifest's complete key set was independently checked against the original lock. The runner requires the unchanged frozen candidate and writes a first-use receipt before native evaluation. A different candidate is refused after first use. This set is now evaluated evidence, not a fresh holdout for future repairs.

## Results and original absolute errors

**Development, 2,557 windows / 162 recordings:** zero failed windows under the adopted checks. Maximum probability absolute error **1.018657691e-6**, maximum signed attribution absolute error **2.536956222e-6**. No robust ranking violations, top-three order changes, candidate disagreements or replay state/metadata/payload failures. Largest continuous-feature error consumes **15.694%** of its frozen local budget.

**Locked validation, 2,464 windows / 155 recordings:** zero failed windows under the adopted checks. Maximum probability absolute error **8.251690626e-7**, maximum signed attribution absolute error **1.579872946e-6**. No robust ranking violations, top-three order changes, candidate disagreements or replay state/metadata/payload failures. Largest continuous-feature error consumes **8.295%** of its frozen local budget.

The original uniform absolute 1e-4 feature comparison still fails on **2,556 / 2,557 development** and **2,464 / 2,464 validation** windows. Per-window reports retain all original absolute errors. Maximum absolute errors by feature are below, in **development / validation** order; maxima can come from different windows:

- Mean: 0.00334581 / 0.00460254.
- Standard deviation: 0.0730760 / 0.0567644.
- Minimum: 0.143084 / 0.196862.
- Maximum: 0.180891 / 0.240570.
- Value range: 0.323975 / 0.328804.
- RMS: 0.125001 / 0.0847052.
- Skewness: 1.18221e-5 / 1.00482e-5.
- Kurtosis: 0.000356322 / 0.000104030.
- Energy: 56,594,706.59375 / 31,041,916.625.
- Derivative standard deviation: 0.0406225 / 0.0660363.
- Dominant frequency: exactly zero error in both sets.
- Lower-band power: 44,739.2903 / 9,772.48457.
- Upper-band power: 115,700.619 / 17,104.7076.
- Bandpower ratio: 0.0417718 / 0.000102921.

These are raw-unit errors; each is assessed against its own window's frozen scale and FP32 spacing. Large raw energy errors are not hidden by the passing status.

The separate native finite-context versus full-recording probability difference is **0.01237930** on development and **0.00854544** on validation. The known development context gap therefore remains. It is not absorbed into the numerical budgets or removed by this repair; see [training-context-decision.md](training-context-decision.md).

Each qualification case resets the actual deployment pipeline and ingests its complete 750- or 1,000-sample context. The checks cover the final target window, preceding warmup candidate, N=2 state, indices, delay and top-three payload. This sparse replay does not establish consecutive-state agreement across entire recorded streams. Existing continuous synthetic regressions cover reset, wrap and longer state transitions separately.

## Tests and target builds

All **11 / 11 CTests passed**, including the unchanged synthetic C++ internal-consistency absolute 1e-4 tests, independent SciPy regression, frozen-criterion checks, qualification-probe reset/metadata/all-delta checks, and the recorded-board protocol tests. Runtime used Python 3.12.0, NumPy 2.4.4 and SciPy 1.17.1.

Both Nano images compile. Initial linker estimates were RAM **84,400 bytes**, flash **340,900 bytes** for `nano33ble`; RAM **85,256 bytes**, flash **335,684 bytes** for `nano33ble_context_parity`. These are build sizes, not measured runtime memory or DWT/GPIO latency. The user then authorized the diagnostic upload and replay below. The normal sensor image was not uploaded.

## Authorized synthetic board replay

The prepared diagnostic was uploaded to the identified Nano and its flash contents were verified. It passed all 21 continuous synthetic decisions. That initial image exposed only three attribution deltas. A diagnostic-only extension then added command `A`, returning `ATTR14`, the latest window index and all 14 signed native FP32 deltas. It recomputes the same mean-perturbation operations from the decision's raw features, as the frozen host qualification probe does. The original top-three payload is checked for exact consistency with those values. Reset invalidates the saved diagnostic values; requesting them before a decision returns `ERR no_decision`.

The filter, pipeline, model, inference and frozen host probe hashes remained unchanged. Only diagnostic observation code and its client were extended. The extended image was archived, uploaded and flash-verified at **12:04:31 UTC**. The latest diagnostic linker sizes are RAM **85,320 bytes**, flash **335,868 bytes**. Additional diagnostic recomputation and USB traffic make this unsuitable as a latency measurement.

All **21 / 21** decisions pass the unchanged local-scale feature, probability, **all-14 signed attribution**, ranking, 0.50 candidate, N=2 state, timing and transmitted-payload checks against independent Python and the native host. There are zero robust ranking violations or top-three index differences. Board/host probability error is exactly zero; maximum signed delta error across all 14 is **5.960464478e-8**. Board/Python maximum probability error is **7.113263512e-8**; maximum all-14 signed delta error is **1.319502340e-7**.

Original uniform absolute 1e-4 raw-feature comparisons still fail on **21 / 21** decisions for both board/Python and board/host. Every feature's original absolute error and frozen local budget remains in `frozen-contract-review.json`; these failures are not omitted from the passing report.

The replay covers normal, normal/risk/recovery and risk-after-reset streams, 750-sample initial emission, 250-sample stride/delay, ring-buffer wrap, mid-warmup reset, incomplete-block rejection and N=2 confirmation/recovery. Five protocol unit tests and the host all-delta preparation also pass. This is accelerated synthetic USB input, not live sensing. The current board image remains the extended context diagnostic with no live sensor/BLE acquisition.

Evidence:

- Initial prepared-image replay: `outputs/parity/context_board/20260910T115350588929Z/`.
- All-14 replay and frozen-rule review: `outputs/parity/context_board/20260910T120454865670Z/`.
- Build and verified-upload receipts: `outputs/parity/qualification/all14-target-build-receipt.json` and `all14-board-upload-receipt.json`.
- Archived diagnostic source/BIN/ELF: `outputs/parity/qualification/all14-diagnostic-image/`.
- Uploaded BIN SHA-256: `30d390c7ad19cc335f8453124e14fbeef7986bd9bb258c5a60dde27fd76b9193`.
- Build/upload logs: `build/parity/all14-context-board-build.log` and `all14-context-bossac-upload.log` (contains `Verify successful`). The initial PlatformIO `nobuild` attempt stopped before touching the board because its generic target expected a HEX; the installed BOSSA uploader then wrote and verified the archived BIN directly.

```powershell
python scripts/replay_context_board.py --probe build/parity/audit-release/context_probe.exe --qualification-probe build/parity/audit-release/qualification_probe.exe --port COM8
python scripts/review_context_board_contract.py --replay-report outputs/parity/context_board/20260910T120454865670Z/report.json --candidate outputs/parity/qualification/frozen-candidate-v1.json --upload-receipt outputs/parity/qualification/all14-board-upload-receipt.json --output outputs/parity/context_board/20260910T120454865670Z/review-repeat.json
```

The replay script never uploads firmware. Use the observed Nano port and the all-14 diagnostic image; host-only preparation omits `--port`. A repeat is additional regression evidence on the same synthetic streams, not a fresh validation set.

## Recorded-data target qualification

The frozen all-14 diagnostic, still on COM8, then replayed every frozen development context over USB. Each case resets the deployment pipeline, rejects `A` before a decision, ingests the 750- or 1,000-sample context in 250-sample blocks, and checks the final window against Python and the frozen host native rows, including all 14 signed deltas, ranking, 0.50 candidate, N=2 state, timing and top-three payload consistency. The locked validation set was not sent to the board. `scripts/qualify_recorded_board.py` never uploads firmware.

All **2,557 / 2,557** development windows pass the adopted checks. Maximum probability absolute error **1.018657691e-6**, maximum signed attribution absolute error **2.536956222e-6**, matching the host development maxima. Candidate disagreements remain zero. Original uniform absolute 1e-4 feature comparisons still fail on **2,556 / 2,557** windows and remain in the report. This is sparse recorded USB replay, not consecutive-stream, live sensing, DWT/GPIO latency or complete parity.

Evidence: `outputs/parity/qualification/recorded-board-development-v1/report.json`, SHA-256 `ea40763522fe70658c80424da3c427bbb6ee010fb4cec4b20e8cbef2970b35bc`. Host-cache preflight: `outputs/parity/qualification/recorded-board-host-cache/`.

```powershell
python scripts/qualify_recorded_board.py --cases outputs/parity/qualification/development-v1 --native-report outputs/parity/qualification/repaired-development/report.json --native-npz outputs/parity/qualification/repaired-development/native.npz --candidate outputs/parity/qualification/frozen-candidate-v1.json --upload-receipt outputs/parity/qualification/all14-board-upload-receipt.json --output outputs/parity/qualification/recorded-board-development-repeat --port COM8
```

## Evidence and reproduction

All recorded inputs, identifiers and detailed results stay in ignored `outputs/`; no recorded fixture was added to source control. The root for this work is `outputs/parity/qualification/`:

- `development-v1/` and `validation-v1/`: immutable input/reference caches, manifests and source provenance.
- `baseline-development/`: reproduced 66-window baseline.
- `baseline-diagnosis/report.json`: all 66 per-window stage diagnoses.
- `baseline-sources/`: preserved original headers and native probe.
- `experiment-*/`: development-only experiment evidence.
- `repaired-development/report.json`: full integrated development pass, SHA-256 `9044032f11ab7b226ed1ad2a78e0a71dfb23e50d74d9366d5fa0fd40f83a4b33`.
- `frozen-candidate-v1.json`: candidate freeze record.
- `target-build-receipt.json`: hashes and byte sizes of both built Nano images, tied to this candidate; neither has been uploaded.
- `locked-validation/report.json`: validation pass, SHA-256 `edd1468b4439d18d290f07730552c06c8c7bdb3d01462e5bf9971266aa395ff1`.

First use is recorded in `outputs/parity/validation_lock/numerical-validation-v1.use.json`. Build/test logs are `build/parity/repair-configure.log`, `repair-build.log`, `repair-tests.log` and `repair-board-build.log`. Final verification confirmed the source, runner, criteria, contract and lock hashes remained frozen. Reproducing the original diagnosis with preserved headers yielded identical results for all 66 cases, saved in `diagnosis-reproduced/`.

The main workflow is `prepare`, complete development `run`, `seal`, then validation `run --candidate`. Outputs are write-once. The following examples use fresh output names and the already prepared caches; repeating the same frozen validation candidate does not create new independent evidence:

```powershell
python scripts/qualify_numerical_parity.py run --cases outputs/parity/qualification/development-v1 --probe build/parity/audit-release/qualification_probe.exe --output outputs/parity/qualification/development-repeat
python scripts/qualify_numerical_parity.py run --cases outputs/parity/qualification/validation-v1 --probe build/parity/audit-release/qualification_probe.exe --candidate outputs/parity/qualification/frozen-candidate-v1.json --output outputs/parity/qualification/validation-repeat
python scripts/diagnose_parity_failures.py --cases outputs/parity/qualification/development-v1 --baseline outputs/parity/qualification/baseline-development --baseline-sources outputs/parity/qualification/baseline-sources --output outputs/parity/qualification/diagnosis-repeat
```

The older audit/review commands describe the preserved pre-repair evidence and require their original source/binary hashes. Use the new qualification runner for the repaired implementation.

The authorized synthetic board replay and recorded development-target qualification are complete within the scope above. The timing continuation below adds bounded cycle measurements; physical GPIO/clock verification, complete runtime memory qualification, live captures and the separate context gap remain. AOPC, PTT-PPG sit/walk/run, the unmet recall/specificity gate and Android ON HOLD are unchanged. Parity is not complete.

## Timing image prepared and measured, 11 September

A separate `nano33ble_context_timing` image now builds around the unchanged frozen pipeline, with DWT counting and a D2 pulse for each 250-sample block. It omits the all-14 diagnostic recomputation and keeps parsing/printing outside the interval. All 13 host CTests and a 21-decision independent host preparation pass. Review found and the added regression reproduced a build-receipt hash-override defect; frozen and build manifests are now checked independently.

The user authorized the exact archived image, SHA-256 `7db771c18ef99fb087933b638ab399bef6f49e74425937e19311374209bbe633`; it was uploaded and flash-verified, then passed all observed functional checks over **210 synthetic decision blocks** (21 distinct decisions repeated ten times). Subsequent-decision cycle-derived median/p95/maximum are **249.702 / 250.535 / 250.797 ms** at the declared 64 MHz clock. Original absolute 1e-4 feature comparisons still fail on all **210/210** and remain in the report. Board/host probability error is zero; maximum top-three signed-attribution error is **5.960464478e-8**. No extra all-14 recomputation was performed.

The coarse millisecond clock reads 1.612–4.060 ms shorter than the cycle-derived decision intervals. This discrepancy is unresolved; physical GPIO verification is still required. The reported main-thread stack watermark is 31,308 / 32,768 bytes before and after replay, and heap allocation snapshots remain 1,040 bytes. These do not establish pipeline-only or whole-device peak RAM. Initial linker estimates remain flash 336,588 bytes and static RAM 85,304 bytes.

The measurement scope, exact-image upload receipt, raw report hash and reproduction steps are in [on-device-timing.md](on-device-timing.md). The board now runs this timing diagnostic. The archived all-14 image, frozen candidate/contract, recorded-development report and core source hashes remain unchanged. No locked validation windows were replayed. Parity is not complete.

## Correction (2026-10-09)

The sentences above that say the locked validation set was not replayed on the board are left standing. They were the scope of this file on 10–11 September. The locked-set board replay was later run: 2464 of 2464 windows pass the adopted local-scale checks, the original uniform absolute 1e-4 comparison fails on 2464 of 2464, and the maximum probability error is about 8.3e-7. Evidence: `docs/evidence-summary-2026-10.md` and `outputs/parity/qualification/locked-board-replay-v1/`. That replay does not make parity complete. The 1e-4 comparison cannot be met in FP32, because the nearest representable energy value already exceeds 1e-4. The local-scale contract was adopted before the repair. It was not retired so the replay would pass.
