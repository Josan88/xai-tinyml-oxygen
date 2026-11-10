# Deployment filtering context — 5-second future delay

## Verified problem

The canonical notebook filters a full recording forward and backward before selecting windows. Its output depends on samples beyond each window's end. A live 10-second window evaluated as soon as it finishes therefore disagrees even when Python and C++ arithmetic match.

`scripts/check_training_context.py` reconstructed 123 retained windows across eight evenly selected recordings. All 123 feature vectors matched the stored canonical export within the stated reconstruction tolerance. No dataset, labels, weights or threshold were modified.

Per-window Python and C++ probability differs by at most **0.0000033964**, with zero candidate disagreements. Full-recording versus per-window filtering differs by up to **0.340203**, with **15 candidate disagreements** at 0.50. This is a deployment input mismatch, not an inference implementation mismatch.

## Offline finite-context experiment

These are Python/SciPy diagnostics at the same retained window positions. Future samples imply additional waiting time in deployment. At recording boundaries, available context is clipped to the actual recording.

| Past context | Future context / added wait | Maximum probability difference from full recording | Windows with difference >= 0.01 | Candidate disagreements |
|---|---|---:|---:|---:|
| 2 seconds | 0 seconds | 0.103553 | 34 | 1 |
| 5 seconds | 0 seconds | 0.104080 | 33 | 0 |
| 10 seconds | 0 seconds | 0.104105 | 33 | 0 |
| 2 seconds | 2 seconds | 0.039079 | 16 | 0 |
| 5 seconds | 5 seconds | 0.002179 | 0 | 0 |
| 10 seconds | 10 seconds | 0.000006144 | 0 | 0 |

## Implemented timing contract

`OxygenRiskPipeline` is `BasicOxygenRiskPipeline<250>`. Features are taken from the central 10 seconds of a 20-second rolling buffer. The 5-second stride, TinyKAN weights, threshold 0.50, and ADR-0005 N=2 are unchanged.

The first target window (seconds 0–10) is emitted at second 15. Later windows follow the 5-second stride with a 5-second decision delay. `RiskDecision` reports `window_start_sample`, `window_end_sample`, and `decision_delay_samples`. Initial windows use available past context. Signal-loss `reset()` returns the pipeline to the 750-sample first decision.

Zero-context `BasicOxygenRiskPipeline<0>` remains the host arithmetic and diagnostic-replay path so existing 500-sample fixtures keep immediate-window semantics.

## Host context probe

Native TEST 6 checks first decision at 750 samples, 250-sample stride, metadata, ring-buffer wrap, and reset. `tests/context_probe.cpp` streams `OxygenRiskPipeline` on the host. Independent SciPy 5 s past / 5 s future on synthetic PPG agrees to about **4.23e-9** probability.

On the same 123 retained windows, streaming C++ matched 122 positions. Maximum C++ versus Python 5 s+5 s probability error is **0.000007555**, with zero candidate disagreements. Maximum C++ versus full-recording probability error is **0.002181**, with zero candidate disagreements, matching the Python 5 s+5 s diagnostic. One retained window was unmatched because the recording ended before 5 seconds of future samples were available. First decisions were at 750 samples for window `[0, 500)` with delay 250.

This sample does not establish agreement across the full dataset. No retraining or retuning was performed.

## Feature precision remains a separate gate

Latest evidence: the [compensated FP32 repair](fp32-filter-repair-validation.md) passes the adopted frozen numerical checks on all 2,557 development windows and all 2,464 validation windows locked before repair. Its authorized 21-decision synthetic Nano replay and sparse recorded development-target replay both pass, including all 14 signed deltas. The original absolute 1e-4 feature comparison still fails on 2,556 development, all 2,464 validation, all 21 synthetic board decisions, and 2,556 recorded board windows; original errors remain recorded. This does not close the full-recording context gap. Board replay figures in the historical sections below describe the earlier image; the linked repair report contains the latest all-14 replay.

The original absolute 1e-4 raw-feature comparison still fails on these windows and remains a reported quantity. Maximum model-standardized feature error is about 0.0001621 across these 123 windows. A raw energy value near 67 million has an FP32 step of 8, so a uniform absolute 1e-4 criterion cannot generally be satisfied under the FP32 contract. The live numerical requirement is now the adopted local-scale rule in [feature-parity-criterion-proposal.md](feature-parity-criterion-proposal.md); that adoption does not by itself close remaining failures.

## Board arithmetic replay (9 September)

The explicit `nano33ble_parity` diagnostic image was uploaded over USB. It uses `BasicOxygenRiskPipeline<0>`, does not acquire the sensor, and does not broadcast BLE. Each of 9 synthetic 500-sample windows resets the pipeline, so this is board arithmetic evidence only; it does not verify N=2 transitions or the 250-sample deployment delay.

Host/board probability and attribution gates passed. Maximum probability error was about **4.26e-10**, maximum signed attribution error about **3.97e-10**, and all candidate states agreed at 0.50. The strict raw-feature 1e-4 gate still fails. Diagnostic build estimates: RAM 78,392 bytes (29.9%), flash 334,908 bytes (34.1%). These are linker sizes, not measured stack/heap, DWT/GPIO latency, live PPG, or physical installation evidence. The normal `nano33ble` image was not uploaded.

`scripts/replay_board_parity.py` writes 64-byte USB chunks. A single 4500-byte line overflows the board receive buffer. Native USB often emits `PARITY_READY_V1` before the host opens the port; the script continues if that banner was missed.

Prepared and completed evidence is in `outputs/parity/board/`. Synthetic fixtures are tracked; recorded OpenOximetry inputs remain local under the PhysioNet Restricted Health Data License. Protocol unit tests use a fake transport and are not hardware evidence.

Detailed context results and source-recording hashes are in `outputs/parity/context/report.json` and `windows.csv`.

## Deployment-context board replay (10 September)

The Nano on COM8 now runs the explicit **`nano33ble_context_parity`** diagnostic image. The normal `nano33ble` sensor firmware has not been uploaded. This newer diagnostic uses the actual `OxygenRiskPipeline` alias (`BasicOxygenRiskPipeline<250>`), preserving state between 250-sample blocks.

`scripts/replay_context_board.py` checked three deterministic synthetic streams: normal, normal/risk/recovery, and risk after reset. Independent SciPy context processing and the host executable passed all **21 decisions**; the Nano then passed all 21 against the host. Verification covers first emission after 750 samples, subsequent 250-sample stride, target-window start/end metadata, 250-sample delay, candidate flags, N=2 confirmation and recovery, consecutive counts, top-three attribution indices/deltas, and ring-buffer wrap.

The host/SciPy maximum probability error is **6.076e-7**, with maximum top-attribution error **1.223e-7**. Board/host maximum probability and top-attribution errors are both **5.961e-8**. Each stream also passed a mid-warmup reset check and rejection of an incomplete input block without advancing the pipeline. These are accelerated USB inputs; sample-index timing was tested, not physical acquisition timing or DWT/GPIO latency.

The existing absolute raw-feature criterion still fails on **19 of 21 board/host decisions**. The client restores exact FP32 host values from their nine-digit decimal representation before comparing them with the board's hexadecimal FP32 values, so that failure is not merely decimal display rounding. Probability, XAI, timing and state agreement do not silently waive the feature criterion.

The protocol identifies firmware on demand before sending data, stages complete blocks before ingestion, and acknowledges each block. The client uses bounded 64-byte USB writes, checks for short writes, and saves partial failure evidence if a run fails. `tests/test_context_board_protocol.py` covers parsing, FP32 transport equivalence, wrong timing/state rejection, malformed results and short writes. CTest now passes **7/7** suites, including the 21-decision independent host regression.

Evidence: `outputs/parity/context_board/20260910T022415659190Z/report.json` contains the host and board results, reset checks, input/source/executable/image hashes, and the explicit completed/verified status. Upload output is in `build/parity/context-board-upload.log`. New diagnostic linker sizes are RAM **85,256 bytes** and flash **335,348 bytes**; neither is a measured runtime peak.

Reproduce host preparation with `python scripts/replay_context_board.py --probe build/parity/context_probe.exe`. Add `--port COM8` only when this diagnostic is present on that verified port. The script does not upload firmware. Synthetic streams are generated deterministically; no restricted recording is included in repository fixtures.

## Expanded host audit (10 September, 162 recordings)

`scripts/audit_feature_errors.py` replayed up to 16 evenly spaced retained windows per recording, requiring a complete 250-sample future context. Result: **2,557 windows / 162 recordings**. Twenty-seven end-of-recording windows were skipped for missing future samples. Stored canonical features reconstructed with **0** failures. This remains the pre-repair recorded-data host audit: no board run, no sensor validation, and no clinical performance claim. The original absolute 1e-4 raw-feature comparison failed on **all 2,557** windows and stays a reported quantity. The later local-scale rule in [feature-parity-criterion-proposal.md](feature-parity-criterion-proposal.md) is now the live requirement; its baseline screen failed 66 windows. The subsequent repair and locked validation pass are recorded separately in [fp32-filter-repair-validation.md](fp32-filter-repair-validation.md).

| Check | Result |
|---|---|
| C++ vs Python 5 s+5 s probability | max **0.0001173**; **0** candidate disagreements at 0.50 |
| C++ vs Python 5 s+5 s Mean-Perturbation | max **0.0001539**; **1** top-3 index-order disagreement |
| 5 s+5 s vs full-recording probability | max **0.01238**; **1** window ≥ 0.01; **0** candidate disagreements |
| Previously sampled 8-recording cohort (122 windows) | max context error **0.002181**, matching the earlier 123-window probe |
| Strict raw-feature 1e-4 (reported, no longer the live gate) | fail on every window |
| First diagnostic proposal (training-scale; not adopted) | 142 feature / 148 numerical window failures |
| Adopted local-scale criterion (same development sample) | 19 feature / 1 probability / 52 attribution; union **66** |

CTest is **8/8**, including `FeatureAuditRegression`. The normal `nano33ble` image was not uploaded.

### One top-3 ranking disagreement

Recording `423f8fc4675d91a37c065bef752e9ce8682ae3f45f42992bc013e4f9115b3e55`, `start_sample` 47250. Probability is **0.33007** on both sides (error **3.02e-6**), far below 0.50.

The top-3 **set** is identical: `std`, `minimum`, `value_range`. Only slots 1 and 2 swap:

| Rank source | #1 | #2 | #3 | \|std\| − \|minimum\| |
|---|---|---|---|---:|
| Python on FP32-quantized 5 s+5 s input | `minimum` (−0.1172115) | `std` (+0.1172070) | `value_range` | **−4.56e-6** |
| Native C++ streaming | `std` (+0.1172099) | `minimum` (−0.1172070) | `value_range` | **+2.86e-6** |

Python re-run on the C++ feature vector matches the native order. The flip is therefore residual feature-value error at a near-tie (~3e-6 to 5e-6), not a different Mean-Perturbation sort. Firmware still ranks by \|Δ\|. Max attribution error on this window is **6.59e-6**, which is larger than the ranking gap, so the order can change without a material explanation change.

### One 5 s+5 s versus full-recording discrepancy ≥ 0.01

Recording `3f4ee88a46393f537651ccfca46a797565a8d3ae5daf20298d6cd01a0ae9a2de`, `start_sample` 8250 (165 s into the recording; past and future 5 s are complete, so this is not start-boundary clipping).

| Probability | Value |
|---|---:|
| Full-recording Python | 0.42343 |
| 5 s+5 s Python | 0.41103 |
| 5 s+5 s C++ | 0.41104 |
| C++ vs full-recording | **0.01238** |
| Python 5 s+5 s vs full-recording | **0.01239** |
| C++ vs Python 5 s+5 s | **2.68e-6** |

Both sides remain below 0.50. The other 15 sampled windows from this recording differ by at most **8.4e-5**. The gap is filtfilt finite-context versus full-recording context, not native arithmetic. The largest standardized feature change is **skewness** (absolute 0.241, standardized **0.365**); kurtosis is next (standardized 0.034). Top-3 ranking is unchanged: `kurtosis`, `bandpower_ratio_0_5_2_to_2_5`, `std`.

The earlier 8-recording sample did not contain a ≥ 0.01 case. Expanding to all 162 recordings shows that the 5-second future-context contract does not keep every window under 0.01 versus full-recording filtering. It still produced **zero** threshold disagreements on this sample.

Evidence: `outputs/parity/feature_audit/20260910T025019351296Z/report.json` and `windows.csv`. Reproduce with `scripts/audit_feature_errors.py` against the audit-release probes. Recorded OpenOximetry inputs stay under ignored `outputs/` / `data/`.
