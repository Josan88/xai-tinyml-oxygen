# Python/C++ probability agreement — preprocessing repair

**Latest continuation, 10 September:** the authorized compensated FP32 filter repair passes the frozen local-scale, probability, all-14-attribution and decision checks on **2,557 development windows** and **2,464 validation windows locked before repair**. Maximum validation probability error is **8.252e-7** and signed attribution error **1.580e-6**. Original absolute 1e-4 feature errors remain reported and fail on 2,556 development and all 2,464 validation windows. See [repair and validation evidence](fp32-filter-repair-validation.md) for the all-66 diagnosis, frozen candidate and full results. The measurements below document earlier repair stages.

**Status: locked host numerical validation, the repaired 21-decision synthetic Nano replay, and recorded development-target qualification on 2,557 windows pass; parity is not complete.** The latest authorized diagnostic exposes all 14 signed attribution deltas. Board/Python maximum probability error is **7.113e-8**, maximum all-14 delta error **1.320e-7**; all adopted feature, ranking and decision checks pass. Original absolute raw-feature comparisons still fail on **21/21** board decisions and remain recorded. The known development finite-context/full-recording probability gap remains about 0.01238. Threshold is 0.50, FP32 TinyKAN is (14, 12, 1), and ADR-0005 is N=2. Weights and operating point were not changed.

The all-14 evidence above is from the archived `nano33ble_context_parity` image. Its board/host maximum probability error is zero, and maximum all-14 delta error is **5.961e-8**. The board now runs the separately authorized `nano33ble_context_timing` diagnostic (no live sensor/BLE), which passed 210 repeated synthetic decision checks with subsequent-decision cycle-derived median **249.702 ms**. Physical GPIO/clock verification and complete runtime memory qualification remain pending; see [on-device-timing.md](on-device-timing.md). The numerical evidence is in [fp32-filter-repair-validation.md](fp32-filter-repair-validation.md); older stages below retain their historical results.

## Repair

Both `OxygenRiskPipeline` and `PPGFeatureExtractor` now use a shared FP32 forward/backward filter with 27-sample odd-extension padding. Their Welch calculations use a periodic Hann window (denominator 256, squared-window sum 96) and the two complete 256-sample segments, excluding the incomplete tail. These choices match per-window SciPy processing.

The filter scratch buffer grows from 500 to 554 floats: 216 additional bytes per pipeline instance. The legacy extractor's static scratch also grows by 216 bytes. No dynamic allocation was added. `verify_features.py` now uses the same Welch contract and a fixed random seed for its demonstration.

The older native reference tests are now labelled C++ internal-consistency tests. Independent verification is supplied by `tests/test_probability_regression.py` and `scripts/check_probability_parity.py`.

## Before and after

The same 17 FP32 replay windows were used before and after the repair; the saved arrays were checked for exact equality. Nine are synthetic and eight are recorded OpenOximetry manifest segments resampled to 50 Hz, using their first 500 samples.

| Check | Before | After |
|---|---:|---:|
| Raw-window probability errors >= 0.01 | 3 / 17 | 0 / 17 |
| Candidate disagreements at 0.50 | 1 / 17 | 0 / 17 |
| Maximum raw-window probability error | 0.0763669 | 0.0000102351 |
| Maximum raw-window signed attribution error, all 14 deltas | Not measured | 0.0000095221 |
| Identical-feature inference, 208 canonical rows: maximum probability error | 1.993e-7 | 1.993e-7 |
| Absolute raw-feature tolerance <= 1e-4 | Fail | Fail |

For `openox_manifest_0`, Python's per-window reference gives **0.434284964**. C++ changed from **0.510651886** to **0.434292495**. Both implementations now classify that window below 0.50.

The full check still exits 1 because the original strict feature gate remains unchanged. For example, an energy value around 67 million has an FP32 spacing of 8, already larger than an absolute 1e-4 tolerance. The measured energy error in that example is about 3513, so rounding to the final FP32 value alone does not explain the whole error. Filter and accumulation precision also need consideration. A suitable feature criterion requires an explicit review; it was not silently relaxed to obtain a pass.

## Validation

- CMake/CTest: native pipeline, stress, independent synthetic/context/board-protocol, and feature-audit unit regressions pass (**8/8** when configured with `-DPARITY_PYTHON=`).
- Independent regression on all 17 local saved windows passes: probability error < 1e-4, signed attribution error < 1e-5, and identical candidate states at 0.50.
- The broader `--probability-only` check passes while continuing to report the strict feature failure.
- Arduino Nano 33 BLE firmware compiles. Diagnostic `nano33ble_parity` USB replay of 9 synthetic windows passed; see [training-context-decision.md](training-context-decision.md). Linker sizes are not measured stack/heap peaks, DWT/GPIO latency, or live PPG evidence. The normal `nano33ble` image was not uploaded.

Nine synthetic fixtures are included under `tests/fixtures/`. Recorded OpenOximetry inputs remain in ignored local `outputs/` because the local dataset carries the PhysioNet Restricted Health Data License. They are not included in repository fixtures.

## Reproduction

Build `tests/probability_probe.cpp` with C++17 and `firmware/Arduino Nano 33/include` on the include path, or build the `probability_probe` CMake target. NumPy and SciPy are required for the independent checks.

```powershell
python tests/test_probability_regression.py --probe build/parity/probability_probe.exe
python tests/test_probability_regression.py --probe build/parity/probability_probe.exe --windows outputs/parity/after/replay_windows.csv
python scripts/check_probability_parity.py --probe build/parity/probability_probe.exe --output outputs/parity/after --probability-only
```

The first command needs no local dataset. The second uses saved local replay inputs. The third regenerates the broader evidence from the canonical CSV, manifest, and local OpenOximetry files. Omit `--probability-only` to run the unchanged strict feature gate, which currently fails. `--inference-only` gates only identical-feature model arithmetic.

To include the independent synthetic regression in CTest, configure CMake with `-DPARITY_PYTHON=` followed by the absolute path of a Python interpreter containing NumPy and SciPy.

Before evidence remains in `outputs/parity/`; repaired evidence is in `outputs/parity/after/`. Each contains its report, feature comparison and replay inputs. JSON reports include source/executable hashes and package versions. Output directories follow the repository's existing ignored-output policy.

## Remaining boundaries

The [feature-parity criterion](feature-parity-criterion-proposal.md) was **adopted 2026-09-10** and replaces the uniform absolute 1e-4 raw-feature requirement. Its 66 baseline failures were subsequently diagnosed individually and repaired without changing the budgets. Locked host validation, the repaired synthetic board replay and recorded development-target qualification pass. The separate context boundary remains. Existing synthetic C++ 1e-4 gates remain unchanged; the current CTest result is **11/11**. The latest board evidence includes the bounded 21-decision synthetic scope and sparse recorded development USB replay.

The canonical notebook filters longer resampled recordings before cutting windows. The repair matches **per-window** SciPy; it cannot establish equivalence to the training export. A diagnostic comparison that filters the complete manifest segment before selecting its first window differs from per-window filtering by as much as **0.011639** probability across the eight recorded examples. This is a segment-context diagnostic, not an exact reconstruction of the full-recording training pipeline.

The 5-second future-context deployment contract, USB diagnostic replay, and 162-recording host audit are documented in [training-context-decision.md](training-context-decision.md). That audit’s one top-3 disagreement is a `std`/`minimum` \|Δ\| near-tie (~3e-6); the top-3 set is unchanged. The adopted local-scale criterion is now the live feature-parity requirement; original absolute 1e-4 errors remain reported. No retraining or threshold tuning was performed.

The inference comparison uses constants currently exported in the firmware header. It does not establish equivalence with an original PyTorch checkpoint, which was not available. Offline recall/specificity, AOPC, live captures, board resource/latency measurements, Android, and the PTT-PPG spike were not advanced by this repair.
