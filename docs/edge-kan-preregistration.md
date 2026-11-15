# Edge KAN (Liu-style spline-per-edge network): host pre-registration (11 Nov 2026)

Written **before** any edge KAN number was computed: no edge KAN model has been trained, and no cross-validation, export, parity or AOPC figure for it exists. Host only, laptop, no board, nothing flashed. Not committed at writing time (no instruction to commit), so commit order cannot prove the timing. The run script records this file's SHA-256 in every output, which binds each result to this text. Results go to the final report. The IEEE paper is not edited, and the shipped TinyKAN is not replaced by this document.

## Why

CONTEXT.md states that TinyKAN is not a KAN in Liu et al.'s sense: each layer is a dense linear mix `u = W x + b` followed by one spline per output node. Jason (11 Nov 2026, chat): build a real edge KAN. Store theta as (d_out, d_in, n_knots) and compute `out_j = sum_i sum_k B_k(x_i) * theta[j, i, k]`, with no W. This study builds that model and measures it with the protocol already used for TinyKAN (repo `v2`). It does not re-freeze anything.

## Model (fixed now)

- Layers `(d_in, 12, 1)`. Each layer: `out[j] = sum_i sum_k B_k(clamp(x[i], -3, 3)) * theta[j, i, k]`. No W, no bias, no SiLU base branch, no grid update.
- Basis: the TinyKAN basis, unchanged. Eight centred cubic B-splines, centres `linspace(-3, 3, 8)`, step 6/7, input clamped to [-3, 3] (`BSpline` in `scripts/train_tinykan_v2.py`).
- Parameters: `12 * d_in * 8 + 12 * 8`. That is 1,632 floats (6,528 B) for 16 inputs and 1,440 floats (5,760 B) for 14. Multiply-adds per forward pass: the same counts. Basis evaluations: `8 * d_in + 8 * 12`.
- Initialisation: `theta[j, i, :] = w[j, i] * c`, where `c` is the least-squares identity coefficient vector of the basis on [-3, 3] (as `KANLayer._init_spline_to_identity`) and `w` is Xavier-uniform with seed 42. The network starts as a clamped linear-linear map.
- Training recipe: TinyKAN v2's, unchanged. StandardScaler fit on the training rows, BCE with `pos_weight = n_neg / n_pos`, Adam lr 0.01, weight decay 1e-4, 100 full-batch epochs, seed 42, `torch.set_num_threads(1)`, smoothness 1e-5 times the summed squared second difference of theta along the knot axis (every edge), gradient-norm clip 10.
- **No hyperparameter search.** If a run gives NaN or a constant predictor, that is reported as the result. No rescue tuning in this study.

## Data and comparators (fixed now)

- Rows and labels: `outputs/openox-v2/window_features.csv` (48,662 windows, header-aligned labels, ADR-0006).
- **Primary model, `ekan16`**: the 16 v2 inputs, in `V2_FEATURES` order. This is the like-for-like replacement of the reported TinyKAN.
- **Secondary model, `ekan14`**: the 14 IR inputs only. It matches the 14 x 8 sizing in Jason's request. Comparator: `a_v2_tinykan14`.
- Comparator predictions are read from the write-once TinyKAN run, not retrained: `outputs/openox-v2/predictions_b_v2_tinykan16.csv`, `predictions_b_v2_tinykan16_patient.csv`, `predictions_a_v2_tinykan14.csv`, and `predictions_c_v2_logreg16.csv` for logistic regression (descriptive).

## Measurements (fixed now)

Script: `scripts/train_edge_kan.py` (new). It imports `scripts/train_tinykan_v2.py` and `scripts/evaluate_xai_faithfulness.py` without editing them. Output directory: `outputs/edge-kan-v1/` (new, write-once).

**E0, same-split checks, run first; the script stops if one fails.** The edge KAN's GroupKFold(5) fold assignment must equal the `fold` column of the comparator prediction file, row by row, for each grouping. The `y_true` column must equal the table's labels.

**E1, out-of-fold discrimination and operating point.** GroupKFold(5) by `recording_id` (primary) and by `patient_id` (secondary). Threshold **0.50**, the frozen TinyKAN operating point, applied unchanged. It is not re-tuned for the new model. Reported: AUC, recall, specificity, precision, F1, accuracy, confusion counts, per-fold AUC, and whether the ETPB dual gate (recall >= 0.80 and specificity >= 0.50) holds at 0.50. All are window level and not N = 2 confirmed.

**E2, paired comparison with TinyKAN.** On the recording-grouped out-of-fold predictions: the differences edge KAN minus TinyKAN in AUC, recall at 0.50 and specificity at 0.50, with a patient-cluster bootstrap (resample the 87 patients with replacement, 2,000 resamples, seed 20261111, percentile 95% interval). The same is done for `ekan14` against `a_v2_tinykan14`.

**E3, export and host parity.** Fit `ekan16` on all 48,662 rows, as the shipped TinyKAN was. Write `firmware/Arduino Nano 33/include/edge_kan_weights_v2.h`, a new file included by no firmware translation unit. Also fit and export `ekan14` to `edge_kan_weights_ir14.h` on the same terms. The C engine is `firmware/Arduino Nano 33/include/edge_kan_inference.h` (new). Host probe: zig 0.16.0 c++ (clang) `-std=c++17 -O2 -ffp-contract=off -mcpu=baseline -target x86_64-windows-gnu`, as in the v2 host parity. Checks on all 48,662 rows:
1. PyTorch vs float64 NumPy forward from the parsed header: max |dP| <= 1e-5.
2. C FP32 vs float64 NumPy from the header: max |dP| <= 1e-5 and max |d attribution| <= 1e-5 (mean-perturbation, training-mean reference, 16 deltas).
3. Top-3 indices and order agree on every window, except where two attributions that decide the top 3 are within 1e-5 of each other. Each mismatch is counted and its gap reported.
4. The engine computes the deltas incrementally. Layer 1 is additive in its inputs, so a perturbed pass changes hidden unit j by `phi_ji(ref_i) - phi_ji(x_i)` and only layer 2 is recomputed. It must match a full 17-pass recomputation in the same C build within 1e-6 on every window.

**E4, deletion AOPC (RQ2 protocol on the new model).** In sample, on the all-row `ekan16` export, through the float64 header forward. Training-mean reference (deletion to 0). Rankings: firmware-absolute (descending |delta|, stable, ties by index), random (5 repeats, seed 20260911), signed_risk and reverse_signed, all from `evaluate_xai_faithfulness.ranked_orders`. Subsets: all, candidate (P >= 0.50), non-candidate, label_1, label_0. Before E4 runs, the same harness must reproduce `outputs/analysis-lamaakal-style-v1/d_aopc_v2_new_run.csv` for the shipped `kan_model_weights_v2.h` within 1e-8 on every subset and ranking. The script stops if it does not.

**E5, Cortex-M4 size.** Compile the engine plus the `ekan16` header, and separately the shipped `kan_inference.h` plus `kan_model_weights_v2.h` (`-DOXY_MODEL_V2`), with the PlatformIO `arm-none-eabi-g++` at `-mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16 -Os -ffunction-sections -fdata-sections`. Report per-symbol sizes of the weight arrays and the inference and XAI functions. These are object-file sizes, not a linked image and not board flash.

## Readings, fixed in advance

- **R1, operating point (primary cell: `ekan16`, recording-grouped).** "Meets the gate at 0.50" if and only if pooled recall >= 0.80 and specificity >= 0.50. The 0.50 cut is TinyKAN's frozen threshold carried over. It was not chosen for this model, and passing or failing at 0.50 does not freeze anything for it.
- **R2, against TinyKAN.** For each paired difference: "edge KAN higher" if the 95% interval lies above 0, "lower" if it lies below 0, otherwise "no difference detected at this resolution". The interval resamples patients, not windows. It is not a test of subject generalisation.
- **R3, AOPC (primary cell: candidates).** "Firmware-absolute beats random" if and only if its candidate deletion AOPC is strictly greater than the random mean. Also reported: the fraction of candidate windows where it beats its own random mean, and a patient-cluster bootstrap 95% interval of the mean gain (2,000 resamples, seed 20261111). The TinyKAN cell (-0.0085 vs -0.0004) is the comparison, and it was measured with the same harness.
- **R4, parity.** E3.1 to E3.4 pass or fail as written. A host pass says the C engine evaluates the exported function on this laptop. It is not board parity.

## Not done here, stated now

- No board replay, no board timing, no peak RAM, no live capture. Flashing needs Jason to name the job. The firmware's `main.cpp`, `platformio.ini`, `kan_inference.h` and every shipped weight header are not edited.
- No new operating point. A threshold for the edge KAN would need its own calibration (as `docs/operating-point-calibration.md`) and freeze.
- The label-0 explainer reference, the rubric, the physio concept and the subgroup analyses are not rerun for the new model.
- CONTEXT.md lists "edge-KAN" under _Avoid_ for TinyKAN. That stays true of TinyKAN. Naming the new model in deliverables is Jason's decision.
