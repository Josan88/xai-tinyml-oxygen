# Lamaakal-style results analysis (host only)

24 October 2026. Requested by Dr Sheena at the 23 Oct 2026 meeting: analysis "similar to Lamaakal et al. 2026" (Explainable KANs for Zero-Shot HAR on TinyML Edge Devices, MAKE 8(3):55; local copy under `TinyKAN/`). The IEEE draft has one results table and no figures. This job builds the results sections that paper has, applied to **our frozen models**, so the Results section can be written from them. Paper due 22 Nov 2026.

**Scope.** v1 is the reported system: TinyKAN `(14, 12, 1)`, 297 parameters, notebook-10 labels (as-coded alignment), operating point 0.50, ADR-0005 N = 2. v2 is a comparison only: header-aligned labels, 16 inputs, 321 parameters (ADR-0006). v3 is out of scope and appears in no table. Host only: no COM8, no flashing, no build, no board script. Nothing in `firmware/`, no weights header, no README number, no training-script default and no existing `outputs/` directory was changed. Write-once directories were hashed before and after the run and are unchanged (see Reproduction).

**What the retrained models are.** Every model retrained here (baselines, seed runs, ablations) is an **analysis model**. None is the shipped model, none replaces a header, and none changes the frozen predictions. The tables say which rows are frozen, which are existing out-of-fold files, and which are analysis retrains.

**Explainer.** Mean-perturbation only: each standardized feature is replaced by its training mean (0), N + 1 passes, ranked by |Δ| with ties broken by feature index, as `kan_sort_attributions` does. No SHAP, LIME, KernelSHAP or gradient×input explainer was added.

Outputs: `outputs/analysis-lamaakal-style-v1/` (tables `*.csv`, figures `fig_*.png`, generated sentences `paper_sentences.md`, `summary.json`, `rerun_check.json`). Code: `scripts/analyze_lamaakal_style.py`, `scripts/lamaakal_metrics.py`, `scripts/lamaakal_models.py`. Tests: `tests/test_lamaakal_metrics.py`, `tests/test_lamaakal_doc_sync.py`.

---

## 0. Provenance findings — read before using any number

1. **v1 folds.** The v1 fold assignment was recovered exactly from `split_index` in `outputs/openox_tinykan_predictions.csv`. Each recording sits in one fold, and the notebook-11 LogReg file uses the same folds. Today's sklearn 1.9 `GroupKFold` gives *different* folds on the same table: 73% of windows land in the same fold. All v1 retraining therefore uses `split_index`, not a fresh `GroupKFold`. The split is **recording-grouped**: a patient can appear in a training fold and the held-out fold.
2. **Two v1 LogReg rows exist on disk.** The same-fold comparator is the notebook-11 out-of-fold file `outputs/openox_logreg_local_explanations.csv`: recall **0.866**, specificity **0.261** at 0.50. The README and `docs/technical-remaining-review.md` quote recall 0.859 and specificity 0.264. That is the `calibrate_operating_point.py` reconstruction on sklearn-1.9 folds, which are **not** the TinyKAN folds. The difference is small, but only the same-fold row supports a paired comparison. This analysis uses the same-fold row. README was not edited. Decide which row the paper prints and footnote it.
3. **`top_attribution_feature` is gradient×input, not mean-perturbation.** In notebook 12 (cell 6) that column, and `outputs/openox_tinykan_feature_attributions.csv`, come from `x.grad * x` on the fold models. They are not the on-device explainer and are **not used** here. Any existing "top driver" summary built from them describes a different explainer. Section D recomputes top drivers with the host replica of the firmware mean-perturbation.
4. **`y_pred` is the 0.30 cut.** This is confirmed (`summary.json` → `checks.v1_stored_y_pred_is_0_30_cut = true`). Every threshold metric here is recomputed from `y_prob` at 0.50.
5. **Reproducing the frozen out-of-fold predictions.** Retraining v1 with the notebook-12 recipe (80 epochs, seed 42, `split_index` folds) reproduces `openox_tinykan_predictions.csv` to a maximum |ΔP| of 1.2e-5, with 0 candidate-flag disagreements. Retraining v2 with the `train_tinykan_v2.py` recipe reproduces `predictions_b_v2_tinykan16.csv` **bit for bit on folds 3–5 but not on folds 1–2**. There the maximum |ΔP| is 0.041, 17 of 48,662 candidate flags differ, and AUC changes by −1.3e-4 (`summary.json` → `checks.seed_42_reproduces_frozen_oof`). In side probes (not archived), the same fold-1 job gave two different results depending only on process state (import order, unpickled inputs), and 64-byte-aligned inputs did not remove the effect. The residual is float-level sensitivity of TinyKAN training to process state, amplified over 100 epochs. It is not a recipe change. The same sensitivity limits rerun determinism (Reproduction).
6. **Host replica of the firmware explainer.** `HeaderKAN` in `scripts/lamaakal_models.py` reads a weights header and evaluates `kan_inference.h` in float64 on FP32 constants, generalised to N inputs. On v1 it matches `evaluate_xai_faithfulness.FrozenKAN` and the write-once AOPC arrays to 1.1e-16 in P and 2.8e-16 in Δ. On v2 it reproduces the resubstitution confusion in `outputs/openox-v2/final_model_report.json` exactly (17,322 / 2,545 / 2,679 / 26,116). Its learned φ(u) matches `scripts/plot_tinykan_architecture_splines.py` to 1.6e-7. The bubble-sort tie-break is tested against a transcription of the firmware sort.

---

## Mapping: their section → our table/figure → result → limit

| Lamaakal et al. | Ours | Files | Result (headline) | Limit |
|---|---|---|---|---|
| Table 3, baselines on the same splits | A. Baseline comparison | `a_baselines.csv` | v1: all 8 models AUC 0.571–0.607; TinyKAN 0.606 ranks 2 of 8. v2: TinyKAN 0.908 ranks 5 of 8 (Random Forest 0.934, HistGB 0.931) | Window-level out-of-fold; baseline hyperparameters fixed and modest, not tuned. Raw-waveform 1D-CNN added later in [analysis-1dcnn.md](analysis-1dcnn.md) |
| Table 5 / Fig. 2, sensitivity of their γ | B. Operating-point sensitivity | `b_operating_point_sweep.csv`, `fig_b_operating_point_sensitivity.png` | v1 is fragile: recall 0.878 → 0.767 → 0.405 at 0.45 / 0.50 / 0.55. v2 balanced accuracy is flat (0.873–0.880) | Descriptive only; 0.50 is not re-selected |
| Table 6, multi-seed mean ± sd and paired test | C. Robustness and uncertainty | `c_*.csv` | v1 TinyKAN − LogReg AUC +0.014 (95% CI +0.001 to +0.029). v2 +0.044 (+0.030 to +0.060). The v1 seed-42 recall/specificity sit at the edge of the 6-seed spread | Recording bootstrap only; folds n = 5; DeLong ignores clustering |
| Fig. 3, local attribution case studies | D(1)–(2). Mean-perturbation case studies and global ranking | `d_case_studies.csv`, `d_global_attribution.csv`, `d_explanation_degeneracy.csv`, `fig_d_case_studies_v1.png`, `fig_d_case_studies_v2.png`, `fig_d_global_attribution.png` | v1 top-1 is kurtosis in about 55% of windows of both classes. On saturated v2 windows 15 of 16 Δ are exactly 0 and the payload is filled by index | In-sample (shipped header on its own training table); one window per cell |
| Fig. 4, learned splines with class-wise activations | D(3). Spline inspection | `d_spline_preactivation.csv`, `d_output_spline_shape.csv`, `fig_d_splines_v1.png`, `fig_d_splines_v2.png` | v1 class-wise pre-activations nearly overlap. The v2 output spline peaks at P 0.905 and clamps at 0.890 | Architecture inspection, not operator XAI; hidden units are not detectors |
| (no AOPC in their paper) | D(4). Deletion AOPC | `d_aopc_v1_cited.csv`, `d_aopc_v2_new_run.csv` | v1 (cited) firmware-absolute 0.0443 vs random 0.0283. v2 (new run) −0.3092 vs −0.1612 | In-sample masking; fixed P(risk) target |
| 5.6 / Table 7, ablation with TinyML columns | E. Ablation; int8/LUT/pruning rows in [analysis-quant-prune.md](analysis-quant-prune.md) | `e_ablation.csv`, `e_sweep_bestf1_reference.csv`, `analysis-quant-prune-v1/` | Width 4/12/32 moves v1 AUC by about 0.01. Labels and the red/IR inputs move it by 0.10 and 0.20 | Different rows and folds between v1 and v2 rows; board latency exists only for the two shipped models |
| Table 10, on-device deployment | F. Deployment table | `f_deployment.csv` | Whole decision block median 249.7 ms (v1), 444.3 ms (v2 r2) | Not inference time; GPIO, peak RAM and energy not measured |
| 6.1 / Table 11, misclassification analysis | G. Error analysis | `g_*.csv`, `fig_g_error_analysis.png` | v2 errors concentrate near the 90 cut and in few recordings; v1 errors do not track distance from the cut | v1 distances are to time-misaligned references |

All figures are IEEE double-column width (7.16 in), 300 dpi. Identity is never carried by colour alone: line style plus marker, or solid vs hatched fill, so the figures read in greyscale. Every figure has a CSV twin. Known cosmetic flaw: the two right-hand panel titles of `fig_d_global_attribution.png` clip by one letter at the right edge. A re-layout for the paper will fix that; the committed PNG was not regenerated because the output directory is write-once.

---

## A. Baseline comparison (Table 3 analogue)

**What it shows.** Eight models on the same folds and the same features, separately for v1 (14 IR features, as-coded labels, 52,090 windows, 162 recordings) and v2 (16 inputs, header-aligned labels, 48,662 windows, 154 recordings). TinyKAN and LogReg rows come from existing out-of-fold files and were not retrained. kNN, SVM-RBF, Random Forest, HistGradientBoosting and two MLPs were retrained here on the same folds, with features standardized inside each training fold only. One MLP has 16 hidden units (257/289 parameters, matched to TinyKAN's 297/321); the other is `(64, 32)`. Hyperparameters were fixed before the run and are recorded in `summary.json` → `recipes.baselines`. v2 also has a patient-grouped pair: the existing TinyKAN patient-fold file, and LogReg retrained on those patient folds.

| v1 (recording-grouped) | Size | AUC | PR-AUC | Bal. acc. | F1 | Recall | Spec. |
|---|---:|---:|---:|---:|---:|---:|---:|
| TinyKAN (frozen OOF) | 297 params | 0.606 | 0.652 | 0.580 | 0.699 | 0.767 | 0.394 |
| LogReg (notebook 11 OOF) | 15 params | 0.592 | 0.647 | 0.564 | 0.726 | 0.866 | 0.261 |
| SVM-RBF | ~106,926 SV values | 0.607 | 0.654 | 0.570 | 0.729 | 0.867 | 0.272 |
| MLP (16) | 257 params | 0.605 | 0.648 | 0.582 | 0.673 | 0.698 | 0.466 |
| Random Forest | ~385,421 nodes | 0.590 | 0.655 | 0.560 | 0.599 | 0.560 | 0.560 |
| MLP (64, 32) | 3,073 params | 0.586 | 0.647 | 0.556 | 0.600 | 0.566 | 0.547 |
| HistGradientBoosting | 12,200 nodes | 0.585 | 0.646 | 0.558 | 0.607 | 0.578 | 0.538 |
| kNN (k = 25) | 583,408 stored values | 0.571 | 0.628 | 0.556 | 0.670 | 0.719 | 0.393 |

| v2 (recording-grouped) | Size | AUC | PR-AUC | Bal. acc. | F1 | Recall | Spec. |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | ~236,331 nodes | 0.934 | 0.939 | 0.884 | 0.903 | 0.895 | 0.874 |
| HistGradientBoosting | 12,200 nodes | 0.931 | 0.932 | 0.880 | 0.897 | 0.880 | 0.880 |
| MLP (16) | 289 params | 0.916 | 0.917 | 0.879 | 0.897 | 0.885 | 0.874 |
| SVM-RBF | ~61,767 SV values | 0.913 | 0.898 | 0.878 | 0.894 | 0.875 | 0.881 |
| TinyKAN (existing OOF) | 321 params | 0.908 | 0.908 | 0.880 | 0.900 | 0.893 | 0.868 |
| MLP (64, 32) | 3,201 params | 0.906 | 0.907 | 0.871 | 0.890 | 0.874 | 0.868 |
| kNN (k = 25) | 622,874 stored values | 0.886 | 0.905 | 0.839 | 0.861 | 0.843 | 0.834 |
| LogReg (existing OOF) | 17 params | 0.863 | 0.858 | 0.846 | 0.860 | 0.821 | 0.870 |

**Limits.** "Size" units differ by model family (`size_unit` column), so only the parameter rows compare directly. kNN and the 0.50 cut are prevalence-unbalanced (kNN has no class weights). The SVM is fitted on a fixed 8,000-window subsample of each training fold for cost; `SVC(probability=True)` is deprecated in sklearn 1.9 but still runs Platt scaling. Baselines were not tuned, and tuning would help the tree models more than TinyKAN. A raw-waveform 1D-CNN was not part of this run; it was added later from a new raw-window export (see the update below this section's sentences). Only the v1 rows support the reported system.

#### Paper-ready sentences

Copied verbatim from `outputs/analysis-lamaakal-style-v1/paper_sentences.md`.

- On the reported v1 inputs (14 IR features, as-coded labels), recording-grouped 5-fold out-of-fold windows (52,090 windows, 162 recordings), TinyKAN reached AUC 0.606 with recall 0.767 and specificity 0.394 at 0.50; logistic regression reached AUC 0.592 (recall 0.866, specificity 0.261). Across 8 models on the same folds, AUC ranged from 0.571 (kNN (k=25)) to 0.607 (SVM-RBF); TinyKAN ranked 2 of 8 by AUC.
- On the v2 inputs (14 IR + 2 red/IR, header-aligned labels), recording-grouped 5-fold out-of-fold windows (48,662 windows, 154 recordings), TinyKAN reached AUC 0.908 with recall 0.893 and specificity 0.868 at 0.50; logistic regression reached AUC 0.863 (recall 0.821, specificity 0.870). Across 8 models on the same folds, AUC ranged from 0.863 (LogReg (L2)) to 0.934 (Random Forest); TinyKAN ranked 5 of 8 by AUC.
- With patient-grouped folds on v2, TinyKAN AUC was 0.907 and logistic regression 0.863.
- No raw-waveform 1D-CNN was run: it needs a raw-window export that this repository does not have.

> **Update, 24 Oct 2026 — the last sentence above is superseded** (it stays because it is the write-once generated text). The raw-window export now exists (`outputs/openox-raw-windows-v1/`) and the 1D-CNN was run on the same folds: [analysis-1dcnn.md](analysis-1dcnn.md). v1: small CNN AUC 0.610 (2,291 params), difference against TinyKAN +0.004 (95% CI −0.008 to +0.016); with both CNNs added, TinyKAN ranks 3 of 10. v2: the pre-registered CNN does not see the raw DC level, so it is not like-for-like (0.784/0.799). A post hoc version with DC side inputs reaches 0.872, still 0.036 below v2 TinyKAN (CI −0.056 to −0.016).

## B. Operating-point sensitivity (Table 5 / Fig. 2 analogue)

**What it shows.** Window-level recall, specificity, balanced accuracy, F1 and Candidate rate from 0.30 to 0.70 in steps of 0.05, for TinyKAN and LogReg on v1 and v2. 0.50 is marked as reported. The v1 TinyKAN rows reproduce `outputs/openox_operating_point_sweep.csv` count for count, with a maximum count difference of 0. That existing sweep (0.01 steps, plus ADR-0005 `confirmed_n2` rows) already answers the v1 TinyKAN question and stays the source for N = 2. Its LogReg rows differ by up to 260 counts because they are the different-fold reconstruction (Provenance item 2).

**Result.** Lamaakal show their γ is robust over a wide range. **v1 is not robust in that sense.** Its out-of-fold probabilities are compressed around 0.5, so recall collapses between 0.50 and 0.60, and LogReg collapses between 0.50 and 0.55. v2 TinyKAN is flat across the whole range. The 97.1% recall quoted for v1 is the 0.30 cut, and stays a footnote.

**Limits.** Descriptive only. 0.50 is not re-selected from this sweep. The ETPB dual gate (recall ≥ 0.80 and specificity ≥ 0.50) remains unreachable for v1 (`docs/operating-point-calibration.md`). v2 TinyKAN meets it at every swept threshold 0.30–0.70; v2 LogReg meets it at 0.35–0.50 (0.821 / 0.870 at 0.50) but not at 0.30 or 0.55–0.70 (`b_operating_point_sweep.csv`). v2 was built after v1 failed, on offline header-aligned labels. Window level, not N = 2 confirmed.

#### Paper-ready sentences

Copied verbatim from `outputs/analysis-lamaakal-style-v1/paper_sentences.md`.

- The v1 operating point is fragile: v1 TinyKAN recall is 0.878 at 0.45, 0.767 at 0.50, 0.405 at 0.55 and 0.007 at 0.60, because its out-of-fold probabilities are compressed around 0.5. v2 TinyKAN balanced accuracy stays between 0.873 and 0.880 across 0.30-0.70.
- Between thresholds 0.30 and 0.70, v1 TinyKAN recall fell from 0.971 to 0.000 while specificity rose from 0.092 to 1.000; at the reported 0.50 it was 0.767 and 0.394 (balanced accuracy 0.580).
- The 97.1% recall quoted for v1 is the 0.30 cut (recall 0.971, specificity 0.092); it is not the operating point.
- At 0.50, v2 TinyKAN gave recall 0.893 and specificity 0.868. The sweep is descriptive; 0.50 was fixed before it and is not re-selected from it.

## C. Robustness and uncertainty (Table 6 analogue)

**What it shows.**
1. **Per fold** (`c_per_fold.csv`, `c_fold_summary.csv`), mean ± sd over the 5 folds. v1 TinyKAN AUC 0.609 ± 0.024, recall 0.766 ± 0.053, specificity 0.394 ± 0.021; v1 LogReg AUC 0.589 ± 0.018. v2 TinyKAN 0.911 ± 0.019; v2 LogReg 0.863 ± 0.028.
2. **Recording bootstrap** (`c_bootstrap_ci.csv`): 2,000 resamples of whole recordings (patients for the patient-grouped v2 pair), seed 20261024 + index, 95% percentile intervals for AUC, recall and specificity at 0.50.
3. **Paired TinyKAN vs LogReg** (`c_paired_comparison.csv`). Three methods are reported: the paired recording bootstrap of the difference (the one to report), a paired DeLong test (implemented and tested against the structural-components formula, but it treats windows as independent, so on this data its p-values are anti-conservative), and a fold-level paired t-test (n = 5, weak). v1 AUC: bootstrap difference +0.014 (CI +0.001 to +0.029; 1.8% of resamples ≤ 0). The 5-fold t-test gives p = 0.12, and TinyKAN wins 4 of 5 folds. The paired bootstrap also shows the v1 trade-off: TinyKAN recall is 0.099 lower and specificity 0.133 higher than LogReg at 0.50. v2 AUC: +0.044 (CI +0.030 to +0.060), with recall +0.071 and no specificity difference (CI −0.026 to +0.015).
4. **Seed spread** (`c_seed_runs.csv`, `c_seed_summary.csv`). Six seeds (42, 1–5) with the exact CV recipe and the same folds: notebook 12 (80 epochs) for v1 and `train_tinykan_v2.py` (100 epochs) for v2. The CV recipe was used rather than `train_final_model.py`, because it is the recipe that produced the frozen out-of-fold predictions. The shipped header is the all-rows 100-epoch fit, so it is not an out-of-fold model. Torch 2.12.0+cpu was already installed in `C:\Python314` (the interpreter `train_tinykan_v2.py` recorded). No venv was needed. The whole run, including all 37 retraining jobs, took about 19 minutes wall on 8 workers.

**Result.** The v1 AUC advantage over LogReg is real but small: its lower bound is barely above zero. AUC itself is stable across seeds (0.611 ± 0.003). **The reported v1 operating point is not.** Seed 42 has the lowest AUC and the highest recall of the six seeds, and the 0.50 recall/specificity split moves by about 0.07 between seeds. v2 is stable on every metric (sd ≤ 0.004).

**Limits.** Out-of-fold window metrics on one dataset. The recording bootstrap treats recordings as independent, but patients repeat across recordings in the recording-grouped split. The patient-level resample is reported for v2 only. The seed runs are analysis models.

#### Paper-ready sentences

Copied verbatim from `outputs/analysis-lamaakal-style-v1/paper_sentences.md`.

- v1: a recording-level bootstrap (2000 resamples of 162 recordings) gives TinyKAN AUC 0.606 (95% CI 0.586–0.626), recall 0.767 (0.720–0.812) and specificity 0.394 (0.346–0.442); logistic regression AUC is 0.592 (0.572–0.612). The paired TinyKAN minus logistic-regression AUC difference is +0.014 (95% CI +0.001 to +0.029). Across the 5 folds TinyKAN AUC was 0.609 ± 0.024 (mean ± sd).
- v2: a recording-level bootstrap (2000 resamples of 154 recordings) gives TinyKAN AUC 0.908 (95% CI 0.887–0.925), recall 0.893 (0.859–0.922) and specificity 0.868 (0.838–0.894); logistic regression AUC is 0.863 (0.836–0.888). The paired TinyKAN minus logistic-regression AUC difference is +0.044 (95% CI +0.030 to +0.060). Across the 5 folds TinyKAN AUC was 0.911 ± 0.019 (mean ± sd).
- Retraining the v1 TinyKAN recipe with 6 seeds (42 1 2 3 4 5) on the same folds gave out-of-fold AUC 0.611 ± 0.003, recall 0.719 ± 0.025 and specificity 0.447 ± 0.027 at 0.50. These are analysis models, not the shipped model.
- Retraining the v2 TinyKAN recipe with 6 seeds (42 1 2 3 4 5) on the same folds gave out-of-fold AUC 0.908 ± 0.002, recall 0.895 ± 0.003 and specificity 0.867 ± 0.003 at 0.50. These are analysis models, not the shipped model.
- The reported v1 seed (42) has the lowest AUC (0.606) and the highest recall (0.767) of the six v1 seeds; the other five give recall 0.699-0.721 and specificity 0.447-0.469. The reported 0.50 trade-off is one draw from a seed-sensitive split, not a stable property of the recipe.
- Windows from one recording are correlated, so window-level DeLong p-values in c_paired_comparison.csv are anti-conservative; the fold-level t-test has n = 5. The recording bootstrap is the comparison to report.

## D. Explanation case studies and spline inspection (Figs. 3–4 analogue)

**What it shows.**
1. **Case studies** (`d_case_studies.csv`, `fig_d_case_studies_v1.png`, `fig_d_case_studies_v2.png`). The shipped header is run on its own training table (in-sample, as in the AOPC run), and its 0.50 decisions define TP/FN/FP/TN. **Selection rule:** in each cell, sort windows by (P, row index) and take position ⌊(n − 1)/2⌋, i.e. the median-probability window. All 14 or 16 signed Δ are plotted with the code name and the dashboard gloss, and the three transmitted features are marked #1–#3. "Toward normal" means the feature lowered P relative to its training-mean replacement. It does not mean the patient is normal.
2. **Global ranking** (`d_global_attribution.csv`, `fig_d_global_attribution.png`): mean |Δ| per feature and the share of windows where each feature is top-1 (and top-3), split by y = 1 and y = 0.
3. **Explanation degeneracy** (`d_explanation_degeneracy.csv`). This is not in Lamaakal; it was added because the case studies exposed it. It reports the share of windows where a top-3 slot carries Δ = 0 (filled by feature index), where the output spline is clamped, and where the top-1 feature works against the decision (lowered P on a Candidate, or raised P on a non-candidate).
4. **Spline inspection** (`d_spline_preactivation.csv`, `d_output_spline_shape.csv`, `fig_d_splines_v1.png`, `fig_d_splines_v2.png`). The learned φ(u) of the 12 hidden units and the output come from the shipped headers. They are cross-checked against `plot_tinykan_architecture_splines.py`, which this script imports rather than modifies. Below each φ(u) are the class-wise distributions of that unit's pre-activation u on the training windows. **Caption: architecture inspection, not operator XAI; hidden units are not detectors.**
5. **AOPC.** The write-once v1 run is cited, not rerun into its directory: `d_aopc_v1_cited.csv` is copied from `openox-aopc-v1/summary.json`, and an in-memory recomputation matches it to 1e-17. A **new** v2 run is written to `d_aopc_v2_new_run.csv`, labelled as new, with the same seed (20260911) and 5 random repeats.

**Result.** v1 explanations are well-formed: no v1 candidate window has a clamped output or a zero-Δ slot in its top-3 (`d_explanation_degeneracy.csv`). They are dominated by the same feature (kurtosis is top-1 on about 55% of windows) in **both** classes, so the top-1 does not separate y = 1 from y = 0. **v2 explanations degenerate on saturated windows.** About 40% of v2 candidate windows sit at the output-spline clamp. On those, every perturbation leaves P unchanged except one that pulls the window back inside the range, where the non-monotone output spline gives a *higher* P. So the transmitted #1 is a feature that lowered P (`ratio_of_ratios` on the median TP and FP windows, Δ = −0.008 and −0.002), and #2 and #3 are `mean` and `std` with Δ = 0, chosen by index. This matches the live sit (score 0.89, top driver `ratio_of_ratios`; ADR-0006). On v2 the median TP and FP windows have the same P and the same top-3: the explanation does not tell a true alarm from a false one.

On AOPC: the ADR-0003 metric fixes the target at P(risk) for every window. On non-candidate windows the |Δ| ranking puts the features that lowered P first, so deleting them *raises* P(risk) and the deletion AOPC goes negative. "Worse than random on non-candidates" is therefore partly a property of the target. With a predicted-class target the non-candidate values flip sign exactly, and firmware-absolute is above random on that subset (derived column `deletion_aopc_predicted_class_target`, arithmetic only, not a new run). The paper should cite the v1 numbers as instructed and state the target. On v2 the firmware ranking is below random even on candidate windows (−0.0085 vs −0.0004; better than its own random mean on 31.7%), which follows from the saturation above.

**Limits.** In-sample: the shipped model is explained on the windows it was fitted to. One window per cell is an illustration, not a sample. Masked feature vectors can be physically impossible because the features are correlated. AOPC uses the full feature order, not the top-3 payload. Δ = 0 here is exact in float64. In FP32 firmware a clamped output also gives identical logits, but that was not checked on the board.

#### Paper-ready sentences

Copied verbatim from `outputs/analysis-lamaakal-style-v1/paper_sentences.md`.

- For the shipped v1 header on its own training windows (in-sample), the median-probability TP window (P = 0.567, 14 of 14 Δ non-zero) has top-3 |Δ| features kurtosis, bandpower_ratio_0_5_2_to_2_5, maximum; the median FP window (P = 0.560, 14 of 14 Δ non-zero) has kurtosis, dominant_frequency_hz, maximum. A feature is 'toward normal' when replacing it with its training mean raises P; it does not mean the patient is normal.
- For the shipped v2 header on its own training windows (in-sample), the median-probability TP window (P = 0.890, 1 of 16 Δ non-zero) has top-3 |Δ| features ratio_of_ratios, mean, std; the median FP window (P = 0.890, 1 of 16 Δ non-zero) has ratio_of_ratios, mean, std. A feature is 'toward normal' when replacing it with its training mean raises P; it does not mean the patient is normal.
- The v2 output spline is not monotone: P peaks at 0.905 near u = 2.67 and is 0.890 at the +3 clamp, where 23.4% of v2 windows sit (d_output_spline_shape.csv). A replacement that pulls a clamped window back just inside the range raises P, so on saturated risk windows the largest |Δ| is often a feature that lowered P.
- v1: the most frequent top-1 feature by |Δ| is kurtosis on y = 1 windows (55.6%) and kurtosis on y = 0 windows (54.4%). A top driver does not identify an ill patient.
- v2: the most frequent top-1 feature by |Δ| is ratio_of_ratios on y = 1 windows (50.8%) and ratio_of_ratios on y = 0 windows (90.4%). A top driver does not identify an ill patient.
- On 36.2% of v2 candidate windows (P >= 0.50) at least one of the three transmitted features has Δ = 0 and is placed by feature index, because the output spline is clamped on both passes (39.7% of candidate windows have a clamped output pre-activation). On 60.8% of v2 candidate windows the top-1 feature lowered P (d_explanation_degeneracy.csv).
- Host deletion AOPC (write-once v1 run): firmware-absolute ranking 0.0443 vs random 0.0283 over all windows; on non-candidate windows firmware-absolute is -0.0536 vs random -0.0273, i.e. worse than random.
- A new in-sample v2 run gives firmware-absolute -0.3092 vs random -0.1612 over all windows, -0.0085 vs -0.0004 on candidate windows, and -0.7402 vs -0.3917 on non-candidate windows (d_aopc_v2_new_run.csv).
- Both AOPC runs fix the target at P(risk). On non-candidate windows a predicted-class target (1 - P) mirrors every curve, so the values change sign (v1 0.0536 vs 0.0273; v2 0.7402 vs 0.3917); 'worse than random on non-candidates' is a statement about the fixed P(risk) target, and should be written with it.
- The spline figures are architecture inspection, not operator XAI; hidden units are not detectors.

## E. Ablation (5.6 / Table 7 analogue)

**What it shows.** `e_ablation.csv` has one row per variant: AUC, PR-AUC, recall and specificity at 0.50, parameters, FP32 weight bytes (parameters × 4), header constant bytes (parameters + scaler + centres + steps + threshold, × 4), and measured board block latency for the two shipped models only.

| Variant | Source | Rows | AUC | Recall@0.50 | Spec@0.50 | Params | FP32 B | Board block median |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Shipped v1 recipe (14 IR, as-coded) | frozen OOF | 52,090 | 0.606 | 0.767 | 0.394 | 297 | 1,188 | 249.7 ms |
| v1 hidden 4 | analysis retrain | 52,090 | 0.602 | 0.818 | 0.331 | 105 | 420 | not measured |
| v1 hidden 32 | analysis retrain | 52,090 | 0.613 | 0.725 | 0.441 | 777 | 3,108 | not measured |
| v1 spline range [−2, 2] | analysis retrain | 52,090 | 0.615 | 0.721 | 0.450 | 297 | 1,188 | not measured |
| v1 6 knots | analysis retrain | 52,090 | 0.600 | 0.772 | 0.379 | 271 | 1,084 | not measured |
| v1 12 knots | analysis retrain | 52,090 | 0.610 | 0.715 | 0.450 | 349 | 1,396 | not measured |
| As-coded labels, 14 IR, 100 epochs | existing `predictions_a_prime_*` | 52,090 | 0.609 | 0.732 | 0.429 | 297 | 1,188 | not measured |
| Header-aligned labels, 14 IR | existing `predictions_a_v2_tinykan14` | 48,662 | 0.712 | 0.610 | 0.680 | 297 | 1,188 | not measured |
| Shipped v2 recipe (16 inputs) | existing `predictions_b_v2_tinykan16` | 48,662 | 0.908 | 0.893 | 0.868 | 321 | 1,284 | 444.3 ms (r2) |
| v2 hidden 4 | analysis retrain | 48,662 | 0.899 | 0.887 | 0.869 | 113 | 452 | not measured |
| v2 hidden 32 | analysis retrain | 48,662 | 0.907 | 0.899 | 0.865 | 841 | 3,364 | not measured |
| 2 red/IR inputs only | analysis retrain | 48,662 | 0.886 | 0.902 | 0.846 | 153 | 612 | not measured |
| LogReg, as-coded, 14 IR | existing (notebook 11) | 52,090 | 0.592 | 0.866 | 0.261 | 15 | 60 | never deployed |
| LogReg, header-aligned, 14 IR | analysis retrain | 48,662 | 0.676 | 0.693 | 0.568 | 15 | 60 | never deployed |
| LogReg, header-aligned, 16 | existing `predictions_c_v2_logreg16` | 48,662 | 0.863 | 0.821 | 0.870 | 17 | 68 | never deployed |
| LogReg, header-aligned, 2 red/IR only | analysis retrain | 48,662 | 0.871 | 0.826 | 0.867 | 3 | 12 | never deployed |

`outputs/openox_tinykan_sweep_results.csv` reports width and range at each configuration's best-F1 cut (0.25–0.35), not at 0.50. `e_sweep_bestf1_reference.csv` lists those rows next to the 0.50 recomputation where per-window predictions exist. The analysis retrains reproduce the sweep's best-F1 recall exactly (hidden 4: 0.9768; hidden 32: 0.9677; range [−2, 2]: 0.9505), so they are the same models the sweep trained.

**Result.** Capacity is not the ceiling. On v1, hidden 4/12/32 span AUC 0.602–0.613 for a 7× change in parameters, and range and knots move it by a similar amount. That is about the size of the seed spread (sd 0.003, range 0.606–0.614). Label alignment moves the 14-IR AUC by +0.10, and the two red/IR inputs move it by another +0.20. LogReg on the two red/IR inputs alone (3 parameters) matches or beats LogReg on all 16. The evidence for "alignment plus red/IR inputs were the ceiling" is ADR-0006's table and `outputs/label-alignment-v1/README.md` (ratio-of-ratios AUC 0.619 as-coded vs 0.860 header-aligned). No file named "ceiling check" exists on disk.

**Limits.** The as-coded and header-aligned rows are different exams: different rows (52,090 vs 48,662), different folds, and the as-coded 100-epoch row uses sklearn-1.9 folds, not `split_index`. Only the two shipped models have board latency. Every other latency is "not measured", because those models never ran on the board.

**int8 + LUT splines and pruning** were later run as host-only, never-deployed rows: [analysis-quant-prune.md](analysis-quant-prune.md). int8 weights leave v1 AUC unchanged (0.606) at 509–597 header bytes against 1,376. int8 activations keep AUC but drop explanation top-1 agreement to 0.82. Pruning 10–25% keeps AUC but moves the 0.50 operating point. 50% costs 0.035 AUC. **Not ported from Table 7:** dropout and window-length rows (see Not ported).

#### Paper-ready sentences

Copied verbatim from `outputs/analysis-lamaakal-style-v1/paper_sentences.md`.

- Hidden width does not move v1: out-of-fold AUC 0.602, 0.606, 0.613 for 4, 12 and 32 hidden units (v2: 0.899, 0.908, 0.907). Capacity is not the ceiling.
- Header-aligned labels lift the 14-IR TinyKAN from AUC 0.609 to 0.712 (different rows and folds), and adding the two red/IR inputs lifts it to 0.908; the two red/IR inputs alone give 0.886.
- Logistic regression on the two red/IR inputs alone reaches AUC 0.871, against 0.863 on all 16 inputs and 0.676 on the 14 IR features: most of the v2 gain comes from the red/IR relation, not from the spline network.

## F. Deployment table (Table 10 analogue)

**What it shows.** `f_deployment.csv`, every value read from a file on disk. No build was run.

| | TinyKAN v1 (14, 12, 1) | TinyKAN v2 (16, 12, 1), r2 | LogReg v1 / v2 |
|---|---|---|---|
| Board | Nano 33 BLE Sense Rev2, nRF52840 Cortex-M4F, 64 MHz | same | not deployed |
| Params / FP32 weight bytes | 297 / 1,188 B | 321 / 1,284 B | 15 / 60 B; 17 / 68 B (host only) |
| Forward passes per decision | 15 | 17 | not measured (never deployed) |
| Live image flash / static RAM | 342,244 B / 88,008 B (`firmware-v1-rollback/rollback-receipt.json`) | 343,620 B / 94,032 B (`firmware-v2-f1a-r2/build.log`; built, not flashed) | not measured (never deployed) |
| Timing image flash / static RAM | 336,588 B / 85,304 B (`docs/on-device-timing.md`) | 337,524 B / 92,320 B (`timing-v2-prepared-r2/build.log`) | not measured (never deployed) |
| Decision block, subsequent (n = 180): median / p95 / max | 249.7 / 250.5 / 250.8 ms | 444.3 / 445.9 / 446.2 ms | not measured (never deployed) |
| First decision after reset, median | 204.2 ms | 355.8 ms | not measured (never deployed) |
| Headroom vs 640 ms FIFO (worst block) | 2.55× | 1.43× | — |
| Board parity | 2,464 / 2,464 locked windows pass the adopted local-scale checks; original 1e-4 failed 2,464 / 2,464; max ΔP 8.25e-7 | 1,243 / 1,243 decisions pass; original 1e-4 vs Python failed 1,243 / 1,243; max board–host ΔP 8.35e-7 | not measured (never deployed) |
| `parity_complete` | false (report field) | not a field in the r2 board report; host r2 run 7 / 71,462 fail and `parity_complete` false (`docs/v2-flash-runbook.md`) | — |
| `physical_gpio_verified` | false | false | — |
| Peak RAM | not measured (stack watermark and heap snapshots are not whole-device peak RAM) | not measured | — |
| Energy | not measured (no power instrumentation). CPU-only **estimate**: 2.47–4.72 mJ per decision (datasheet 3.3 / 6.3 mA × 3 V × 249.7 ms; `outputs/energy-estimate-v1/`) | not measured. Estimate 4.40–8.40 mJ (× 444.3 ms) | — |

The decision block is the **whole** 250-sample block: pushes, filtering, features, one baseline inference, the perturbation passes, the sort, and the debounce/payload step. It is **not inference time**, and the file does not split inference from explanation. Latency is DWT cycle-derived at 64 MHz on accelerated synthetic USB input, not physically calibrated (D2 GPIO unmeasured), with no sensor or BLE latency. The pre-fix v2 image (15 Oct) measured 433.4 ms; the r2 numeric source added about 11 ms.

**Limits.** Static RAM is linker output, not peak RAM. The v2 r2 live image was built but never flashed; its sizes come from the build log.

#### Paper-ready sentences

Copied verbatim from `outputs/analysis-lamaakal-style-v1/paper_sentences.md`.

- On the Nano 33 BLE Sense (nRF52840, 64 MHz), the whole 250-sample decision block, including the explanation passes, has a cycle-derived median of 249.7 ms for v1 and 444.3 ms for v2 on synthetic USB input; this is not inference time, and GPIO timing, peak RAM and energy were not measured.
- Board parity: v1 2464/2464 locked windows pass the adopted local-scale checks; v2 r2 1243/1243 decisions pass. parity_complete stays false.

## G. Error analysis (6.1 analogue)

**What it shows.** For TinyKAN v1 and v2 out-of-fold at 0.50:
- error rate by the reference median ScalcO2's distance from 90 (`g_error_by_spo2_distance.csv`; no window has a median in [89, 91], so the bins jump from −1 to +1);
- concentration of FP and FN by recording, as the number of recordings that hold half of each error type (`g_error_concentration.csv`);
- v2 error rates by `ir_perfusion_index` quartile, window-level quartiles (`g_error_by_perfusion_quartile.csv`). There is no perfusion input in the v1 table.

`fig_g_error_analysis.png` draws all three. The subgroup analyses already on disk are linked, not redone: [tinykan-fairness-subgroups.md](tinykan-fairness-subgroups.md) and [tinykan-motion-subgroups.md](tinykan-motion-subgroups.md).

**Result.** v2 behaves like a model that tracks the reference: its errors pile up within a few points of the cut and in a minority of recordings. v1 does not. Its FP rate is 0.55–0.66 in every bin above 90 and its FN rate barely falls with distance. That is consistent with its labels being time-misaligned (ADR-0006), but this analysis cannot separate misalignment from inability. v2 errors are highest at both perfusion extremes: mostly false positives at the lowest quartile, mostly false negatives at the highest.

**Limits.** Window-level. Recordings are not independent of patients. The v1 distances are to as-coded references. Quartiles are of the training-table distribution, not of any live sensor.

#### Paper-ready sentences

Copied verbatim from `outputs/analysis-lamaakal-style-v1/paper_sentences.md`.

- v1: half of the 13,032 false positives sit in 47 of 162 recordings, and half of the 7,133 false negatives in 23.
- v2: half of the 2,622 false positives sit in 20 of 154 recordings, and half of the 3,090 false negatives in 13.
- v2 errors concentrate near the cut: the FN rate is 0.417 for reference ScalcO2 1-2 points below 90 and 0.062 more than 10 below; the FP rate is 0.177 1-2 points above and 0.009 more than 10 above (115 windows). v1 shows no such gradient: its FP rate is 0.548-0.655 in every bin above 90.
- v2 error rate is highest in the lowest ir_perfusion_index quartile (0.155, mostly FP: FP rate 0.211) and the highest quartile (0.144, mostly FN: FN rate 0.190).
- v1 distances are to the as-coded (time-misaligned) reference, so v1 error-by-distance does not measure closeness to a true saturation cut.

---

## Not ported, with reasons

- **Zero-shot learning, semantic embeddings, calibration factor γ** (their 5.2–5.3, 5.7.1–5.7.2, Tables 4, 8, 9). This is a binary single-task problem with no unseen classes. Section B ports only the idea of a sensitivity sweep.
- **UCI HAR, WISDM, PAMAP2.** These datasets have no PPG and no oxygen labels.
- **Deploying int8 + LUT splines or a pruned net** (Table 10 int8 columns). The shipped net is native FP32 (ADR-0004). Quantising changes the weights, invalidates parity, timing and the on-device mean-perturbation, and would create a model that never ran. Host-only ablation rows (Table 7 analogue) now exist, labelled never deployed: [analysis-quant-prune.md](analysis-quant-prune.md).
- **Gradient×input and SHAP-style scores** (their 3.5.1–3.5.4). Neither is the on-device explainer. The only gradient×input on disk (notebook 12's `top_attribution_feature`) is flagged in Provenance item 3 and not used.
- **Measured energy per inference.** No power instrumentation exists; not measured. A CPU-only datasheet estimate exists (`scripts/estimate_energy_datasheet.py`, `outputs/energy-estimate-v1/`): v1 2.47 mJ (DC/DC) to 4.72 mJ (LDO) per decision block, v2 4.40–8.40 mJ. It excludes the sensor LEDs, IMU, radio and idle current, and must be labelled an estimate.
- **Dropout and window-length ablations.** TinyKAN has no dropout, and the window length is fixed by the feature pipeline and the firmware (500 samples, 250 stride).
- **Their headline numbers are not a target.** The in-repo reproduction (`TinyKAN/reproduction/zsl_diagnostics_findings.md`) did not recover their zero-shot accuracies: pure ZSL 47.7% against the paper's 96.4% on UCI HAR, with seen-class accuracy 95.0%. Their tables show which analyses exist, not what our numbers should look like.
- ~~Raw-waveform 1D-CNN~~ — now run: [analysis-1dcnn.md](analysis-1dcnn.md).

---

## What this means for the paper (advisor verdict)

**Supported.**
- On the reported v1 inputs, TinyKAN's discrimination is in line with seven other learners. It is higher than LogReg by a small margin whose recording-bootstrap interval excludes zero (+0.014, CI +0.001 to +0.029). At 0.50 it trades about 10 points of recall for 13 points of specificity relative to LogReg.
- The deployed model is small (297 parameters, 1.2 kB FP32) and its whole decision block, explanation included, fits the 5 s stride with 2.55× FIFO headroom on synthetic input.
- Capacity is not the limit; inputs and labels are (Section E). A raw-waveform 1D-CNN with 8× the parameters reaches the same v1 ceiling (AUC 0.610; paired difference +0.004, CI includes zero), so the hand-built features are not what holds v1 back ([analysis-1dcnn.md](analysis-1dcnn.md)).

**Not supported. Do not write it.**
- "TinyKAN outperforms the baselines." On v1 it ranks 2 of 8, inside a 0.571–0.607 band (3 of 10 with the two 1D-CNNs, band 0.571–0.610). On v2 it ranks 5 of 8.
- "The operating point is robust." On v1 recall moves from 0.878 to 0.405 between 0.45 and 0.55. The reported recall/specificity is one seed's draw, at the edge of a six-seed spread.
- Any v2 explanation claim made without the degeneracy result. If v2 appears in the paper, Section D's sentence on zero-Δ slots and the non-monotone output spline must go with it.
- Crediting the spline network for the v2 jump. LogReg on two red/IR inputs gets 0.871.

**Concrete next actions (paper, not new experiments).**
1. Pick one v1 LogReg row (same-fold 0.866/0.261 recommended), use it everywhere, and footnote the other.
2. Put the B figure in the paper with one sentence on v1 fragility. It is the honest analogue of their Fig. 2.
3. Quote the C bootstrap intervals instead of bare point estimates in the v1 results table.
4. When citing the AOPC non-candidate result, state the fixed P(risk) target in the same sentence.
5. Check the IEEE draft for any use of notebook-12 `top_attribution_feature` and remove it.

---

## Reproduction

```text
C:\Python314\python.exe scripts/analyze_lamaakal_style.py            # writes outputs/analysis-lamaakal-style-v1/ (refuses if it exists)
C:\Python314\python.exe scripts/analyze_lamaakal_style.py --output <new dir> --jobs 8
C:\Python314\python.exe -m pytest tests/test_lamaakal_metrics.py tests/test_lamaakal_doc_sync.py
```

Environment (recorded in `summary.json`): Python 3.14.5 (`C:\Python314`), numpy 2.4.6, pandas 3.0.3, scipy 1.17.1, scikit-learn 1.9.0, torch 2.12.0+cpu, matplotlib 3.10.9. Thread counts are forced to 1, and each retraining job runs in its own freshly spawned process. Seeds: TinyKAN retrains 42 and 1–5; bootstrap 20261024 (+0/+1/+2 per comparison), 2,000 resamples; AOPC 20260911 with 5 random repeats; baselines 42. The default `python` on this machine (hermes venv) has no torch; use `C:\Python314\python.exe`. The committed run took 1,158 s on 8 workers. `--job-cache` exists for development only, and `summary.json` → `job_cache_used` is false for the committed run.

**Rerun check** (`rerun_check.json`; written by the verification step, not by the script). A second full run into `build/lamaakal-scratch/rerun` produced **26 of 32 files byte-identical**. That includes every baseline, LogReg, bootstrap, paired test, AOPC, case study, figure and `paper_sentences.md`. The 6 files that differ all derive from retrained TinyKANs (`c_seed_runs.csv`, `c_seed_summary.csv`, `e_ablation.csv`, `e_sweep_bestf1_reference.csv`, and the two `oof_*` archives). The largest difference in any reported metric is 1.7e-4 (v2 seed 2 recall). Seed means and sds move by at most 6.7e-5, and ablation AUCs by at most 2.1e-6. The affected jobs differ mostly on fold 1. The same fold run standalone was bit-identical across four fresh processes, so the variation comes from process state inside the concurrent pool (Provenance item 5). **Full byte-level determinism of the retrained TinyKANs was not achieved.** Rounded to three decimals, every sentence in the paper-ready blocks is unchanged.

**Write-once directories.** `outputs/xai-faithfulness/openox-aopc-v1/`, `outputs/ptt-ppg/subject-1-v1/`, `outputs/ptt_ppg_motion_spike.csv` and `outputs/parity/qualification/locked-board-replay-v1/` were hashed file by file before and after the run, and the hashes are unchanged (`summary.json` → `write_once_unchanged`, `write_once_sha256`). The recorded key hashes still match: AOPC `summary.json` `3524d43a…bf74c`, PTT `report.json` `fed42702…67b8`, locked replay `report.json` `76341731…4d9a`, motion-spike CSV `27a91e01…4734`.
