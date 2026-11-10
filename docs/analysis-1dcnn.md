# Raw-waveform 1D-CNN comparator (host only)

24 October 2026. Closes the Section A gap in [analysis-lamaakal-style.md](analysis-lamaakal-style.md): Lamaakal et al. (MAKE 8(3):55) compare against deep raw-signal baselines; this repository had none because no raw-window export existed. The CNNs here are **analysis models**. They never replace TinyKAN, never become a header, never ran on the board, and have no on-device explanation.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Raw-window export | `scripts/export_openox_raw_windows.py` | `outputs/openox-raw-windows-v1/` (`.npz` not committed, 132 MB; regenerate and compare `SHA256SUMS`) |
| 2. Pre-registered CNN run | `scripts/train_1dcnn_baseline.py` | `outputs/analysis-1dcnn-v1/` (`recipe.json` written before training) |
| 3. **Post hoc** v2 DC supplement | `scripts/train_1dcnn_v2_dc_supplement.py` | `outputs/analysis-1dcnn-v2dc-supplement/` |

**Export join check.** Every row of the two frozen window tables has its filtered window exported in CSV order (v1: 52,090 × IR; v2: 48,662 × IR + red + raw DC means). The 14 IR features (plus ratio_of_ratios and ir_perfusion_index for v2) were recomputed from the stored arrays and compared with the frozen CSVs over all rows. Max scaled difference: 1.4e-14 (v1) and 9.3e-13 (v2) from the float64 waveform, 4.2e-6 and 2.8e-6 from the float32 arrays a model reads. The CNN therefore sees exactly the windows TinyKAN's features came from.

**Recipe (fixed before training).** Same windows, labels and folds as Section A (v1: `split_index`; v2: the `predictions_b_v2_tinykan16` fold column). Each window is z-scored per channel. Its own log(std) and mean/std go to the head as side inputs, which makes waveform plus side inputs a lossless re-encoding of the exported window. Two sizes: small (2.3k parameters) and large (37k). Class-weighted BCE, Adam 1e-3, ≤ 40 epochs, early stop on 15% of the training recordings. Seed 42, plus seeds 0–4 for v1 small. Paired recording bootstrap with Section C's seeds, so the TinyKAN intervals reproduce Section C exactly (checked).

## Results

| | Params | MACs / window | AUC | Recall@0.50 | Spec@0.50 | ΔAUC vs TinyKAN (95% CI) |
|---|---:|---:|---:|---:|---:|---|
| **v1** TinyKAN (shipped) | 297 | 284 per pass, 4,260 per decision (15 passes) | 0.606 | 0.767 | 0.394 | — |
| v1 1D-CNN small | 2,291 | 179,698 | 0.610 | 0.611 | 0.542 | +0.004 (−0.008 to +0.016) |
| v1 1D-CNN large | 37,153 | 3,519,904 | 0.599 | 0.516 | 0.614 | −0.008 (−0.028 to +0.011) |
| v1 1D-CNN small, seeds 0–4 | 2,291 | | 0.615 ± 0.002 | 0.586 ± 0.027 | 0.562 ± 0.021 | |
| **v2** TinyKAN (shipped) | 321 | 308 per pass | 0.908 | 0.893 | 0.868 | — |
| v2 1D-CNN small (pre-registered) | 2,349 | 207,700 | 0.784 | 0.646 | 0.780 | −0.124 (−0.144 to −0.101) |
| v2 1D-CNN large (pre-registered) | 37,361 | 3,591,968 | 0.799 | 0.627 | 0.822 | −0.109 (−0.130 to −0.086) |
| v2 1D-CNN small + raw DC (**post hoc**) | 2,351 | | 0.837 | 0.746 | 0.814 | −0.071 (−0.090 to −0.051) |
| v2 1D-CNN large + raw DC (**post hoc**) | 37,425 | | 0.872 | 0.790 | 0.809 | −0.036 (−0.056 to −0.016) |

TinyKAN MACs count the mixes and spline weights only. The B-spline basis and the 14-feature extraction (Welch FFT) are not counted, so the MAC ratio overstates TinyKAN's advantage. No CNN latency, RAM or energy exists, because no CNN was deployed.

With both pre-registered CNNs added to Section A, v1 TinyKAN ranks 3 of 10 by AUC (range 0.571–0.610), and v2 TinyKAN ranks 5 of 10 (range 0.784–0.934). Combined table: `outputs/analysis-1dcnn-v1/a_baselines_with_cnn.csv`.

## What it means

- **v1 (reported system): a raw-waveform CNN does no better.** It reaches the same ceiling as every feature-level learner, AUC about 0.60–0.62. TinyKAN's own six-seed spread is 0.606–0.614, and the small CNN's is 0.612–0.618. The paired interval includes zero. This supports Section E: the v1 ceiling is in the as-coded labels, not in the choice of features or model. The seed-42 v1 CNNs stop early (best epoch 1–12; the large CNN's best epoch is 1 on three of five folds), which is what a weak label signal looks like.
- **At 0.50 the CNN sits at a different operating point** (recall 0.61, specificity 0.54 against TinyKAN's 0.77 / 0.39). This is a calibration difference, not a discrimination difference. Do not compare recall alone.
- **v2: the pre-registered CNN is not a like-for-like comparator.** `recipe.json` already declared that the CNN does not see the raw DC level, which ratio_of_ratios and ir_perfusion_index divide by. The post hoc supplement adds log(DC) for IR and red as two side inputs. It lifts the CNN by +0.053 (small) and +0.073 (large) AUC, with both intervals above zero. The large CNN with DC is still 0.036 below v2 TinyKAN (CI −0.056 to −0.016). The remaining gap is plausibly the explicit ratio-of-ratios feature, which the CNN would have to learn from 48k windows. Section E shows LogReg on the two red/IR inputs alone already reaches 0.871. This is a v2 comparison only, and v2 is not the reported system.
- **The size and explanation argument holds either way.** The small CNN has about 8× TinyKAN's parameters and about 42× its multiply-accumulates per decision, perturbation passes included and feature extraction excluded. It has no named-feature explanation: mean-perturbation over 500 samples is not operator-readable.

## Limits

- The v2 DC supplement is **post hoc**: it was added after seeing the first v2 result. It is reported next to the pre-registered rows, not instead of them.
- No hyperparameter tuning. The v2 small + DC model hit the 40-epoch cap on 3 of 5 folds, so it is under-trained relative to its own early-stopping rule. The cap was not raised after the fact.
- Window-level, out-of-fold, recording-grouped folds (n = 5). Seed spread was run for v1 small only.
- The CNN training is deterministic per job (`torch.use_deterministic_algorithms`, one thread). The bootstrap reproduces Section C's TinyKAN intervals exactly.

## Paper-ready sentences (verbatim from `outputs/analysis-1dcnn-v1/paper_sentences.md`, pre-registered run)

- On the reported v1 windows (IR waveform, as-coded labels, 52,090 windows, 162 recordings), with the same recording-grouped folds, a raw-waveform 1D-CNN reached out-of-fold AUC 0.610 with 2,291 parameters (recall 0.611, specificity 0.542 at 0.50) and 0.599 with 37,153 parameters (recall 0.516, specificity 0.614). The paired recording-bootstrap AUC difference against TinyKAN is +0.004 (95% CI -0.008 to +0.016) for the small CNN and -0.008 (-0.028 to +0.011) for the large CNN; with both CNNs added, TinyKAN ranks 3 of 10 by AUC.
- Size gap (v1): the small CNN has 2,291 parameters and 179,698 multiply-accumulates per window, the large CNN 37,153 and 3,519,904, against TinyKAN's 297 parameters and 284 per forward pass (feature extraction not counted). Neither CNN was deployed, so its board latency is not measured.
- Across seeds 0-4 the small v1 CNN gave AUC 0.615 ± 0.002, recall 0.586 ± 0.027 and specificity 0.562 ± 0.021 at 0.50 (mean ± sd). These are analysis models, not the shipped model.

The generated v2 sentence (AUC 0.784 / 0.799, −0.124 / −0.109) is correct for the pre-registered run. It must not be quoted without the DC caveat above.

## Reproduction

```text
C:\Python314\python.exe scripts/export_openox_raw_windows.py --workers 6
C:\Python314\python.exe scripts/train_1dcnn_baseline.py --workers 14          # 45 jobs, about 17 min on 20 cores
C:\Python314\python.exe scripts/train_1dcnn_v2_dc_supplement.py --workers 10  # 10 jobs, about 16 min
```

Every script refuses to overwrite an existing output directory. `build/1dcnn-smoke/` is a one-epoch development run and is not evidence.
