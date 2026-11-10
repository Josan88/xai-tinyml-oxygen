# Analysis addenda (29 Oct 2026)

Two host-only analyses that fill gaps the IEEE draft stated as limits. Script `scripts/analyze_addenda.py`; outputs `outputs/analysis-addenda-v1/` (README, `summary.json` with input hashes, seeds, versions, checks and headlines). No board I/O. Every retrained model is an analysis model; no header, weight, threshold (0.50), N=2, label, input or recipe changed. Write-once directories (`openox-aopc-v1`, `subject-1-v1`, `locked-board-replay-v1`, `analysis-lamaakal-style-v1`) were hashed before and after the run and are unchanged.

## Reproduction checks (run before any result is written)

- **Recording folds, v1 seed 42.** Retrain AUC 0.6063, recall 0.7668, specificity 0.3941 at 0.50, identical to the frozen out-of-fold predictions; 0 candidate-flag disagreements. Max |ΔP| was 1.2e-5 in one run and 2.2e-4 in another: CPU training of the TinyKAN recipe is not bit-deterministic across runs. LogReg-14 retrain: same recall/specificity, max |ΔP| 1.3e-5.
- **AOPC same code path.** The shipped v1 and v2 headers reproduce the cited deletion AOPC (v1 0.0443 vs random 0.0283; v2 −0.3092 vs −0.1612). Every cell is within 1e-8 of the cited CSVs, which print 8 significant digits.
- Two check bounds in the first draft of the script were tighter than the arithmetic allows and stopped the run with identical metrics: the LogReg probability bound (1e-6, now the TinyKAN criterion: identical flags at 0.50 and AUC within 1e-4) and the AOPC bound (1e-9, now 1e-8 because of the CSV rounding). Neither bound is a frozen project gate.

## Analysis 1: v1 (baseline) patient-grouped

GroupKFold(5) on `patient_id` over the same 52,090 windows (91 patients; labels as coded, so a grouping change only, not a realignment). The generator regenerates the v2 patient fold column exactly; on the v1 rows the patient-to-fold assignment differs because the window counts differ. Under the recording folds, 35 of 91 patients (33,801 windows) spanned more than one fold.

| Model | AUC (95% patient bootstrap) | Recall | Specificity |
|---|---|---|---|
| TinyKAN seed 42 | 0.604 (0.584–0.625) | 0.762 | 0.395 |
| TinyKAN, six seeds | 0.608 ± 0.003 | 0.706 ± 0.030 | 0.453 ± 0.031 |
| LogReg-14 | 0.571 (0.550–0.591) | 0.823 | 0.282 |

Paired TinyKAN − LogReg AUC +0.033 (+0.017 to +0.053). Recording-grouped v1 was 0.606 (seed spread 0.611 ± 0.003), so patient grouping barely moves the baseline. The determinism rerun reproduced seed 42 byte for byte; seeds 1 and 5 moved by at most 1.4e-5 AUC and 1.3e-4 recall.

## Analysis 2: logistic regression on the fidelity axis, and cost

No exported all-window LogReg exists in the repository, so one was fitted per version (`BASELINE_SETTINGS['logreg']`, own StandardScaler; coefficients in `a2_logreg_coefficients.csv`). Same procedure as the TinyKAN runs: in-sample, training-mean replacement, target fixed at the window's P, firmware-absolute order of the n+1-forward deltas, random order 5 repeats at seed 20260911.

| Version | Model | All: abs / random | Candidates: abs / random | Non-candidates: abs / random |
|---|---|---|---|---|
| v1 | TinyKAN | 0.0443 / 0.0283 | 0.0958 / 0.0576 | −0.0536 / −0.0273 |
| v1 | LogReg | 0.0046 / 0.0050 | 0.0275 / 0.0177 | −0.0885 / −0.0466 |
| v2 | TinyKAN | −0.3092 / −0.1612 | −0.0085 / −0.0004 | −0.7402 / −0.3917 |
| v2 | LogReg | −0.0851 / −0.0393 | 0.0943 / 0.0543 | −0.2989 / −0.1509 |

Reading: a linear model is not faithful by construction under this test. The firmware-absolute sort with a fixed P target makes both models worse than random on non-candidates. On the baseline, LogReg's absolute order does not beat random on all windows, and its gain on candidates (+0.0098) is smaller than TinyKAN's (+0.0382). On the header-aligned table, LogReg's order beats random on candidates where TinyKAN's does not. Subsets are each model's own candidates, and AOPC scales with each model's P range, so compare each model with its own random order, not raw AOPC across models. This is host, in-sample, exported-model behaviour. It is not board XAI and not clinical faithfulness.

Cost (counted from the model form, `a2_cost_facts.csv`): LogReg has 15 / 17 parameters and 14 / 16 multiply-adds per forward. Its n+1 mean-perturbation forwards cost 210 / 272 multiply-adds, and the logit contributions w_i·z_i come free from one forward. TinyKAN v1 / v2 has 297 / 321 parameters, about 284 / 308 multiply-adds per forward excluding basis arithmetic, and 15 / 17 forwards. LogReg was never deployed, so it has no board time, flash or RAM.

## Where it went

- IEEE draft, 29 Oct (backup `…backup-2026-10-29-pre-patient-grouped-lr-aopc.docx`): Table I baseline patient-grouped AUC 0.604; the Data paragraph points at the rerun; one Discussion sentence on the LogReg AOPC and its dot-product form. Paid for with cuts, Word still 6 pages.
- The fuller tables (LogReg patient-grouped, seed spread, all subsets, cost) are queued for the final report.
- 5 Nov 2026: the seven-metric patient-bootstrap table (v1 and v2, recording- and patient-grouped folds, paired TinyKAN − LogReg differences, recording-vs-patient interval widths) is `outputs/patient-bootstrap-v1/`. Script `scripts/analyze_patient_bootstrap.py`. Not edited into the IEEE draft: panel S22 / R1-W6 was deferred to the final report.
