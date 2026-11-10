# Patient-grouped rerun with patient-level bootstrap intervals (5 Nov 2026)

Host only. Closes review finding R1-W6 (`docs/ieee-draft-panel-review-2026-11-03/seats/r1-phase2.md`): the published paired bootstrap resampled recordings although 162 recordings come from 91 patients; the patient-grouped rerun was reported for AUC only; and the recall/specificity difference intervals were not printed. Panel S22 deferred this from the IEEE draft to the final report. This file is that evidence. It is not in the draft.

Script `scripts/analyze_patient_bootstrap.py`. Tables `outputs/patient-bootstrap-v1/`. Nothing was retrained. Every probability vector is a frozen out-of-fold artefact. Threshold stays 0.50. Bootstrap: 2,000 resamples of whole patients, 95% percentile intervals, base seed 20261105 + cell index, paired within each cell. A same-environment rerun reproduced `bootstrap_ci.csv`, `paired_difference.csv`, both width tables, `reproduction_check.csv` and the seed tables byte for byte. Point AUCs match the published figures within 1e-5 (the gzipped out-of-fold tables store 7 significant digits).

## What changes

**Patient grouping changes the baseline comparison. It does not change the header-aligned one.** On v1 (52,090 windows, 162 recordings, 91 patients), patient folds leave TinyKAN where it was (AUC 0.606 to 0.604) and cost logistic regression 0.021 (0.592 to 0.571). Under recording folds, 35 of 91 patients (33,801 windows) spanned more than one fold, and the linear model was the larger beneficiary. The paired AUC difference widens from +0.014 (+0.002 to +0.029) to +0.033 (+0.017 to +0.054). On v2 (48,662 windows, 154 recordings, 87 patients) the grouping is nearly inert: TinyKAN 0.908 to 0.907, logistic regression 0.863 either way, difference +0.044 either way, despite 32 of 87 patients spanning recording folds.

**Resampling patients instead of recordings, on the same predictions, does not change a conclusion.** Interval width moves by at most about 12% either way (model-interval width ratio 0.87 to 1.12, median 0.98). No interval changes whether it excludes zero. The baseline AUC difference W6 flagged, lower bound +0.001 under the recording bootstrap, is +0.002 under the patient bootstrap and still excludes zero. At 2,000 resamples the endpoints carry Monte-Carlo noise, and the two units used different seeds, so ratios within a few percent of 1 are not a real effect.

**Three paired differences include zero, and are printed as such.** v1 patient-grouped accuracy +0.011 (−0.003 to +0.024). v2 patient-grouped specificity +0.000 (−0.024 to +0.020). v2 patient-grouped precision +0.008 (−0.007 to +0.019). The v1 recall/specificity trade holds with both intervals: recall −0.061 (−0.097 to −0.027) for specificity +0.113 (+0.073 to +0.153). Across six seeds the v1 patient-grouped TinyKAN gives AUC 0.608 ± 0.003, still above the logistic-regression interval's upper bound of 0.592. Its threshold metrics move more (recall 0.706 ± 0.030, specificity 0.453 ± 0.031).

Paste-ready cells: `outputs/patient-bootstrap-v1/paper_table_patient_bootstrap.csv`.

## Limits

Analysis models in the source out-of-fold tables are not the shipped model. Patient identifiers are the dataset's encounter metadata (`recording_id` to `encounter_id` to `patient_id`), not a clinical record. A cluster bootstrap corrects the variance for patient clustering. It does not correct for anything else the folds share, and it is not an external cohort.
