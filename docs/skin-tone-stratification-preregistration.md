# Skin-tone stratification of RQ1: host analysis pre-registration (11 Nov 2026)

**Status: approved by Jason on 11 Nov 2026 ("Approve and run") and committed before the script was written. No outcome had been computed at that point.** Host only, laptop, no board, no firmware or header writes. Nothing shipped changes.

**Honest limit of this pre-registration.** Drafted by Claude. Before writing it, Claude read, for the 154 analysed recordings, only the availability and level counts of the skin-tone fields (listed under "Seen before fixing"). No probability, label or metric was joined to a skin-tone field. The ordering rests on this file's commit, which precedes the script's.

## Why

The draft says the descriptor's skin-colour fields are not scored (¶16) and lists that as a limit beside the occult-hypoxaemia finding of [20] (¶110). Roadmap item S17 of `docs/ieee-draft-panel-review-2026-11-10/` asks for the stratified result. `docs/tinykan-fairness-subgroups.md` (26 Aug) skipped this axis only because the fields had not been exported; they are in `encounter.csv` and `spectrophotometer.csv` of the local dataset copy.

**What this can and cannot show.** The label is ScalcO2, estimated from end-tidal oxygen, not a pulse-oximeter reading, so the label does not carry the oximeter bias of [20]. A difference between groups would therefore be a difference in what the PPG inputs let the model do, not a replay of [20]. With 87 patients, an interval that includes zero is **not** evidence of equal performance.

## Seen before fixing (no outcomes)

- All 154 recordings match an `encounter.csv` row. Monk fields: no missing value. Fitzpatrick: 1 missing. Spectrophotometer "Dorsal (B)" rows exist for all 154.
- `monk_dorsal` is coded A to J (taken as Monk 1 to 10). Recordings per level: A 2, B 17, C 6, D 22, E 25, F 19, G 19, H 33, I 10, J 1.
- 19 of 87 patients have more than one `monk_dorsal` level across their recordings (14 for Fitzpatrick), so the group is a property of the **recording**, not the patient.
- Dorsal ITA (median per recording, from `lab_l` and `lab_b`) ranges from about −85 to +56 degrees.

## Measurements (fixed now)

Script `scripts/analyze_skin_tone_stratification.py`, new, editing no existing file. Output `outputs/skin-tone-stratification-v1/` (new, write-once). Scores: the stored out-of-fold probabilities in `outputs/openox-v2/predictions_b_v2_tinykan16.csv` (recording-grouped folds), threshold 0.50, all 48,662 windows. No model is trained or retuned.

**M0 same-code checks, run first; the script stops before writing a number if one fails.**
1. Pooled AUC, recall and specificity from this script reproduce the paper's 0.908, 89.3 and 86.8 percent to the printed precision.
2. Every recording joins exactly one `encounter.csv` row with the same `patient_id`; 154 recordings, 87 patients, 48,662 windows.
3. Group sizes in recordings equal the counts above (light 47, medium 63, dark 44).

**Grouping (primary).** `monk_dorsal` of the recording's encounter: light = A to D (Monk 1 to 4), medium = E to G (5 to 7), dark = H to J (8 to 10). The bands are fixed on external grounds and not from the counts: they are the three Monk groups (1-4, 5-7, 8-10) of the FDA's January 2025 draft guidance "Pulse Oximeters for Medical Purposes", checked against the FDA document on 11 Nov before this file was committed. **Difference from the FDA:** the guidance assigns the group from the forehead; the primary grouping here uses the dorsal site because the PPG is recorded at a finger. The forehead grouping is reported in S2.

**S1 (primary).** Per group: recordings, patients, windows, prevalence, AUC, recall and specificity at 0.50, each with a 95 percent percentile interval from 2,000 resamples of whole patients (seed 20261111 + cell index; a resampled patient brings all their recordings, each keeping its own group). **Primary contrast: recall, dark minus light**, with its paired patient-bootstrap interval. Recall is primary because a missed low-oxygen window is the harm [20] describes.

**S2 (secondary, descriptive).** The same table for specificity and AUC contrasts (dark minus light); for logistic regression (`predictions_c_v2_logreg16.csv`); and for two other groupings: `monk_forehead` in the same three bands (the FDA's site), Fitzpatrick I–II, III–IV, V–VI (the recording with a missing value dropped), and dorsal ITA in thirds of the 154 recordings (ITA = arctan((L* − 50)/b*) in degrees, median of the recording's "Dorsal (B)" rows). No reading is attached to S2.

**S3 (context, descriptive).** Per primary group, the median of the standardised perfusion-index input and of the infrared AC amplitude input, to show whether signal strength differs between groups. No test.

## Readings (fixed now)

- **Primary.** If the interval of the dark-minus-light recall difference excludes zero: "recall differs between the darkest and lightest groups by X points (interval)", with its sign. If it includes zero: "no difference was detected; the interval (a to b) does not rule out a difference of up to max(|a|, |b|) points". The words "fair", "unbiased" and "equivalent" are not used in either case.
- **Minimum size.** A group with fewer than 10 patients is reported with its counts and point estimates only, and no contrast is read from it.
- **Paper wording.** At most two sentences, in the limits paragraph (¶110), replacing "the skin-colour fields were not scored". Stated as exploratory, added after review. No abstract, Table I or conclusion change, whatever the result.
- No headline number is replaced and no threshold is changed.

## Limits known in advance

- 87 patients from one laboratory; the study is not powered for a subgroup difference.
- Groups differ in more than skin tone (which patients, how many recordings, prevalence, perfusion). The contrast is descriptive, not causal.
- Monk ratings are an observer's judgement and vary between a patient's encounters.
- Out-of-fold scores come from recording-grouped folds; 32 of 87 patients span folds.
- The scored set excludes windows with a median from 89 to 91 percent.
- Host scores only. Nothing here speaks to the MAX30102 on the board.

## Amendment A1 (11 Nov 2026, after the first run stopped; before any stratified number)

The first run (script f73555f) stopped while building the groupings and wrote nothing. All 154 recordings have "Dorsal (B)" spectrophotometer rows, as stated above, but in two of them `lab_l` and `lab_b` are empty, so no dorsal ITA can be computed. No probability or metric had been computed when it stopped beyond the pooled M0.1 check.

- The dorsal ITA grouping (S2, secondary) drops those two recordings, as the Fitzpatrick grouping drops its one; thirds are taken over the remaining 152.
- The primary grouping, S1, the readings and every other rule are unchanged.

## Erratum (11 Nov 2026, after the run; no rule or number changes)

"Why" cites roadmap item S17. The skin-tone item of the 10 Nov roadmap is S32.
