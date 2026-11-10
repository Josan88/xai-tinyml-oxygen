# Pre-registration: ScalcO2 (the label) against arterial blood SaO2 (8 Nov 2026)

Written and committed before any paired statistic (difference, cross-table, agreement) between ScalcO2, SaO2 and SpO2 is computed. Host only, no training, no firmware, no board. Results go to the final report only; the IEEE paper is not edited. Named by Jason on 8 Nov 2026.

Plan: `.hermes/plans/2026-11-08-scalco2-vs-sao2.md` (cited below as "plan:line"). Precedent for structure: `docs/watch-head-preregistration.md` (c2a6e8c).

## What this is and is not

- It **is** a check of the label. Every reported Header-aligned model number is graded against window-median ScalcO2 < 90. This study asks how closely ScalcO2 tracks arterial co-oximetry SaO2 at the blood draws in OpenOximetry, using the pulse oximeters tested at the same draws as the yardstick.
- It is **not**:
  - a test of the MAX30102, the board, or any firmware image;
  - a retraining, re-thresholding or re-scoring of the shipped TinyKAN (S1 reads the stored out-of-fold vectors only);
  - a hypoxia or hypoxaemia result;
  - an ISO 80601-2-61 accuracy claim for anything.
- PASS or FAIL changes neither the shipped model, its 0.50 threshold, N=2, nor any reported number (plan:40).
- **The IEEE paper is unaffected.** The vault docx is not edited.

## Question

At blood draws, does the label (ScalcO2) agree with arterial SaO2 at least as closely as the median pulse oximeter tested in the same lab, both as a number (ARMS) and as the below-90 decision (Cohen's kappa)?

## Wording rules (fixed now)

- **ScalcO2** = SaO2 calculated from end-tidal O2 (ETO2) via Severinghaus 1979 (Fong et al. 2025, ref. 20). Column `bloodgas.csv` `ScalcO2` (per draw) and 2 Hz `ScalcO2` (per window, via the v2 export).
- The label is **window-median ScalcO2**. The CSV column `median_spo2` in `outputs/openox-v2/window_features.csv` is ScalcO2 (written at `scripts/export_openox_windows_v2.py:251`, `:281`). It is not SpO2.
- **SaO2** = arterial co-oximetry, `bloodgas.csv` column `so2`.
- **SpO2** = pulse-oximeter reading, `pulseoximeter.csv` column `saturation`.
- Never call ScalcO2 "SpO2", "oxygen level" or "blood oxygen". Never call a ScalcO2 < 90 window "hypoxia".
- In deliverables the model is "TinyKAN" / "Header-aligned model", not "v2" (CONTEXT.md:17). `v2` below names repo files only.

## Source facts (Fong et al., Sci. Data 12:570, 2025, read 8 Nov from nature.com/articles/s41597-025-04870-8)

Copied verbatim from plan:9–16.

- "These plateaus were estimated using a calculated SaO2 [20]"; ref. 20 = Severinghaus, "Simple, accurate equations for human blood O2 dissociation computations", J. Appl. Physiol. 46, 599–602 (1979).
- "The calculated saturation (ScalcO2) is the instantaneously estimated SaO2 based on ETO2. Blood sample draws are labeled and correspond to the 'sample' number in other tables."
- "Participants remained motionless, with each desaturation lasting approximately 20 minutes. Pulse oximeter readings (SpO2) were recorded with each arterial blood sample, and research coordinators were blinded to arterial SaO2."
- "The saturation reading of each test device at the time the sample is taken was simultaneously recorded and was denoted incrementally as the 'sample number' within an encounter."
- Raw PPG "are not time-synchronized to the other waveform data from the same encounter".
- The paper does not say ScalcO2 is fitted to the blood samples.

## Already seen before this pre-registration

### From the plan (copied verbatim from plan:18–26)

(read-only snippets, 8 Nov; marginal counts only — no paired difference, cross-table or agreement statistic was computed)

- `bloodgas.csv` (SHA-256 73c96a32e95ff1c4a483194eb4a35718cca963411ad46c3bba48bfe461e47afc): 32,877 rows, 759 encounters, 224 patients; `type` = 1 on every row. so2 numeric 31,467; ScalcO2 numeric 30,692; both numeric 29,295 rows (748 encounters, 218 patients).
- On the 29,295 both-numeric rows: so2 58.1–100.8 (< 70: 875; 70–100: 28,277; > 100: 143; < 90: 18,853). ScalcO2 59.82–100.00 (< 90: 18,994).
- Rows per (encounter_id, sample): 1 → 5,568 groups; 2 → 12,558; 3 → 302; 4 → 256; 6 → 1. No group mixes `type`. Draws (groups with ≥ 1 both-numeric row): 16,709; in the 154 v2 recordings: 3,458 (6,974 rows).
- `pulseoximeter.csv` (SHA-256 979a37049a94eca0fdcc2803d8a5fd205d0b21d47fda07aecaa5201d39e88d6d): 89,404 rows, 90 devices, 753 encounters; exactly one row per (encounter_id, device, sample_number). saturation NaN 11, > 100: 2 (101, 493), < 50: 76. `probe_location` NaN on 45,670 rows. 79,841 rows sit on a draw with both so2 and ScalcO2; 64 devices have ≥ 100 such rows, 51 have ≥ 200.
- v2 table (`outputs/openox-v2/window_features.csv`): 48,662 windows, 154 recordings, 87 patients. Stored out-of-fold at 0.50: TinyKAN (`predictions_b_v2_tinykan16.csv`) recall 0.8927, specificity 0.8680; logistic regression (`predictions_c_v2_logreg16.csv`) recall 0.8214, specificity 0.8701 (28,795 positives).
- 2 Hz `Sample` markers: present in all 154 v2 recordings, 7,172 marker rows; a sample number can mark more than one row (e.g. 1, 2, 2). 6,471 marker rows fall inside at least one v2 window's reference span; 6,365 inside two (5 s stride, 10 s window).
- v2 export (`scripts/export_openox_windows_v2.py`): reference elapsed time from the first 2 Hz row with a valid Timestamp (l.224–226); window reference span = [start + δ, end + δ) with δ = ppg_start_time_seconds − hz2_first_time_seconds from `outputs/openox_ppg_eligible_encounters.csv` (l.194, 236–241); windows with median within 90 ± 1.0 are dropped (BORDERLINE_MARGIN, l.46, 253–255).

### Re-checked while writing this (8 Nov, `py -3.14`, pandas 3.0.3, numpy 2.4.6, sklearn 1.9.0; schemas, join keys and marginal counts only)

No paired statistic was computed. Confirmed exactly: both SHA-256s above; every `bloodgas.csv` count in plan:20–22 (with `encounter_id` read as str and pandas' default `dropna=True` grouping); 89,404 pulse rows, 753 encounters, 79,841 rows on a draw, > 100: 2 (101, 493), < 50: 76, `probe_location` NaN 45,670; the v2 table counts and both stored recall/specificity pairs; 7,172 / 6,471 / 6,365 marker rows, with 0 marker rows inside three spans, 0 marker rows on a row with an invalid Timestamp, and 0 non-integer `Sample` values.

**Plan statements that are false or incomplete against the files** (the plan text above is left as written; the rules below use these corrected facts):

1. **"90 devices"** counts raw spellings. `device` mixes spellings of one device: `2` and `2.0`, `59` and `59.0`, and one typo `56 .01` (25 rows). A normalised label (`float(device.replace(" ", ""))` formatted `%.2f`) gives **68 devices**. `2` and `2.0` (and every such pair) never appear together in one encounter, so they are the same device spelled two ways. Suffixes `.01`/`.02`/`.03` (e.g. `42.01`, `42.02`) are kept as separate devices: they are separate labels and 5 (encounter, model) pairs carry two suffixed units side by side. By integer model number there are 40.
2. **"64 devices have ≥ 100 such rows, 51 have ≥ 200"** are raw-spelling counts. With the normalised label, on valid readings on draws: **54 devices ≥ 100, 47 ≥ 200**. (By integer model: 39 ≥ 100, 39 ≥ 200.)
3. **"exactly one row per (encounter_id, device, sample_number)"** holds only among rows with a device. **190 rows have no `device`** (3 encounters; 168 of them on draws), and among them (encounter_id, sample_number) repeats 160 times. With the normalised label and device present, the key is unique (0 duplicates).
4. **"saturation NaN 11"**: there are no empty cells. The 11 are **free-text notes** in the `saturation` column (e.g. "very unstable", "Device 42 was reading in the 60s for the last plateau."), so under pandas 3 the column reads as `str` and numeric comparison raises. All 11 have `sample_number` NaN (the only 11 NaN `sample_number`s), so they never join a draw.
5. **Rows per (encounter_id, sample)** omits **257 `bloodgas.csv` rows with no `encounter_id`** (12 patients; 26 groups if NaN keys are kept). None has a numeric ScalcO2, so none forms a draw. Reading `encounter_id` without `dtype=str` gives a mixed-type warning.
6. **Patients per encounter**: 7 encounters carry a second `patient_id` on 1–3 rows. None of those rows is both-numeric. On the 29,295 both-numeric rows every draw has one `patient_id`, and it equals `encounter.csv`'s `patient_id` for that encounter (0 mismatches, 0 missing).

## Data and inputs (frozen; SHA-256 recorded in summary.json)

| Input | Path | SHA-256 (8 Nov) |
|---|---|---|
| Blood gas | `data/physionet.org/files/openox-repo/1.1.1/bloodgas.csv` | 73c96a32e95ff1c4a483194eb4a35718cca963411ad46c3bba48bfe461e47afc |
| Pulse oximeters | `data/physionet.org/files/openox-repo/1.1.1/pulseoximeter.csv` | 979a37049a94eca0fdcc2803d8a5fd205d0b21d47fda07aecaa5201d39e88d6d |
| v2 windows | `outputs/openox-v2/window_features.csv` | d96d93989f1ffa85f72689a2e084c898dacf18d4eafce50712fb0dcf3ad1b3d6 |
| TinyKAN out-of-fold | `outputs/openox-v2/predictions_b_v2_tinykan16.csv` | e1449fcac7d32bd691d4e1168fa3d7254dffb63c832cf937124162aa546a1611 |
| Logistic regression out-of-fold | `outputs/openox-v2/predictions_c_v2_logreg16.csv` | 4ce1e78f08e43e26452cc21b6e5aa1165abe32256b710f9a79dcb1e49e2c80b7 |
| Eligible encounters (δ, 2 Hz paths) | `outputs/openox_ppg_eligible_encounters.csv` | 120734a1a4a460f962b58d5aaeb5b52dbbeeb9aa9bc909d184bacb9c888662b5 |

Also read, not hashed as gated inputs: `data/physionet.org/files/openox-repo/1.1.1/encounter.csv` (patient check only) and the 154 per-recording 2 Hz CSVs named in the eligible-encounters CSV `hz2_csv_path` (S1 only). The script asserts each SHA-256 above at start and stops on a mismatch.

## Unit and cleaning (fixed now)

### Blood gas → draws (plan:30)

- Read with `dtype={"encounter_id": str, "patient_id": str}`. Drop rows with no `encounter_id` (257; counted).
- `so2` and `ScalcO2` through `pd.to_numeric(errors="coerce")`. Both-numeric row = both finite.
- `sample`: assert no NaN and integer-valued, cast to int64.
- **Draw** = (encounter_id, sample) with ≥ 1 both-numeric row. **SaO2_draw** = arithmetic mean of `so2` over those rows; **ScalcO2_draw** = arithmetic mean of `ScalcO2` over the same rows; `n_rows` = their count. Rows where only one of the two is numeric do not enter either mean.
- so2 > 100 values are kept (no clipping). The ranges below are applied to SaO2_draw.
- `patient_id` of a draw: assert exactly one per draw, and equal to `encounter.csv` `patient_id` for that encounter (both hold on 8 Nov). Stop otherwise.
- Expected: 16,709 draws, 748 encounters, 218 patients. v2 subset: draws whose `encounter_id` is a `recording_id` in `window_features.csv` (read as str): 3,458 draws, 6,974 rows, 154 recordings. Asserted.

### Pulse oximeters (plan:32–33)

- Read with `dtype=str` for `encounter_id`, `device`, `saturation`. `sample_number` to float.
- Drop and count, in this order: rows with no `device` (190); rows with `sample_number` NaN (11); then the validity rule.
- **Validity rule:** `sat = pd.to_numeric(saturation, errors="coerce")`; valid iff `10 <= sat <= 100`. Invalid rows are counted by reason (non-numeric, > 100, < 10) and dropped. Expected on 8 Nov: 11 non-numeric (already gone with `sample_number` NaN), 2 above 100, 27 below 10 on draws.
- **Why 10, not the plan's 0 (Hermes review, 8 Nov, before commit; marginal SpO2 values only, no pairing seen):** the 76 readings below 50 are 1 (×1), 4 (×34), 8 (×1), 9 (×4), then 33–49 (×36). Single-digit readings sit far below the lowest arterial value in the file (58.1) and below any saturation a conscious volunteer survives; 33 of the 40 come from device `64`. They are treated as entry or error codes. Readings 33–49 are kept as possible real device errors. The change can only lower device ARMS and raise device kappa, so it makes the gate harder for ScalcO2, not easier. The plan's `0 < sat <= 100` rule is reported as a sensitivity (below), not gated.
- **Device label:** `device.str.replace(" ", "")` → float → `f"{x:.2f}"`. Assert (encounter_id, device_label, sample_number) is unique after cleaning.
- **Join keys:** `sample_number` cast to int64 after an integer-valued assert. Inner join to draws on `encounter_id` and `sample_number == sample`. A valid paired reading = one cleaned pulse row joined to one draw; SpO2 = `sat`. Expected on 8 Nov: 79,644 valid paired readings (79,841 raw rows on draws, less 168 with no device, less 2 above 100, less 27 below 10); 79,671 under the plan's `0 < sat <= 100` rule.

### Benchmark devices (plan:33)

- A device is a benchmark device **for a metric** if it has ≥ 100 valid paired readings on the draws that metric uses:
  - P1 (ARMS): draws with 70 ≤ SaO2_draw ≤ 100;
  - P2 (kappa): all draws.
- The two sets can differ. Both are listed in `devices.csv` with their counts. Expected for P2: 54 devices. The P1 count is not computed before the run.

## Formulas (fixed now)

For a set of paired values, test value x (ScalcO2_draw or SpO2) and reference r = SaO2_draw, d = x − r, all in float64:

- **ARMS** = `sqrt(mean(d**2))`.
- **Bias** = `mean(d)`. **SD** = `std(d, ddof=1)`. **95 % limits of agreement** = bias ± 1.96 × SD. This is the plain Bland–Altman form. It ignores repeated draws per patient; the patient bootstrap is the only correction for that.
- **Cohen's kappa:** `sklearn.metrics.cohen_kappa_score(y1, y2, labels=[0, 1])` with `y1 = (x < 90).astype(int)`, `y2 = (r < 90).astype(int)`, no `weights`. In the bootstrap, `sample_weight` = the per-unit multiplicity. Source: `sklearn/metrics/_classification.py:875–1038` (sklearn 1.9.0). The weighted confusion matrix is built at l.978. An undefined kappa returns `replace_undefined_by` (default NaN, l.1010, l.1035). `labels=[0, 1]` keeps the table 2×2 when a resample lacks a class.
- **Sensitivity / specificity** of ScalcO2 < 90 with truth SaO2_draw < 90 (both strict): sens = #(x < 90 ∧ r < 90) / #(r < 90); spec = #(x ≥ 90 ∧ r ≥ 90) / #(r ≥ 90). Weighted version: the same sums with weights. The four counts come straight from the two boolean vectors. `lamaakal_metrics.confusion_at` (`scripts/lamaakal_metrics.py:21–30`) is not used, because it thresholds a probability with ≥ and these are strict < 90 tests on saturations. An empty denominator gives NaN (as `_ratio`, l.33–34).
- Per-device values: ARMS_d over that device's valid paired readings on P1 draws; kappa_d over that device's valid paired readings on all draws (x = SpO2).
- **Median over devices** = `np.median` over the benchmark devices' values (mean of the two middle values for an even count). A device whose kappa_d is not finite is left out of the P2 median and counted in `gate.json`. ARMS_d is always finite (n ≥ 100).
- **Pooled all-device ARMS / kappa** = the same formula over all valid paired readings of all benchmark devices for that metric, pooled.

## Gate (point estimates; fixed before any paired statistic is computed)

Copied verbatim from plan:35–40.

- **P1 accuracy.** On draws with 70 ≤ SaO2_draw ≤ 100: ARMS_S = sqrt(mean((ScalcO2_draw − SaO2_draw)^2)). For each benchmark device, ARMS_d over its valid paired readings on draws in the same range. P1 passes if ARMS_S ≤ median over devices of ARMS_d.
- **P2 below-90 agreement.** On all draws: Cohen's kappa between ScalcO2_draw < 90 and SaO2_draw < 90. For each benchmark device (≥ 100 valid paired readings on all draws), kappa between SpO2 < 90 and SaO2_draw < 90. P2 passes if kappa_S ≥ median over devices of kappa_d.
- Verdict PASS if P1 and P2 both pass; otherwise FAIL naming the failed part(s).
- Meaning of PASS: at blood draws, the label tracks arterial blood at least as closely as the median pulse oximeter tested in the same lab. Meaning of FAIL: it does not; the label is noisier than a typical oximeter at draws. Neither changes the shipped model, threshold or reported numbers.

Added precision (not a change): the gate runs on the **primary population (all draws, all encounters)** (plan:31). Gate quantities are compared unrounded in float64, with ≤ and ≥ as written. The gate uses point estimates only; CIs never enter it. The v2-subset tables are reported, not gated.

## Reported, not gated

From plan:42–50, with the mechanisms fixed:

- Bias, SD, 95 % limits of agreement, and ARMS by SaO2_draw band [70, 80), [80, 90), [90, 100], with counts per band. Draws with SaO2_draw < 70 or > 100 are counted, not banded.
- Sensitivity and specificity of ScalcO2_draw < 90 with SaO2_draw < 90 as truth. Disagreements split by |SaO2_draw − 90| ≤ 2 vs > 2.
- **Patient-grouped bootstrap 95 % CIs** for ARMS_S, bias, kappa_S, sensitivity, specificity:
  - B = 2000, seed 20261108.
  - Groups: `_, group_index = np.unique(patient_id.astype(str), return_inverse=True)` over the population's draws. `n_groups = group_index.max() + 1`.
  - Counts: `lamaakal_metrics.group_bootstrap_counts(group_index, n_groups, 2000, 20261108)` (`scripts/lamaakal_metrics.py:103–112`). That is exactly `np.random.default_rng(seed).integers(0, n_groups, size=(B, n_groups))` followed by a per-row `np.bincount(..., minlength=n_groups)`.
  - Unit weights: `w_b = counts[b][group_index]` (a draw's weight is its patient's multiplicity; plan:46, same convention as `scripts/lamaakal_metrics.py:9–10`). Weighted ARMS = sqrt(Σw d² / Σw); weighted bias = Σw d / Σw; weighted kappa via `sample_weight=w_b`; weighted sens/spec from weighted counts. Weighted SD is not bootstrapped.
  - CI: `lamaakal_metrics.percentile_ci(values, 0.95)` (`scripts/lamaakal_metrics.py:151–155`; 2.5th and 97.5th percentiles over finite values). The number of non-finite resamples per statistic is reported.
  - ARMS_S uses the same count matrix as the other statistics, restricted to the P1 draws (patients absent from P1 draws simply contribute nothing).
  - One count matrix per population: all draws; v2-subset draws; S1 matched draws. Each comes from its own `np.unique`-sorted patients with the same seed.
  - Devices are a fixed benchmark (no bootstrap).
- **Device distribution:** number of benchmark devices (P1 and P2 sets), min / quartiles / max of ARMS_d and kappa_d, pooled all-device ARMS and kappa, and ScalcO2's rank among devices (1 = best; ARMS ascending, kappa descending; rank = 1 + number of devices strictly better).
- **Device-unit sensitivity** (added here; reported only): P1 and P2 recomputed with devices pooled by integer model number (`int(float(label))`, 40 models), same ≥ 100 rule. This answers whether the `.01`/`.02` split moves the median. It never changes the verdict.
- **Validity-rule sensitivity** (added in review; reported only): P1 and P2, device medians and ScalcO2's rank recomputed with the plan's `0 < sat <= 100` rule. It never changes the verdict.
- **Row-level sensitivity:** each both-numeric `bloodgas.csv` row as a unit (no draw means); ARMS_S, kappa_S, sens, spec. Device comparison stays at the draw level.
- The same tables on the v2-recording subset (3,458 draws).

## S1 mapping: TinyKAN against blood (reported, not gated; descriptive only)

Plan:50, with every mechanism fixed.

1. **Draws:** the v2-subset draws (3,458).
2. **2 Hz reading**, per recording, exactly as `scripts/export_openox_windows_v2.py:220–226`:
   - read the CSV at `hz2_csv_path` of the eligible-encounters row;
   - `Timestamp = pd.to_datetime(Timestamp, errors="coerce")` (l.221);
   - keep rows with a valid Timestamp (l.223);
   - `t0` = the first kept row's Timestamp (l.225);
   - `elapsed_seconds = (Timestamp − t0).dt.total_seconds()` (l.226).
   - The export's next step, the reference-column `notna` filter (l.227), is **not** applied: marker rows are kept whether or not ScalcO2 is present on them.
3. **Draw time:** marker rows = rows with `Sample` not NaN; `Sample` asserted integer-valued and cast to int64. t_draw = `np.median` of `elapsed_seconds` over the rows whose `Sample` equals the draw's `sample`. A draw with no such row is counted as **no marker**.
4. **δ:** `ppg_start_time_seconds − hz2_first_time_seconds` from the recording's row of `outputs/openox_ppg_eligible_encounters.csv` (`scripts/export_openox_windows_v2.py:194`). All 154 recordings are present (checked 8 Nov), so the export's `delta_map.get(..., 0.0)` default (l.231) never fires. The script asserts this.
5. **Window span:** for each window of that recording in `window_features.csv`, span = [start_time_seconds + δ, end_time_seconds + δ), half-open (l.236–241). end − start = 10.0 s on every row (checked).
6. **Matching:**
   - candidate windows = those whose span contains t_draw;
   - none → counted as **no window**;
   - one → matched;
   - two → the one whose span centre (start + end)/2 + δ is nearest t_draw by absolute difference;
   - **tie** (exactly equal absolute differences in float64) → the window with the smaller `start_time_seconds` (earlier).
   - Three or more cannot occur at a 5 s stride with 10 s windows (0 marker rows inside three spans, 8 Nov). The script asserts it.
7. **Unit:** the draw. If one window is matched by more than one draw, it counts once per draw; the number of such windows is reported.
8. **Join to the stored out-of-fold row:**
   - The matched window's 0-based row position in `window_features.csv` is its `row_index`.
   - Checked 8 Nov for both `predictions_b_v2_tinykan16.csv` and `predictions_c_v2_logreg16.csv`: `row_index` is unique and equals 0..48661 in file order. At equal `row_index`, `recording_id`, `patient_id` and `start_time_seconds` equal the window table's. `y_true == label` on every row. (`recording_id`, `start_time_seconds`) is unique in the table and in both files. `window_features.csv` is sorted by (`recording_id`, `start_time_seconds`), as written at `scripts/export_openox_windows_v2.py:295`. A `validate="one_to_one"` merge on (`recording_id`, `start_time_seconds`) gives 48,662 rows, with table index == `row_index` on every row.
   - Rule: join on `row_index`. Then assert `recording_id` and `start_time_seconds` equal, and stop if not. Both key choices are one-to-one and give the same rows.
9. **Reported on matched draws**, separately for TinyKAN and logistic regression:
   - flag = `y_prob >= 0.50` (single window, not N = 2);
   - recall and specificity of flag against SaO2_draw < 90;
   - recall and specificity of flag against `y_true` on the same matched windows;
   - agreement of `y_true` with SaO2_draw < 90 (kappa, sens, spec; formulas above).
   - Patient-grouped bootstrap CIs, own count matrix, seed 20261108.
   - Counts: draws in v2 recordings, no marker, no window, matched, windows matched by > 1 draw.

## Limits fixed in advance

Copied verbatim from plan:52–59.

- Draws are taken on stable plateaus with the participant motionless. Agreement at draws does not describe transitions, where end-tidal O2 leads arterial saturation; many windows sit in transitions.
- Borderline v2 windows (median 89–91) were dropped at export, so S1 under-represents draws near 90.
- The PPG is not time-synced to the 2 Hz stream (PhysioNet); header alignment is better, not exact. S1 inherits it.
- If ScalcO2 were fitted to the blood samples, agreement would be optimistic; the paper does not say it is.
- The benchmark devices are devices under test in a lab study, of mixed quality and several finger sites; the median device is a benchmark, not a standard. No ISO 80601-2-61 limit is used as the gate (the 2026 edition's figure was not checked).
- Nothing here tests the MAX30102 or the board.

Added while writing (from the re-check above): device identity is inferred from label spelling. The `.01`/`.02` suffix meaning is not documented in the local dataset files (`devices.csv` lists integer numbers only). The integer-model sensitivity is the only check on that choice.

## Planned outputs (`outputs/scalco2-vs-sao2-v1/`, write-once)

Script `scripts/analyze_scalco2_vs_sao2.py`. It refuses to run if `outputs/scalco2-vs-sao2-v1/` exists, and it creates the directory only after every input assert has passed. The pre-registration commit hash is a required command-line argument (`--prereg-commit <hash>`), and the script checks that it resolves with `git cat-file -e`.

- `summary.json`:
  - SHA-256 of the six inputs in the table above;
  - the pre-registration commit hash;
  - versions (python, numpy, pandas, sklearn);
  - B and seed;
  - every count from the cleaning: rows dropped by reason, draws, encounters, patients, valid paired readings, devices, benchmark devices per metric;
  - the S1 counts.
- `gate.json`: ARMS_S, the device ARMS median and the number of devices behind it, P1; kappa_S, the device kappa median, the number of devices behind it and the number excluded as non-finite, P2; verdict and failed part(s).
- `draws.csv`: one row per draw: encounter_id, patient_id, sample, n_rows, SaO2_draw, ScalcO2_draw, in_v2, n_valid_pulse_readings.
- `devices.csv`: one row per device label: model number, valid paired readings (P1 range, all draws), benchmark flags, ARMS_d, bias_d, kappa_d.
- `bands.csv`: band × population (all draws, v2 subset): n, bias, SD, LoA, ARMS.
- `s1_windows.csv`: one row per v2-subset draw: draw keys, t_draw, n_marker_rows, status (matched / no marker / no window), row_index, recording_id, start_time_seconds, y_true, y_prob TinyKAN, y_prob logistic regression, SaO2_draw.
- `README.md`: the readings, the verdict in the wording rules above, and the limits.

Then (outside this file): `docs/scalco2-vs-sao2-results.md`. On FAIL, also a numbered section appended to the vault `paper/Final report - queued results.md` (plan:66).

## Tests written first (`tests/test_scalco2_vs_sao2.py`, written and failing before the script)

Pure helpers on synthetic data unless marked:

- **Draw aggregation:**
  - rows with only one numeric value are excluded from both means;
  - a group with no both-numeric row is not a draw;
  - means are over the same rows;
  - missing `encounter_id` rows are dropped;
  - a draw with two `patient_id`s raises.
- **Validity filter:**
  - text, NaN, 0, −1, 4, 9.99, 10, 100, 100.0001, 101 and 493 are classified correctly (only 10 ≤ s ≤ 100 kept), and the sensitivity rule keeps 0 < s ≤ 100;
  - drop counts by reason;
  - rows with no device or NaN `sample_number` are dropped first.
- **Device label:** `2`, `2.0` → `2.00`; `56 .01` → `56.01`; `42.01` stays distinct from `42.02` and from `42.00`; integer model of `42.02` is 42.
- **Join:** an inner join on (encounter_id, sample_number == sample) drops readings on non-draws and keeps one row per (draw, device).
- **ARMS / bias / SD / LoA** on a hand-computed vector; weighted ARMS with integer weights equals ARMS on the replicated vector.
- **Kappa wrapper:**
  - equals `cohen_kappa_score` on a hand case;
  - with integer `sample_weight` equals kappa on the replicated vector;
  - returns NaN, not an error, when both vectors are one class;
  - the threshold is strict (`89.999` → 1, `90.0` → 0).
- **Benchmark rule:** a device with 99 readings in range is excluded and one with 100 is included; the P1 and P2 sets are computed on their own draw sets.
- **Gate rule:**
  - pass/pass, P1-fail, P2-fail and both-fail on synthetic inputs;
  - equality passes on both parts (≤, ≥);
  - even-count median;
  - a non-finite kappa_d is excluded and counted.
- **Bootstrap indices:** for a fixed seed, the count matrix equals `np.bincount` of `np.random.default_rng(seed).integers(0, n, (B, n))` row by row; rows sum to n; groups are `np.unique`-sorted (string order).
- **Draw time:** median of marker rows' elapsed seconds, including an even count (mean of two middle values); elapsed is measured from the first valid-Timestamp row even when that row has no ScalcO2; invalid-Timestamp rows are skipped before t0; no marker → status "no marker".
- **Window matching:**
  - one containing window;
  - t_draw exactly at a span start (included) and exactly at a span end (excluded);
  - **two windows** → nearest centre;
  - **exact tie** → earlier start;
  - no window → status "no window";
  - δ shifts the span.
- **Out-of-fold join:** the `row_index` join returns the same rows as the (recording_id, start_time_seconds) join on a small synthetic table, and raises on a mismatch.
- **Write-once guard:** the entry point raises before writing anything when the output dir exists; it raises when `--prereg-commit` is missing.
- **Real-file asserts** (read-only, marked slow): the input SHA-256s; 16,709 draws / 748 encounters / 218 patients; 3,458 v2-subset draws; 79,644 valid paired readings (79,671 under the sensitivity rule); 68 device labels; 54 P2 benchmark devices; the `row_index` / key checks in S1 step 8. None of these computes a paired statistic.

## Amendments

**A1 (8 Nov 2026, before the real run; no paired statistic on the full data seen).** Fixed for reported-only numbers that this file left open. None touches the gate.
- Headline bias, SD and 95 % limits of agreement, and the bootstrapped bias, use the P1 draws (70 ≤ SaO2_draw ≤ 100), the same draws as ARMS_S. All-draw values are reported beside them.
- Row-level sensitivity: ARMS on both-numeric rows with 70 ≤ so2 ≤ 100; kappa, sensitivity and specificity on all both-numeric rows.
- v2-subset device tables use only pulse readings at v2-subset draws, with the same ≥ 100 rule; reported, not gated.
- "68 device labels" counts normalised labels over all pulse rows that have a device, before the validity rule.
- The real-file count asserts listed under "Tests written first" live in the script (checked before the output directory is created), not in pytest, which uses synthetic data only. The smoke run skips them.
- Smoke mode keeps the first 20 `bloodgas.csv` encounter_ids (string sort); its numbers are not results.

None yet.

This file is not edited after any paired statistic between ScalcO2, SaO2 and SpO2 is seen, or after any file in `outputs/scalco2-vs-sao2-v1/` is read. Any change after that is a dated amendment appended here, saying what changed, why, and which results had been seen. This file's commit must predate every file in `outputs/scalco2-vs-sao2-v1/`.
