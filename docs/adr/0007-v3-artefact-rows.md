# 0007 - v3: the v2 table minus frozen-sensor artefact rows

**Status:** Recorded 17 Oct 2026, after the waveform gate. v3 is not the reported system. The paper reports v1. v2 is unchanged.

## Context

`outputs/openox-v2/window_features.csv` (48,662 rows, 154 recordings) contains seven long runs of windows whose features do not change. The 15 Oct host-parity residual traced 96 of its 114 failures to one of them, `9c01688d…` (ADR-0006, correction of 17 Oct 2026).

## Drop rule

The table is sorted by recording and start time. A run is consecutive rows of one recording whose `std`, `kurtosis`, `ir_perfusion_index` and `ratio_of_ratios` are identical to 6 decimals. Runs of at least 146 rows are candidates. The longest run anywhere else in the table is 2 rows.

A candidate run is dropped only if its raw waveform is a sensor fault, a probe-off spike train or a frozen flat line, not physiology. The raw encounters were opened with the loader from `scripts/export_openox_windows_v2.py`. Evidence: `outputs/openox-v3/artefact-audit.md`, `artefact_segments.json` and `artefact_segments.png`, from `outputs/openox-v3/audit_artefact_segments.py`.

All seven passed the gate. In each one, all 8 recorded channels repeat bit for bit with a 43-frame (0.5 s at 86 Hz) period, through to the end of the recording. Within each run the Drive, DC and Modulation channels are constant, and red sits at the 2^20 rail. In six of the seven a single sensor buffer is being replayed. The seventh, `9c01688d2219`, is a probe-off spike train inside the same loop: IR about 450–520 with a spike every 0.5 s.

| Recording | Rows dropped | Positive | v2 rows | v3 rows |
|---|---|---|---|---|
| `1be0cb9a60a1` | 404 | 192 | 414 | 10 |
| `cec145b67e4c` | 401 | 256 | 401 | **0** |
| `1af3b3944b46` | 352 | 239 | 352 | **0** |
| `1880c36e2fa3` | 291 | 128 | 291 | **0** |
| `13a54dd79863` | 225 | 148 | 435 | 210 |
| `9c01688d2219` | 187 | 136 | 342 | 155 |
| `3719e6dfe6b8` | 146 | 136 | 386 | 240 |
| **Total** | **2,006** | **1,235** | | |

Row manifest: `outputs/openox-v3/artefact_rows.csv`. Per-recording manifest: `outputs/openox-v3/drop_manifest.csv`.

## Decision

v3 is the v2 recipe, unchanged, on the v2 csv with exactly these 2,006 rows removed. Nothing was re-exported and no recording was dropped as a unit. The script is `scripts/train_tinykan_v3.py`, and its outputs are in `outputs/openox-v3/`. The weights are in `firmware/Arduino Nano 33/include/kan_model_weights_v3.h` (SHA-256 `c393161d…6860`), which no translation unit includes.

**Three recordings leave the table.** Every window of `1880c36e2fa3`, `cec145b67e4c` and `1af3b3944b46` lies inside a gated run. The v3 table therefore has 46,656 rows from **151 recordings, not 154**, and from 86 patients, not 87. This is a consequence of the drop. It is not a new eligibility rule, and the export's minimum of 5 windows per encounter was not re-applied.

**Not dropped.** 20 v2 rows lie inside the repeat but outside the gated runs: 6 in `1be0cb9a60a1`, 9 in `3719e6dfe6b8` and 5 in `9c01688d2219`. They sit at the run edges, and their features differ slightly from the frozen value. They stay in v3, because widening the drop would need a rule this decision does not make. Frozen loops shorter than 146 windows were not searched for.

## Consequences

- **v2 is unchanged.** `outputs/openox-v2/`, `kan_model_weights_v2.h` and `scripts/train_tinykan_v2.py` are not modified. v2's evidence, and its open items in ADR-0006, stand as they were.
- **v3 is not the reported system.** v3 is not on the paper and has no board, timing, parity or live evidence.
- **The CV numbers are different exams.** v2 was cross-validated on the table with the artefact rows in it, v3 on the table without them, with different rows and folds. `outputs/openox-v3/summary.json` reports them side by side and says so. v2's own out-of-fold predictions, scored on the v3 rows only, give AUC 0.919. On the 2,006 dropped rows alone they give 0.559. Most of the difference between v2's published 0.908 and v3's 0.921 is therefore a change of exam, not of model. Neither number ranks the two models. The final-fit metrics in `final_model_report.json` are resubstitution, not generalisation.
- **Stripping is not a parity fix.** Host parity is judged on the full v2 recording set, `9c01688d` included. Removing those windows from the exam does not repair the port. The port defect is `window_mean` (ADR-0006, correction of 17 Oct 2026), and its repair is a separate firmware change. v3 host parity, if run, gets its own directory.
