# PTT-PPG sit/walk/run frozen TinyKAN spike (subject 1)

11 September 2026, 11:29 MYT. Jason said continue until help is needed. The remaining laptop job was **PTT-PPG**: finish the interrupted PhysioNet 1.1.0 fetch for subject 1, then frozen TinyKAN **inference** only. No training, fine-tune, upload, COM8, GPIO, live capture, FIFO repair, or AOPC overwrite. Weights, threshold **0.50**, ADR-0005 N=2, and rankings were not changed.

**This is out-of-domain host inference on one healthy subject. It is not on-device P, not live still/wiggle/walk, and not a window-level SpO2 accuracy result.** Header SpO2 at activity start/end stays ≥96% on these three records; a high candidate rate is domain shift, not low-oxygen risk.

## What was measured

`scripts/evaluate_ptt_ppg.py --download --subjects 1 --channel pleth_2` on Pulse Transit Time PPG **1.1.0** records `s1_sit`, `s1_walk`, `s1_run`. Channel **pleth_2** (dataset Methods vs Data Description disagree on whether that is red or infrared). Physical units via WFDB `physical=True`. Full-record `resample_poly` 500 Hz → 50 Hz (`padtype=line`). Windows **500** samples, hop **250** (frozen firmware `STRIDE=250`, not the 2.0 s line in `CONTEXT.md`). Features from `cpp_extract_features`; P and top-3 from `model_attributions` on `kan_model_weights.h`. Dataset IMU is a **context column** only.

Subjects 2–22 were not downloaded. Consecutive-state N=2 was not applied (candidate only).

## Activity table (exported TinyKAN, P ≥ 0.50)

**294** complete windows. Mean unperturbed P is high on all three activities.

| Activity | Windows | Mean P | Median P | P05 | P95 | Candidates | Fraction P≥0.50 | Top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| sit | 100 | 0.5522 | 0.5520 | 0.5330 | 0.5759 | 99 | 0.990 | kurtosis 98; maximum 1; std 1 |
| walk | 97 | 0.5759 | 0.5802 | 0.5536 | 0.5944 | 95 | 0.979 | kurtosis 70; bandpower_ratio_0_5_2_to_2_5 24; maximum 3 |
| run | 97 | 0.5378 | 0.5382 | 0.5086 | 0.5681 | 96 | 0.990 | kurtosis 87; maximum 9; bandpower_ratio_0_5_2_to_2_5 1 |

Header numerics (not window labels): sit SpO2 98/98; walk 96/97; run 99/98.

IMU vector-magnitude RMS (header g, not a model input). Dynamic RMS subtracts per-axis mean inside each window:

| Activity | IMU RMS mean | Dynamic RMS mean |
|---|---:|---:|
| sit | 9.586 | 0.023 |
| walk | 11.402 | 2.816 |
| run | 9.938 | 2.323 |

Sit vs motion separates on **dynamic** RMS. Inclusive RMS does not rank run above walk. Do not treat IMU as a TinyKAN feature.

## Checks that passed

- `python tests/test_ptt_ppg.py`: **7/7**.
- Published SHA-256 for the six waveform/header files matches `SHA256SUMS.txt`.
- Same-features Python P error **0**; candidate disagreements **0** (NumPy path only; no `probability_probe`).
- Header SHA-256 `123a305f0bb91a0bff79c298f7d35df3c113c660eede9afebdcd30ba96a3fe55` (same frozen weights as host AOPC).

## What this does not show

- Not a 22-subject cohort and not a complete 1.1.0 tree (no CSV bundle, no `.atr`, subjects 2–22 absent).
- Not board execution; backend is the NumPy feature twin + FP32 NumPy TinyKAN.
- Not live still/wiggle/walk (`outputs/live/` still absent; timing image still on the board).
- Not recall/spec vs SpO2 < 90 (no window labels).
- Not a reason to retune 0.50 or claim motion robustness.
- `pleth_2` wavelength mapping is internally inconsistent in the dataset README.
- Physical NU (~7.3×10^4) is not Nano MAX30105 counts.

## Evidence

Directory (gitignored, write-once): `outputs/ptt-ppg/subject-1-v1/`

- `report.json` SHA-256 `fed427028bef91002d512453c9b198e700c2911f32ef7abba913c970f18767b8`
- `activity_summary.csv` / `outputs/ptt_ppg_motion_spike.csv` SHA-256 `27a91e01ede1c675e8a4db7268ba094fb6c646dc3c189cf976c2b7b8f95d4734`
- `windows.csv` SHA-256 `d0df6f4b953a31bc62bc9609ffbe8227919b91462800000e3f55612d6a1e596a`
- Started 2026-09-11 03:29:32 UTC.

Reproduce into a **new empty** directory only:

```powershell
python tests/test_ptt_ppg.py
python scripts/evaluate_ptt_ppg.py --download --subjects 1 --channel pleth_2 --output outputs/ptt-ppg/subject-1-repeat --success-csv outputs/ptt_ppg_motion_spike_repeat.csv
```
