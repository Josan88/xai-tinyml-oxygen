# Existing FYRP literature for the EAT40006 research paper

Inventory of sources **already present** in this repository and in Jason's EAT40005 submissions (research plan, poster, progress report). This is a related-work map for the 10% conference paper. It is not a new paper draft and does not rewrite `conference-template-A4.docx`.

**Claim this map serves.** TinyKAN (14, 12, 1), 297 parameters (the 201-parameter figure formerly given here belonged to the EAT40005 model, TinyKAN (14, 8, 1)) + mean-perturbation XAI on Arduino Nano 33 BLE, OpenOximetry windows, operating point 0.50, recall ≈ 0.77, specificity ≈ 0.39. XAI sample n=20 (mean 5.70/6; too small to claim the research-plan pass). No pigmentation labels in the scored export. IMU unused.

**Inventory order.** Repo `docs/`, `*.md`, README, ADRs, TinyKAN paper dump → EAT40005 plan/poster/progress-report reference lists → notebook/script comments. No `.bib` existed in the repo before this note. No PDFs of the conference paper were in the repo.

**Tag rule.** Every row is `in-repo` (already cited or stored in the project). No `gap-fill` rows: all five paper themes already have existing citations.

**Year note.** Table years follow Crossref/DOI or arXiv metadata fetched 2026-08-27. A few EAT40005 bibliography years differ by a month or two (Assis, Wei, Salih); those are provenance, not new papers.

## Source table

| Citation | Year | Venue / URL or local path | Theme | How this project uses it | Tag |
|---|---|---|---|---|---|
| Liu, Z. et al., KAN: Kolmogorov–Arnold Networks | 2024 | arXiv:2404.19756; https://arxiv.org/abs/2404.19756 | KAN | Nested in the in-repo Lamaakal markdown (TinyKAN ref. [46]). Defines learnable univariate splines on edges. Not independently numbered in EAT40005 lists. | in-repo |
| Lamaakal, I. et al., Explainable Kolmogorov–Arnold Networks for Zero-Shot HAR on TinyML Edge Devices | 2026 | MAKE 8(3):55; doi:10.3390/make8030055; local `TinyKAN/Lamaakal et al. - 2026 - ….md` | KAN / TinyML-MCU / XAI-attribution | Seed TinyKAN-HAR paper. Repo reproduces UCI HAR ZSL (not the oxygen task). EAT40005 [10]/progress [6]; poster comparator. Notebook 12 explicitly is **not** a TinyKAN-HAR reproduction. | in-repo |
| Heydari, S. and Mahmoud, Q. H., Tiny Machine Learning and On-Device Inference: A Survey | 2025 | Sensors 25(10):3191; doi:10.3390/s25103191 | TinyML-MCU | Frames accuracy + memory + latency + quantisation as one benchmark contract (plan [11]; progress [7]). | in-repo |
| Banbury, C. et al., MLPerf Tiny Benchmark | 2021 | NeurIPS Datasets and Benchmarks; arXiv:2106.07597; https://arxiv.org/abs/2106.07597 | TinyML-MCU | Progress [17]: MCU-class inference must be reported with a reproducible resource protocol, not accuracy alone. | in-repo |
| Burrello, A. et al., Q-PPG: Energy-Efficient PPG Heart Rate Monitoring | 2021 | IEEE TBioCAS 15(6):1196–1209; doi:10.1109/TBCAS.2021.3122017 | TinyML-MCU / pulse-oximetry-OpenOximetry | Progress [18]: Cortex-M4 PPG inference is feasible; task is HR, not oxygen-risk, and has no XAI layer. | in-repo |
| Dekimpe, R. and Bol, D., ECG Arrhythmia Classification on an Ultra-Low-Power MCU | 2022 | IEEE TBioCAS 16(3):456–466; doi:10.1109/TBCAS.2022.3182159 | TinyML-MCU | Progress [19]: biomedical TinyML precedent; ECG not PPG, no explanation method. | in-repo |
| Zheng, Y. et al., Tiny-PPG: Lightweight DNN for PPG Motion-Artifact Detection on Edge Devices | 2024 | Internet of Things 25:101007; doi:10.1016/j.iot.2023.101007 | TinyML-MCU / pulse-oximetry-OpenOximetry | Progress [20]: on-device PPG artefact CNN. Supports SQI thinking; not oxygen-risk classification. | in-repo |
| Zhang, Z. et al., Energy-Efficient ECG Processor with Ultra-Low-Parameter Network | 2024 | IEEE TBioCAS 18(6):1296–1307; doi:10.1109/TBCAS.2024.3385993 | TinyML-MCU | Progress [21]: compact biomedical classifier on MCU; ECG, no XAI. | in-repo |
| ARM, CMSIS-DSP Library | 2024 | https://github.com/ARM-software/CMSIS-DSP | TinyML-MCU | Progress [32]: planned Cortex-M4 FFT/vector path for feature parity. Firmware already uses native FP32 (ADR-0004); do not retune. | in-repo |
| Chan, M. et al., Enabling Continuous Wearable Reflectance Pulse Oximetry at the Sternum | 2021 | Biosensors 11(12):521; doi:10.3390/bios11120521 | pulse-oximetry-OpenOximetry | Plan [3] / progress [1]: wearable reflectance PPG is a credible oxygen-sensing context. Not an ML or MCU study. | in-repo |
| Zhong, Y. et al., Enhanced SpO2 estimation using explainable ML and neck PPG | 2023 | Artif. Intell. Med. 145:102685; doi:10.1016/j.artmed.2023.102685 | pulse-oximetry-OpenOximetry / XAI-attribution | Closest oxygen+XAI precedent (plan [1]; progress [3]; poster). Continuous SpO2 + post-hoc SHAP; no grouped split, no MCU. | in-repo |
| Baciu, V.-E. et al., Anomaly Detection in Multi-Wavelength PPG Using Lightweight ML | 2023 | Sensors 23(15):6947; doi:10.3390/s23156947 | pulse-oximetry-OpenOximetry / TinyML-MCU | Compact PPG anomaly model (plan [9]; progress [4]; poster). Encounter-aware; no SaO2/SpO2 risk label, no XAI. | in-repo |
| Badiola, I. et al., Real-Time Subject-Specific PPG Modelling under Hypoxia | 2025 | Sensors 25(23):7176; doi:10.3390/s25237176 | pulse-oximetry-OpenOximetry | Plan [8] / progress [16]: artefact-aware SpO2 under controlled hypoxia. Subject-specific, so not transferable to GroupKFold. | in-repo |
| Fong, N. et al., Open Access Dataset and Common Data Model for Pulse Oximeter Performance Data | 2025 | Sci. Data 12:570; doi:10.1038/s41597-025-04870-8 | pulse-oximetry-OpenOximetry / fairness-quality | Descriptor for OpenOximetry (plan [14]; progress [9]). Justifies SaO2/SpO2 supervision and skin-colour fields that this export does **not** currently score. | in-repo |
| Fong, N. et al., OpenOximetry Repository | 2025 | PhysioNet; doi:10.13026/BE2E-CN29; https://physionet.org/content/openox-repo/1.1.1/; local `data/physionet.org/files/openox-repo/1.1.1/` | pulse-oximetry-OpenOximetry | Canonical training set (52,090 windows / 162 recordings). Restricted Health Data License v1.5.0. Plan [15]; progress [8]. | in-repo |
| Pimentel, M. A. F. et al., Toward a Robust Estimation of Respiratory Rate From Pulse Oximeters | 2017 | IEEE TBME 64(8):1914–1923; doi:10.1109/TBME.2016.2613124 | pulse-oximetry-OpenOximetry | BIDMC source paper (plan [12]; progress [27]). RR from PPG, not oxygen-risk. | in-repo |
| Pimentel, M. et al., BIDMC PPG and Respiration Dataset | 2018 | PhysioNet; doi:10.13026/C2208R; local `data/physionet.org/files/bidmc/1.0.0/` | pulse-oximetry-OpenOximetry | Semester-1 pipeline sandbox (5,017 windows; positives in 2/53 recordings). Not the main generalisation set. Plan [13]; progress [10]. | in-repo |
| Charlton, P. H. et al., The 2023 Wearable Photoplethysmography Roadmap | 2023 | Physiol. Meas. 44(11):111001; doi:10.1088/1361-6579/acead2 | pulse-oximetry-OpenOximetry | Progress [23]: wearable PPG agenda (quality, motion, standards). Context, not a model. | in-repo |
| Simonson, T. S. et al., Silent Hypoxaemia in COVID-19 Patients | 2021 | J. Physiol. 599(4):1057–1065; doi:10.1113/JP280769 | pulse-oximetry-OpenOximetry | Progress [26]: cited for SpO2 < 90% as a hypoxaemia-relevant engineering threshold. | in-repo |
| Assis, A., Dantas, J. and Andrade, E., The performance–interpretability trade-off | 2024 | J. Reliable Intell. Environ. 11(1); doi:10.1007/s40860-024-00240-0 | XAI-attribution | Plan [2] / progress [5]: interpretability is an engineering cost, not a free extra. Stronger on tabular than images. | in-repo |
| Rudin, C., Stop explaining black box models for high-stakes decisions | 2019 | Nat. Mach. Intell. 1:206–215; doi:10.1038/s42256-019-0048-x | XAI-attribution | Progress [28]: prefer inherently interpretable models over post-hoc approximations. Motivates LogReg coefficients and on-device TinyKAN attributions. | in-repo |
| Salih, A. M. et al., A Perspective on XAI Methods: SHAP and LIME | 2024 | Adv. Intell. Syst. 7(1):2400304; doi:10.1002/aisy.202400304 | XAI-attribution | Progress [29]: LIME/SHAP instability under correlated features. README/ADR-0001 reject KernelSHAP/LIME on-device for this reason. | in-repo |
| Kim, S. Y. et al., XAI-Based Clinical Decision Support Systems: A Systematic Review | 2024 | Appl. Sci. 14(15):6638; doi:10.3390/app14156638 | XAI-attribution | Progress [24]: clinical XAI review. Informs the rubric; not an MCU method. | in-repo |
| Vimbi, V., Shaffi, N. and Mahmud, M., LIME and SHAP in Alzheimer’s Detection | 2024 | Brain Inform. 11; doi:10.1186/s40708-024-00222-1 | XAI-attribution | Progress [25]: post-hoc XAI in a clinical imaging domain. Contrast class, not a method to port. | in-repo |
| Cotton, S. A. et al., Do Differences in Skin Pigmentation Affect Detection of Hypoxemia by Pulse Oximetry | 2025 | Clin. Nurs. Res. 34(8):403–411; doi:10.1177/10547738251374746 | fairness-quality | Plan [4] / progress [13]: pigmentation as a reliability constraint. **Not** usable as a scored axis here (no pigmentation labels in the 0.50 table). | in-repo |
| Benner, K. J., Goel, N. N. and Rea, M. S., Cause, effect, and remediation of melanin-associated bias in pulse oximetry | 2026 | J. Biomed. Opt. 31(2):028002; doi:10.1117/1.JBO.31.2.028002 | fairness-quality | Plan [5] / progress [14]: mechanism of melanin bias. External caution only. | in-repo |
| Barros, K. B. et al., Accuracy and Bias of Pulse Oximetry in the ICU | 2026 | Nurs. Crit. Care 31(3):e70433; doi:10.1111/nicc.70433 | fairness-quality | Plan [6] / progress [15]: ICU pulse-ox accuracy/bias. Not this hardware or feature set. | in-repo |
| Wei, L. et al., Assessing PPG Signal Quality for Wearable Devices during Unrestricted Daily Activities | 2025 | Biomed. Phys. Eng. Express 12(1):015016; doi:10.1088/2057-1976/ae250f | fairness-quality / pulse-oximetry-OpenOximetry | Plan [7] / progress [2]; poster. Forehead IR PPG SQI in healthy wearers. Motivates quality screening; site/population do not transfer. | in-repo |
| Feli, M. et al., Energy-Efficient Semi-Supervised On-Device PPG Signal Quality Assessment | 2023 | Smart Health 28:100390; doi:10.1016/j.smhl.2023.100390 | fairness-quality / TinyML-MCU | Progress [22]: on-device PPG SQI. Quality gate, not an oxygen-risk model. | in-repo |
| WHO, Ethics and Governance of Artificial Intelligence for Health | 2021 | https://www.who.int/publications/i/item/9789240029200 | fairness-quality | Progress [12]: equity language for a later fairness gate. Current scored axes are `patient_id` and `recording_id` only. | in-repo |
| United Nations DESA, Goal 3: Ensure healthy lives and promote well-being for all at all ages | 2015 (page accessed 2026-10-23) | https://sdgs.un.org/goals/goal3 | SDG | Added 23 Oct 2026 with Jason's approval for the Week 7 SDG requirement (link one objective to an SDG in the paper and report). IEEE draft [13]: the aim is linked to target 3.8 (access to quality essential health-care services) as a possible route only; the paper does not show a contribution. Not evidence for any result. | added (Jason, 2026-10-23) |
| Nordic Semiconductor, nRF52840 Product Specification v1.1 (4413_417) | 2019 | https://content.arduino.cc/assets/Nano_BLE_MCU-nRF52840_PS_v1.1.pdf (Arduino-hosted copy; sec. 5.2.1.3, p. 58) | TinyML-MCU | Source of the typical CPU-running currents (3.3 mA DC/DC, 6.3 mA LDO, CoreMark @ 64 MHz from flash, 3 V, 25 C) for the host energy ESTIMATE `outputs/energy-estimate-v1/` (current x 3 V x board decision-block time, CPU only). Not a measurement; never cite it as measured energy. | added (energy estimate, 2026-10-24) |

### Also present, not a paper-theme row

| Item | Local / URL | Role |
|---|---|---|
| EPU, Malaysia Digital Economy Blueprint (MyDIGITAL) | Progress [11], 2021 | Policy context only. |
| IEC 60601-1-11:2015 | Progress [30] | Home-healthcare device standard; out of current firmware scope. |
| ISO/IEEE 21451-1:2010 | Progress [31] | Transducer interface standard; not used. |
| ADR-0001 Mean-Perturbation XAI | `docs/adr/0001-on-device-mean-perturbation-xai.md` | Project decision: N+1 mean-perturbation instead of KernelSHAP/LIME. Names the methods; does not cite Ribeiro or Lundberg. |
| ADR-0003 Deletion AUC | `docs/adr/0003-xai-faithfulness-evaluation-via-deletion-auc.md` | Project decision: deletion/insertion AOPC on 52,090 windows. No Petsiuk/Samek paper in the bibliography. |
| ADR-0004 Native FP32 TinyKAN | `docs/adr/0004-native-fp32-execution-for-tinykan.md` | Keep FP32 on nRF52840 FPU; do not quantise for this card. |
| ADR-0005 N=2 debounce | `docs/adr/0005-consecutive-window-risk-confirmation.md` | Confirmed-risk rule. Not literature. |
| CONTEXT.md / README.md | repo root | Define TinyKAN, Mean-Perturbation, KernelSHAP-as-offline-reference. No bibliographic entries. |
| TinyKAN reproduction notes | `TinyKAN/reproduction/intake.md`, `AGENTS.md` | Record that Lamaakal ZSL numbers were **not** reproduced from the described method. |
| Notebooks 05–15 | `notebooks/` | Operational use of OpenOximetry / TinyKAN / LogReg XAI. Notebook 12 disclaims TinyKAN-HAR ZSL/HAR claims. Notebook 15 notes SHAP was not required for LogReg. |
| `docs/tinykan-xai-rubric.md`, fairness and PI notes | `docs/` | Project evaluation artefacts, not external papers. Do not rewrite. |

## Related-work paragraphs (paper-ready)

### KAN

Kolmogorov–Arnold Networks replace fixed node activations with learnable univariate functions on edges. The formulation used throughout this project is the spline-KAN of Liu et al. (2024), which appears in the in-repo TinyKAN-HAR paper as reference [46] rather than as a standalone EAT40005 numbered item. Lamaakal et al. (2026) then compile a compact KAN for microcontroller HAR with spline inspection plus post-hoc gradient/SHAP-style scores, and store that paper and its figures under `TinyKAN/`. This FYRP keeps the spline-layer idea and a 14→12→1 binary head on OpenOximetry windows (notebook 12). It does not keep TinyKAN-HAR’s semantic projection, zero-shot activity split, or UCI HAR labels, and the local reproduction notes that the paper’s ZSL accuracies were not recovered from the described method. The oxygen paper should therefore cite Lamaakal as the edge-KAN precedent and Liu as the architecture, then state the domain transfer explicitly.

### TinyML-MCU

Heydari and Mahmoud (2025) and Banbury et al. (2021) treat on-device inference as a joint accuracy–memory–latency problem. Biomedical TinyML evidence already exists for PPG heart-rate (Burrello et al., 2021), PPG artefact detection (Zheng et al., 2024), on-device PPG quality (Feli et al., 2023), and ECG classifiers (Dekimpe and Bol, 2022; Zhang et al., 2024). None of those systems classifies low-oxygen risk or emits a local explanation on the same MCU pass. Baciu et al. (2023) is the closest compact PPG classifier, but it is an anomaly detector without an arterial-oxygen label or an explanation budget. This project’s contribution sits in that hole: native FP32 TinyKAN plus 15-pass mean-perturbation on a Nano 33 BLE (ADR-0004, ADR-0001), with CMSIS-DSP cited only as the planned DSP library (progress [32]), not as a result.

### Pulse oximetry and OpenOximetry

Wearable reflectance PPG is an established oxygen-sensing path (Chan et al., 2021). Modelling papers diverge on the endpoint: Zhong et al. (2023) regress continuous SpO2 from neck PPG; Badiola et al. (2025) fit subject-specific artefact-aware estimators under hypoxia; Baciu et al. (2023) detect multi-wavelength PPG anomalies. Charlton et al. (2023) survey the wearable-PPG agenda; Simonson et al. (2021) is the EAT40005 source for treating SpO2 < 90% as a hypoxaemia-relevant cut. Dataset routing is two-step. BIDMC (Pimentel et al., 2017, 2018) validated ingestion but is too sparse for generalisation (positives in 2 of 53 recordings). OpenOximetry (Fong et al., 2025 descriptor and PhysioNet 1.1.1 repository) is the canonical set: 52,090 windows, 162 recordings, SaO2/SpO2 supervision. SpO2 is label evidence only. The paper should keep that split and not imply certified oximetry.

### XAI and attribution

Rudin (2019) argues that high-stakes systems should be interpretable rather than post-hoc explained. Assis et al. (2024) show the accuracy–interpretability gap is smaller on tabular features than on images. Clinical XAI reviews (Kim et al., 2024; Vimbi et al., 2024) and Salih et al. (2024) document LIME/SHAP use and their instability under correlated inputs. Zhong et al. (2023) apply SHAP to neck-PPG SpO2 on a server-side model. Lamaakal et al. (2026) mix intrinsic spline plots with gradient and SHAP-style scores, still in HAR. README, CONTEXT, and ADR-0001 reject KernelSHAP/LIME on the Cortex-M4 because they need a background set and many more forward passes. The firmware instead emits a 14-dimensional mean-perturbation vector (N+1 passes, feature replaced by the training mean 0 in normalised space). ADR-0003 evaluates those ranks with deletion/insertion AOPC. Neither the original LIME (Ribeiro) nor SHAP (Lundberg) paper, nor a deletion-AUC source paper, is in the existing bibliography; the paper may cite the surveys already on the list, or add those originals later.

### Fairness and signal quality

Pulse-oximetry error varies with pigmentation and care setting (Cotton et al., 2025; Benner et al., 2026; Barros et al., 2026). Wei et al. (2025) and Feli et al. (2023) support short-window quality screening; Wei’s evidence is forehead infrared PPG in healthy daily activity and does not transfer automatically to this reflectance/OpenOximetry route. Fong et al. (2025) document skin-colour fields in OpenOximetry. This repo’s scored fairness table only has `patient_id` (87/91 eligible) and `recording_id` (155/162). The quality axis that exists is encounter-mean `pulseox_pi_mean` tertiles, not motion and not IMU. WHO (2021) is cited for equity language. The paper must not claim a pigmentation or motion result.

### Synthesis (what to argue in one paragraph)

No existing source already combines (i) grouped OpenOximetry oxygen-risk classification, (ii) on-device KAN inference, and (iii) a microcontroller-budget local explanation. Zhong has oxygen + XAI without grouping or an MCU. Baciu has a compact PPG model without an oxygen-risk label or XAI. Lamaakal has an explainable TinyML KAN without PPG or oxygen labels. TinyML PPG work is HR or artefact detection. Fairness papers concern clinical pulse oximeters, not this 14-feature classifier. The paper’s related-work close is therefore an integration claim at operating point 0.50 (recall ≈ 0.77, specificity ≈ 0.39), with the n=20 XAI sample reported as preliminary.

## Gap list (existing set does not cover)

1. **On-device KAN XAI for PPG/oxygen.** Lamaakal is HAR + SHAP/gradient/spline plots. This firmware’s mean-perturbation vector has no peer-reviewed PPG-KAN precedent in the existing list.
2. **Original KAN paper as a first-class EAT40005 cite.** Liu et al. (2024) is only nested inside the Lamaakal dump. Safe to promote in the paper; it is not a new discovery.
3. **Original LIME/SHAP and deletion-AUC papers.** CONTEXT/ADR name KernelSHAP, LIME, and AOPC, but Ribeiro, Lundberg, Petsiuk, and Samek are absent. Surveys (Salih, Kim, Vimbi) are what the paper can cite today.
4. **Pigmentation / device / site / sex / age / race.** Cited as external caution (Cotton, Benner, Barros, Fong descriptor). Not in the scored export. Do not open another fairness card.
5. **Motion.** No IMU channel is used. PI tertiles are a perfusion/quality axis, not motion (see `docs/tinykan-motion-subgroups.md`).
6. **Lamaakal ZSL/HAR numbers.** In-repo reproduction did not recover the paper’s unseen-activity accuracies. Do not recycle those HAR figures as oxygen evidence.
7. **ETPB operating-point gate.** Existing TinyML/PPG papers do not discuss the recall ≥ 0.80 and specificity ≥ 0.50 gate. Calibration left the firmware at 0.50; do not retune here.
8. **XAI sample size.** Research-plan rubric contemplated a larger sample. n=20 at 5.70/6 cannot claim that gate.

## Provenance of this inventory

| Source | What was taken |
|---|---|
| EAT40005 Research Plan (Canvas 2172 / assignment 37780, file 1805469) | Numbered refs [1]–[15]; §3 literature synthesis. |
| EAT40005 Progress Report (assignment 37784, file 1881309) | Expanded refs [1]–[32]; Ch. 2 four-strand review. |
| EAT40005 Poster (assignment 37781, file 1827414) | Zhong / Baciu / Wei / Lamaakal comparison grid. |
| `TinyKAN/` Lamaakal markdown + reproduction notes | Full paper text, Liu [46], ZSL non-reproduction. |
| `data/physionet.org/files/openox-repo/1.1.1/` and `bidmc/1.0.0/` | Dataset versions, license, BIDMC README citation. |
| `docs/adr/0001`–`0005`, README, CONTEXT, notebooks 12 and 15 | Method names and non-claims. |

No firmware, Android, threshold, model, rubric, fairness table, or PI table was edited. No child cards. `n_gap_fill = 0` because every theme already had at least one in-repo citation.

Companion BibTeX (Crossref/DOI + arXiv, no invented keys): `docs/tinykan-existing-literature.bib`.
