# Explainable TinyML for Real-Time PPG Oxygen Risk Monitoring

An embedded machine learning research framework that integrates an on-device low-oxygen risk score with a local Mean-Perturbation explanation on a resource-constrained microcontroller. Detection of oxygenation is not established.

## Core Terminology

**Low-Oxygen Risk**:
A binary window label: a window median ScalcO2 below 90 percent; ScalcO2 is the dataset's SaO2 estimated from ETO2, not a pulse-oximeter SpO2. It is not a blood-oxygen measurement.
_Avoid_: Hypoxia prediction, SpO2 regression, disease diagnosis

**TinyKAN**:
A compact spline network for microcontroller inference: a dense linear mix from 14 inputs to 12 hidden units, one cubic B-spline on each hidden node, a second linear mix, and one cubic B-spline on the scalar output (297 parameters). Liu et al.'s KAN definition (learnable univariate functions on edges) is not what this firmware implements.
_Avoid_: edge-KAN, MLP, deep network

**Baseline Model / Header-Aligned Model**:
Reader-facing names (report, slides, dashboard) for the two TinyKAN models, chosen 25 Oct 2026. Since 6 Nov 2026 the IEEE paper reports one model, called TinyKAN, and uses neither name; the label defect appears there only as a Methods correction to an earlier export (ADR-0006, 6 Nov section). Baseline model = repo `v1`: 14 IR features on the as-coded notebook-10 labels, the comparison (the reported system until 5 Nov 2026). Header-aligned model = repo `v2`: 14 IR features plus ratio of ratios and IR perfusion index (16 inputs) on header-aligned labels, the reported system since 5 Nov 2026, when Jason chose to lead the paper's title, abstract and results with it (ADR-0006, appended note). Repo IDs (`v1`/`v2` in scripts, output folders, headers, firmware envs, ADRs) are unchanged. `v3` is not reported and has no reader-facing name.
_Avoid_: v1, v2 (in deliverables); "baseline" for anything else in the same text (say training-mean reference, logistic regression, reference build)

**Mean-Perturbation Attribution**:
A local feature attribution method that replaces a single normalized feature with its training dataset mean (0.0) and measures the resulting output probability drop.
_Avoid_: SHAP, LIME, ablation study, feature importance

**Linear Coefficient Attribution**:
The baseline local explanation method for Logistic Regression, computed directly as the product of the standardized feature value and its model weight ($x_i \times w_i$).
_Avoid_: Feature weight, static ranking

**PPG Sliding Window**:
A contiguous 10-second buffer of raw photoplethysmogram samples (500 samples at 50 Hz) used to extract 14 infrared features.
_Avoid_: Epoch, slice, frame

**Top-3 Explanation**:
The three features with the largest absolute attribution $|\Delta_i|$, ranked in descending order (ties broken by feature index) and transmitted over BLE with their signed values for operator transparency. Because the ranking is sign-agnostic, a top-3 feature may be one that lowers the risk probability.
_Avoid_: Global summary, model weights, explanation vector

**Deletion AUC (AOPC)**:
The quantitative faithfulness metric evaluating the decay rate of low-oxygen risk probability as features are progressively masked in order of their attribution ranking.
_Avoid_: Accuracy drop, ablation curve

**Sliding Stride**:
The 5.0-second temporal hop (250 samples at 50 Hz; 50% overlap between successive 10-second windows) at which 14 PPG features are re-extracted and TinyKAN inference is executed on-device. Each window is filtered inside a 20-second buffer with 5 seconds of past and 5 seconds of future context, so its decision is emitted 5 seconds after the window ends.
_Avoid_: Hop length, refresh rate

**KernelSHAP Reference**:
An unused host nickname. KernelSHAP was not run. It is not the on-device method and not the AOPC faithfulness evaluation.
_Avoid_: Ground truth, exact Shapley values

**Standardization Header**:
A generated C/C++ header storing the 14 feature means and standard deviations as compile-time flash constants.
_Avoid_: Scaler file, normalization matrix

**ASCII Telemetry Payload**:
A pipe-delimited string broadcast via BLE GATT containing live SpO2, risk state, raw optical levels, and top-3 feature attributions.
_Avoid_: Binary packet, JSON stream

**Dual Latency Profiling**:
The intended benchmarking methodology combining internal ARM Cortex-M DWT cycle counting with physical GPIO pin toggling for oscilloscope validation. GPIO toggling has not been measured. Reported timing is cycle-derived only.
_Avoid_: Software timing, print profiling

**Temporal Debounce**:
The policy requiring 2 consecutive positive sliding windows before transitioning the system state to Low-Oxygen Risk.
_Avoid_: Smoothing filter, hysteresis threshold

**SpO2 est**:
The display-only red/IR value in BLE field 0 and the serial `SpO2_est:` field. On `nano33ble_v2_f1a_spo2` it comes from Maxim's reference algorithm (ratio of ratios through Maxim's curve, shown only in 70–100, ADR-0010). Other images show a demo formula. It is uncalibrated for this sensor, not a TinyKAN input, not used by any decision or explanation, and not a blood-oxygen reading.
_Avoid_: SpO2 (alone), oxygen level, oxygen saturation reading

**Session CSV Export**:
The client-side browser mechanism that downloads raw PPG streams, 14 features, risk states, and top-3 attributions as a timestamped CSV file.
_Avoid_: Database sync, cloud dumping



