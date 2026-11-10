# Project: fyrp-xai-tinyml

## Architecture
The `fyrp-xai-tinyml` project encapsulates real-time PPG feature extraction, TinyKAN neural network inference, Mean-Perturbation XAI attribution, and ADR-0005 temporal risk debounce confirmation into a unified, stateful C++ streaming pipeline class: `OxygenRiskPipeline`.

### System Data Flow
```
Raw PPG IR Sample (float) -> [push_sample]
       |
       v
[500-Sample Circular Buffer (50 Hz)]
       | (Every 250 samples / 5.0s step)
       v
[4th-Order Butterworth DF2T Bandpass Filter (0.5 - 8.0 Hz)]
       |
       v
[14 Feature Extraction (10 Statistical + 4 Welch 256-pt FFT Spectral)]
       |
       v
[TinyKAN FP32 Inference (14 -> 12 -> 1 B-Spline Net)]
       |
       v
[15-Pass Mean-Perturbation XAI Attribution -> Top-3 Feature Drivers]
       |
       v
[ADR-0005 Temporal Debounce Confirmation Machine (N=2 Consecutive Windows >= 0.50)]
       |
       v
Returns std::optional<RiskDecision>
```

### Portability Layer
The pipeline header `firmware/Arduino Nano 33/include/oxygen_risk_pipeline.h` uses hardware decoupling macros (`#ifndef ARDUINO`) to allow compilation under native desktop C++ compilers (MSVC `cl.exe`, `g++`, `clang++`) without requiring physical Arduino hardware libraries.

---

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Raw PPG Ring Buffer | 500-sample window at 50 Hz with 50% overlap / 250-sample step | M1 (R1) | ORIGINAL_REQUEST §R1 |
| 2 | 4th-Order Butterworth Filter | Direct Form II Transposed bandpass [0.5, 8.0] Hz with step response zi | M1 (R1) | ORIGINAL_REQUEST §R1 |
| 3 | 10 Statistical Features | mean, std, min, max, range, rms, skewness, kurtosis, energy, derivative_std | M1 (R1) | feature_extraction.h |
| 4 | 4 Spectral Features | dominant_freq, bandpower_0_5_2, bandpower_2_5, ratio via 256-pt Welch FFT | M1 (R1) | feature_extraction.h |
| 5 | TinyKAN FP32 Forward Inference | $14 \to 12 \to 1$ B-spline neural network forward pass | M1 (R1) | kan_inference.h |
| 6 | 15-Pass Mean-Perturbation XAI | Calculate top-3 feature attributions by perturbing each feature to dataset mean | M1 (R1) | kan_inference.h |
| 7 | Zero-Allocation Streaming Interface | `push_sample(float ir_val) -> std::optional<RiskDecision>` with no caller dynamic arrays | M1 (R1) | ORIGINAL_REQUEST §R1 |
| 8 | Hardware Decoupling Layer | `#ifndef ARDUINO` standard C++ wrapper headers | M1 (R1) | explorer_survey_1 |
| 9 | ADR-0005 Debounce Machine | Assert `is_confirmed_risk = true` iff $N=2$ consecutive windows produce $P \ge 0.50$ | M2 (R2) | ADR-0005 |
| 10 | Transient Spike Suppression | Single high-risk window flags candidate risk but suppresses confirmed alert | M2 (R2) | ADR-0005 |
| 11 | Pipeline Reset & Recovery | Clear consecutive window counter on signal loss or finger disconnect | M2 (R2) | ADR-0005 |
| 12 | `RiskDecision` Struct Payload | Holds risk_prob, candidate flag, confirmed flag, top-3 attributions, raw features | M2 (R2) | ORIGINAL_REQUEST §R1-R2 |
| 13 | Firmware Main Loop Integration | Refactor `firmware/Arduino Nano 33/src/main.cpp` to use `OxygenRiskPipeline` | M3 (R3) | ORIGINAL_REQUEST §R3 |
| 14 | Firmware Serial & BLE Notifications | Update serial telemetry and BLE characteristic with confirmed decision & top-3 XAI | M3 (R3) | ORIGINAL_REQUEST §R3 |
| 15 | Host Native C++ Test Suite | Desktop test runner (`tests/test_oxygen_risk_pipeline.cpp`) compiled with MSVC / g++ | M4 (R4) | ORIGINAL_REQUEST §R4 |
| 16 | Filter & 14-Feature Parity Unit Tests | Assert feature extraction parity within $\le 10^{-4}$ tolerance vs SciPy reference | M4 (R4) | ORIGINAL_REQUEST §R4 |
| 17 | KAN & XAI Parity Unit Tests | Assert KAN prediction parity ($< 0.01$) and top-3 feature attribution ranking | M4 (R4) | ORIGINAL_REQUEST §R4 |
| 18 | Debounce Transition Unit Tests | Exhaustive test coverage of all 4 state machine scenarios (Normal, Candidate, Confirmed, Resolution) | M4 (R4) | ORIGINAL_REQUEST §R4 |

---

## Code Layout
```
fyrp-xai-tinyml/
├── docs/
│   └── adr/
│       └── 0005-consecutive-window-risk-confirmation.md
├── firmware/
│   └── Arduino Nano 33/
│       ├── include/
│       │   ├── feature_extraction.h
│       │   ├── kan_inference.h
│       │   ├── kan_model_weights.h
│       │   └── oxygen_risk_pipeline.h       (M1, M2 - Header & implementation)
│       └── src/
│           └── main.cpp                    (M3 - Firmware main loop)
├── scripts/
│   └── verify_features.py
├── tests/
│   ├── test_oxygen_risk_pipeline.cpp        (M4 - Host native test runner)
│   └── CMakeLists.txt                      (M4 - Native CMake build script)
├── .agents/                                (Agent metadata only)
└── PROJECT.md                              (Global architecture & plan)
```

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Deep OxygenRiskPipeline Module | Create `oxygen_risk_pipeline.h` stateful class encapsulating 500-sample buffer, 4th-order Butterworth filter, 14-feature Welch FFT extraction, TinyKAN FP32 inference, 15-pass XAI, zero caller allocation | none | DONE |
| M2 | ADR-0005 Debounce Machine | Implement ADR-0005 temporal debounce state logic in `OxygenRiskPipeline` ($N=2$ consecutive windows $P \ge 0.50$), `RiskDecision` struct, candidate vs confirmed states, reset handling | M1 | DONE |
| M3 | Arduino Nano 33 BLE Firmware | Integrate `OxygenRiskPipeline` into `firmware/Arduino Nano 33/src/main.cpp`, update Serial telemetry and BLE characteristic notifications | M1, M2 | DONE |
| M4 | Native Desktop C++ Test Suite | Build native desktop test runner `tests/test_oxygen_risk_pipeline.cpp` verifying feature/filter parity ($\le 10^{-4}$), KAN probability parity ($< 0.01$), and exhaustive state machine debounce transitions | M1, M2 | DONE |

---

## Interface Contracts

### `struct FeatureAttribution`
```cpp
struct FeatureAttribution {
    uint8_t feature_index;
    float importance_delta;
    const char* feature_name;
};
```

### `struct RiskDecision`
```cpp
struct RiskDecision {
    uint32_t window_index;
    uint64_t total_samples_ingested;
    float risk_probability;
    bool is_candidate_risk;
    bool is_confirmed_risk;
    uint8_t consecutive_risk_windows;
    FeatureAttribution top_attributions[3];
    float raw_features[14];
};
```

### `class OxygenRiskPipeline` Interface
```cpp
class OxygenRiskPipeline {
public:
    OxygenRiskPipeline(float decision_threshold = 0.50f, uint8_t required_consecutive_windows = 2);
    
    // Primary streaming interface
    // Returns std::nullopt during buffering (samples 0..499, and step gaps 1..249)
    // Emits populated RiskDecision upon window completion (every 250 samples)
    std::optional<RiskDecision> push_sample(float ir_val);
    
    // State management
    void reset();
    
    // Status queries
    bool is_buffer_ready() const;
    uint64_t get_total_samples() const;
    uint8_t get_consecutive_risk_count() const;
    bool is_confirmed_risk() const;
};
```
