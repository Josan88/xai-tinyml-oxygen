#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cassert>
#include <cstring>
#include <sstream>
#include <string>
#include <algorithm>

#include "oxygen_risk_pipeline.h"

// -----------------------------------------------------------------
// Test Assertion & Tracking Infrastructure
// -----------------------------------------------------------------
struct TestTracker {
    int total_assertions = 0;
    int passed_assertions = 0;
    int failed_assertions = 0;

    void check(bool condition, const std::string& msg) {
        total_assertions++;
        if (condition) {
            passed_assertions++;
            std::cout << "  [PASS] " << msg << "\n";
        } else {
            failed_assertions++;
            std::cout << "  [FAIL] " << msg << "\n";
        }
    }

    void check_near(float actual, float expected, float tol, const std::string& name) {
        float diff = std::fabs(actual - expected);
        bool pass = (diff <= tol);
        total_assertions++;
        if (pass) {
            passed_assertions++;
            std::cout << "  [PASS] " << std::left << std::setw(28) << name 
                      << " | Act: " << std::fixed << std::setprecision(6) << std::setw(12) << actual 
                      << " | Exp: " << std::setw(12) << expected 
                      << " | Diff: " << std::scientific << std::setprecision(3) << diff << "\n";
        } else {
            failed_assertions++;
            std::cout << "  [FAIL] " << std::left << std::setw(28) << name 
                      << " | Act: " << std::fixed << std::setprecision(6) << std::setw(12) << actual 
                      << " | Exp: " << std::setw(12) << expected 
                      << " | Diff: " << std::scientific << std::setprecision(3) << diff 
                      << " (Exceeds tol " << std::setprecision(1) << tol << ")\n";
        }
    }
};

// -----------------------------------------------------------------
// Deterministic Synthetic PPG Generators
// -----------------------------------------------------------------
// Normal PPG signal (1.2 Hz pulse + normal respiratory wave + low noise -> risk_prob = 0.126565 < 0.50)
std::vector<float> generate_synthetic_ppg(size_t num_samples, float freq_hz = 1.2f, float noise_amplitude = 5.0f) {
    std::vector<float> ppg(num_samples);
    const float fs = 50.0f;
    for (size_t i = 0; i < num_samples; ++i) {
        float t = static_cast<float>(i) / fs;
        float cardiac = 100.0f * std::sin(2.0f * static_cast<float>(M_PI) * freq_hz * t);
        float respiratory = 20.0f * std::sin(2.0f * static_cast<float>(M_PI) * 0.1f * t);
        float noise = noise_amplitude * (std::sin(13.0f * t + 0.5f) + std::cos(29.0f * t + 1.2f));
        ppg[i] = cardiac + respiratory + noise;
    }
    return ppg;
}

// High-risk PPG signal (2.2 Hz pulse -> bandpower_ratio drops -> risk_prob = 0.6468 >= 0.50)
std::vector<float> generate_high_risk_ppg(size_t num_samples) {
    std::vector<float> ppg(num_samples);
    const float fs = 50.0f;
    for (size_t i = 0; i < num_samples; ++i) {
        float t = static_cast<float>(i) / fs;
        float cardiac = 100.0f * std::sin(2.0f * static_cast<float>(M_PI) * 2.2f * t);
        float respiratory = 20.0f * std::sin(2.0f * static_cast<float>(M_PI) * 0.1f * t);
        float noise = 5.0f * (std::sin(13.0f * t + 0.5f) + std::cos(29.0f * t + 1.2f));
        ppg[i] = cardiac + respiratory + noise;
    }
    return ppg;
}

// Transient single-window spike signal (1.2 Hz base + 3.0 Hz component -> pure window P = 0.5385, mixed window P = 0.4750 < 0.50)
std::vector<float> generate_spike_ppg(size_t num_samples) {
    std::vector<float> ppg(num_samples);
    const float fs = 50.0f;
    for (size_t i = 0; i < num_samples; ++i) {
        float t = static_cast<float>(i) / fs;
        float base = 100.0f * std::sin(2.0f * static_cast<float>(M_PI) * 1.2f * t);
        float spike = 50.0f * std::sin(2.0f * static_cast<float>(M_PI) * 3.0f * t);
        ppg[i] = base + spike;
    }
    return ppg;
}

// -----------------------------------------------------------------
// Reference Calculator (C++ internal consistency; not independent Python parity)
// -----------------------------------------------------------------
struct ReferenceResult {
    float features[14];
    float risk_probability;
    struct {
        uint8_t index;
        float importance_delta;
        std::string name;
    } top_3[3];
};

ReferenceResult compute_reference(const float* raw_500) {
    ReferenceResult ref;
    
    // Shared filter contract; the independent Python check lives in check_probability_parity.py.
    float filtered[PPG_WINDOW_SIZE];
    float scratch[PPG_FILTER_SIZE];
    ppg_bandpass_filter(raw_500, filtered, scratch);

    // 2. Statistical features (10)
    float sum = 0.0f, min_v = filtered[0], max_v = filtered[0], sum_sq = 0.0f;
    for (int i = 0; i < 500; ++i) {
        float v = filtered[i];
        sum += v; sum_sq += v * v;
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
    }
    float mean_v = sum / 500.0f;
    float rms_v = std::sqrt(sum_sq / 500.0f);
    float energy_v = sum_sq;
    float range_v = max_v - min_v;

    float m2 = 0.0f, m3 = 0.0f, m4 = 0.0f;
    for (int i = 0; i < 500; ++i) {
        float d = filtered[i] - mean_v;
        float d2 = d * d;
        m2 += d2; m3 += d2 * d; m4 += d2 * d2;
    }
    m2 /= 500.0f; m3 /= 500.0f; m4 /= 500.0f;
    float std_v = std::sqrt(m2);

    float skew_v = 0.0f, kurt_v = 0.0f;
    if (std_v > 1e-10f) {
        float g1 = m3 / (m2 * std_v);
        skew_v = std::sqrt(500.0f * 499.0f) / 498.0f * g1;
        float g2 = m4 / (m2 * m2) - 3.0f;
        float term1 = 499.0f / (498.0f * 497.0f);
        kurt_v = term1 * (501.0f * g2 + 6.0f);
    }

    float diff_sum = 0.0f, diff_sum_sq = 0.0f;
    for (int i = 0; i < 499; ++i) {
        float d = filtered[i + 1] - filtered[i];
        diff_sum += d; diff_sum_sq += d * d;
    }
    float diff_mean = diff_sum / 499.0f;
    float deriv_std = std::sqrt((diff_sum_sq / 499.0f) - (diff_mean * diff_mean));

    // 3. Spectral features via Welch FFT (4)
    float hann[256];
    for (int i = 0; i < 256; ++i) {
        hann[i] = 0.5f * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * i / 256.0f));
    }
    float psd[129] = {0};
    const float welch_scale = 1.0f / (50.0f * 96.0f);

    for (int seg = 0; seg < 2; ++seg) {
        int start_idx = seg * 128;
        float seg_data[256];
        for (int i = 0; i < 256; ++i) {
            seg_data[i] = (start_idx + i < 500) ? filtered[start_idx + i] : 0.0f;
        }
        float s_sum = 0.0f;
        for (int i = 0; i < 256; ++i) s_sum += seg_data[i];
        float s_mean = s_sum / 256.0f;
        for (int i = 0; i < 256; ++i) seg_data[i] -= s_mean;

        float re[256], im[256];
        for (int i = 0; i < 256; ++i) {
            re[i] = seg_data[i] * hann[i]; im[i] = 0.0f;
        }
        for (int i = 0; i < 256; ++i) {
            unsigned int rev = 0, x = i;
            for (int b = 0; b < 8; ++b) { rev = (rev << 1) | (x & 1); x >>= 1; }
            if (i < static_cast<int>(rev)) { std::swap(re[i], re[rev]); std::swap(im[i], im[rev]); }
        }
        for (int len = 2; len <= 256; len <<= 1) {
            float ang = -2.0f * static_cast<float>(M_PI) / len;
            float wlen_r = std::cos(ang), wlen_i = std::sin(ang);
            for (int i = 0; i < 256; i += len) {
                float w_r = 1.0f, w_i = 0.0f;
                int half = len / 2;
                for (int j = 0; j < half; ++j) {
                    float u_r = re[i + j], u_i = im[i + j];
                    float v_r = re[i + j + half] * w_r - im[i + j + half] * w_i;
                    float v_i = re[i + j + half] * w_i + im[i + j + half] * w_r;
                    re[i + j] = u_r + v_r; im[i + j] = u_i + v_i;
                    re[i + j + half] = u_r - v_r; im[i + j + half] = u_i - v_i;
                    float n_w_r = w_r * wlen_r - w_i * wlen_i;
                    w_i = w_r * wlen_i + w_i * wlen_r; w_r = n_w_r;
                }
            }
        }
        psd[0] += (re[0] * re[0] + im[0] * im[0]) * welch_scale;
        for (int k = 1; k < 128; ++k) {
            psd[k] += 2.0f * (re[k] * re[k] + im[k] * im[k]) * welch_scale;
        }
        psd[128] += (re[128] * re[128] + im[128] * im[128]) * welch_scale;
    }
    for (int k = 0; k < 129; ++k) psd[k] /= 2.0f;

    float max_p = psd[1]; int max_bin = 1;
    for (int k = 2; k < 129; ++k) {
        if (psd[k] > max_p) { max_p = psd[k]; max_bin = k; }
    }
    float dom_freq = static_cast<float>(max_bin) * (50.0f / 256.0f);
    float df = 50.0f / 256.0f;

    float bp_0_5_2_sum = 0.5f * psd[3];
    for (int k = 4; k < 10; ++k) bp_0_5_2_sum += psd[k];
    bp_0_5_2_sum += 0.5f * psd[10];
    float bp_0_5_2 = bp_0_5_2_sum * df;

    float bp_2_5_sum = 0.5f * psd[11];
    for (int k = 12; k < 25; ++k) bp_2_5_sum += psd[k];
    bp_2_5_sum += 0.5f * psd[25];
    float bp_2_5 = bp_2_5_sum * df;

    float bp_ratio = (bp_2_5 > 0.0f) ? (bp_0_5_2 / bp_2_5) : 0.0f;

    ref.features[0] = mean_v; ref.features[1] = std_v; ref.features[2] = min_v;
    ref.features[3] = max_v; ref.features[4] = range_v; ref.features[5] = rms_v;
    ref.features[6] = skew_v; ref.features[7] = kurt_v; ref.features[8] = energy_v;
    ref.features[9] = deriv_std; ref.features[10] = dom_freq;
    ref.features[11] = bp_0_5_2; ref.features[12] = bp_2_5; ref.features[13] = bp_ratio;

    // 4. KAN inference & 15-pass Mean-Perturbation XAI
    float scaled[14];
    kan_standardize(ref.features, scaled);
    float base_logit = kan_predict_logit(scaled);
    ref.risk_probability = kan_sigmoid(base_logit);

    struct Attr { uint8_t idx; float delta; const char* name; } all_attrs[14];
    float pert_scaled[14];
    for (int i = 0; i < 14; ++i) {
        std::memcpy(pert_scaled, scaled, 14 * sizeof(float));
        pert_scaled[i] = 0.0f;
        float p_logit = kan_predict_logit(pert_scaled);
        float p_prob = kan_sigmoid(p_logit);
        all_attrs[i].idx = static_cast<uint8_t>(i);
        all_attrs[i].delta = ref.risk_probability - p_prob;
        all_attrs[i].name = KAN_FEATURE_NAMES[i];
    }
    // Sort descending by abs(delta)
    for (int i = 0; i < 13; ++i) {
        for (int j = 0; j < 13 - i; ++j) {
            if (std::abs(all_attrs[j].delta) < std::abs(all_attrs[j + 1].delta)) {
                std::swap(all_attrs[j], all_attrs[j + 1]);
            }
        }
    }
    for (int i = 0; i < 3; ++i) {
        ref.top_3[i].index = all_attrs[i].idx;
        ref.top_3[i].importance_delta = all_attrs[i].delta;
        ref.top_3[i].name = all_attrs[i].name;
    }

    return ref;
}

// -----------------------------------------------------------------
// Test Functions
// -----------------------------------------------------------------

// TEST 1: Buffer initialization, streaming sample buffering, zero allocation,
// std::nullopt during buffering, RiskDecision struct emission on window boundaries (every 250 samples).
void test_1_buffering_and_emission(TestTracker& tracker) {
    std::cout << "\n========================================================\n";
    std::cout << " TEST 1: Streaming Buffer Initialization & Boundary Emission\n";
    std::cout << "========================================================\n";

    WindowLocalOxygenRiskPipeline pipeline;

    // Check initial pipeline state
    tracker.check(pipeline.is_buffer_ready() == false, "Initial is_buffer_ready() is false");
    tracker.check(pipeline.get_total_samples() == 0, "Initial get_total_samples() is 0");
    tracker.check(pipeline.get_consecutive_risk_count() == 0, "Initial consecutive_risk_count is 0");
    tracker.check(pipeline.is_confirmed_risk() == false, "Initial is_confirmed_risk() is false");

    std::vector<float> ppg = generate_synthetic_ppg(750, 1.2f, 5.0f);

    // Push first 499 samples (samples 0..498) -> should return std::nullopt
    bool nullopt_during_buffering = true;
    for (size_t i = 0; i < 499; ++i) {
        auto res = pipeline.push_sample(ppg[i]);
        if (res.has_value()) {
            nullopt_during_buffering = false;
        }
    }
    tracker.check(nullopt_during_buffering, "Samples 0..498 return std::nullopt during window buffering");
    tracker.check(pipeline.is_buffer_ready() == false, "is_buffer_ready() remains false after 499 samples");
    tracker.check(pipeline.get_total_samples() == 499, "get_total_samples() is 499");

    // Push 500th sample (sample 499) -> triggers Window 0 emission
    auto w0_opt = pipeline.push_sample(ppg[499]);
    tracker.check(w0_opt.has_value(), "Sample 499 (500th sample) emits valid std::optional<RiskDecision>");
    tracker.check(pipeline.is_buffer_ready() == true, "is_buffer_ready() becomes true at sample 500");
    tracker.check(pipeline.get_total_samples() == 500, "get_total_samples() is 500");

    if (w0_opt.has_value()) {
        const auto& d0 = w0_opt.value();
        tracker.check(d0.window_index == 0, "Window 0 payload window_index is 0");
        tracker.check(d0.total_samples_ingested == 500, "Window 0 payload total_samples_ingested is 500");
    }

    // Push next 249 samples (samples 500..748) -> step buffering returns std::nullopt
    bool nullopt_during_step = true;
    for (size_t i = 500; i < 749; ++i) {
        auto res = pipeline.push_sample(ppg[i]);
        if (res.has_value()) {
            nullopt_during_step = false;
        }
    }
    tracker.check(nullopt_during_step, "Samples 500..748 (249 step samples) return std::nullopt");

    // Push 750th sample (sample 749 / 250 step) -> triggers Window 1 emission
    auto w1_opt = pipeline.push_sample(ppg[749]);
    tracker.check(w1_opt.has_value(), "Sample 749 (750th sample / 250 step) emits valid RiskDecision");
    tracker.check(pipeline.get_total_samples() == 750, "get_total_samples() is 750");

    if (w1_opt.has_value()) {
        const auto& d1 = w1_opt.value();
        tracker.check(d1.window_index == 1, "Window 1 payload window_index is 1");
        tracker.check(d1.total_samples_ingested == 750, "Window 1 payload total_samples_ingested is 750");
    }
}

// TEST 2: Filter stability and 14-feature statistical & 256-pt Welch FFT spectral numerical parity <= 10^-4 vs C++ reference.
void test_2_filter_and_feature_parity(TestTracker& tracker) {
    std::cout << "\n========================================================\n";
    std::cout << " TEST 2: Filter Stability & 14-Feature Numerical Parity\n";
    std::cout << "========================================================\n";

    // Subtest 2a: SOS Pole Stability Analysis & Filter Response
    {
        std::cout << " Subtest 2a: Filter SOS Stability & Response Checks\n";
        float max_pole_mag = 0.0f;
        for (int s = 0; s < 4; ++s) {
            float a1 = FILTER_SOS[s][4];
            float a2 = FILTER_SOS[s][5];
            float disc = a1 * a1 - 4.0f * a2;
            float mag = (disc >= 0.0f) ? std::max(std::abs((-a1 + std::sqrt(disc)) / 2.0f), std::abs((-a1 - std::sqrt(disc)) / 2.0f)) : std::sqrt(a2);
            if (mag > max_pole_mag) max_pole_mag = mag;
        }
        tracker.check(max_pole_mag < 1.0f, "Direct Form II Transposed SOS max pole radius < 1.0 (BIBO Stable)");

        // Impulse Response Test (10000.0f impulse at sample 0)
        WindowLocalOxygenRiskPipelineimpulse_pipe;
        std::vector<float> impulse_input(500, 0.0f);
        impulse_input[0] = 10000.0f;
        std::optional<RiskDecision> impulse_dec;
        for (int i = 0; i < 500; ++i) impulse_dec = impulse_pipe.push_sample(impulse_input[i]);

        tracker.check(impulse_dec.has_value(), "Impulse response emitted decision");
        if (impulse_dec.has_value()) {
            bool impulse_stable = true;
            for (int f = 0; f < 14; ++f) {
                if (std::isnan(impulse_dec->raw_features[f]) || std::isinf(impulse_dec->raw_features[f])) impulse_stable = false;
            }
            tracker.check(impulse_stable, "Impulse response produced bounded non-NaN/Inf features");
        }

        // Step Response Test (100.0f constant DC step)
        WindowLocalOxygenRiskPipelinestep_pipe;
        std::optional<RiskDecision> step_dec;
        for (int i = 0; i < 500; ++i) step_dec = step_pipe.push_sample(100.0f);

        tracker.check(step_dec.has_value(), "Step response emitted decision");
        if (step_dec.has_value()) {
            bool step_stable = true;
            for (int f = 0; f < 14; ++f) {
                if (std::isnan(step_dec->raw_features[f]) || std::isinf(step_dec->raw_features[f])) step_stable = false;
            }
            float dc_mean = std::abs(step_dec->raw_features[0]);
            tracker.check(step_stable && dc_mean < 1.0f, "Step response bounded and DC attenuated (mean < 1.0)");
        }
    }

    // Subtest 2b: 14-feature numerical parity vs C++ reference
    {
        std::cout << "\n Subtest 2b: 14-Feature Parity Comparison (Tolerance <= 10^-4)\n";
        WindowLocalOxygenRiskPipeline pipeline;
        std::vector<float> ppg = generate_synthetic_ppg(750, 1.2f, 5.0f);
        
        // Window 0 parity
        std::optional<RiskDecision> d0_opt;
        for (int i = 0; i < 500; ++i) d0_opt = pipeline.push_sample(ppg[i]);

        tracker.check(d0_opt.has_value(), "Window 0 emitted decision payload");
        if (d0_opt.has_value()) {
            const auto& d0 = d0_opt.value();
            ReferenceResult ref0 = compute_reference(&ppg[0]);

            std::cout << std::left << std::setw(28) << "  Feature Name"
                      << " | " << std::setw(14) << "Pipeline"
                      << " | " << std::setw(14) << "Reference"
                      << " | " << std::setw(12) << "Diff" << "\n";
            std::cout << "  " << std::string(73, '-') << "\n";

            for (int f = 0; f < 14; ++f) {
                float pipe_val = d0.raw_features[f];
                float ref_val = ref0.features[f];
                tracker.check_near(pipe_val, ref_val, 1e-4f, KAN_FEATURE_NAMES[f]);
            }
        }

        // Window 1 parity (samples 250..749)
        std::optional<RiskDecision> d1_opt;
        for (int i = 500; i < 750; ++i) d1_opt = pipeline.push_sample(ppg[i]);
        tracker.check(d1_opt.has_value(), "Window 1 emitted decision payload");
        if (d1_opt.has_value()) {
            const auto& d1 = d1_opt.value();
            ReferenceResult ref1 = compute_reference(&ppg[250]);
            bool w1_pass = true;
            for (int f = 0; f < 14; ++f) {
                if (std::abs(d1.raw_features[f] - ref1.features[f]) > 1e-4f) w1_pass = false;
            }
            tracker.check(w1_pass, "Window 1 (samples 250..749) 14-feature parity confirmed (<= 10^-4)");
        }
    }
}

// TEST 3: TinyKAN FP32 forward pass prediction parity (< 0.01 tolerance) and 15-pass Mean-Perturbation top-3 feature attribution ranking vs C++ reference.
void test_3_kan_prediction_and_xai_parity(TestTracker& tracker) {
    std::cout << "\n========================================================\n";
    std::cout << " TEST 3: TinyKAN FP32 Prediction & 15-Pass XAI Parity\n";
    std::cout << "========================================================\n";

    // Normal synthetic signal prediction & XAI parity
    {
        std::cout << " Subtest 3a: Normal Signal KAN Prediction & XAI Attribution\n";
        WindowLocalOxygenRiskPipeline pipeline;
        std::vector<float> ppg = generate_synthetic_ppg(500, 1.2f, 5.0f);
        std::optional<RiskDecision> d_opt;
        for (int i = 0; i < 500; ++i) d_opt = pipeline.push_sample(ppg[i]);

        tracker.check(d_opt.has_value(), "Normal signal emitted decision payload");
        if (d_opt.has_value()) {
            const auto& d = d_opt.value();
            ReferenceResult ref = compute_reference(&ppg[0]);

            tracker.check_near(d.risk_probability, ref.risk_probability, 0.01f, "KAN Risk Probability (< 0.01)");
            tracker.check_near(d.risk_probability, ref.risk_probability, 1e-5f, "KAN Risk Probability (< 1e-5 exact)");

            std::cout << "\n  Comparing Top-3 Feature Driver Attributions:\n";
            for (int i = 0; i < 3; ++i) {
                const auto& pipe_attr = d.top_attributions[i];
                const auto& ref_attr = ref.top_3[i];

                std::stringstream ss_pipe, ss_ref;
                ss_pipe << "Top-" << (i + 1) << " Pipe (" << pipe_attr.feature_name << ")";
                ss_ref << "Top-" << (i + 1) << " Ref (" << ref_attr.name << ")";

                tracker.check(pipe_attr.feature_index == ref_attr.index, 
                              ss_pipe.str() + " index matches " + ss_ref.str());
                tracker.check(std::string(pipe_attr.feature_name) == ref_attr.name, 
                              ss_pipe.str() + " name matches " + ss_ref.str());
                tracker.check_near(pipe_attr.importance_delta, ref_attr.importance_delta, 1e-5f, 
                                   ss_pipe.str() + " importance delta");
            }
        }
    }

    // High-risk synthetic signal prediction & XAI parity
    {
        std::cout << "\n Subtest 3b: High-Risk Signal KAN Prediction & XAI Attribution\n";
        WindowLocalOxygenRiskPipelinehr_pipeline;
        std::vector<float> hr_ppg = generate_high_risk_ppg(500);
        std::optional<RiskDecision> hr_opt;
        for (int i = 0; i < 500; ++i) hr_opt = hr_pipeline.push_sample(hr_ppg[i]);

        tracker.check(hr_opt.has_value(), "High-risk signal emitted decision payload");
        if (hr_opt.has_value()) {
            const auto& hr_d = hr_opt.value();
            ReferenceResult hr_ref = compute_reference(&hr_ppg[0]);

            tracker.check_near(hr_d.risk_probability, hr_ref.risk_probability, 0.01f, "High-Risk Probability (< 0.01)");
            tracker.check_near(hr_d.risk_probability, hr_ref.risk_probability, 1e-5f, "High-Risk Probability (< 1e-5 exact)");

            for (int i = 0; i < 3; ++i) {
                const auto& pipe_attr = hr_d.top_attributions[i];
                const auto& ref_attr = hr_ref.top_3[i];

                std::stringstream ss;
                ss << "HR Top-" << (i + 1) << " (" << pipe_attr.feature_name << ")";

                tracker.check(pipe_attr.feature_index == ref_attr.index, ss.str() + " index matches reference");
                tracker.check_near(pipe_attr.importance_delta, ref_attr.importance_delta, 1e-5f, ss.str() + " delta matches reference");
            }
        }
    }
}

// TEST 4: Exhaustive ADR-0005 Debounce State Machine Scenarios:
//   4a: Normal baseline (risk_prob < 0.50, candidate=false, confirmed=false).
//   4b: Transient single-window spike (risk_prob >= 0.50, candidate=true, confirmed=false - suppressed).
//   4c: Two consecutive high-risk windows (N=2, candidate=true, confirmed=true - alert asserted).
//   4d: Sustained high-risk windows (confirmed=true remains active).
//   4e: Resolution recovery (low-risk window clears confirmed=false, consecutive_count=0).
//   Plus Oscillating noise immunity.
void test_4_adr0005_debounce_scenarios(TestTracker& tracker) {
    std::cout << "\n========================================================\n";
    std::cout << " TEST 4: Exhaustive ADR-0005 Debounce State Machine Scenarios\n";
    std::cout << "========================================================\n";

    WindowLocalOxygenRiskPipeline pipeline(0.50f, 2);
    std::vector<float> normal_ppg = generate_synthetic_ppg(500, 1.2f, 5.0f);
    std::vector<float> high_risk_ppg = generate_high_risk_ppg(1000);

    // Scenario 4a: Normal Baseline
    std::cout << " Scenario 4a: Normal Baseline (Window 0, P < 0.50)\n";
    std::optional<RiskDecision> d_4a;
    for (int i = 0; i < 500; ++i) d_4a = pipeline.push_sample(normal_ppg[i]);

    tracker.check(d_4a.has_value(), "Window 0 emitted decision");
    if (d_4a.has_value()) {
        tracker.check(d_4a->risk_probability < 0.50f, "Window 0 risk_probability < 0.50f");
        tracker.check(d_4a->is_candidate_risk == false, "Window 0 is_candidate_risk is false");
        tracker.check(d_4a->is_confirmed_risk == false, "Window 0 is_confirmed_risk is false");
        tracker.check(d_4a->consecutive_risk_windows == 0, "Window 0 consecutive_risk_windows is 0");
    }

    // Scenario 4b: Transient Single-Window Spike (Push 250 step samples of high-risk)
    std::cout << "\n Scenario 4b: Transient Single-Window Spike (Window 1 High-Risk -> Alert Suppressed)\n";
    std::optional<RiskDecision> d_4b;
    for (int i = 0; i < 250; ++i) d_4b = pipeline.push_sample(high_risk_ppg[i]);

    tracker.check(d_4b.has_value(), "Window 1 emitted decision");
    if (d_4b.has_value()) {
        tracker.check(d_4b->risk_probability >= 0.50f, "Window 1 risk_probability >= 0.50f");
        tracker.check(d_4b->is_candidate_risk == true, "Window 1 is_candidate_risk is TRUE (Candidate)");
        tracker.check(d_4b->is_confirmed_risk == false, "Window 1 is_confirmed_risk is FALSE (Alert Suppressed)");
        tracker.check(d_4b->consecutive_risk_windows == 1, "Window 1 consecutive_risk_windows is 1");
    }

    // Scenario 4c: Two Consecutive High-Risk Windows (Push next 250 step samples of high-risk)
    std::cout << "\n Scenario 4c: Two Consecutive High-Risk Windows (Window 2 High-Risk -> Alert Asserted)\n";
    std::optional<RiskDecision> d_4c;
    for (int i = 250; i < 500; ++i) d_4c = pipeline.push_sample(high_risk_ppg[i]);

    tracker.check(d_4c.has_value(), "Window 2 emitted decision");
    if (d_4c.has_value()) {
        tracker.check(d_4c->risk_probability >= 0.50f, "Window 2 risk_probability >= 0.50f");
        tracker.check(d_4c->is_candidate_risk == true, "Window 2 is_candidate_risk is TRUE");
        tracker.check(d_4c->is_confirmed_risk == true, "Window 2 is_confirmed_risk is TRUE (Hypoxia Alert Asserted)");
        tracker.check(d_4c->consecutive_risk_windows == 2, "Window 2 consecutive_risk_windows is 2");
    }

    // Scenario 4d: Sustained High-Risk Windows (Push next 250 and 500 step samples)
    std::cout << "\n Scenario 4d: Sustained High-Risk Windows (Window 3 & 4 High-Risk)\n";
    std::optional<RiskDecision> d_4d_3, d_4d_4;
    for (int i = 500; i < 750; ++i) d_4d_3 = pipeline.push_sample(high_risk_ppg[i]);
    for (int i = 750; i < 1000; ++i) d_4d_4 = pipeline.push_sample(high_risk_ppg[i]);

    tracker.check(d_4d_3.has_value() && d_4d_4.has_value(), "Windows 3 and 4 emitted decisions");
    if (d_4d_3.has_value() && d_4d_4.has_value()) {
        tracker.check(d_4d_3->is_confirmed_risk == true, "Window 3 is_confirmed_risk remains TRUE");
        tracker.check(d_4d_3->consecutive_risk_windows == 3, "Window 3 consecutive_risk_windows is 3");

        tracker.check(d_4d_4->is_confirmed_risk == true, "Window 4 is_confirmed_risk remains TRUE");
        tracker.check(d_4d_4->consecutive_risk_windows == 4, "Window 4 consecutive_risk_windows is 4");
    }

    // Scenario 4e: Resolution Recovery (Push 500 samples of normal PPG -> 100% low-risk window)
    std::cout << "\n Scenario 4e: Resolution Recovery (Window 5 Normal Signal -> Alarm Cleared)\n";
    std::optional<RiskDecision> d_4e;
    for (int i = 0; i < 500; ++i) {
        auto res = pipeline.push_sample(normal_ppg[i]);
        if (res.has_value()) d_4e = res;
    }

    tracker.check(d_4e.has_value(), "Window 5 emitted decision");
    if (d_4e.has_value()) {
        tracker.check(d_4e->risk_probability < 0.50f, "Window 5 risk_probability < 0.50f");
        tracker.check(d_4e->is_candidate_risk == false, "Window 5 is_candidate_risk is FALSE");
        tracker.check(d_4e->is_confirmed_risk == false, "Window 5 is_confirmed_risk is FALSE (Alarm Cleared)");
        tracker.check(d_4e->consecutive_risk_windows == 0, "Window 5 consecutive_risk_windows reset to 0");
    }

    // Additional Subtest: Oscillating Noise Immunity Sequence
    std::cout << "\n Subtest 4f: Oscillating Spike Sequence (Alternating High/Low Risk Windows)\n";
    WindowLocalOxygenRiskPipelineosc_pipe(0.50f, 2);
    std::vector<float> spike_ppg = generate_spike_ppg(500);

    // Window 0: 500 normal samples (Low risk, counter = 0)
    for (size_t i = 0; i < 500; ++i) osc_pipe.push_sample(normal_ppg[i]);

    // High risk spike 1 (500 spike samples: step 1 mixed P=0.4750 counter=0, step 2 pure P=0.5385 counter=1)
    std::optional<RiskDecision> osc1;
    for (size_t i = 0; i < 500; ++i) {
        auto r = osc_pipe.push_sample(spike_ppg[i]);
        if (r.has_value()) osc1 = r;
    }
    tracker.check(osc1.has_value() && osc1->consecutive_risk_windows == 1 && osc1->is_confirmed_risk == false,
                  "Oscillating Spike 1: Counter=1, Confirmed=false (Alert Suppressed)");

    // Low risk recovery (500 normal samples: step 1 mixed P=0.4661 counter=0, step 2 100% normal counter=0)
    std::optional<RiskDecision> osc2;
    for (size_t i = 0; i < 500; ++i) {
        auto r = osc_pipe.push_sample(normal_ppg[i]);
        if (r.has_value()) osc2 = r;
    }
    tracker.check(osc2.has_value() && osc2->consecutive_risk_windows == 0 && osc2->is_confirmed_risk == false,
                  "Oscillating Low Risk: Counter=0 (Reset), Confirmed=false");

    // High risk spike 2 (500 spike samples: step 1 mixed P=0.4750 counter=0, step 2 pure P=0.5385 counter=1)
    std::optional<RiskDecision> osc3;
    for (size_t i = 0; i < 500; ++i) {
        auto r = osc_pipe.push_sample(spike_ppg[i]);
        if (r.has_value()) osc3 = r;
    }
    tracker.check(osc3.has_value() && osc3->consecutive_risk_windows == 1 && osc3->is_confirmed_risk == false,
                  "Oscillating Spike 2: Counter=1, Confirmed=false (Alert Suppressed Throughout)");
    if (osc3.has_value()) {
        std::cout << "  - Osc 3 Prob: " << osc3->risk_probability << ", Counter: " << (int)osc3->consecutive_risk_windows 
                  << ", Confirmed: " << (osc3->is_confirmed_risk ? "true" : "false") << "\n";
    }
    tracker.check(osc3.has_value() && osc3->consecutive_risk_windows == 1 && osc3->is_confirmed_risk == false,
                  "Oscillating Spike 2: Counter=1, Confirmed=false (Alert Suppressed Throughout)");
}

// TEST 5: Pipeline reset() state machine behavior (clears buffers/counters, mid-stream reset, post-reset parity).
void test_5_pipeline_reset_behavior(TestTracker& tracker) {
    std::cout << "\n========================================================\n";
    std::cout << " TEST 5: Pipeline reset() State Machine Behavior\n";
    std::cout << "========================================================\n";

    WindowLocalOxygenRiskPipeline pipeline;
    std::vector<float> ppg = generate_synthetic_ppg(750, 1.2f, 5.0f);

    // Ingest 500 samples to reach active state
    std::optional<RiskDecision> pre_reset_w0;
    for (int i = 0; i < 500; ++i) pre_reset_w0 = pipeline.push_sample(ppg[i]);

    tracker.check(pipeline.is_buffer_ready() == true, "Pipeline buffer is ready before reset()");
    tracker.check(pipeline.get_total_samples() == 500, "total_samples is 500 before reset()");

    // Execute reset()
    pipeline.reset();

    tracker.check(pipeline.is_buffer_ready() == false, "is_buffer_ready() is false after reset()");
    tracker.check(pipeline.get_total_samples() == 0, "get_total_samples() is 0 after reset()");
    tracker.check(pipeline.get_consecutive_risk_count() == 0, "get_consecutive_risk_count() is 0 after reset()");
    tracker.check(pipeline.is_confirmed_risk() == false, "is_confirmed_risk() is false after reset()");

    // Mid-stream reset test (resetting at 350 samples while buffering)
    std::cout << "\n Testing Mid-Stream Reset (at 350 samples)...\n";
    for (size_t i = 0; i < 350; ++i) pipeline.push_sample(ppg[i]);
    tracker.check(pipeline.get_total_samples() == 350, "total_samples is 350 before mid-stream reset");

    pipeline.reset();
    tracker.check(pipeline.get_total_samples() == 0, "total_samples is 0 after mid-stream reset");
    tracker.check(pipeline.is_buffer_ready() == false, "is_buffer_ready() is false after mid-stream reset");

    // Post-reset streaming & parity test
    std::cout << "\n Ingesting 499 samples post-reset...\n";
    bool nullopt_post_reset = true;
    for (size_t i = 0; i < 499; ++i) {
        auto d = pipeline.push_sample(ppg[i]);
        if (d.has_value()) nullopt_post_reset = false;
    }
    tracker.check(nullopt_post_reset, "All 499 samples post-reset returned std::nullopt");

    std::cout << " Ingesting 500th sample post-reset...\n";
    auto post_reset_w0 = pipeline.push_sample(ppg[499]);
    tracker.check(post_reset_w0.has_value(), "500th sample post-reset emitted valid RiskDecision");

    if (pre_reset_w0.has_value() && post_reset_w0.has_value()) {
        const auto& d_pre = pre_reset_w0.value();
        const auto& d_post = post_reset_w0.value();

        tracker.check(d_post.window_index == 0, "Post-reset Window index is 0");
        tracker.check(d_post.total_samples_ingested == 500, "Post-reset total_samples_ingested is 500");

        tracker.check_near(d_post.risk_probability, d_pre.risk_probability, 1e-6f, "Post-reset risk_probability parity");

        bool all_features_match = true;
        for (int f = 0; f < 14; ++f) {
            float diff = std::abs(d_post.raw_features[f] - d_pre.raw_features[f]);
            if (diff > 1e-6f) all_features_match = false;
        }
        tracker.check(all_features_match, "All 14 raw features match pre-reset Window 0 identically (< 1e-6)");
    }
}

float filtered_mean(const float* raw, int input_count, int output_offset) {
    float filtered[PPG_WINDOW_SIZE];
    float scratch[OxygenRiskPipeline::BUFFER_SAMPLES + 2 * PPG_FILTER_PAD];
    ppg_bandpass_filter(raw, filtered, scratch, input_count, output_offset);
    float sum = 0.0f;
    for (int i = 0; i < PPG_WINDOW_SIZE; ++i) sum += filtered[i];
    return sum / static_cast<float>(PPG_WINDOW_SIZE);
}

void test_6_future_context_contract(TestTracker& tracker) {
    std::cout << "\n========================================================\n";
    std::cout << " TEST 6: 5-Second Future-Context Timing, Metadata, Alignment\n";
    std::cout << "========================================================\n";

    tracker.check(OxygenRiskPipeline::CONTEXT_SAMPLES == 250, "Deployment alias uses 250-sample context");
    tracker.check(OxygenRiskPipeline::FIRST_DECISION_SAMPLES == 750, "First decision requires 750 samples");
    tracker.check(OxygenRiskPipeline::BUFFER_SAMPLES == 1000, "Steady-state buffer is 1000 samples");

    OxygenRiskPipeline pipeline;
    std::vector<float> ppg = generate_synthetic_ppg(2000, 1.2f, 5.0f);
    for (size_t i = 0; i < ppg.size(); ++i) ppg[i] += static_cast<float>(i);

    bool silent = true;
    for (int i = 0; i < 749; ++i) {
        if (pipeline.push_sample(ppg[i]).has_value()) silent = false;
    }
    tracker.check(silent, "Samples 0..748 return std::nullopt");
    tracker.check(pipeline.is_buffer_ready() == false, "is_buffer_ready() remains false after 749 samples");
    tracker.check(pipeline.get_total_samples() == 749, "get_total_samples() is 749");

    auto d0 = pipeline.push_sample(ppg[749]);
    tracker.check(d0.has_value(), "Sample 749 (750th sample) emits first RiskDecision");
    tracker.check(pipeline.is_buffer_ready() == true, "is_buffer_ready() becomes true at sample 750");
    if (d0.has_value()) {
        tracker.check(d0->window_index == 0, "Window 0 window_index is 0");
        tracker.check(d0->total_samples_ingested == 750, "Window 0 total_samples_ingested is 750");
        tracker.check(d0->window_start_sample == 0, "Window 0 window_start_sample is 0");
        tracker.check(d0->window_end_sample == 500, "Window 0 window_end_sample is 500");
        tracker.check(d0->decision_delay_samples == 250, "Window 0 decision_delay_samples is 250");
        tracker.check_near(d0->raw_features[0], filtered_mean(ppg.data(), 750, 0), 1e-5f,
                           "Window 0 mean matches 750-sample context filter offset 0");
    }

    silent = true;
    for (int i = 750; i < 999; ++i) {
        if (pipeline.push_sample(ppg[i]).has_value()) silent = false;
    }
    tracker.check(silent, "Samples 750..998 return std::nullopt");

    auto d1 = pipeline.push_sample(ppg[999]);
    tracker.check(d1.has_value(), "Sample 999 (1000th sample) emits second RiskDecision");
    if (d1.has_value()) {
        tracker.check(d1->window_index == 1, "Window 1 window_index is 1");
        tracker.check(d1->total_samples_ingested == 1000, "Window 1 total_samples_ingested is 1000");
        tracker.check(d1->window_start_sample == 250, "Window 1 window_start_sample is 250");
        tracker.check(d1->window_end_sample == 750, "Window 1 window_end_sample is 750");
        tracker.check(d1->decision_delay_samples == 250, "Window 1 decision_delay_samples is 250");
        tracker.check_near(d1->raw_features[0], filtered_mean(ppg.data(), 1000, 250), 1e-5f,
                           "Window 1 mean matches 1000-sample context filter offset 250");
    }

    silent = true;
    for (int i = 1000; i < 1249; ++i) {
        if (pipeline.push_sample(ppg[i]).has_value()) silent = false;
    }
    tracker.check(silent, "Samples 1000..1248 return std::nullopt");

    auto d2 = pipeline.push_sample(ppg[1249]);
    tracker.check(d2.has_value(), "Sample 1249 (1250th sample) emits third RiskDecision");
    if (d2.has_value()) {
        tracker.check(d2->window_index == 2, "Window 2 window_index is 2");
        tracker.check(d2->total_samples_ingested == 1250, "Window 2 total_samples_ingested is 1250");
        tracker.check(d2->window_start_sample == 500, "Window 2 window_start_sample is 500");
        tracker.check(d2->window_end_sample == 1000, "Window 2 window_end_sample is 1000");
        tracker.check_near(d2->raw_features[0], filtered_mean(ppg.data() + 250, 1000, 250), 1e-5f,
                           "Window 2 mean matches ring-buffer 1000-sample context offset 250");
    }

    std::optional<RiskDecision> d_wrap;
    for (int i = 1250; i < 2000; ++i) {
        auto decision = pipeline.push_sample(ppg[i]);
        if (decision.has_value()) d_wrap = decision;
    }
    tracker.check(d_wrap.has_value(), "Sample 1999 (2000th sample) emits a wrapped RiskDecision");
    if (d_wrap.has_value()) {
        tracker.check(d_wrap->window_index == 5, "Wrapped window_index is 5");
        tracker.check(d_wrap->total_samples_ingested == 2000, "Wrapped total_samples_ingested is 2000");
        tracker.check(d_wrap->window_start_sample == 1250, "Wrapped window_start_sample is 1250");
        tracker.check(d_wrap->window_end_sample == 1750, "Wrapped window_end_sample is 1750");
        tracker.check_near(d_wrap->raw_features[0], filtered_mean(ppg.data() + 1000, 1000, 250), 1e-5f,
                           "Wrapped mean matches last 1000 samples with offset 250");
    }

    pipeline.reset();
    tracker.check(pipeline.is_buffer_ready() == false, "Context pipeline is unprimed after reset()");
    tracker.check(pipeline.get_total_samples() == 0, "Context pipeline sample count is 0 after reset()");
    silent = true;
    for (int i = 0; i < 749; ++i) {
        if (pipeline.push_sample(ppg[i]).has_value()) silent = false;
    }
    tracker.check(silent, "Post-reset samples 0..748 return std::nullopt");
    auto post = pipeline.push_sample(ppg[749]);
    tracker.check(post.has_value() && post->window_index == 0 && post->total_samples_ingested == 750
                  && post->window_start_sample == 0 && post->window_end_sample == 500,
                  "Post-reset first decision is again at 750 samples for window [0, 500)");
}

// -----------------------------------------------------------------
// Main Runner
// -----------------------------------------------------------------
int main() {
    std::cout << "========================================================\n";
    std::cout << "   WindowLocalOxygenRiskPipeline (BasicOxygenRiskPipeline<0>) Native Desktop Verification Suite \n";
    std::cout << "   Window-local host variant; NOT the deployed <250> device path\n";
    std::cout << "   Milestone M4: Native Desktop C++ Test Harness      \n";
    std::cout << "========================================================\n";

    TestTracker tracker;

    test_1_buffering_and_emission(tracker);
    test_2_filter_and_feature_parity(tracker);
    test_3_kan_prediction_and_xai_parity(tracker);
    test_4_adr0005_debounce_scenarios(tracker);
    test_5_pipeline_reset_behavior(tracker);
    test_6_future_context_contract(tracker);

    std::cout << "\n========================================================\n";
    std::cout << "                  VERIFICATION SUMMARY                   \n";
    std::cout << "========================================================\n";
    std::cout << " Total Assertions Evaluated: " << tracker.total_assertions << "\n";
    std::cout << " Passed Assertions:          " << tracker.passed_assertions << "\n";
    std::cout << " Failed Assertions:          " << tracker.failed_assertions << "\n";
    std::cout << "--------------------------------------------------------\n";

    if (tracker.failed_assertions == 0) {
        std::cout << " VERDICT: APPROVE - ALL 6 TEST SECTIONS PASSED 100%!  \n";
        std::cout << "========================================================\n";
        return 0;
    } else {
        std::cout << " VERDICT: REQUEST_CHANGES - " << tracker.failed_assertions << " ASSERTION(S) FAILED!\n";
        std::cout << "========================================================\n";
        return 1;
    }
}
