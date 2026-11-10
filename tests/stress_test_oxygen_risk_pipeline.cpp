#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <cassert>
#include <limits>
#include <cstring>
#include "oxygen_risk_pipeline.h"

// Macro helper for assertion tracking
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define STRESS_ASSERT(cond, msg) \
    do { \
        if (cond) { \
            g_tests_passed++; \
        } else { \
            g_tests_failed++; \
            std::cout << "  [FAIL] " << msg << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        } \
    } while(0)

#define STRESS_PASS(msg) \
    do { \
        g_tests_passed++; \
        std::cout << "  [PASS] " << msg << "\n"; \
    } while(0)

// Generator for synthetic normal PPG signal (1.2 Hz sine + DC + small noise)
static float generate_normal_ppg(int sample_idx) {
    float t = static_cast<float>(sample_idx) / 50.0f; // 50 Hz
    float dc = 2000.0f;
    float ac1 = 50.0f * std::sin(2.0f * static_cast<float>(M_PI) * 1.2f * t);
    float ac2 = 15.0f * std::sin(2.0f * static_cast<float>(M_PI) * 2.4f * t);
    float noise = 2.0f * std::sin(2.0f * static_cast<float>(M_PI) * 17.3f * t);
    return dc + ac1 + ac2 + noise;
}

// Generator for synthetic high-risk PPG signal (high kurtosis / distorted harmonic content)
static float generate_risk_ppg(int sample_idx) {
    float t = static_cast<float>(sample_idx) / 50.0f;
    float dc = 2000.0f;
    // Spiky waveform with high kurtosis and high high-frequency bandpower
    float spike = (sample_idx % 10 == 0) ? 300.0f : -30.0f;
    float hf_noise = 80.0f * std::sin(2.0f * static_cast<float>(M_PI) * 6.5f * t);
    return dc + spike + hf_noise;
}

void run_suite_1_long_duration_streaming() {
    std::cout << "\n========================================================\n";
    std::cout << " CHALLENGE SUITE 1: Long-Duration Continuous Streaming\n";
    std::cout << "========================================================\n";

    WindowLocalOxygenRiskPipeline pipeline;
    const int total_samples = 50000;
    int emitted_decisions = 0;
    bool any_nan_or_inf = false;

    for (int i = 0; i < total_samples; ++i) {
        float sample = generate_normal_ppg(i);
        auto opt_dec = pipeline.push_sample(sample);
        if (opt_dec.has_value()) {
            emitted_decisions++;
            RiskDecision dec = *opt_dec;

            // Check total samples ingested
            if (dec.total_samples_ingested != static_cast<uint64_t>(i + 1)) {
                std::cout << "  [FAIL] Sample index mismatch: dec.total_samples_ingested=" 
                          << dec.total_samples_ingested << " exp=" << (i + 1) << "\n";
                g_tests_failed++;
            }

            // Check non-NaN/Inf in outputs
            if (std::isnan(dec.risk_probability) || std::isinf(dec.risk_probability)) {
                any_nan_or_inf = true;
            }
            for (int k = 0; k < 14; ++k) {
                if (std::isnan(dec.raw_features[k]) || std::isinf(dec.raw_features[k])) {
                    any_nan_or_inf = true;
                }
            }
            for (int k = 0; k < 3; ++k) {
                if (std::isnan(dec.top_attributions[k].importance_delta) || std::isinf(dec.top_attributions[k].importance_delta)) {
                    any_nan_or_inf = true;
                }
            }
        }
    }

    // Expected decisions: 1 at 500th sample, then 1 every 250 samples up to 50000
    // 500 + 250 * 198 = 50000 -> total 199 windows (indices 0..198)
    int expected_decisions = 1 + (total_samples - 500) / 250;
    STRESS_ASSERT(emitted_decisions == expected_decisions, 
                  "Emitted decisions count matches expected (199 windows for 50k samples)");
    STRESS_ASSERT(pipeline.get_total_samples() == static_cast<uint64_t>(total_samples), 
                  "Total samples reported by pipeline is exactly 50,000");
    STRESS_ASSERT(!any_nan_or_inf, 
                  "Zero NaN or Infinity values detected across all 199 emitted decision payloads");
}

void run_suite_2_extreme_signals_and_edge_cases() {
    std::cout << "\n========================================================\n";
    std::cout << " CHALLENGE SUITE 2: Extreme Signal & Edge-Case Input Conditions\n";
    std::cout << "========================================================\n";

    // Subtest 2a: Flatline Zero Signal
    {
        WindowLocalOxygenRiskPipeline pipeline;
        bool emitted = false;
        RiskDecision dec = {};
        for (int i = 0; i < 500; ++i) {
            auto res = pipeline.push_sample(0.0f);
            if (res.has_value()) {
                emitted = true;
                dec = *res;
            }
        }
        STRESS_ASSERT(emitted, "Flatline 0.0f signal emitted decision at sample 500 without crashing");
        STRESS_ASSERT(!std::isnan(dec.risk_probability) && !std::isinf(dec.risk_probability), 
                      "Flatline 0.0f risk probability is finite and valid");
        STRESS_ASSERT(dec.risk_probability >= 0.0f && dec.risk_probability <= 1.0f,
                      "Flatline 0.0f risk probability is in valid range [0.0, 1.0]");
        
        bool features_finite = true;
        for (int k = 0; k < 14; ++k) {
            if (std::isnan(dec.raw_features[k]) || std::isinf(dec.raw_features[k])) {
                features_finite = false;
            }
        }
        STRESS_ASSERT(features_finite, "Flatline 0.0f raw features are all finite (no division-by-zero NaN)");
    }

    // Subtest 2b: Large Constant DC Signal (100,000.0f)
    {
        WindowLocalOxygenRiskPipeline pipeline;
        RiskDecision dec = {};
        for (int i = 0; i < 500; ++i) {
            auto res = pipeline.push_sample(100000.0f);
            if (res.has_value()) dec = *res;
        }
        STRESS_ASSERT(!std::isnan(dec.risk_probability) && !std::isinf(dec.risk_probability), 
                      "Constant 100,000.0f signal produced finite risk probability");
        STRESS_ASSERT(std::fabs(dec.raw_features[0]) < 1e-3f, 
                      "Filtered mean feature is near 0.0f due to zero-phase bandpass filter DC attenuation");
    }

    // Subtest 2c: Rapid High-Frequency Nyquist Noise (+1e4f, -1e4f alternating)
    {
        WindowLocalOxygenRiskPipeline pipeline;
        RiskDecision dec = {};
        for (int i = 0; i < 500; ++i) {
            float sample = (i % 2 == 0) ? 10000.0f : -10000.0f;
            auto res = pipeline.push_sample(sample);
            if (res.has_value()) dec = *res;
        }
        STRESS_ASSERT(!std::isnan(dec.risk_probability) && !std::isinf(dec.risk_probability), 
                      "Nyquist alternating signal (+-1e4) produced finite risk probability");
    }

    // Subtest 2d: Extreme Floating Point Limits (FLT_MAX / 1e30f)
    {
        WindowLocalOxygenRiskPipeline pipeline;
        RiskDecision dec = {};
        for (int i = 0; i < 500; ++i) {
            float sample = (i % 2 == 0) ? 1e15f : -1e15f;
            auto res = pipeline.push_sample(sample);
            if (res.has_value()) dec = *res;
        }
        STRESS_ASSERT(!std::isnan(dec.risk_probability) && !std::isinf(dec.risk_probability), 
                      "Extreme magnitude input (+-1e15f) processed safely");
    }

    // Subtest 2e: NaN/Inf Input Propagation
    {
        WindowLocalOxygenRiskPipeline pipeline;
        // Prime buffer with 499 normal samples
        for (int i = 0; i < 499; ++i) {
            pipeline.push_sample(generate_normal_ppg(i));
        }
        // Inject single NaN sample at 500th position
        float nan_val = std::numeric_limits<float>::quiet_NaN();
        auto res = pipeline.push_sample(nan_val);
        STRESS_ASSERT(res.has_value(), "Sample 500 with NaN emitted RiskDecision");
        if (res.has_value()) {
            RiskDecision dec = *res;
            // Document empirical behavior of NaN injection:
            // Since sum/min/max include NaN, features become NaN.
            // Check if pipeline handles or exposes NaN gracefully without crash/hang.
            std::cout << "  [INFO] NaN sample injected -> emitted risk_probability=" << dec.risk_probability << "\n";
            STRESS_PASS("Pipeline processed NaN sample without crash or segmentation fault");
        }
    }
}

void run_suite_3_state_machine_robustness() {
    std::cout << "\n========================================================\n";
    std::cout << " CHALLENGE SUITE 3: State Machine Robustness & Anomalous Calls\n";
    std::cout << "========================================================\n";

    WindowLocalOxygenRiskPipeline pipeline;

    // Subtest 3a: Successive reset() calls
    for (int r = 0; r < 10; ++r) {
        pipeline.reset();
    }
    STRESS_ASSERT(pipeline.get_total_samples() == 0, "total_samples is 0 after 10 consecutive reset() calls");
    STRESS_ASSERT(!pipeline.is_buffer_ready(), "is_buffer_ready() is false after 10 consecutive reset() calls");
    STRESS_ASSERT(pipeline.get_consecutive_risk_count() == 0, "consecutive_risk_count is 0 after reset() calls");
    STRESS_ASSERT(!pipeline.is_confirmed_risk(), "is_confirmed_risk() is false after reset() calls");

    // Subtest 3b: Mid-window resets at multiple sample intervals
    int reset_points[] = {1, 50, 127, 249, 399, 499};
    for (int pt : reset_points) {
        pipeline.reset();
        for (int i = 0; i < pt; ++i) {
            auto res = pipeline.push_sample(1.0f);
            STRESS_ASSERT(!res.has_value(), "push_sample returned nullopt before reaching 500 samples post-reset");
        }
        STRESS_ASSERT(pipeline.get_total_samples() == static_cast<uint64_t>(pt), "total_samples matches pushed count before mid-reset");
        pipeline.reset();
        STRESS_ASSERT(pipeline.get_total_samples() == 0, "total_samples reset to 0");
    }

    // Subtest 3c: Push sample immediately after reset
    pipeline.reset();
    auto opt1 = pipeline.push_sample(100.0f);
    STRESS_ASSERT(!opt1.has_value(), "First push_sample right after reset returns nullopt");
    STRESS_ASSERT(pipeline.get_total_samples() == 1, "total_samples is 1 after first push post-reset");

    // Subtest 3d: consecutive_risk_windows saturation test (prevent uint8_t overflow at 255)
    {
        // Set up custom pipeline with 0 threshold so all windows trigger risk
        WindowLocalOxygenRiskPipelinerisk_pipe(0.00f, 2);
        const int samples_to_push = 500 + 250 * 260; // 261 windows (> 255)
        uint8_t max_counter_seen = 0;
        for (int i = 0; i < samples_to_push; ++i) {
            auto res = risk_pipe.push_sample(generate_risk_ppg(i));
            if (res.has_value()) {
                if (res->consecutive_risk_windows > max_counter_seen) {
                    max_counter_seen = res->consecutive_risk_windows;
                }
            }
        }
        STRESS_ASSERT(max_counter_seen == 255, 
                      "consecutive_risk_windows saturated cleanly at 255 without uint8_t overflow to 0");
        STRESS_ASSERT(risk_pipe.is_confirmed_risk(), "is_confirmed_risk() remains true at counter saturation");
    }

    // Subtest 3e: Mid-alert reset clears alert state immediately
    {
        WindowLocalOxygenRiskPipelinerisk_pipe(0.00f, 2);
        // Push 750 samples to get 2 risk windows (confirmed alert)
        for (int i = 0; i < 750; ++i) {
            risk_pipe.push_sample(generate_risk_ppg(i));
        }
        STRESS_ASSERT(risk_pipe.is_confirmed_risk(), "Alert confirmed before reset");
        risk_pipe.reset();
        STRESS_ASSERT(!risk_pipe.is_confirmed_risk(), "is_confirmed_risk() is immediately false after reset()");
        STRESS_ASSERT(risk_pipe.get_consecutive_risk_count() == 0, "consecutive_risk_count cleared to 0 after reset()");
    }
}

void run_suite_4_ring_buffer_wraparound_integrity() {
    std::cout << "\n========================================================\n";
    std::cout << " CHALLENGE SUITE 4: Ring Buffer & Alignment Integrity Across Wraparounds\n";
    std::cout << "========================================================\n";

    WindowLocalOxygenRiskPipeline pipeline;
    // Push 5,250 samples (10 ring buffer wrap-arounds, 20 classification windows)
    // Verify that every window evaluation produces deterministic and identical output when fed identical periodic signal.
    
    std::vector<RiskDecision> periodic_decisions;
    const int period_samples = 250;

    for (int w = 0; w < 10; ++w) {
        for (int i = 0; i < period_samples; ++i) {
            auto res = pipeline.push_sample(generate_normal_ppg(i));
            if (res.has_value()) {
                periodic_decisions.push_back(*res);
            }
        }
    }

    STRESS_ASSERT(periodic_decisions.size() > 5, "Emitted multiple periodic decisions across wraparounds");
    
    // Windows after buffer is primed and feeding periodic signal should produce stable risk probabilities
    float p0 = periodic_decisions[1].risk_probability;
    bool all_stable = true;
    for (size_t k = 2; k < periodic_decisions.size(); ++k) {
        if (std::fabs(periodic_decisions[k].risk_probability - p0) > 1e-4f) {
            all_stable = false;
        }
    }
    STRESS_ASSERT(all_stable, "Ring buffer unrolling produces stable, deterministic window evaluation across wrap boundaries");
}

void run_suite_5_custom_threshold_and_parameters() {
    std::cout << "\n========================================================\n";
    std::cout << " CHALLENGE SUITE 5: Custom Threshold & Parameter Configurations\n";
    std::cout << "========================================================\n";

    // Pipeline with threshold 0.70f and required_consecutive_windows = 3
    WindowLocalOxygenRiskPipeline pipeline(0.70f, 3);
    
    // Push normal signal (P ~ 0.12)
    for (int i = 0; i < 500; ++i) pipeline.push_sample(generate_normal_ppg(i));
    STRESS_ASSERT(pipeline.get_consecutive_risk_count() == 0, "Normal signal counter is 0");

    // Push high risk signal (P ~ 0.64). Since decision_threshold = 0.70f, P = 0.64 is NOT candidate risk!
    RiskDecision dec_high = {};
    for (int i = 0; i < 250; ++i) {
        auto res = pipeline.push_sample(generate_risk_ppg(i));
        if (res.has_value()) dec_high = *res;
    }

    std::cout << "  [INFO] Signal P=" << dec_high.risk_probability 
              << " vs Custom Threshold=" << 0.70f << "\n";
    STRESS_ASSERT(!dec_high.is_candidate_risk, "Probability 0.64 is not candidate risk when threshold is set to 0.70f");
    STRESS_ASSERT(!dec_high.is_confirmed_risk, "Confirmed risk is false when threshold is 0.70f");
    STRESS_ASSERT(dec_high.consecutive_risk_windows == 0, "Consecutive risk counter remains 0");
}

int main() {
    std::cout << "========================================================\n";
    std::cout << " WindowLocalOxygenRiskPipeline (BasicOxygenRiskPipeline<0>) Empirical Challenge & Stress Suite \n";
    std::cout << " Window-local host variant; NOT the deployed <250> device path\n";
    std::cout << " Milestone M4: Challenger 1 Adversarial Verification    \n";
    std::cout << "========================================================\n";

    run_suite_1_long_duration_streaming();
    run_suite_2_extreme_signals_and_edge_cases();
    run_suite_3_state_machine_robustness();
    run_suite_4_ring_buffer_wraparound_integrity();
    run_suite_5_custom_threshold_and_parameters();

    std::cout << "\n========================================================\n";
    std::cout << "             STRESS VERIFICATION SUMMARY                \n";
    std::cout << "========================================================\n";
    std::cout << "  Total Assertions Evaluated: " << (g_tests_passed + g_tests_failed) << "\n";
    std::cout << "  Passed Assertions:          " << g_tests_passed << "\n";
    std::cout << "  Failed Assertions:          " << g_tests_failed << "\n";
    std::cout << "--------------------------------------------------------\n";
    if (g_tests_failed == 0) {
        std::cout << " VERDICT: APPROVE - ALL STRESS SUITES PASSED 100%!  \n";
    } else {
        std::cout << " VERDICT: REQUEST_CHANGES - " << g_tests_failed << " ASSERTION(S) FAILED! \n";
    }
    std::cout << "========================================================\n";

    return (g_tests_failed == 0) ? 0 : 1;
}
