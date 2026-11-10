#pragma once

#ifndef ARDUINO
#include <cmath>
#include <cstring>
#include <cstdint>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif
#else
#include <Arduino.h>
#endif

#if __cplusplus >= 201703L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201703L)
#include <optional>
#else
#include <experimental/optional>
namespace std {
    template<typename T>
    class optional : public experimental::optional<T> {
    public:
        using experimental::optional<T>::optional;
        optional() : experimental::optional<T>() {}
        optional(const experimental::optional<T>& other) : experimental::optional<T>(other) {}
        optional(experimental::optional<T>&& other) : experimental::optional<T>(std::move(other)) {}
        optional(const T& val) : experimental::optional<T>(val) {}
        optional(T&& val) : experimental::optional<T>(std::move(val)) {}
        optional& operator=(const experimental::optional<T>& other) {
            experimental::optional<T>::operator=(other);
            return *this;
        }
        optional& operator=(experimental::optional<T>&& other) {
            experimental::optional<T>::operator=(std::move(other));
            return *this;
        }
        bool has_value() const noexcept {
            return static_cast<bool>(*this);
        }
    };
    static const auto nullopt = experimental::nullopt;
}
#endif

#include "feature_extraction.h"
#include "kan_inference.h"
#if defined(OXY_MODEL_P5)
#include "kan_model_weights_p5.h"
#include "kan_sat_inference_p5.h"
#include "physio_features.h"
#elif defined(OXY_MODEL_V2)
#include "kan_model_weights_v2.h"
#else
#include "kan_model_weights.h"
#endif

// OXY_MODEL_V2 (ADR-0006) selects the 16-input model: the 14 IR features, then
// ratio_of_ratios and ir_perfusion_index from paired red/IR samples. Without it
// the build is the v1 14-input IR-only pipeline.
// OXY_MODEL_P5 (ADR-0008) selects the 5-input P5 model on the same paired context
// filter: beat features from the filtered IR/red window and the raw-window DC
// means (include/physio_features.h). The 14-feature/FFT path is not compiled.
#if defined(OXY_MODEL_P5)
static_assert(KAN_INPUT_DIM == physio::PHYSIO_NUM_FEATURES, "P5 model takes the five beat features");
#elif defined(OXY_MODEL_V2)
static_assert(KAN_INPUT_DIM == NUM_FEATURES + 2, "v2 model takes 14 IR features plus two red/IR inputs");
#else
static_assert(KAN_INPUT_DIM == NUM_FEATURES, "v1 model takes the 14 IR features");
#endif

// Individual feature attribution score from Mean-Perturbation XAI
struct FeatureAttribution {
    uint8_t feature_index;    // Feature index (0..KAN_INPUT_DIM-1)
    float importance_delta;   // baseline_prob - perturbed_prob
    const char* feature_name; // Feature string identifier
};

// Window evaluation payload returned by push_sample
struct RiskDecision {
    uint32_t window_index;                 // 0-indexed classification window counter
    uint64_t total_samples_ingested;       // Total PPG IR samples pushed since reset
    uint64_t window_start_sample;           // Inclusive target-window start since reset
    uint64_t window_end_sample;             // Exclusive target-window end since reset
    uint16_t decision_delay_samples;        // Future context beyond target window
    float risk_probability;                // Raw TinyKAN FP32 output risk probability [0.0, 1.0]
    bool is_candidate_risk;                // True if risk_probability >= decision_threshold
    bool is_confirmed_risk;                // True iff N=2 consecutive windows produce P >= 0.50f (ADR-0005)
    uint8_t consecutive_risk_windows;      // Counter of consecutive high-risk windows
    FeatureAttribution top_attributions[3];// Top-3 XAI feature drivers sorted by |importance_delta| descending
    float raw_features[KAN_INPUT_DIM];     // Extracted feature vector (v1: 14; v2: 16)
#ifdef OXY_MODEL_V2
    bool two_wavelength_valid;             // False: features 14-15 hold their training means (see compute_two_wavelength_features)
#endif
#ifdef OXY_MODEL_P5
    bool beat_valid;                       // False: the five inputs hold their imputation means (z = 0); see compute_beat_features
    uint8_t n_beats;                       // IR peaks found by the beat detector (0 when the window is unusable)
    int8_t polarity;                       // +1 / -1 chosen from the data; 0 when the window is unusable
    float s_hat;                           // estimated ScalcO2 (model saturation estimate); display only, shown only when beat_valid
#endif
};

template<uint16_t ContextSamples = 250>
class BasicOxygenRiskPipeline {
    static_assert(ContextSamples == 0 || ContextSamples == 250, "Supported context is 0 or 250 samples");
public:
    static constexpr uint16_t CONTEXT_SAMPLES = ContextSamples;
    static constexpr uint16_t FIRST_DECISION_SAMPLES = PPG_WINDOW_SIZE + ContextSamples;
    static constexpr uint16_t BUFFER_SAMPLES = PPG_WINDOW_SIZE + 2 * ContextSamples;
private:
#ifndef ARDUINO
    // Desktop audit access only; no additional board code or state.
    friend struct PipelineAuditAccess;
#endif
    static constexpr uint16_t PPG_STEP_SIZE = 250;

    // Signal & Feature Buffers (Pre-allocated for Zero Dynamic Allocation)
    float raw_ring_buffer[BUFFER_SAMPLES];  // 500 raw IR samples ring buffer
    float filtered_buffer[PPG_WINDOW_SIZE];  // 500 4th-order Butterworth DF2T filtered samples
    float aligned_raw[BUFFER_SAMPLES];      // Chronologically ordered window samples
    float filter_temp_buf[BUFFER_SAMPLES + 2 * PPG_FILTER_PAD];  // Temp buffer for zero-phase forward-backward filter
    float raw_features[KAN_INPUT_DIM];       // Model inputs: 14 statistical/spectral IR features (+2 red/IR in v2)
#ifndef OXY_MODEL_P5
    float hann_window[FFT_SIZE];             // Precomputed 256-point Hann window for Welch FFT
#endif
#if defined(OXY_MODEL_V2) || defined(OXY_MODEL_P5)
    float red_ring_buffer[BUFFER_SAMPLES];       // Red samples, written at the same index as their IR record
    float red_filtered_buffer[PPG_WINDOW_SIZE];  // Red target window after the same zero-phase filter as IR
    float ir_raw_mean;                           // Mean of the raw IR target window
    float red_raw_mean;                          // Mean of the raw red target window
#endif
#ifdef OXY_MODEL_V2
    bool two_wavelength_valid;
#endif
#ifdef OXY_MODEL_P5
    physio::PhysioWorkspace physio_workspace;    // beat-detector buffers (~6.7 KB, fixed size)
    physio::PhysioResult physio_result;          // beat features of the last scored window
#endif

    // Pipeline State Variables
    int write_idx;                           // Ring buffer circular write index (0..499)
    uint64_t total_samples_ingested;         // Total raw PPG samples pushed
    uint32_t samples_since_last_window;      // Samples pushed since last classification window
    uint32_t window_index;                   // Completed classification window index
    uint8_t consecutive_risk_windows;        // ADR-0005 consecutive high-risk window counter
    bool is_buffer_primed;                   // Set to true after first 500 samples ingested

    // ADR-0005 Parameters
    float decision_threshold;                // Probability cutoff for candidate risk (default 0.50f)
    uint8_t required_consecutive_windows;    // Windows required for confirmed risk alert (default 2)

    // Internal Helper Methods
#ifndef OXY_MODEL_P5
    void initialize_hann_window() {
        for (int i = 0; i < FFT_SIZE; ++i) {
            hann_window[i] = 0.5f * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * static_cast<float>(i) / static_cast<float>(FFT_SIZE)));
        }
    }
#endif

    void prepare_filtered_window() {
        const int count = total_samples_ingested < BUFFER_SAMPLES
                        ? static_cast<int>(total_samples_ingested) : BUFFER_SAMPLES;
        int read_idx = total_samples_ingested < BUFFER_SAMPLES ? 0 : write_idx;
        for (int i = 0; i < count; ++i) {
            aligned_raw[i] = raw_ring_buffer[read_idx];
            read_idx = (read_idx + 1) % BUFFER_SAMPLES;
        }
        const int target_offset = count - FIRST_DECISION_SAMPLES;
        ppg_bandpass_filter(aligned_raw, filtered_buffer, filter_temp_buf, count, target_offset);
#if defined(OXY_MODEL_V2) || defined(OXY_MODEL_P5)
        // Red takes the same context buffer and filter call as IR. aligned_raw is
        // reused, so the IR raw mean is taken before it is overwritten.
        ir_raw_mean = window_mean_two_pass(aligned_raw + target_offset);
        read_idx = total_samples_ingested < BUFFER_SAMPLES ? 0 : write_idx;
        for (int i = 0; i < count; ++i) {
            aligned_raw[i] = red_ring_buffer[read_idx];
            read_idx = (read_idx + 1) % BUFFER_SAMPLES;
        }
        ppg_bandpass_filter(aligned_raw, red_filtered_buffer, filter_temp_buf, count, target_offset);
        red_raw_mean = window_mean_two_pass(aligned_raw + target_offset);
#endif
    }

#if defined(OXY_MODEL_V2) || defined(OXY_MODEL_P5)
    // np.mean of a raw 500-sample target window. Summed relative to the centre
    // sample so large DC levels do not swamp the FP32 sum.
    static float window_mean(const float* raw_500) {
        const float ref = raw_500[PPG_WINDOW_SIZE / 2];
        float sum = 0.0f;
        for (int i = 0; i < PPG_WINDOW_SIZE; ++i) sum += raw_500[i] - ref;
        return ref + sum / static_cast<float>(PPG_WINDOW_SIZE);
    }

    // np.mean of a raw 500-sample target window, two passes in FP32: a plain
    // mean, then the mean of the residuals about it. Unlike window_mean, no
    // single sample (e.g. a spike at the centre) sets the reference (ADR-0006,
    // correction of 17 Oct 2026). Used for ir_raw_mean and red_raw_mean.
    static float window_mean_two_pass(const float* raw_500) {
        float sum = 0.0f;
        for (int i = 0; i < PPG_WINDOW_SIZE; ++i) sum += raw_500[i];
        const float first = sum / static_cast<float>(PPG_WINDOW_SIZE);
        float residual = 0.0f;
        for (int i = 0; i < PPG_WINDOW_SIZE; ++i) residual += raw_500[i] - first;
        return first + residual / static_cast<float>(PPG_WINDOW_SIZE);
    }
#endif

#ifdef OXY_MODEL_P5
    // The five P5 inputs of the scored window (job F2, ADR-0008): beat features of the
    // filtered IR/red target windows from the same context filter as v2, with the raw-window
    // FP32 DC means that v2's ratio inputs use (include/physio_features.h, a port of
    // scripts/physio_features.py). Beat-invalid: every input takes KAN_P5_IMPUTATION_MEAN,
    // which equals SCALER_MEAN in FP32, so it standardizes to 0 (the training-time imputation).
    void compute_beat_features(float* features_out) {
        physio::extract(filtered_buffer, red_filtered_buffer, ir_raw_mean, red_raw_mean,
                        physio_workspace, physio_result);
        for (int i = 0; i < KAN_INPUT_DIM; ++i) {
            features_out[i] = physio_result.beat_valid ? static_cast<float>(physio_result.features[i])
                                                       : KAN_P5_IMPUTATION_MEAN[i];
        }
    }

    // Estimated ScalcO2 from the same standardized inputs as the risk score (the regressor's
    // scaler is the P5 scaler). Display only: it does not enter the score or the explanation.
    static float evaluate_sat_estimate(const float* features) {
        float scaled[KAN_INPUT_DIM];
        kan_standardize(features, scaled);
        return kan_sat_estimate(scaled);
    }
#endif

#ifdef OXY_MODEL_V2
    // np.std (ddof 0) of a filtered 500-sample window: the same two passes as
    // the std feature in extract_features_from_filtered.
    static float window_std(const float* filtered_500) {
        float sum = 0.0f;
        for (int i = 0; i < PPG_WINDOW_SIZE; ++i) sum += filtered_500[i];
        const float mean_val = sum / static_cast<float>(PPG_WINDOW_SIZE);
        float m2 = 0.0f;
        for (int i = 0; i < PPG_WINDOW_SIZE; ++i) {
            const float diff = filtered_500[i] - mean_val;
            m2 += diff * diff;
        }
        return std::sqrt(m2 / static_cast<float>(PPG_WINDOW_SIZE));
    }

    // two_wavelength_features() in scripts/export_openox_windows_v2.py, on the
    // scored window:
    //   ir_perfusion_index = std(filtered IR) / mean(raw IR)
    //   ratio_of_ratios    = (std(filtered red) / mean(raw red)) / ir_perfusion_index
    // Where the export would give NaN (a mean <= 0, IR perfusion <= 0), both
    // inputs take their training means instead: they standardize to 0, so no
    // NaN reaches the KAN and Mean-Perturbation gives them zero attribution.
    // (Training-mean reference only. Under OXY_XAI_REF_LABEL0 the fill is the
    // same, so their deltas are the move from the training mean to the label-0 median.)
    void compute_two_wavelength_features(float* features_out) {
        float ir_perfusion = 0.0f;
        float ratio = 0.0f;
        two_wavelength_valid = false;
        if (ir_raw_mean > 0.0f && red_raw_mean > 0.0f) {
            ir_perfusion = window_std(filtered_buffer) / ir_raw_mean;
            if (ir_perfusion > 0.0f) {
                ratio = (window_std(red_filtered_buffer) / red_raw_mean) / ir_perfusion;
                two_wavelength_valid = std::isfinite(ir_perfusion) && std::isfinite(ratio);
            }
        }
        features_out[NUM_FEATURES] = two_wavelength_valid ? ratio : SCALER_MEAN[NUM_FEATURES];
        features_out[NUM_FEATURES + 1] = two_wavelength_valid ? ir_perfusion : SCALER_MEAN[NUM_FEATURES + 1];
    }
#endif

    unsigned int reverse_bits(unsigned int x, int bits) const {
        unsigned int y = 0;
        for (int i = 0; i < bits; ++i) {
            y = (y << 1) | (x & 1);
            x >>= 1;
        }
        return y;
    }

    void cooley_tukey_fft(float* re, float* im, int n) const {
        int bits = 0;
        while ((1 << bits) < n) bits++;

        for (int i = 0; i < n; ++i) {
            int rev = reverse_bits(i, bits);
            if (i < rev) {
                float temp = re[i]; re[i] = re[rev]; re[rev] = temp;
                temp = im[i]; im[i] = im[rev]; im[rev] = temp;
            }
        }

        for (int len = 2; len <= n; len <<= 1) {
            float angle = -2.0f * static_cast<float>(M_PI) / static_cast<float>(len);
            float wlen_re = std::cos(angle);
            float wlen_im = std::sin(angle);
            for (int i = 0; i < n; i += len) {
                float w_re = 1.0f;
                float w_im = 0.0f;
                int half = len / 2;
                for (int j = 0; j < half; ++j) {
                    float u_re = re[i + j];
                    float u_im = im[i + j];
                    float v_re = re[i + j + half] * w_re - im[i + j + half] * w_im;
                    float v_im = re[i + j + half] * w_im + im[i + j + half] * w_re;

                    re[i + j] = u_re + v_re;
                    im[i + j] = u_im + v_im;
                    re[i + j + half] = u_re - v_re;
                    im[i + j + half] = u_im - v_im;

                    float next_w_re = w_re * wlen_re - w_im * wlen_im;
                    w_im = w_re * wlen_im + w_im * wlen_re;
                    w_re = next_w_re;
                }
            }
        }
    }

#ifndef OXY_MODEL_P5
    void extract_features_from_filtered(const float* filtered_500, float* features_out) {
        // Statistical Features
        float sum = 0.0f;
        float min_val = filtered_500[0];
        float max_val = filtered_500[0];
        float sum_sq = 0.0f;

        for (int i = 0; i < PPG_WINDOW_SIZE; ++i) {
            float val = filtered_500[i];
            sum += val;
            sum_sq += val * val;
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }

        float mean_val = sum / static_cast<float>(PPG_WINDOW_SIZE);
        float rms_val = std::sqrt(sum_sq / static_cast<float>(PPG_WINDOW_SIZE));
        float energy_val = sum_sq;
        float value_range = max_val - min_val;

        float m2 = 0.0f;
        float m3 = 0.0f;
        float m4 = 0.0f;
#ifdef OXY_MODEL_V2
        // v2 only: compensated (Kahan) sums for m2, m3, m4, so kurtosis tracks
        // the float64 reference (17 Oct 2026 host-parity residual). v1 keeps
        // the plain FP32 adds below, so v2 std, skewness and kurtosis may differ
        // from v1 on the same window in the last bits. That is intended; do not
        // "repair" it by changing the v1 branch. Needs no -ffast-math.
        float c2 = 0.0f;
        float c3 = 0.0f;
        float c4 = 0.0f;
        for (int i = 0; i < PPG_WINDOW_SIZE; ++i) {
            float diff = filtered_500[i] - mean_val;
            float diff2 = diff * diff;
            float y2 = diff2 - c2;
            float t2 = m2 + y2;
            c2 = (t2 - m2) - y2;
            m2 = t2;
            float y3 = diff2 * diff - c3;
            float t3 = m3 + y3;
            c3 = (t3 - m3) - y3;
            m3 = t3;
            float y4 = diff2 * diff2 - c4;
            float t4 = m4 + y4;
            c4 = (t4 - m4) - y4;
            m4 = t4;
        }
#else
        for (int i = 0; i < PPG_WINDOW_SIZE; ++i) {
            float diff = filtered_500[i] - mean_val;
            float diff2 = diff * diff;
            m2 += diff2;
            m3 += diff2 * diff;
            m4 += diff2 * diff2;
        }
#endif
        m2 /= static_cast<float>(PPG_WINDOW_SIZE);
        m3 /= static_cast<float>(PPG_WINDOW_SIZE);
        m4 /= static_cast<float>(PPG_WINDOW_SIZE);

        float std_val = std::sqrt(m2);

        float skewness_val = 0.0f;
        float kurtosis_val = 0.0f;
        if (std_val > 1e-10f) {
            float g1 = m3 / (m2 * std_val);
            skewness_val = std::sqrt(static_cast<float>(PPG_WINDOW_SIZE) * static_cast<float>(PPG_WINDOW_SIZE - 1)) / static_cast<float>(PPG_WINDOW_SIZE - 2) * g1;

            float g2 = m4 / (m2 * m2) - 3.0f;
            float term1 = static_cast<float>(PPG_WINDOW_SIZE - 1) / (static_cast<float>(PPG_WINDOW_SIZE - 2) * static_cast<float>(PPG_WINDOW_SIZE - 3));
            kurtosis_val = term1 * (static_cast<float>(PPG_WINDOW_SIZE + 1) * g2 + 6.0f);
        }

        float diff_sum = 0.0f;
        float diff_sum_sq = 0.0f;
        for (int i = 0; i < PPG_WINDOW_SIZE - 1; ++i) {
            float d = filtered_500[i + 1] - filtered_500[i];
            diff_sum += d;
            diff_sum_sq += d * d;
        }
        float diff_mean = diff_sum / static_cast<float>(PPG_WINDOW_SIZE - 1);
        float derivative_std = std::sqrt((diff_sum_sq / static_cast<float>(PPG_WINDOW_SIZE - 1)) - (diff_mean * diff_mean));

        // Spectral Features via Welch FFT
        float psd[FFT_SIZE / 2 + 1];
        std::memset(psd, 0, sizeof(psd));

        float segment[FFT_SIZE];
        float fft_re[FFT_SIZE];
        float fft_im[FFT_SIZE];

        const float welch_scale = 1.0f / (50.0f * 96.0f);

        for (int seg = 0; seg < 2; ++seg) {
            int start_idx = seg * 128;
            for (int i = 0; i < FFT_SIZE; ++i) {
                int data_idx = start_idx + i;
                float sample = 0.0f;
                if (data_idx < PPG_WINDOW_SIZE) {
                    sample = filtered_500[data_idx];
                }
                segment[i] = sample;
            }

            float seg_sum = 0.0f;
            for (int i = 0; i < FFT_SIZE; ++i) seg_sum += segment[i];
            float seg_mean = seg_sum / static_cast<float>(FFT_SIZE);
            for (int i = 0; i < FFT_SIZE; ++i) segment[i] -= seg_mean;

            for (int i = 0; i < FFT_SIZE; ++i) {
                fft_re[i] = segment[i] * hann_window[i];
                fft_im[i] = 0.0f;
            }

            cooley_tukey_fft(fft_re, fft_im, FFT_SIZE);

            psd[0] += (fft_re[0] * fft_re[0] + fft_im[0] * fft_im[0]) * welch_scale;
            for (int k = 1; k < FFT_SIZE / 2; ++k) {
                float r = fft_re[k];
                float img = fft_im[k];
                psd[k] += 2.0f * (r * r + img * img) * welch_scale;
            }
            float nyq_re = fft_re[FFT_SIZE / 2];
            float nyq_im = fft_im[FFT_SIZE / 2];
            psd[FFT_SIZE / 2] += (nyq_re * nyq_re + nyq_im * nyq_im) * welch_scale;
        }

        for (int k = 0; k < FFT_SIZE / 2 + 1; ++k) {
            psd[k] /= 2.0f;
        }

        float max_power = psd[1];
        int max_power_bin = 1;
        for (int k = 2; k < FFT_SIZE / 2 + 1; ++k) {
            if (psd[k] > max_power) {
                max_power = psd[k];
                max_power_bin = k;
            }
        }
        float dominant_frequency_hz = static_cast<float>(max_power_bin) * (50.0f / static_cast<float>(FFT_SIZE));

        const float df = 50.0f / static_cast<float>(FFT_SIZE);

        float bp_0_5_2_sum = 0.5f * psd[3];
        for (int k = 4; k < 10; ++k) {
            bp_0_5_2_sum += psd[k];
        }
        bp_0_5_2_sum += 0.5f * psd[10];
        float bandpower_0_5_2_hz = bp_0_5_2_sum * df;

        float bp_2_5_sum = 0.5f * psd[11];
        for (int k = 12; k < 25; ++k) {
            bp_2_5_sum += psd[k];
        }
        bp_2_5_sum += 0.5f * psd[25];
        float bandpower_2_5_hz = bp_2_5_sum * df;

        float bandpower_ratio = (bandpower_2_5_hz > 0.0f) ? (bandpower_0_5_2_hz / bandpower_2_5_hz) : 0.0f;

        features_out[0] = mean_val;
        features_out[1] = std_val;
        features_out[2] = min_val;
        features_out[3] = max_val;
        features_out[4] = value_range;
        features_out[5] = rms_val;
        features_out[6] = skewness_val;
        features_out[7] = kurtosis_val;
        features_out[8] = energy_val;
        features_out[9] = derivative_std;
        features_out[10] = dominant_frequency_hz;
        features_out[11] = bandpower_0_5_2_hz;
        features_out[12] = bandpower_2_5_hz;
        features_out[13] = bandpower_ratio;
    }
#endif

    // One baseline pass plus one Mean-Perturbation pass per model input.
    float evaluate_kan_and_xai(const float* features, FeatureAttribution top_3_out[3]) {
        float scaled[KAN_INPUT_DIM];
        kan_standardize(features, scaled);

        float baseline_logit = kan_predict_logit(scaled);
        float baseline_prob = kan_sigmoid(baseline_logit);

        FeatureAttribution all_attrs[KAN_INPUT_DIM];
        float perturbed_scaled[KAN_INPUT_DIM];

        for (int i = 0; i < KAN_INPUT_DIM; ++i) {
            std::memcpy(perturbed_scaled, scaled, KAN_INPUT_DIM * sizeof(float));
            perturbed_scaled[i] = KAN_XAI_REFERENCE(i);

            float perturbed_logit = kan_predict_logit(perturbed_scaled);
            float perturbed_prob = kan_sigmoid(perturbed_logit);

            all_attrs[i].feature_index = static_cast<uint8_t>(i);
            all_attrs[i].importance_delta = baseline_prob - perturbed_prob;
            all_attrs[i].feature_name = KAN_FEATURE_NAMES[i];
        }

        // Sort descending by |importance_delta|
        for (int i = 0; i < KAN_INPUT_DIM - 1; ++i) {
            for (int j = 0; j < KAN_INPUT_DIM - 1 - i; ++j) {
                if (std::fabs(all_attrs[j].importance_delta) < std::fabs(all_attrs[j + 1].importance_delta)) {
                    FeatureAttribution temp = all_attrs[j];
                    all_attrs[j] = all_attrs[j + 1];
                    all_attrs[j + 1] = temp;
                }
            }
        }

        for (int i = 0; i < 3; ++i) {
            top_3_out[i] = all_attrs[i];
        }

        return baseline_prob;
    }

    RiskDecision evaluate_window() {
        prepare_filtered_window();
#ifdef OXY_MODEL_P5
        compute_beat_features(raw_features);
#else
        extract_features_from_filtered(filtered_buffer, raw_features);
#endif
#ifdef OXY_MODEL_V2
        compute_two_wavelength_features(raw_features);
#endif

        FeatureAttribution top_3[3];
        float risk_prob = evaluate_kan_and_xai(raw_features, top_3);

        bool is_candidate = (risk_prob >= decision_threshold);
        if (is_candidate) {
            if (consecutive_risk_windows < 255) {
                consecutive_risk_windows++;
            }
        } else {
            consecutive_risk_windows = 0;
        }

        bool is_confirmed = (consecutive_risk_windows >= required_consecutive_windows);

        RiskDecision decision;
        decision.window_index = window_index++;
        decision.total_samples_ingested = total_samples_ingested;
        decision.window_end_sample = total_samples_ingested - ContextSamples;
        decision.window_start_sample = decision.window_end_sample - PPG_WINDOW_SIZE;
        decision.decision_delay_samples = ContextSamples;
        decision.risk_probability = risk_prob;
        decision.is_candidate_risk = is_candidate;
        decision.is_confirmed_risk = is_confirmed;
        decision.consecutive_risk_windows = consecutive_risk_windows;

        for (int i = 0; i < 3; ++i) {
            decision.top_attributions[i] = top_3[i];
        }
        for (int i = 0; i < KAN_INPUT_DIM; ++i) {
            decision.raw_features[i] = raw_features[i];
        }
#ifdef OXY_MODEL_V2
        decision.two_wavelength_valid = two_wavelength_valid;
#endif
#ifdef OXY_MODEL_P5
        decision.beat_valid = physio_result.beat_valid;
        decision.n_beats = static_cast<uint8_t>(physio_result.n_beats);
        decision.polarity = static_cast<int8_t>(physio_result.polarity);
        decision.s_hat = evaluate_sat_estimate(raw_features);
#endif

        return decision;
    }

public:
    explicit BasicOxygenRiskPipeline(float decision_threshold = 0.50f, uint8_t required_consecutive_windows = 2)
        : write_idx(0),
          total_samples_ingested(0),
          samples_since_last_window(0),
          window_index(0),
          consecutive_risk_windows(0),
          is_buffer_primed(false),
          decision_threshold(decision_threshold),
          required_consecutive_windows(required_consecutive_windows)
    {
#ifndef OXY_MODEL_P5
        initialize_hann_window();
#endif
        reset();
    }

#if defined(OXY_MODEL_V2) || defined(OXY_MODEL_P5)
    // Pass IR and red from the SAME FIFO record; both share one write index.
    std::optional<RiskDecision> push_sample(float ir_val, float red_val) {
        red_ring_buffer[write_idx] = red_val;
#else
    std::optional<RiskDecision> push_sample(float ir_val) {
#endif
        raw_ring_buffer[write_idx] = ir_val;
        write_idx = (write_idx + 1) % BUFFER_SAMPLES;
        total_samples_ingested++;
        samples_since_last_window++;

        if (!is_buffer_primed) {
            if (total_samples_ingested >= FIRST_DECISION_SAMPLES) {
                is_buffer_primed = true;
                samples_since_last_window = 0;
                return evaluate_window();
            } else {
                return std::nullopt;
            }
        } else {
            if (samples_since_last_window >= PPG_STEP_SIZE) {
                samples_since_last_window = 0;
                return evaluate_window();
            } else {
                return std::nullopt;
            }
        }
    }

    void reset() {
        write_idx = 0;
        total_samples_ingested = 0;
        samples_since_last_window = 0;
        window_index = 0;
        consecutive_risk_windows = 0;
        is_buffer_primed = false;
        std::memset(raw_ring_buffer, 0, sizeof(raw_ring_buffer));
        std::memset(filtered_buffer, 0, sizeof(filtered_buffer));
        std::memset(aligned_raw, 0, sizeof(aligned_raw));
        std::memset(filter_temp_buf, 0, sizeof(filter_temp_buf));
        std::memset(raw_features, 0, sizeof(raw_features));
#ifdef OXY_MODEL_V2
        std::memset(red_ring_buffer, 0, sizeof(red_ring_buffer));
        std::memset(red_filtered_buffer, 0, sizeof(red_filtered_buffer));
        ir_raw_mean = 0.0f;
        red_raw_mean = 0.0f;
        two_wavelength_valid = false;
#endif
#ifdef OXY_MODEL_P5
        std::memset(red_ring_buffer, 0, sizeof(red_ring_buffer));
        std::memset(red_filtered_buffer, 0, sizeof(red_filtered_buffer));
        ir_raw_mean = 0.0f;
        red_raw_mean = 0.0f;
        std::memset(&physio_result, 0, sizeof(physio_result));
#endif
    }

    bool is_buffer_ready() const { return is_buffer_primed; }
    uint64_t get_total_samples() const { return total_samples_ingested; }
    uint8_t get_consecutive_risk_count() const { return consecutive_risk_windows; }
    bool is_confirmed_risk() const { return consecutive_risk_windows >= required_consecutive_windows; }
};

// Deployment contract: 5 seconds past/future context; first decision at 15 s.
using OxygenRiskPipeline = BasicOxygenRiskPipeline<250>;

// Host arithmetic / diagnostic variant: filters only the isolated 500-sample
// window (no past/future context). NOT the deployed path; results from it
// (e.g. candidate or alert-flip rates) do not describe device behaviour.
using WindowLocalOxygenRiskPipeline = BasicOxygenRiskPipeline<0>;
