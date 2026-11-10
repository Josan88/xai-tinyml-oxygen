// PPG Feature Extraction on-device
// Matches per-window SciPy features; full-recording filtering context differs.
// Uses a self-contained C++ FFT to avoid linker dependencies

#pragma once

#ifdef ARDUINO
#include <Arduino.h>
#else
#include <cmath>
#include <cstring>
#include <cstdint>
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif
#endif
#include <math.h>

#define PPG_WINDOW_SIZE 500
#define FFT_SIZE 256
#define NUM_FEATURES 14

// Filter coefficients for 4th-order Butterworth bandpass [0.5 Hz, 8.0 Hz] at fs=50 Hz
// Generated from SciPy's signal.butter, printed with 9 significant digits
// to preserve the nearest FP32 coefficients.
const float FILTER_SOS[4][6] = {
    {0.0185630106f, 0.0371260213f, 0.0185630106f, 1.0f, -0.704780708f, 0.163919936f},
    {1.0f, 2.0f, 1.0f, 1.0f, -0.846256804f, 0.550904629f},
    {1.0f, -2.0f, 1.0f, 1.0f, -1.87717618f, 0.881742658f},
    {1.0f, -2.0f, 1.0f, 1.0f, -1.95297743f, 0.956947676f}
};

// SciPy sosfiltfilt default: 3 * (2 * 4 + 1) odd-extension samples.
constexpr int PPG_FILTER_PAD = 27;
constexpr int PPG_FILTER_SIZE = PPG_WINDOW_SIZE + 2 * PPG_FILTER_PAD;

// Caller owns scratch space; all arithmetic remains FP32.
inline void ppg_bandpass_filter(const float* input, float* output, float* scratch,
                                int input_count = PPG_WINDOW_SIZE, int output_offset = 0) {
    const int padded_count = input_count + 2 * PPG_FILTER_PAD;
    // Center on the target window, where accuracy matters. Centering on the
    // first context sample can leave a large DC level during a small target
    // signal. The bandpass rejects this constant; it is not added back.
    const float dc = input[output_offset + PPG_WINDOW_SIZE / 2];
    for (int i = 0; i < PPG_FILTER_PAD; ++i) {
        scratch[i] = 2.0f * (input[0] - dc) - (input[PPG_FILTER_PAD - i] - dc);
        scratch[PPG_FILTER_PAD + input_count + i] =
            2.0f * (input[input_count - 1] - dc) - (input[input_count - 2 - i] - dc);
    }
    for (int i = 0; i < input_count; ++i) scratch[PPG_FILTER_PAD + i] = input[i] - dc;
    static const float zi[4][2] = {
        {0.143157091f, -0.00794613801f},
        {0.756299344f, -0.344021061f},
        {-0.918019446f, 0.918019446f},
        {-0.0f, 0.0f}
    };
    // Equivalent high-pass feedback coordinates, computed from the canonical
    // FP64 SOS before rounding once to FP32: k=1+a1+a2, h=1-a2.
    // Storing these small terms directly avoids subtracting rounded a1/a2
    // near -2/+1. This changes the realization, not the intended bandpass.
    static const float hp_k[2] = {0.00456648134f, 0.00397024211f};
    static const float hp_h[2] = {0.118257344f, 0.0430523232f};
#ifdef __FAST_MATH__
#error "PPG compensated FP32 filtering requires non-reassociated arithmetic"
#endif
    for (int pass = 0; pass < 2; ++pass) {
        const float x0 = scratch[0];
        for (int section = 0; section < 4; ++section) {
            const float* c = FILTER_SOS[section];
            if (section >= 2) {
                // d integrates y, v integrates d. This has transfer function
                // (1-z^-1)^2 / (1 + (k+h-2)z^-1 + (1-h)z^-2).
                // Carry FP32 rounding residuals in the state updates. Every
                // stored value and arithmetic operation remains single precision.
                const float k = hp_k[section - 2], h = hp_h[section - 2];
                float v = scratch[0] / k, d = 0.0f;
                float vc = fmaf(k, v, -scratch[0]) / k, dcorr = 0.0f;
                for (int i = 0; i < padded_count; ++i) {
                    float y = fmaf(-h, d, fmaf(-k, v, scratch[i]));
                    y += fmaf(k, vc, h * dcorr);
                    const float di = y - dcorr;
                    const float dn = d + di;
                    dcorr = (dn - d) - di;
                    d = dn;
                    const float vi = (d - dcorr) - vc;
                    const float vn = v + vi;
                    vc = (vn - v) - vi;
                    v = vn;
                    scratch[i] = y;
                }
                continue;
            }
            float w1 = zi[section][0] * x0;
            float w2 = zi[section][1] * x0;
            for (int i = 0; i < padded_count; ++i) {
                const float x = scratch[i];
                const float y = fmaf(c[0], x, w1);
                w1 = fmaf(-c[4], y, fmaf(c[1], x, w2));
                w2 = fmaf(-c[5], y, c[2] * x);
                scratch[i] = y;
            }
        }
        for (int i = 0; i < padded_count / 2; ++i) {
            const float value = scratch[i];
            scratch[i] = scratch[padded_count - 1 - i];
            scratch[padded_count - 1 - i] = value;
        }
    }
    memcpy(output, scratch + PPG_FILTER_PAD + output_offset, PPG_WINDOW_SIZE * sizeof(float));
}

class PPGFeatureExtractor {
private:
    float raw_buffer[PPG_WINDOW_SIZE];
    float filtered_buffer[PPG_WINDOW_SIZE];
    int write_idx;
    bool buffer_full;

    // Hann Window for 256-point FFT (Welch)
    float hann_window[FFT_SIZE];

    // Helper: Bit reversal for Cooley-Tukey FFT
    unsigned int reverse_bits(unsigned int x, int bits) {
        unsigned int y = 0;
        for (int i = 0; i < bits; ++i) {
            y = (y << 1) | (x & 1);
            x >>= 1;
        }
        return y;
    }

    // Helper: Self-contained Cooley-Tukey Radix-2 FFT
    void cooley_tukey_fft(float* re, float* im, int n) {
        int bits = 0;
        while ((1 << bits) < n) bits++;
        
        // Bit-reversal permutation
        for (int i = 0; i < n; ++i) {
            int rev = reverse_bits(i, bits);
            if (i < rev) {
                float temp = re[i]; re[i] = re[rev]; re[rev] = temp;
                temp = im[i]; im[i] = im[rev]; im[rev] = temp;
            }
        }
        
        // Cooley-Tukey decimation-in-time
        for (int len = 2; len <= n; len <<= 1) {
            float angle = -2.0f * M_PI / len;
            float wlen_re = cosf(angle);
            float wlen_im = sinf(angle);
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
                    
                    // Update twiddle factor
                    float next_w_re = w_re * wlen_re - w_im * wlen_im;
                    w_im = w_re * wlen_im + w_im * wlen_re;
                    w_re = next_w_re;
                }
            }
        }
    }

    // Helper: Apply the 4th-order Butterworth bandpass filter forward-backward in-place
    void apply_bandpass_filter(const float* input, float* output) {
        static float scratch[PPG_FILTER_SIZE];
        ppg_bandpass_filter(input, output, scratch);
    }

public:
    PPGFeatureExtractor() : write_idx(0), buffer_full(false) {
        // Precompute Hann window
        for (int i = 0; i < FFT_SIZE; ++i) {
            hann_window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / FFT_SIZE));
        }

        // Initialize buffers
        memset(raw_buffer, 0, sizeof(raw_buffer));
        memset(filtered_buffer, 0, sizeof(filtered_buffer));
    }

    // Add a new sample to the sliding window
    void add_sample(float val) {
        raw_buffer[write_idx] = val;
        write_idx++;
        if (write_idx >= PPG_WINDOW_SIZE) {
            write_idx = 0;
            buffer_full = true;
        }
    }

    bool is_full() const {
        return buffer_full;
    }

    // Process the window and extract 14 features
    void extract_features(float* features_out) {
        if (!buffer_full) return;

        // 1. Align buffer chronologically (oldest to newest)
        float aligned_raw[PPG_WINDOW_SIZE];
        int idx = write_idx;
        for (int i = 0; i < PPG_WINDOW_SIZE; ++i) {
            aligned_raw[i] = raw_buffer[idx];
            idx = (idx + 1) % PPG_WINDOW_SIZE;
        }

        // 2. Bandpass filter the aligned raw buffer
        apply_bandpass_filter(aligned_raw, filtered_buffer);

        // 3. Compute Basic Statistical Features
        float sum = 0.0f;
        float min_val = filtered_buffer[0];
        float max_val = filtered_buffer[0];
        float sum_sq = 0.0f;

        for (int i = 0; i < PPG_WINDOW_SIZE; ++i) {
            float val = filtered_buffer[i];
            sum += val;
            sum_sq += val * val;
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }

        float mean_val = sum / PPG_WINDOW_SIZE;
        float rms_val = sqrtf(sum_sq / PPG_WINDOW_SIZE);
        float energy_val = sum_sq;
        float value_range = max_val - min_val;

        // Higher Moments (skewness and kurtosis)
        float m2 = 0.0f;
        float m3 = 0.0f;
        float m4 = 0.0f;
        for (int i = 0; i < PPG_WINDOW_SIZE; ++i) {
            float diff = filtered_buffer[i] - mean_val;
            float diff2 = diff * diff;
            m2 += diff2;
            m3 += diff2 * diff;
            m4 += diff2 * diff2;
        }
        m2 /= PPG_WINDOW_SIZE;
        m3 /= PPG_WINDOW_SIZE;
        m4 /= PPG_WINDOW_SIZE;

        float std_val = sqrtf(m2);
        
        float skewness_val = 0.0f;
        float kurtosis_val = 0.0f;
        if (std_val > 1e-10f) {
            // Unbiased skewness (scipy stats.skew bias=False)
            float g1 = m3 / (m2 * std_val);
            skewness_val = sqrtf((float)PPG_WINDOW_SIZE * (PPG_WINDOW_SIZE - 1)) / (PPG_WINDOW_SIZE - 2) * g1;

            // Unbiased kurtosis (scipy stats.kurtosis fisher=True bias=False)
            float g2 = m4 / (m2 * m2) - 3.0f;
            float term1 = (float)(PPG_WINDOW_SIZE - 1) / ((float)(PPG_WINDOW_SIZE - 2) * (PPG_WINDOW_SIZE - 3));
            kurtosis_val = term1 * ((float)(PPG_WINDOW_SIZE + 1) * g2 + 6.0f);
        }

        // Standard deviation of the first derivative
        float diff_sum = 0.0f;
        float diff_sum_sq = 0.0f;
        for (int i = 0; i < PPG_WINDOW_SIZE - 1; ++i) {
            float d = filtered_buffer[i+1] - filtered_buffer[i];
            diff_sum += d;
            diff_sum_sq += d * d;
        }
        float diff_mean = diff_sum / (PPG_WINDOW_SIZE - 1);
        float derivative_std = sqrtf((diff_sum_sq / (PPG_WINDOW_SIZE - 1)) - (diff_mean * diff_mean));

        // 4. Spectral Features via Welch's Method (nperseg = 256, 50% overlap, 2 complete segments)
        // Segment 1: samples 0 to 255
        // Segment 2: samples 128 to 383
        // Incomplete trailing segment is excluded, matching SciPy Welch.
        
        float psd[FFT_SIZE / 2 + 1]; // One-sided PSD (size 129)
        memset(psd, 0, sizeof(psd));

        float segment[FFT_SIZE];
        float fft_re[FFT_SIZE];
        float fft_im[FFT_SIZE];

        // Welch scale factor: 1.0 / (fs * sum(w_n^2))
        // Hann window of size 256 sum of squares is 96.0
        // Scale factor: 1.0f / (50.0f * 96.0f)
        const float welch_scale = 1.0f / (50.0f * 96.0f);

        for (int seg = 0; seg < 2; ++seg) {
            int start_idx = seg * 128;
            
            // Extract segment
            for (int i = 0; i < FFT_SIZE; ++i) {
                int data_idx = start_idx + i;
                float sample = 0.0f;
                if (data_idx < PPG_WINDOW_SIZE) {
                    sample = filtered_buffer[data_idx];
                }
                segment[i] = sample;
            }

            // Constant detrend (subtract segment mean)
            float seg_sum = 0.0f;
            for (int i = 0; i < FFT_SIZE; ++i) seg_sum += segment[i];
            float seg_mean = seg_sum / FFT_SIZE;
            for (int i = 0; i < FFT_SIZE; ++i) segment[i] -= seg_mean;

            // Apply Hann Window and initialize FFT complex inputs
            for (int i = 0; i < FFT_SIZE; ++i) {
                fft_re[i] = segment[i] * hann_window[i];
                fft_im[i] = 0.0f;
            }

            // Run Cooley-Tukey Radix-2 FFT
            cooley_tukey_fft(fft_re, fft_im, FFT_SIZE);

            // Compute Magnitude Squared (Periodogram)
            // Bin 0 (DC)
            psd[0] += (fft_re[0] * fft_re[0] + fft_im[0] * fft_im[0]) * welch_scale;

            // Bins 1 to 127
            for (int k = 1; k < FFT_SIZE / 2; ++k) {
                float r = fft_re[k];
                float img = fft_im[k];
                float mag_sq = r * r + img * img;
                // One-sided PSD multiplies bins 1 to 127 by 2
                psd[k] += 2.0f * mag_sq * welch_scale;
            }

            // Bin 128 (Nyquist)
            float nyq_re = fft_re[FFT_SIZE / 2];
            float nyq_im = fft_im[FFT_SIZE / 2];
            psd[FFT_SIZE / 2] += (nyq_re * nyq_re + nyq_im * nyq_im) * welch_scale;
        }

        // Average PSD over the 2 complete segments
        for (int k = 0; k < FFT_SIZE / 2 + 1; ++k) {
            psd[k] /= 2.0f;
        }

        // Extract Dominant Frequency (excluding DC bin 0)
        float max_power = psd[1];
        int max_power_bin = 1;
        for (int k = 2; k < FFT_SIZE / 2 + 1; ++k) {
            if (psd[k] > max_power) {
                max_power = psd[k];
                max_power_bin = k;
            }
        }
        float dominant_frequency_hz = max_power_bin * (50.0f / FFT_SIZE);

        // Compute Bandpowers via Trapezoidal rule
        // Frequency resolution: df = 50.0f / 256 = 0.1953125f
        const float df = 50.0f / FFT_SIZE;

        float bandpower_0_5_2_hz = 0.0f;
        float bandpower_2_5_hz = 0.0f;

        // Bandpower [0.5, 2.0] Hz: Bins 3 to 10
        float bp_0_5_2_sum = 0.0f;
        bp_0_5_2_sum += 0.5f * psd[3];
        for (int k = 4; k < 10; ++k) {
            bp_0_5_2_sum += psd[k];
        }
        bp_0_5_2_sum += 0.5f * psd[10];
        bandpower_0_5_2_hz = bp_0_5_2_sum * df;

        // Bandpower [2.0, 5.0] Hz: Bins 11 to 25
        float bp_2_5_sum = 0.0f;
        bp_2_5_sum += 0.5f * psd[11];
        for (int k = 12; k < 25; ++k) {
            bp_2_5_sum += psd[k];
        }
        bp_2_5_sum += 0.5f * psd[25];
        bandpower_2_5_hz = bp_2_5_sum * df;

        float bandpower_ratio = (bandpower_2_5_hz > 0.0f) ? (bandpower_0_5_2_hz / bandpower_2_5_hz) : 0.0f;

        // 5. Fill Features Output Array
        // Matches FEATURE_COLUMN_NAMES ordering:
        // "mean", "std", "minimum", "maximum", "value_range", "rms", "skewness", "kurtosis",
        // "energy", "derivative_std", "dominant_frequency_hz", "bandpower_0_5_2_hz",
        // "bandpower_2_5_hz", "bandpower_ratio_0_5_2_to_2_5"
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
};
