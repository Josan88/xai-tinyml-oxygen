// Host-only development experiments; not included in firmware builds.
#pragma once
#include "feature_extraction.h"

inline void ppg_experiment_filter(const float* input, float* output, float* scratch,
                                  int count = 500, int offset = 0) {
    const int padded = count + 54;
#if PPG_FILTER_EXPERIMENT >= 6
    const float dc = input[offset + 250];
#else
    const float dc = input[0];
#endif
    for (int i = 0; i < 27; ++i) {
#if PPG_FILTER_EXPERIMENT >= 6
        scratch[i] = 2.0f * (input[0] - dc) - (input[27 - i] - dc);
#else
        scratch[i] = -(input[27 - i] - dc);
#endif
        scratch[27 + count + i] = 2.0f * (input[count - 1] - dc) - (input[count - 2 - i] - dc);
    }
    for (int i = 0; i < count; ++i) scratch[27 + i] = input[i] - dc;
    const float zi[4][2] = {{.143157091f, -.00794613801f}, {.756299344f, -.344021061f},
                            {-.918019446f, .918019446f}, {0.f, 0.f}};
    // Algebraic transformation of canonical SciPy SOS, rounded once to FP32:
    // k = 1+a1+a2; h = 1-a2. Values were computed before FP32 rounding of a1/a2.
    const float k[2] = {.00456648134f, .00397024211f};
    const float h[2] = {.118257344f, .0430523232f};
    for (int pass = 0; pass < 2; ++pass) {
        const float x0 = scratch[0];
        for (int section = 0; section < 4; ++section) {
            const float* c = FILTER_SOS[section];
#if PPG_FILTER_EXPERIMENT >= 2
            if (section >= 2) {
                // Equivalent high-pass transfer (1-z^-1)^2 / (1+a1*z^-1+a2*z^-2).
                // v integrates d; d integrates y. Initial constant input has y=d=0.
                const float kk = k[section - 2], hh = h[section - 2];
                float v = scratch[0] / kk, d = 0.f;
                float vc = 0.f, dcorr = 0.f;
#if PPG_FILTER_EXPERIMENT >= 5
                vc = fmaf(kk, v, -scratch[0]) / kk;
#endif
                for (int i = 0; i < padded; ++i) {
#if PPG_FILTER_EXPERIMENT >= 4
                    float y = fmaf(-hh, d, fmaf(-kk, v, scratch[i]));
#if PPG_FILTER_EXPERIMENT >= 5
                    y += fmaf(kk, vc, hh * dcorr);
#endif
#else
                    const float y = fmaf(-kk, v, fmaf(-hh, d, scratch[i]));
#endif
#if PPG_FILTER_EXPERIMENT >= 3
                    const float di = y - dcorr;
                    const float dn = d + di;
                    dcorr = (dn - d) - di;
                    d = dn;
#if PPG_FILTER_EXPERIMENT >= 5
                    const float vi = (d - dcorr) - vc;
#else
                    const float vi = d - vc;
#endif
                    const float vn = v + vi;
                    vc = (vn - v) - vi;
                    v = vn;
#else
                    d += y;
                    v += d;
#endif
                    scratch[i] = y;
                }
                continue;
            }
#endif
            float w1 = zi[section][0] * x0, w2 = zi[section][1] * x0;
            for (int i = 0; i < padded; ++i) {
                const float x = scratch[i];
                const float y = fmaf(c[0], x, w1);
                w1 = fmaf(-c[4], y, fmaf(c[1], x, w2));
                w2 = fmaf(-c[5], y, c[2] * x);
                scratch[i] = y;
            }
        }
        for (int i = 0; i < padded / 2; ++i) {
            const float saved = scratch[i];
            scratch[i] = scratch[padded - 1 - i];
            scratch[padded - 1 - i] = saved;
        }
    }
    memcpy(output, scratch + 27 + offset, 500 * sizeof(float));
}
