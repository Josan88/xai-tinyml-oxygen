#pragma once
// Physiological beat features of one filtered PPG window, for the P5 model (job F2, ADR-0008).
//
// A step-for-step port of scripts/physio_features.py, which stays the frozen reference (this file
// changes nothing there). Inputs: the filtered IR window x and red window r (500 samples at 50 Hz,
// FP32 as the deployed filter writes them) and the raw-window DC means dc_ir, dc_red. Outputs, in
// model order: r_beat, pi_beat, hr_bpm, sqi_template, sqi_ibi_ratio, plus n_beats, beat_valid and
// the chosen polarity. Every step is evaluated in double on the FP32 inputs, as NumPy/SciPy do on
// float64 arrays, so given the same input window the integer peaks are identical and the features
// agree to double rounding (tests/physio_features_probe.cpp, scripts/check_physio_features_cpp.py).
//
// Semantics reproduced from the installed NumPy 2.4 / SciPy 1.17 sources:
//  * np.std / np.mean: numpy's pairwise summation (8 accumulators for <= 128 items, halving above),
//    then division by the count; np.std = sqrt(mean((x - mean)^2)), ddof 0.
//  * np.percentile(d, q), method "linear": virtual index (n - 1) * q / 100, gamma = index - floor,
//    lerp a + (b - a) * gamma for gamma < 0.5 and b - (b - a) * (1 - gamma) otherwise.
//  * scipy.signal.find_peaks(y, distance=17, prominence=p):
//      _local_maxima_1d: a rise then a fall, plateaus allowed, the peak is the plateau midpoint
//        (left + right) // 2; the first and last samples are never peaks;
//      _select_by_peak_distance BEFORE prominence: peaks visited in descending height, each kept
//        peak removes every still-kept peak closer than ceil(17) = 17 samples;
//      _peak_prominences with wlen=None: bases searched over the whole window, prominence =
//        height - max(left base minimum, right base minimum); kept iff p <= prominence.
//    Height ties in the distance step: SciPy orders peaks with np.argsort (quicksort). On NumPy 2.x
//    x86 builds that dispatches to a vectorized sort whose tie order depends on the CPU, so it is not
//    portable and is not reproduced. Here a tie is broken by position (the later peak is visited
//    first, the order of a stable sort). Every window where a tie decides a removal sets
//    distance_tie, so a run can list them; on float64 or FP32 filtered PPG they are not expected.
//  * 3-point parabola refinement with the offset clipped to [-0.5, 0.5]; ibi = diff / 50 s.
//  * np.median: middle order statistic, or (a + b) / 2 of the two middle ones.
//  * np.linspace(0, 1, L) = k * (1 / (L - 1)) with the last point exactly 1; np.interp on the grid
//    0..n-1: f[j] when x == j or x is the last point, else (f[j+1] - f[j]) * (x - j) + f[j].
//  * resampled.mean(axis=0): rows added in segment order, then divided by the row count.
//  * Pearson r: np.dot is a BLAS call whose summation order is the library's; here it is a plain
//    sequential sum. This is the one step that is not bit-exact (residual ~1e-16 relative).
//
// Fixed-size buffers only, no heap. Worst cases for a 500-sample window (local maxima lie in [1, 498]):
//  * local maxima: at most 249 (PHYSIO_MAX_LOCAL_MAXIMA = 250 = x.size // 2, SciPy's allocation);
//  * peaks after the 17-sample distance rule: at most (498 - 1) / 17 + 1 = 30 (PHYSIO_MAX_PEAKS);
//  * segments: at most 29;
//  * template length L = round(median segment length): with k >= 3 segments of >= 18 samples
//    spanning at most 497 gaps, the median gap is at most 240 (k = 3), so L <= 241.
// PhysioWorkspace is about 6.7 KB; the caller keeps it in static storage (not the thread stack).
#ifdef ARDUINO
#include <Arduino.h>
#endif
#include <math.h>
#include <stdint.h>
#include <string.h>

namespace physio {

static const int PHYSIO_WINDOW = 500;
static const int PHYSIO_NUM_FEATURES = 5;
static const int PHYSIO_MIN_PEAK_DISTANCE = 17;   // samples (0.34 s); firmware HR_MIN_PEAK_GAP
static const int PHYSIO_MIN_BEATS = 4;            // >= 3 intervals
static const double PHYSIO_SAMPLE_RATE_HZ = 50.0;
static const double PHYSIO_PROMINENCE_STD = 1.0;
static const int PHYSIO_MAX_LOCAL_MAXIMA = PHYSIO_WINDOW / 2;
static const int PHYSIO_MAX_PEAKS = (PHYSIO_WINDOW - 3) / PHYSIO_MIN_PEAK_DISTANCE + 1;
static const int PHYSIO_MAX_SEGMENTS = PHYSIO_MAX_PEAKS - 1;
static const int PHYSIO_MAX_TEMPLATE = 241;
static_assert(PHYSIO_MAX_PEAKS == 30, "peaks >= 17 apart in [1, 498]");

// Model input order (scripts/physio_features.py FEATURES).
enum { R_BEAT = 0, PI_BEAT = 1, HR_BPM = 2, SQI_TEMPLATE = 3, SQI_IBI_RATIO = 4 };

struct PhysioResult {
    double features[PHYSIO_NUM_FEATURES];  // NAN when !beat_valid
    int n_beats;                           // detected peaks (0 when the window is unusable)
    bool beat_valid;
    int polarity;                          // +1 or -1; 0 when the window is unusable
    int16_t peaks[PHYSIO_MAX_PEAKS];       // the first n_beats entries are the integer IR peaks
    bool distance_tie;                     // a height tie decided a distance removal (see above)
};

struct PhysioWorkspace {
    int16_t maxima[PHYSIO_MAX_LOCAL_MAXIMA];
    int16_t order[PHYSIO_MAX_LOCAL_MAXIMA];
    uint8_t keep[PHYSIO_MAX_LOCAL_MAXIMA];
    double refined[PHYSIO_MAX_PEAKS];
    double ibi[PHYSIO_MAX_SEGMENTS];
    double ratios[PHYSIO_MAX_SEGMENTS];
    double perfusion[PHYSIO_MAX_SEGMENTS];
    double pearson[PHYSIO_MAX_SEGMENTS];
    double scratch[PHYSIO_MAX_PEAKS];
    double mean_beat[PHYSIO_MAX_TEMPLATE];
    double beat[PHYSIO_MAX_TEMPLATE];
};

// numpy pairwise_sum (loops_utils.h.src) over f(start), ..., f(start + n - 1).
template <class F>
double pairwise_sum(const F& f, int start, int n) {
    if (n < 8) {
        double res = -0.0;
        for (int i = 0; i < n; ++i) res += f(start + i);
        return res;
    }
    if (n <= 128) {
        double r[8];
        for (int k = 0; k < 8; ++k) r[k] = f(start + k);
        int i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int k = 0; k < 8; ++k) r[k] += f(start + i + k);
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += f(start + i);
        return res;
    }
    int n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum(f, start, n2) + pairwise_sum(f, start + n2, n - n2);
}

// np.median of values[0..n-1] (n >= 1); sorts a copy in scratch (insertion sort, n <= 30).
inline double median(const double* values, int n, double* scratch) {
    for (int i = 0; i < n; ++i) {
        const double v = values[i];
        int j = i;
        while (j > 0 && scratch[j - 1] > v) { scratch[j] = scratch[j - 1]; --j; }
        scratch[j] = v;
    }
    return (n % 2) ? scratch[n / 2] : (scratch[n / 2 - 1] + scratch[n / 2]) / 2.0;
}

// np.percentile linear interpolation between order statistics a = sorted[prev], b = sorted[prev + 1].
inline double lerp_quantile(double a, double b, double gamma) {
    const double diff = b - a;
    return gamma >= 0.5 ? b - diff * (1.0 - gamma) : a + diff * gamma;
}

// polarity(): +1 if P99(diff(x)) >= -P1(diff(x)), else -1. Only order statistics 4, 5, 493 and 494
// of the 499 differences are needed, so the 6 smallest and 6 largest are kept instead of a sort.
inline int polarity(const float* x) {
    const int n = PHYSIO_WINDOW - 1;
    double lo[6], hi[6];  // lo ascending (smallest first), hi descending (largest first)
    int nlo = 0, nhi = 0;
    for (int i = 0; i < n; ++i) {
        const double d = static_cast<double>(x[i + 1]) - static_cast<double>(x[i]);
        if (nlo < 6 || d < lo[nlo - 1]) {
            int j = nlo < 6 ? nlo++ : 5;
            while (j > 0 && lo[j - 1] > d) { lo[j] = lo[j - 1]; --j; }
            lo[j] = d;
        }
        if (nhi < 6 || d > hi[nhi - 1]) {
            int j = nhi < 6 ? nhi++ : 5;
            while (j > 0 && hi[j - 1] < d) { hi[j] = hi[j - 1]; --j; }
            hi[j] = d;
        }
    }
    // sorted[k] for k = 493, 494 is hi[n - 1 - k] = hi[5], hi[4]; for k = 4, 5 it is lo[4], lo[5].
    const double v99 = static_cast<double>(n - 1) * 0.99;
    const double v01 = static_cast<double>(n - 1) * 0.01;
    const int p99 = static_cast<int>(floor(v99)), p01 = static_cast<int>(floor(v01));
    const double q99 = lerp_quantile(hi[n - 1 - p99], hi[n - 2 - p99], v99 - p99);
    const double q01 = lerp_quantile(lo[p01], lo[p01 + 1], v01 - p01);
    return q99 >= -q01 ? 1 : -1;
}

// Height order for the distance rule: a before b iff (y[a], a) < (y[b], b). Heapsort, so the cost
// is bounded on noisy windows with ~250 maxima.
template <class Y>
inline bool height_less(const Y& y, int a, int b) {
    const double ya = y(a), yb = y(b);
    return ya < yb || (ya == yb && a < b);
}

template <class Y>
inline void sift_down(int16_t* order, int start, int end, const Y& y, const int16_t* peaks) {
    int root = start;
    while (2 * root + 1 <= end) {
        int child = 2 * root + 1;
        if (child + 1 <= end && height_less(y, peaks[order[child]], peaks[order[child + 1]])) ++child;
        if (!height_less(y, peaks[order[root]], peaks[order[child]])) return;
        const int16_t t = order[root]; order[root] = order[child]; order[child] = t;
        root = child;
    }
}

// scipy.signal.find_peaks(y, distance=17, prominence=min_prominence) with y(i) = s * x[i].
// Writes the kept peak positions to out (ascending) and returns their count.
inline int find_peaks(const float* x, int s, double min_prominence, PhysioWorkspace& ws, int16_t* out,
                      bool* distance_tie) {
    auto y = [x, s](int i) { return s > 0 ? static_cast<double>(x[i]) : -static_cast<double>(x[i]); };
    // _local_maxima_1d
    int m = 0;
    int i = 1;
    const int i_max = PHYSIO_WINDOW - 1;
    while (i < i_max) {
        if (y(i - 1) < y(i)) {
            int i_ahead = i + 1;
            while (i_ahead < i_max && y(i_ahead) == y(i)) ++i_ahead;
            if (y(i_ahead) < y(i)) {
                ws.maxima[m++] = static_cast<int16_t>((i + i_ahead - 1) / 2);
                i = i_ahead;
            }
        }
        ++i;
    }
    // _select_by_peak_distance: visit in descending height (ties: later peak first).
    for (int k = 0; k < m; ++k) { ws.order[k] = static_cast<int16_t>(k); ws.keep[k] = 1; }
    for (int start = m / 2 - 1; start >= 0; --start) sift_down(ws.order, start, m - 1, y, ws.maxima);
    for (int end = m - 1; end > 0; --end) {
        const int16_t t = ws.order[0]; ws.order[0] = ws.order[end]; ws.order[end] = t;
        sift_down(ws.order, 0, end - 1, y, ws.maxima);
    }
    *distance_tie = false;
    for (int r = m - 1; r >= 0; --r) {
        const int j = ws.order[r];
        if (!ws.keep[j]) continue;
        for (int k = j - 1; k >= 0 && ws.maxima[j] - ws.maxima[k] < PHYSIO_MIN_PEAK_DISTANCE; --k) {
            if (ws.keep[k] && y(ws.maxima[k]) == y(ws.maxima[j])) *distance_tie = true;
            ws.keep[k] = 0;
        }
        for (int k = j + 1; k < m && ws.maxima[k] - ws.maxima[j] < PHYSIO_MIN_PEAK_DISTANCE; ++k) {
            if (ws.keep[k] && y(ws.maxima[k]) == y(ws.maxima[j])) *distance_tie = true;
            ws.keep[k] = 0;
        }
    }
    // _peak_prominences (wlen=None) on the survivors, then the prominence condition.
    int n = 0;
    for (int k = 0; k < m; ++k) {
        if (!ws.keep[k]) continue;
        const int peak = ws.maxima[k];
        const double height = y(peak);
        double left_min = height, right_min = height;
        for (int t = peak; t >= 0 && y(t) <= height; --t)
            if (y(t) < left_min) left_min = y(t);
        for (int t = peak; t <= PHYSIO_WINDOW - 1 && y(t) <= height; ++t)
            if (y(t) < right_min) right_min = y(t);
        const double prominence = height - (left_min > right_min ? left_min : right_min);
        if (min_prominence <= prominence) out[n++] = static_cast<int16_t>(peak);
    }
    return n;
}

inline void set_invalid(PhysioResult& result, int n_beats, int s) {
    for (int k = 0; k < PHYSIO_NUM_FEATURES; ++k) result.features[k] = NAN;
    result.n_beats = n_beats;
    result.beat_valid = false;
    result.polarity = s;
}

// np.interp(grid[j] * (len - 1), arange(len), seg) for the segment y(lo .. lo + len - 1).
template <class Y>
inline double resample(const Y& y, int lo, int len, int L, int j) {
    const double step = 1.0 / static_cast<double>(L - 1);
    const double grid = (j == L - 1) ? 1.0 : static_cast<double>(j) * step;
    const double xv = grid * static_cast<double>(len - 1);
    const int k = static_cast<int>(floor(xv));
    if (k >= len - 1) return y(lo + len - 1);
    if (static_cast<double>(k) == xv) return y(lo + k);
    const double slope = (y(lo + k + 1) - y(lo + k)) / 1.0;
    return slope * (xv - static_cast<double>(k)) + y(lo + k);
}

// template_sqi(s * x, peaks): mean Pearson r of each resampled segment with the mean beat.
inline double template_sqi(const float* x, int s, const int16_t* peaks, int n_peaks, PhysioWorkspace& ws) {
    auto y = [x, s](int i) { return s > 0 ? static_cast<double>(x[i]) : -static_cast<double>(x[i]); };
    const int segments = n_peaks - 1;
    // Segment lengths go in ws.pearson, which is not filled until the second pass below.
    for (int k = 0; k < segments; ++k) ws.pearson[k] = static_cast<double>(peaks[k + 1] - peaks[k] + 1);
    const int L = static_cast<int>(floor(median(ws.pearson, segments, ws.scratch) + 0.5));
    for (int j = 0; j < L; ++j) ws.mean_beat[j] = 0.0;
    for (int k = 0; k < segments; ++k) {
        const int lo = peaks[k], len = peaks[k + 1] - peaks[k] + 1;
        for (int j = 0; j < L; ++j) ws.mean_beat[j] += resample(y, lo, len, L, j);
    }
    for (int j = 0; j < L; ++j) ws.mean_beat[j] /= static_cast<double>(segments);
    const double* mb = ws.mean_beat;
    const double mb_mean = pairwise_sum([mb](int j) { return mb[j]; }, 0, L) / static_cast<double>(L);
    double bb = 0.0;
    for (int j = 0; j < L; ++j) { const double b = mb[j] - mb_mean; bb += b * b; }
    for (int k = 0; k < segments; ++k) {
        const int lo = peaks[k], len = peaks[k + 1] - peaks[k] + 1;
        for (int j = 0; j < L; ++j) ws.beat[j] = resample(y, lo, len, L, j);
        const double* bt = ws.beat;
        const double bt_mean = pairwise_sum([bt](int j) { return bt[j]; }, 0, L) / static_cast<double>(L);
        double ab = 0.0, aa = 0.0;
        for (int j = 0; j < L; ++j) {
            const double a = bt[j] - bt_mean, b = mb[j] - mb_mean;
            ab += a * b;
            aa += a * a;
        }
        const double denom = sqrt(aa * bb);
        ws.pearson[k] = denom > 0 ? ab / denom : 0.0;
    }
    const double* pr = ws.pearson;
    return pairwise_sum([pr](int k) { return pr[k]; }, 0, segments) / static_cast<double>(segments);
}

// extract(ir, red, ir_dc, red_dc) of scripts/physio_features.py.
inline void extract(const float* x, const float* r, float dc_ir_f, float dc_red_f, PhysioWorkspace& ws,
                    PhysioResult& result) {
    const double ir_dc = static_cast<double>(dc_ir_f), red_dc = static_cast<double>(dc_red_f);
    result.distance_tie = false;
    bool finite = true;
    for (int i = 0; i < PHYSIO_WINDOW; ++i)
        if (!isfinite(x[i]) || !isfinite(r[i])) { finite = false; break; }
    double std_x = 0.0;
    if (finite) {
        const double n = static_cast<double>(PHYSIO_WINDOW);
        const double mean = pairwise_sum([x](int i) { return static_cast<double>(x[i]); }, 0, PHYSIO_WINDOW) / n;
        const double var = pairwise_sum([x, mean](int i) {
            const double d = static_cast<double>(x[i]) - mean;
            return d * d;
        }, 0, PHYSIO_WINDOW) / n;
        std_x = sqrt(var);
    }
    if (!(finite && std_x > 0 && ir_dc > 0 && red_dc > 0)) {
        set_invalid(result, 0, 0);
        return;
    }
    const int s = polarity(x);
    const int n_peaks = find_peaks(x, s, PHYSIO_PROMINENCE_STD * std_x, ws, result.peaks, &result.distance_tie);
    if (n_peaks < PHYSIO_MIN_BEATS) {
        set_invalid(result, n_peaks, s);
        return;
    }
    auto y = [x, s](int i) { return s > 0 ? static_cast<double>(x[i]) : -static_cast<double>(x[i]); };
    for (int k = 0; k < n_peaks; ++k) {
        const int p = result.peaks[k];
        double out = static_cast<double>(p);
        if (p > 0 && p < PHYSIO_WINDOW - 1) {
            const double a = y(p - 1), b = y(p), c = y(p + 1);
            const double denom = a - 2.0 * b + c;
            double offset = 0.0;
            if (denom < 0) offset = 0.5 * (a - c) / denom;
            if (offset < -0.5) offset = -0.5;
            if (offset > 0.5) offset = 0.5;
            out += offset;
        }
        ws.refined[k] = out;
    }
    const int segments = n_peaks - 1;
    double ibi_max = 0.0, ibi_min = 0.0;
    for (int k = 0; k < segments; ++k) {
        ws.ibi[k] = (ws.refined[k + 1] - ws.refined[k]) / PHYSIO_SAMPLE_RATE_HZ;
        if (k == 0 || ws.ibi[k] > ibi_max) ibi_max = ws.ibi[k];
        if (k == 0 || ws.ibi[k] < ibi_min) ibi_min = ws.ibi[k];
    }
    int kept = 0;
    for (int k = 0; k < segments; ++k) {
        const int lo = result.peaks[k], hi = result.peaks[k + 1] + 1;
        double ir_hi = x[lo], ir_lo = x[lo], red_hi = r[lo], red_lo = r[lo];
        for (int t = lo + 1; t < hi; ++t) {
            if (x[t] > ir_hi) ir_hi = x[t];
            if (x[t] < ir_lo) ir_lo = x[t];
            if (r[t] > red_hi) red_hi = r[t];
            if (r[t] < red_lo) red_lo = r[t];
        }
        const double ac_ir = ir_hi - ir_lo, ac_red = red_hi - red_lo;
        if (ac_ir <= 0) continue;
        ws.ratios[kept] = (ac_red / red_dc) / (ac_ir / ir_dc);
        ws.perfusion[kept] = ac_ir / ir_dc;
        ++kept;
    }
    if (!kept) {
        set_invalid(result, n_peaks, s);
        return;
    }
    result.features[R_BEAT] = median(ws.ratios, kept, ws.scratch);
    result.features[PI_BEAT] = median(ws.perfusion, kept, ws.scratch);
    result.features[HR_BPM] = 60.0 / median(ws.ibi, segments, ws.scratch);
    result.features[SQI_TEMPLATE] = template_sqi(x, s, result.peaks, n_peaks, ws);
    result.features[SQI_IBI_RATIO] = ibi_max / ibi_min;
    result.n_beats = n_peaks;
    result.beat_valid = true;
    result.polarity = s;
}

}  // namespace physio
