// v2 (ADR-0006) pipeline checks: paired IR/red ingest, the two red/IR inputs,
// and the divide-by-zero guard. Build with -DOXY_MODEL_V2.
#ifndef OXY_MODEL_V2
#error "Build with -DOXY_MODEL_V2"
#endif
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include "oxygen_risk_pipeline.h"

static int failures = 0;
static void check(bool ok, const std::string& what) {
    std::cout << (ok ? "  [PASS] " : "  [FAIL] ") << what << '\n';
    if (!ok) ++failures;
}

static std::vector<float> ppg(size_t n, float dc, float ac, float hz) {
    std::vector<float> x(n);
    for (size_t i = 0; i < n; ++i) {
        const float t = static_cast<float>(i) / 50.0f;
        x[i] = dc + ac * std::sin(2.0f * 3.14159265f * hz * t) + 0.2f * ac * std::sin(2.0f * 3.14159265f * 0.1f * t);
    }
    return x;
}

static OxygenRiskPipeline pipeline;  // keep buffers off the stack

static std::vector<RiskDecision> run(const std::vector<float>& ir, const std::vector<float>& red) {
    pipeline.reset();
    std::vector<RiskDecision> out;
    for (size_t i = 0; i < ir.size(); ++i) {
        const auto d = pipeline.push_sample(ir[i], red[i]);
        if (d.has_value()) out.push_back(*d);
    }
    return out;
}

static bool all_finite(const RiskDecision& d) {
    if (!std::isfinite(d.risk_probability)) return false;
    for (float f : d.raw_features) if (!std::isfinite(f)) return false;
    for (const auto& a : d.top_attributions) if (!std::isfinite(a.importance_delta)) return false;
    return true;
}

int main() {
    const auto ir = ppg(1750, 100000.0f, 400.0f, 1.2f);
    const auto red = ppg(1750, 60000.0f, 300.0f, 1.2f);

    std::cout << "Decision schedule is unchanged (750 then every 250)\n";
    const auto normal = run(ir, red);
    check(normal.size() == 5, "five decisions from 1750 samples");
    bool schedule = !normal.empty();
    for (size_t k = 0; k < normal.size(); ++k) {
        const auto& d = normal[k];
        schedule = schedule && d.window_index == k && d.total_samples_ingested == 750 + 250 * k &&
                   d.window_start_sample == 250 * k && d.window_end_sample == 250 * k + 500 &&
                   d.decision_delay_samples == 250;
    }
    check(schedule, "window index, sample counts and 250-sample delay");

    std::cout << "Two red/IR inputs\n";
    bool valid = true, finite = true, sane = true;
    for (const auto& d : normal) {
        valid = valid && d.two_wavelength_valid;
        finite = finite && all_finite(d);
        sane = sane && d.raw_features[14] > 0.0f && d.raw_features[15] > 0.0f;
        for (const auto& a : d.top_attributions) sane = sane && a.feature_index < KAN_INPUT_DIM;
    }
    check(valid && finite && sane, "valid, finite, positive, top-3 indices < 16");
    const float expected_pi = 400.0f / 100000.0f / std::sqrt(2.0f);
    check(std::fabs(normal[2].raw_features[15] / expected_pi - 1.0f) < 0.05f,
          "ir_perfusion_index near (AC/sqrt 2)/DC for a sinusoid");

    // Linear filter and power-of-two scaling: identical arithmetic on both channels gives exactly 1.
    std::vector<float> doubled(ir.size());
    for (size_t i = 0; i < ir.size(); ++i) doubled[i] = 2.0f * ir[i];
    const auto same = run(ir, ir);
    const auto twice = run(ir, doubled);
    bool exact_one = !same.empty() && !twice.empty();
    for (const auto& d : same) exact_one = exact_one && d.raw_features[14] == 1.0f;
    for (const auto& d : twice) exact_one = exact_one && d.raw_features[14] == 1.0f;
    check(exact_one, "ratio_of_ratios == 1 exactly when red == IR or red == 2*IR");
    bool ir_unchanged = true;
    for (size_t k = 0; k < same.size(); ++k)
        for (int f = 0; f < NUM_FEATURES; ++f)
            ir_unchanged = ir_unchanged && same[k].raw_features[f] == normal[k].raw_features[f];
    check(ir_unchanged, "14 IR features do not depend on the red channel");

    std::cout << "Guard: no NaN reaches the KAN\n";
    const std::vector<float> zeros(1750, 0.0f), flat(1750, 50000.0f);
    std::vector<float> negative(1750);
    for (size_t i = 0; i < negative.size(); ++i) negative[i] = -red[i];
    const std::vector<std::pair<std::string, std::vector<RiskDecision>>> guarded = {
        {"red all zero", run(ir, zeros)}, {"IR all zero", run(zeros, red)},
        {"both zero", run(zeros, zeros)}, {"IR flat (zero perfusion)", run(flat, red)},
        {"red negative mean", run(ir, negative)}};
    for (const auto& item : guarded) {
        bool ok = !item.second.empty();
        for (const auto& d : item.second) {
            ok = ok && !d.two_wavelength_valid && all_finite(d) &&
                 d.raw_features[14] == SCALER_MEAN[14] && d.raw_features[15] == SCALER_MEAN[15];
            for (const auto& a : d.top_attributions)
                ok = ok && (a.feature_index < 14 || a.importance_delta == 0.0f);
        }
        check(ok, item.first + ": filled with training means, zero attribution, finite");
    }
    const auto red_flat = run(ir, flat);
    bool zero_ratio = !red_flat.empty();
    for (const auto& d : red_flat) zero_ratio = zero_ratio && d.two_wavelength_valid && d.raw_features[14] == 0.0f;
    check(zero_ratio, "flat red with positive mean is valid with ratio_of_ratios 0 (as the export)");

    std::cout << "Reset clears both channels\n";
    pipeline.reset();
    for (int i = 0; i < 600; ++i) pipeline.push_sample(ir[i], zeros[i]);
    pipeline.reset();
    std::vector<RiskDecision> after;
    for (size_t i = 0; i < ir.size(); ++i) {
        const auto d = pipeline.push_sample(ir[i], red[i]);
        if (d.has_value()) after.push_back(*d);
    }
    bool identical = after.size() == normal.size();
    for (size_t k = 0; identical && k < after.size(); ++k)
        for (int f = 0; f < KAN_INPUT_DIM; ++f)
            identical = identical && after[k].raw_features[f] == normal[k].raw_features[f];
    check(identical, "decisions after reset equal a fresh run");

    std::cout << (failures ? "FAILED " : "ALL PASSED ") << failures << " failure(s)\n";
    return failures ? 1 : 0;
}
