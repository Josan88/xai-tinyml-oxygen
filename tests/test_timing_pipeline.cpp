#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include "timing_pipeline.h"

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

// Clock/GPIO is the external hardware boundary; pipeline math is real.
struct SimulatedClock {
    bool high = false;
    uint32_t begin() { high = true; return 0xfffffff0u; }
    uint32_t end() { high = false; return 0x30u; }
};

int main() {
    try {
        TimedOxygenRiskPipeline timed;
        OxygenRiskPipeline reference;
        SimulatedClock clock;
        float block[250];
        unsigned decisions = 0;
        for (unsigned b = 0; b < 12; ++b) {
            std::optional<RiskDecision> expected;
            for (unsigned i = 0; i < 250; ++i) {
                const float t = (b * 250 + i) / 50.0f;
                block[i] = 2000.0f + 100.0f * std::sin(2.0f * 3.14159265f * 1.2f * t);
                const auto d = reference.push_sample(block[i]);
                if (d) expected = d;
            }
            const auto actual = timed.push_block(block, clock);
            require(actual.cycles == 64, "unsigned cycle subtraction must tolerate one wrap");
            require(!clock.high, "GPIO must return low after every block");
            require(timed.get_total_samples() == (b + 1) * 250, "block must ingest exactly 250 samples");
            require(bool(actual.decision) == bool(expected), "timing must preserve emission cadence");
            if (!expected) continue;
            ++decisions;
            const auto& a = *actual.decision;
            const auto& e = *expected;
            require(a.window_index == e.window_index && a.total_samples_ingested == e.total_samples_ingested &&
                    a.window_start_sample == e.window_start_sample && a.window_end_sample == e.window_end_sample &&
                    a.decision_delay_samples == e.decision_delay_samples, "decision metadata changed");
            require(a.risk_probability == e.risk_probability && a.is_candidate_risk == e.is_candidate_risk &&
                    a.is_confirmed_risk == e.is_confirmed_risk && a.consecutive_risk_windows == e.consecutive_risk_windows,
                    "probability or debounce state changed");
            for (int i = 0; i < 14; ++i) require(a.raw_features[i] == e.raw_features[i], "raw feature changed");
            for (int i = 0; i < 3; ++i) {
                require(a.top_attributions[i].feature_index == e.top_attributions[i].feature_index &&
                        a.top_attributions[i].importance_delta == e.top_attributions[i].importance_delta,
                        "top-three attribution changed");
            }
        }
        require(decisions == 10, "expected ten decisions including ring wrap");
        timed.reset();
        require(timed.get_total_samples() == 0, "reset must discard previous samples");
        require(!timed.push_block(block, clock).decision, "reset warmup block one");
        timed.reset();
        require(!timed.push_block(block, clock).decision, "mid-warmup reset block one");
        require(!timed.push_block(block, clock).decision, "reset warmup block two");
        const auto first = timed.push_block(block, clock);
        require(first.decision && first.decision->window_index == 0 &&
                first.decision->total_samples_ingested == 750 && first.decision->consecutive_risk_windows <= 1,
                "reset must restart cadence and debounce");
        std::cout << "Timing preserves ten real pipeline decisions and counter wrap\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
