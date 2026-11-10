#pragma once
#include "oxygen_risk_pipeline.h"

struct TimedBlock {
    uint32_t cycles;
    std::optional<RiskDecision> decision;
};

// Diagnostic wrapper only. The frozen deployment pipeline is used unchanged.
// One interval includes all 250 pushes, at most one decision and its normal XAI.
class TimedOxygenRiskPipeline {
    OxygenRiskPipeline pipeline;
public:
    template<class Clock>
    TimedBlock push_block(const float (&samples)[250], Clock& clock) {
        TimedBlock result;
        const uint32_t start = clock.begin();
        for (float value : samples) {
            const auto decision = pipeline.push_sample(value);
            if (decision) result.decision = decision;
        }
        const uint32_t stop = clock.end();
        result.cycles = stop - start;
        return result;
    }
    uint64_t get_total_samples() const { return pipeline.get_total_samples(); }
    void reset() { pipeline.reset(); }
};
