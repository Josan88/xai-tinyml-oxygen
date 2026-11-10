#pragma once
#if !defined(OXY_MODEL_V2) && !defined(OXY_MODEL_P5)
#error "timing_pipeline_v2.h requires OXY_MODEL_V2 or OXY_MODEL_P5 (paired IR/red); v1 uses timing_pipeline.h"
#endif
#include "oxygen_risk_pipeline.h"

// v2 (ADR-0006) of timing_pipeline.h: one block is 250 IR/red pairs. Also the P5 timing
// image (ADR-0008), whose pipeline takes the same paired samples.
// Diagnostic wrapper only. The frozen deployment pipeline is used unchanged.
// One interval includes all 250 pushes, at most one decision and its normal XAI.
struct TimedBlock {
    uint32_t cycles;
    std::optional<RiskDecision> decision;
};

class TimedOxygenRiskPipeline {
    OxygenRiskPipeline pipeline;
public:
    template<class Clock>
    TimedBlock push_block(const float (&ir)[250], const float (&red)[250], Clock& clock) {
        TimedBlock result;
        const uint32_t start = clock.begin();
        for (int i = 0; i < 250; ++i) {
            const auto decision = pipeline.push_sample(ir[i], red[i]);
            if (decision) result.decision = decision;
        }
        const uint32_t stop = clock.end();
        result.cycles = stop - start;
        return result;
    }
    uint64_t get_total_samples() const { return pipeline.get_total_samples(); }
    void reset() { pipeline.reset(); }
};
