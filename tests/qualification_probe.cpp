// Batch cold-start context replay through the actual deployment pipeline.
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#ifdef PPG_FILTER_EXPERIMENT
#include "precision_filter_experiments.h"
#define ppg_bandpass_filter ppg_experiment_filter
#endif
#include "oxygen_risk_pipeline.h"
#include "probe_variant.h"

struct PipelineAuditAccess {
    static const float* filtered(const OxygenRiskPipeline& p) { return p.filtered_buffer; }
};

int main() {
    announce_pipeline_variant<OxygenRiskPipeline>("qualification_probe");
    std::cout << std::setprecision(9);
    std::string line;
    unsigned case_index = 0;
    OxygenRiskPipeline pipeline;
    while (std::getline(std::cin, line)) {
        std::istringstream input(line);
        int count;
        if (!(input >> count) || (count != 750 && count != 1000)) return 2;
        pipeline.reset();
        float first_probability = 0;
        unsigned emissions = 0;
        std::optional<RiskDecision> final;
        for (int i = 0; i < count; ++i) {
            float value;
            if (!(input >> value) || !std::isfinite(value)) return 3;
            const auto d = pipeline.push_sample(value);
            if (d.has_value()) {
                if (emissions++ == 0) first_probability = d->risk_probability;
                final = d;
            }
        }
        std::string extra;
        if (input >> extra || !final.has_value()) return 4;
        const auto& d = *final;
        std::cout << case_index++ << ' ' << emissions << ' ' << first_probability << ' '
                  << d.window_index << ' ' << d.total_samples_ingested << ' '
                  << d.window_start_sample << ' ' << d.window_end_sample << ' ' << d.decision_delay_samples
                  << ' ' << d.risk_probability;
        for (float value : d.raw_features) std::cout << ' ' << value;
        std::cout << ' ' << d.is_candidate_risk << ' ' << d.is_confirmed_risk
                  << ' ' << static_cast<unsigned>(d.consecutive_risk_windows);
        for (const auto& a : d.top_attributions) std::cout << ' ' << static_cast<unsigned>(a.feature_index) << ' ' << a.importance_delta;
        float scaled[14];
        kan_standardize(d.raw_features, scaled);
        for (int i = 0; i < 14; ++i) {
            const float saved = scaled[i];
            scaled[i] = 0;
            std::cout << ' ' << d.risk_probability - kan_sigmoid(kan_predict_logit(scaled));
            scaled[i] = saved;
        }
        const float* filtered = PipelineAuditAccess::filtered(pipeline);
        for (int i = 0; i < 500; ++i) std::cout << ' ' << filtered[i];
        std::cout << '\n';
    }
    return 0;
}
