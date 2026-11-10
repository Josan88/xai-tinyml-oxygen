#include <iomanip>
#include <iostream>
#include <string>
#include "oxygen_risk_pipeline.h"
#include "probe_variant.h"

int main(int argc, char** argv) {
    const bool extended = argc == 2 && std::string(argv[1]) == "--extended";
    if (argc > 1 && !extended) return 2;
    announce_pipeline_variant<OxygenRiskPipeline>("context_probe");
    std::cout << std::setprecision(9);
    OxygenRiskPipeline pipeline;
    float value;
    while (std::cin >> value) {
        if (!std::isfinite(value)) return 3;
        const auto decision = pipeline.push_sample(value);
        if (!decision.has_value()) continue;
        std::cout << decision->window_index << ' '
                  << decision->total_samples_ingested << ' '
                  << decision->window_start_sample << ' '
                  << decision->window_end_sample << ' '
                  << decision->decision_delay_samples << ' '
                  << decision->risk_probability;
        for (float feature : decision->raw_features) std::cout << ' ' << feature;
        if (extended) {
            std::cout << ' ' << decision->is_candidate_risk << ' ' << decision->is_confirmed_risk
                      << ' ' << static_cast<unsigned>(decision->consecutive_risk_windows);
            for (const auto& attribution : decision->top_attributions) {
                std::cout << ' ' << static_cast<unsigned>(attribution.feature_index)
                          << ' ' << attribution.importance_delta;
            }
        }
        std::cout << '\n';
    }
    return 0;
}
