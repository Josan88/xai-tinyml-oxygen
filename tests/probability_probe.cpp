// Desktop-only bridge: run the actual firmware headers on Python-supplied inputs.
// Raw mode filters each isolated 500-sample window (WindowLocalOxygenRiskPipeline);
// it is NOT the deployed 5 s past/future context path. Use context_probe for that.
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include "oxygen_risk_pipeline.h"
#include "probe_variant.h"

int main(int argc, char** argv) {
    if (argc != 2 || (std::string(argv[1]) != "raw" && std::string(argv[1]) != "features")) {
        std::cerr << "Usage: probability_probe raw|features\n"
                  << "  raw: window-local filtering (BasicOxygenRiskPipeline<0>), not the deployed path\n";
        return 2;
    }
    announce_pipeline_variant<WindowLocalOxygenRiskPipeline>("probability_probe");
    const bool raw = std::string(argv[1]) == "raw";
    std::cout << std::setprecision(9);
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream input(line);
        float values[PPG_WINDOW_SIZE];
        const int count = raw ? PPG_WINDOW_SIZE : KAN_INPUT_DIM;
        for (int i = 0; i < count; ++i) {
            if (!(input >> values[i]) || !std::isfinite(values[i])) return 3;
        }
        std::string extra;
        if (input >> extra) return 3;
        float features[KAN_INPUT_DIM];
        if (raw) {
            WindowLocalOxygenRiskPipeline pipeline;
            for (int i = 0; i < count; ++i) {
                const auto decision = pipeline.push_sample(values[i]);
                if (i == count - 1) {
                    if (!decision.has_value()) return 4;
                    std::memcpy(features, decision->raw_features, sizeof(features));
                }
            }
        } else {
            std::memcpy(features, values, sizeof(features));
        }
        float scaled[KAN_INPUT_DIM];
        kan_standardize(features, scaled);
        const float probability = kan_sigmoid(kan_predict_logit(scaled));
        std::cout << probability;
        for (float feature : features) std::cout << ' ' << feature;
        for (float feature : scaled) std::cout << ' ' << feature;
        for (int i = 0; i < KAN_INPUT_DIM; ++i) {
            const float saved = scaled[i];
            scaled[i] = 0.0f;
            std::cout << ' ' << probability - kan_sigmoid(kan_predict_logit(scaled));
            scaled[i] = saved;
        }
        std::cout << '\n';
    }
    return 0;
}
