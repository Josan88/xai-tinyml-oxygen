// Desktop-only access to the deployment filter and feature calculation stages.
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include "oxygen_risk_pipeline.h"
#include "probe_variant.h"

struct PipelineAuditAccess {
    static void features(OxygenRiskPipeline& pipeline, const float* filtered, float* out) {
        pipeline.extract_features_from_filtered(filtered, out);
    }
};

int main(int argc, char** argv) {
    if (argc != 2 || (std::string(argv[1]) != "raw" && std::string(argv[1]) != "filtered")) return 2;
    const bool raw = std::string(argv[1]) == "raw";
    announce_pipeline_variant<OxygenRiskPipeline>("feature_audit_probe");
    OxygenRiskPipeline pipeline;
    std::cout << std::setprecision(9);
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream input(line);
        int count = 500, offset = 0;
        if (raw && (!(input >> count >> offset) ||
            !((count == 750 && offset == 0) || (count == 1000 && offset == 250)))) return 3;
        float samples[1000], filtered[500], scratch[1054], features[14];
        for (int i = 0; i < count; ++i) {
            if (!(input >> samples[i]) || !std::isfinite(samples[i])) return 3;
        }
        std::string extra;
        if (input >> extra) return 3;
        if (raw) ppg_bandpass_filter(samples, filtered, scratch, count, offset);
        else std::memcpy(filtered, samples, sizeof(filtered));
        PipelineAuditAccess::features(pipeline, filtered, features);
        for (float value : filtered) std::cout << value << ' ';
        for (int i = 0; i < 14; ++i) std::cout << features[i] << (i == 13 ? '\n' : ' ');
    }
    return 0;
}
