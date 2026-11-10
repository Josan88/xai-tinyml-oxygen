// v2 (ADR-0006) host probe: streams IR/red pairs through the deployed
// OxygenRiskPipeline (BasicOxygenRiskPipeline<250>) built with OXY_MODEL_V2.
// Input file, little-endian: repeated segments of
//   uint32 segment_id, uint32 pair_count, pair_count x (float32 ir, float32 red).
// Each segment starts from a reset pipeline. One output line per decision:
//   segment window total start end delay probability f0..f15 candidate confirmed
//   streak top_idx0 top_delta0 top_idx1 top_delta1 top_idx2 top_delta2
//   two_wavelength_valid delta0..delta15
// Floats print with 9 significant digits, which round-trips FP32 exactly.
#ifndef OXY_MODEL_V2
#error "Build with -DOXY_MODEL_V2"
#endif
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include "oxygen_risk_pipeline.h"
#include "probe_variant.h"

static_assert(KAN_INPUT_DIM == 16, "v2 probe expects the 16-input model");

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    announce_pipeline_variant<OxygenRiskPipeline>("v2_stream_probe");
#ifdef OXY_XAI_REF_LABEL0
    std::cerr << "v2_stream_probe: xai_reference=" KAN_XAI_REF_TAG " (KAN_XAI_REF_Z)\n";
#endif
    std::ifstream input(argv[1], std::ios::binary);
    if (!input) return 3;
    std::cout << std::setprecision(9);
    static OxygenRiskPipeline pipeline;  // ~16 KB of buffers; keep it off the stack
    uint32_t header[2];
    std::vector<float> pairs;
    while (input.read(reinterpret_cast<char*>(header), sizeof(header))) {
        const uint32_t segment = header[0], count = header[1];
        pairs.resize(2u * count);
        if (count && !input.read(reinterpret_cast<char*>(pairs.data()), pairs.size() * sizeof(float))) return 4;
        pipeline.reset();
        for (uint32_t i = 0; i < count; ++i) {
            const float ir = pairs[2u * i], red = pairs[2u * i + 1];
            if (!std::isfinite(ir) || !std::isfinite(red)) return 5;
            const auto d = pipeline.push_sample(ir, red);
            if (!d.has_value()) continue;
            std::cout << segment << ' ' << d->window_index << ' ' << d->total_samples_ingested << ' '
                      << d->window_start_sample << ' ' << d->window_end_sample << ' '
                      << d->decision_delay_samples << ' ' << d->risk_probability;
            for (float f : d->raw_features) std::cout << ' ' << f;
            std::cout << ' ' << d->is_candidate_risk << ' ' << d->is_confirmed_risk << ' '
                      << static_cast<unsigned>(d->consecutive_risk_windows);
            for (const auto& a : d->top_attributions)
                std::cout << ' ' << static_cast<unsigned>(a.feature_index) << ' ' << a.importance_delta;
            std::cout << ' ' << d->two_wavelength_valid;
            // All 16 signed deltas, recomputed exactly as the on-device XAI loop does
            // (reference 0, or KAN_XAI_REF_Z under -DOXY_XAI_REF_LABEL0).
            float scaled[KAN_INPUT_DIM];
            kan_standardize(d->raw_features, scaled);
            for (int k = 0; k < KAN_INPUT_DIM; ++k) {
                const float saved = scaled[k];
                scaled[k] = KAN_XAI_REFERENCE(k);
                std::cout << ' ' << d->risk_probability - kan_sigmoid(kan_predict_logit(scaled));
                scaled[k] = saved;
            }
            std::cout << '\n';
        }
    }
    return input.eof() ? 0 : 6;
}
