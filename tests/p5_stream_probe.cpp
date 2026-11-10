// P5 (job F2, ADR-0008) host probe: streams IR/red pairs through the deployed OxygenRiskPipeline
// (BasicOxygenRiskPipeline<250>) built with -DOXY_MODEL_P5 [-DOXY_XAI_REF_LABEL0].
// Input file, little-endian: repeated segments of
//   uint32 segment_id, uint32 pair_count, pair_count x (float32 ir, float32 red).
// Each segment starts from a reset pipeline. One output line per decision (63 fields):
//   segment window total start end delay probability f0..f4 candidate confirmed streak
//   top_idx0 top_delta0 top_idx1 top_delta1 top_idx2 top_delta2 beat_valid n_beats polarity
//   distance_tie s_hat delta0..delta4 ir_dc red_dc peak0..peak29 (unused peak slots -1)
// Floats print with 9 significant digits, which round-trips FP32 exactly.
// Optional: p5_stream_probe <segments> <dump-list> <dump-out>. dump-list holds "segment window"
// pairs; for each, the C++ FP32 filtered IR and red target windows and the two raw-window DC
// means are written to dump-out as uint32 segment, uint32 window, float32 ir[500], float32
// red[500], float32 ir_dc, float32 red_dc (for the cause analysis of docs/p5-parity-criteria.md).
#ifndef OXY_MODEL_P5
#error "Build with -DOXY_MODEL_P5"
#endif
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <utility>
#include <vector>
#include "oxygen_risk_pipeline.h"
#include "probe_variant.h"

static_assert(KAN_INPUT_DIM == 5, "P5 probe expects the 5-input model");

// Desktop-only read access (the pipeline declares this friend under !ARDUINO).
struct PipelineAuditAccess {
    template <class P> static const physio::PhysioResult& beat(const P& p) { return p.physio_result; }
    template <class P> static const float* ir(const P& p) { return p.filtered_buffer; }
    template <class P> static const float* red(const P& p) { return p.red_filtered_buffer; }
    template <class P> static float ir_dc(const P& p) { return p.ir_raw_mean; }
    template <class P> static float red_dc(const P& p) { return p.red_raw_mean; }
};

int main(int argc, char** argv) {
    if (argc != 2 && argc != 4) return 2;
    announce_pipeline_variant<OxygenRiskPipeline>("p5_stream_probe");
#ifdef OXY_XAI_REF_LABEL0
    std::cerr << "p5_stream_probe: xai_reference=" KAN_XAI_REF_TAG " (KAN_XAI_REF_Z)\n";
#endif
    std::ifstream input(argv[1], std::ios::binary);
    if (!input) return 3;
    std::set<std::pair<uint32_t, uint32_t>> dump_keys;
    std::ofstream dump;
    if (argc == 4) {
        std::ifstream list(argv[2]);
        uint32_t s, w;
        while (list >> s >> w) dump_keys.insert({s, w});
        dump.open(argv[3], std::ios::binary);
        if (!dump) return 7;
    }
    std::cout << std::setprecision(9);
    static OxygenRiskPipeline pipeline;  // ~30 KB of buffers; keep it off the stack
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
            const physio::PhysioResult& beat = PipelineAuditAccess::beat(pipeline);
            std::cout << segment << ' ' << d->window_index << ' ' << d->total_samples_ingested << ' '
                      << d->window_start_sample << ' ' << d->window_end_sample << ' '
                      << d->decision_delay_samples << ' ' << d->risk_probability;
            for (float f : d->raw_features) std::cout << ' ' << f;
            std::cout << ' ' << d->is_candidate_risk << ' ' << d->is_confirmed_risk << ' '
                      << static_cast<unsigned>(d->consecutive_risk_windows);
            for (const auto& a : d->top_attributions)
                std::cout << ' ' << static_cast<unsigned>(a.feature_index) << ' ' << a.importance_delta;
            std::cout << ' ' << d->beat_valid << ' ' << static_cast<unsigned>(d->n_beats) << ' '
                      << static_cast<int>(d->polarity) << ' ' << beat.distance_tie << ' ' << d->s_hat;
            // All five signed deltas, recomputed exactly as the on-device XAI loop does.
            float scaled[KAN_INPUT_DIM];
            kan_standardize(d->raw_features, scaled);
            for (int k = 0; k < KAN_INPUT_DIM; ++k) {
                const float saved = scaled[k];
                scaled[k] = KAN_XAI_REFERENCE(k);
                std::cout << ' ' << d->risk_probability - kan_sigmoid(kan_predict_logit(scaled));
                scaled[k] = saved;
            }
            std::cout << ' ' << PipelineAuditAccess::ir_dc(pipeline) << ' ' << PipelineAuditAccess::red_dc(pipeline);
            for (int k = 0; k < physio::PHYSIO_MAX_PEAKS; ++k)
                std::cout << ' ' << (k < beat.n_beats ? static_cast<int>(beat.peaks[k]) : -1);
            std::cout << '\n';
            if (dump_keys.count({segment, d->window_index})) {
                const float dc[2] = {PipelineAuditAccess::ir_dc(pipeline), PipelineAuditAccess::red_dc(pipeline)};
                const uint32_t key[2] = {segment, d->window_index};
                dump.write(reinterpret_cast<const char*>(key), sizeof(key));
                dump.write(reinterpret_cast<const char*>(PipelineAuditAccess::ir(pipeline)), PPG_WINDOW_SIZE * sizeof(float));
                dump.write(reinterpret_cast<const char*>(PipelineAuditAccess::red(pipeline)), PPG_WINDOW_SIZE * sizeof(float));
                dump.write(reinterpret_cast<const char*>(dc), sizeof(dc));
            }
        }
    }
    return input.eof() ? 0 : 6;
}
