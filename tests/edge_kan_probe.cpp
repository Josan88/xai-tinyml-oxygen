// Host probe for firmware/Arduino Nano 33/include/edge_kan_inference.h.
// Built and run by scripts/train_edge_kan.py parity (zig c++, -ffp-contract=off); not a unit test.
//
// Input file:  uint32 rows, uint32 dims, then rows * dims float32 raw features (little endian).
// Output file: per row, float32
//   [P_incremental, delta_incremental[dims], top3_incremental[3],
//    P_fullpass,    delta_fullpass[dims],    top3_fullpass[3]]
// with the top-3 feature indices stored as float32 (exact for small integers).

#define EKAN_WITH_FULLPASS_REFERENCE
#include "edge_kan_inference.h"

#include <cstdint>
#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: edge_kan_probe <in.bin> <out.bin>\n");
        return 2;
    }
    std::FILE* in = std::fopen(argv[1], "rb");
    if (!in) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
    uint32_t header[2];
    if (std::fread(header, sizeof(uint32_t), 2, in) != 2) { std::fprintf(stderr, "short header\n"); return 2; }
    const uint32_t rows = header[0];
    const uint32_t dims = header[1];
    if (dims != EKAN_INPUT_DIM) {
        std::fprintf(stderr, "input has %u dims, engine expects %d\n", dims, EKAN_INPUT_DIM);
        return 2;
    }
    std::vector<float> raw(static_cast<size_t>(rows) * dims);
    if (std::fread(raw.data(), sizeof(float), raw.size(), in) != raw.size()) { std::fprintf(stderr, "short body\n"); return 2; }
    std::fclose(in);

    const size_t cols = 2 * (1 + EKAN_INPUT_DIM + 3);
    std::vector<float> out(static_cast<size_t>(rows) * cols);
    for (uint32_t r = 0; r < rows; ++r) {
        const float* x = &raw[static_cast<size_t>(r) * dims];
        float* o = &out[static_cast<size_t>(r) * cols];
        EkanXaiAttribution top3[3];

        o[0] = ekan_predict_with_xai(x, top3, o + 1);
        for (int t = 0; t < 3; ++t) o[1 + EKAN_INPUT_DIM + t] = static_cast<float>(top3[t].feature_index);

        float* f = o + 1 + EKAN_INPUT_DIM + 3;
        f[0] = ekan_predict_with_xai_fullpass(x, top3, f + 1);
        for (int t = 0; t < 3; ++t) f[1 + EKAN_INPUT_DIM + t] = static_cast<float>(top3[t].feature_index);
    }

    std::FILE* dst = std::fopen(argv[2], "wb");
    if (!dst) { std::fprintf(stderr, "cannot open %s\n", argv[2]); return 2; }
    std::fwrite(out.data(), sizeof(float), out.size(), dst);
    std::fclose(dst);
    std::fprintf(stderr, "rows=%u dims=%u params=%d\n", rows, dims,
                 EKAN_HIDDEN_DIM * EKAN_INPUT_DIM * EKAN_KNOTS + EKAN_HIDDEN_DIM * EKAN_KNOTS);
    return 0;
}
