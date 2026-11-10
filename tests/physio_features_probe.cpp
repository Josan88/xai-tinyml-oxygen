// Host probe for include/physio_features.h (job F2): beat features of supplied windows.
// Input file, little-endian: repeated records of
//   uint32 id, float32 ir[500], float32 red[500], float32 dc_ir, float32 dc_red.
// One output line per record:
//   id beat_valid n_beats polarity distance_tie f0..f4 peak0..peak29
// Features print with 17 significant digits (round-trips float64; "nan" when beat-invalid);
// unused peak slots are -1. Desktop only; no pipeline, no filter, no model.
#include <cstdint>
#include <cstdio>
#include <fstream>
#include "physio_features.h"

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    std::ifstream input(argv[1], std::ios::binary);
    if (!input) return 3;
    static physio::PhysioWorkspace ws;
    physio::PhysioResult result;
    uint32_t id;
    float ir[physio::PHYSIO_WINDOW], red[physio::PHYSIO_WINDOW], dc[2];
    while (input.read(reinterpret_cast<char*>(&id), sizeof(id))) {
        if (!input.read(reinterpret_cast<char*>(ir), sizeof(ir)) || !input.read(reinterpret_cast<char*>(red), sizeof(red)) ||
            !input.read(reinterpret_cast<char*>(dc), sizeof(dc)))
            return 4;
        physio::extract(ir, red, dc[0], dc[1], ws, result);
        std::printf("%u %d %d %d %d", id, result.beat_valid ? 1 : 0, result.n_beats, result.polarity, result.distance_tie ? 1 : 0);
        for (double f : result.features) std::printf(" %.17g", f);
        for (int k = 0; k < physio::PHYSIO_MAX_PEAKS; ++k)
            std::printf(" %d", k < result.n_beats ? static_cast<int>(result.peaks[k]) : -1);
        std::printf("\n");
    }
    return input.eof() ? 0 : 5;
}
