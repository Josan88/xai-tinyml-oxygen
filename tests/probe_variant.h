// Desktop-only: announce on stderr which pipeline variant a probe runs, so probe
// output is never mistaken for device behaviour. stdout stays machine-parsed.
#pragma once
#include <iostream>
#include "oxygen_risk_pipeline.h"

template<class Pipeline>
void announce_pipeline_variant(const char* probe) {
    if (Pipeline::CONTEXT_SAMPLES == 0) {
        std::cerr << probe << ": pipeline=WindowLocalOxygenRiskPipeline (BasicOxygenRiskPipeline<0>); "
                  << "window-local filtering, NOT the deployed device path\n";
    } else {
        std::cerr << probe << ": pipeline=OxygenRiskPipeline (BasicOxygenRiskPipeline<"
                  << Pipeline::CONTEXT_SAMPLES << ">); deployed device path\n";
    }
}
