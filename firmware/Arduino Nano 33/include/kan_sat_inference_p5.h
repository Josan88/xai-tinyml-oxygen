#pragma once
// Forward pass of the P5 concept regressor (job F2, ADR-0008): estimated ScalcO2, s_hat, the model
// saturation estimate. Displayed only; it never feeds the risk score or the explanation. Not an
// oxygen reading; uncalibrated for the MAX30102.
//
// The arithmetic of kan_layer1_forward / kan_layer2_forward (kan_inference.h), operation for
// operation, on the KANSAT_* constants of kan_sat_regressor_p5.h, with a linear output: the
// returned value is the regressor output before the target scaling, not a logit.
#ifndef OXY_MODEL_P5
#error "kan_sat_inference_p5.h is for -DOXY_MODEL_P5 builds"
#endif
#include "kan_sat_regressor_p5.h"

inline float kan_sat_output(const float* scaled_features) {
    float hidden[KANSAT_HIDDEN_DIM];
    for (int j = 0; j < KANSAT_HIDDEN_DIM; ++j) {
        float u_j = KANSAT1_B[j];
        for (int k = 0; k < KAN_INPUT_DIM; ++k) {
            u_j += KANSAT1_W[j][k] * scaled_features[k];
        }
        float spline_out = 0.0f;
        float u_clamped = u_j;
        if (u_clamped < -3.0f) u_clamped = -3.0f;
        if (u_clamped > 3.0f) u_clamped = 3.0f;
        for (int k = 0; k < KANSAT_KNOTS; ++k) {
            float dist = fabsf((u_clamped - KANSAT1_CENTERS[k]) / (KANSAT1_STEP + 1e-10f));
            float basis_val = 0.0f;
            if (dist < 1.0f) {
                basis_val = (2.0f / 3.0f) - dist * dist + 0.5f * dist * dist * dist;
            } else if (dist < 2.0f) {
                float term = 2.0f - dist;
                basis_val = (term * term * term) / 6.0f;
            }
            spline_out += basis_val * KANSAT1_THETA[j][k];
        }
        hidden[j] = spline_out;
    }

    float u_0 = KANSAT2_B[0];
    for (int k = 0; k < KANSAT_HIDDEN_DIM; ++k) {
        u_0 += KANSAT2_W[0][k] * hidden[k];
    }
    float spline_out = 0.0f;
    float u_clamped = u_0;
    if (u_clamped < -3.0f) u_clamped = -3.0f;
    if (u_clamped > 3.0f) u_clamped = 3.0f;
    for (int k = 0; k < KANSAT_KNOTS; ++k) {
        float dist = fabsf((u_clamped - KANSAT2_CENTERS[k]) / (KANSAT2_STEP + 1e-10f));
        float basis_val = 0.0f;
        if (dist < 1.0f) {
            basis_val = (2.0f / 3.0f) - dist * dist + 0.5f * dist * dist * dist;
        } else if (dist < 2.0f) {
            float term = 2.0f - dist;
            basis_val = (term * term * term) / 6.0f;
        }
        spline_out += basis_val * KANSAT2_THETA[0][k];
    }
    return spline_out;
}

// s_hat from standardized P5 inputs: output * target std + target mean, in FP32. Not clipped.
inline float kan_sat_estimate(const float* scaled_features) {
    return kan_sat_output(scaled_features) * KANSAT_TARGET_STD + KANSAT_TARGET_MEAN;
}
