// Edge KAN inference engine (spline on every edge, no W, no bias).
//
//   layer:  out[j] = sum_i sum_k B_k(clamp(x[i], EKAN_RANGE_MIN, EKAN_RANGE_MAX)) * THETA[j][i][k]
//   model:  z = standardize(raw);  h = layer1(z);  logit = layer2(h)[0];  P = sigmoid(logit)
//
// B_k is the TinyKAN cubic B-spline basis (kan_inference.h): centred bumps at EKAN_CENTERS,
// spacing EKAN_STEP, same FP32 expressions. Weights come from scripts/train_edge_kan.py.
// Pre-registration: docs/edge-kan-preregistration.md.
//
// NOT included by any firmware translation unit. The shipped images use kan_inference.h.
// Host checks: tests/edge_kan_probe.cpp through scripts/train_edge_kan.py parity.
//
// Weights header: edge_kan_weights_v2.h (16 v2 inputs) by default, or
// edge_kan_weights_ir14.h (14 IR inputs) with -DEKAN_WEIGHTS_IR14.

#pragma once

#ifdef ARDUINO
#include <Arduino.h>
#else
#include <cmath>
#include <cstring>
#endif
#include <math.h>

#if defined(EKAN_WEIGHTS_IR14)
#include "edge_kan_weights_ir14.h"
#else
#include "edge_kan_weights_v2.h"
#endif

// Mean-perturbation reference in standardized units: 0 is the training mean, as on the
// shipped images. A build may define its own (for example a label-0 median table).
// -DEKAN_XAI_REF_LABEL0: the label-0 median (edge_kan_xai_reference.h,
// docs/edge-kan-label0-board-preregistration.md). Under that flag nothing may fall back to 0.
#ifdef EKAN_XAI_REF_LABEL0
#ifdef EKAN_XAI_REFERENCE
#error "EKAN_XAI_REF_LABEL0: EKAN_XAI_REFERENCE was defined before edge_kan_inference.h"
#endif
#include "edge_kan_xai_reference.h"
#if !defined(EKAN_XAI_REFERENCE) || !defined(EKAN_XAI_REF_TAG)
#error "EKAN_XAI_REF_LABEL0: edge_kan_xai_reference.h did not define the label-0 reference"
#endif
#endif
#ifndef EKAN_XAI_REFERENCE
#define EKAN_XAI_REFERENCE(i) 0.0f
#endif

// All EKAN_KNOTS basis values of one scalar input (zeros included; at most four are nonzero).
static inline void ekan_basis(float x, float* basis) {
    if (x < EKAN_RANGE_MIN) x = EKAN_RANGE_MIN;
    if (x > EKAN_RANGE_MAX) x = EKAN_RANGE_MAX;
    for (int k = 0; k < EKAN_KNOTS; ++k) {
        float dist = fabsf((x - EKAN_CENTERS[k]) / (EKAN_STEP + 1e-10f));
        float value = 0.0f;
        if (dist < 1.0f) {
            value = (2.0f / 3.0f) - dist * dist + 0.5f * dist * dist * dist;
        } else if (dist < 2.0f) {
            float term = 2.0f - dist;
            value = (term * term * term) / 6.0f;
        }
        basis[k] = value;
    }
}

static inline void ekan_standardize(const float* raw, float* z) {
    for (int i = 0; i < EKAN_INPUT_DIM; ++i) {
        z[i] = (raw[i] - EKAN_SCALER_MEAN[i]) / (EKAN_SCALER_STD[i] + 1e-10f);
    }
}

// Layer 1, EKAN_INPUT_DIM -> EKAN_HIDDEN_DIM. The basis of each input is evaluated once and
// shared by every output: EKAN_INPUT_DIM * EKAN_KNOTS basis values, then
// EKAN_HIDDEN_DIM * EKAN_INPUT_DIM * EKAN_KNOTS multiply-adds.
static inline void ekan_layer1(const float* z, float* hidden) {
    float basis[EKAN_KNOTS];
    for (int j = 0; j < EKAN_HIDDEN_DIM; ++j) hidden[j] = 0.0f;
    for (int i = 0; i < EKAN_INPUT_DIM; ++i) {
        ekan_basis(z[i], basis);
        for (int j = 0; j < EKAN_HIDDEN_DIM; ++j) {
            const float* theta = EKAN1_THETA[j][i];
            float acc = 0.0f;
            for (int k = 0; k < EKAN_KNOTS; ++k) acc += basis[k] * theta[k];
            hidden[j] += acc;
        }
    }
}

// Layer 2, EKAN_HIDDEN_DIM -> 1: EKAN_HIDDEN_DIM * EKAN_KNOTS basis values and multiply-adds.
static inline float ekan_layer2(const float* hidden) {
    float basis[EKAN_KNOTS];
    float logit = 0.0f;
    for (int i = 0; i < EKAN_HIDDEN_DIM; ++i) {
        ekan_basis(hidden[i], basis);
        const float* theta = EKAN2_THETA[0][i];
        float acc = 0.0f;
        for (int k = 0; k < EKAN_KNOTS; ++k) acc += basis[k] * theta[k];
        logit += acc;
    }
    return logit;
}

static inline float ekan_sigmoid(float logit) {
    return 1.0f / (1.0f + expf(-logit));
}

static inline float ekan_logit_standardized(const float* z) {
    float hidden[EKAN_HIDDEN_DIM];
    ekan_layer1(z, hidden);
    return ekan_layer2(hidden);
}

static inline float ekan_predict_probability(const float* raw) {
    float z[EKAN_INPUT_DIM];
    ekan_standardize(raw, z);
    return ekan_sigmoid(ekan_logit_standardized(z));
}

struct EkanXaiAttribution {
    int feature_index;
    const char* feature_name;
    float attribution_score;
};

// Descending |score|; stable, so ties keep input order (as kan_sort_attributions).
static inline void ekan_sort_attributions(EkanXaiAttribution* attrs, int count) {
    for (int i = 0; i < count - 1; ++i) {
        for (int j = 0; j < count - i - 1; ++j) {
            if (fabsf(attrs[j].attribution_score) < fabsf(attrs[j + 1].attribution_score)) {
                EkanXaiAttribution temp = attrs[j];
                attrs[j] = attrs[j + 1];
                attrs[j + 1] = temp;
            }
        }
    }
}

static inline void ekan_rank_top3(const float* deltas, EkanXaiAttribution* top3) {
    EkanXaiAttribution all[EKAN_INPUT_DIM];
    for (int i = 0; i < EKAN_INPUT_DIM; ++i) {
        all[i].feature_index = i;
        all[i].feature_name = EKAN_FEATURE_NAMES[i];
        all[i].attribution_score = deltas[i];
    }
    ekan_sort_attributions(all, EKAN_INPUT_DIM);
    for (int r = 0; r < 3; ++r) top3[r] = all[r];
}

// Mean-perturbation attribution, delta_i = P(z) - P(z with z_i = EKAN_XAI_REFERENCE(i)),
// computed incrementally. Layer 1 is a sum of per-edge functions, so replacing z_i moves hidden
// unit j by sum_k (B_k(ref_i) - B_k(z_i)) * EKAN1_THETA[j][i][k] and leaves every other edge
// unchanged. Each perturbed pass therefore costs 2 * EKAN_KNOTS input basis values,
// EKAN_HIDDEN_DIM * EKAN_KNOTS multiply-adds, and one layer 2, instead of a full forward pass.
// Same quantity as ekan_predict_with_xai_fullpass up to FP32 rounding.
// Fills top3 (by |delta|, ties by input order); writes all deltas to all_deltas if not null.
static inline float ekan_predict_with_xai(const float* raw, EkanXaiAttribution* top3, float* all_deltas) {
    float z[EKAN_INPUT_DIM];
    float hidden[EKAN_HIDDEN_DIM];
    float perturbed[EKAN_HIDDEN_DIM];
    float deltas[EKAN_INPUT_DIM];
    float basis_x[EKAN_KNOTS];
    float basis_r[EKAN_KNOTS];

    ekan_standardize(raw, z);
    ekan_layer1(z, hidden);
    const float probability = ekan_sigmoid(ekan_layer2(hidden));

    for (int i = 0; i < EKAN_INPUT_DIM; ++i) {
        const float reference = EKAN_XAI_REFERENCE(i);
        if (z[i] == reference) {
            deltas[i] = 0.0f;
            continue;
        }
        ekan_basis(z[i], basis_x);
        ekan_basis(reference, basis_r);
        for (int k = 0; k < EKAN_KNOTS; ++k) basis_r[k] -= basis_x[k];
        for (int j = 0; j < EKAN_HIDDEN_DIM; ++j) {
            const float* theta = EKAN1_THETA[j][i];
            float acc = 0.0f;
            for (int k = 0; k < EKAN_KNOTS; ++k) acc += basis_r[k] * theta[k];
            perturbed[j] = hidden[j] + acc;
        }
        deltas[i] = probability - ekan_sigmoid(ekan_layer2(perturbed));
    }

    ekan_rank_top3(deltas, top3);
    if (all_deltas) {
        for (int i = 0; i < EKAN_INPUT_DIM; ++i) all_deltas[i] = deltas[i];
    }
    return probability;
}

#ifdef EKAN_WITH_FULLPASS_REFERENCE
// Reference only (host probe): the shipped firmware's method, one full forward pass per input.
static inline float ekan_predict_with_xai_fullpass(const float* raw, EkanXaiAttribution* top3, float* all_deltas) {
    float z[EKAN_INPUT_DIM];
    float changed[EKAN_INPUT_DIM];
    float deltas[EKAN_INPUT_DIM];
    ekan_standardize(raw, z);
    const float probability = ekan_sigmoid(ekan_logit_standardized(z));
    for (int i = 0; i < EKAN_INPUT_DIM; ++i) {
        memcpy(changed, z, sizeof(z));
        changed[i] = EKAN_XAI_REFERENCE(i);
        deltas[i] = probability - ekan_sigmoid(ekan_logit_standardized(changed));
    }
    ekan_rank_top3(deltas, top3);
    if (all_deltas) {
        for (int i = 0; i < EKAN_INPUT_DIM; ++i) all_deltas[i] = deltas[i];
    }
    return probability;
}
#endif
