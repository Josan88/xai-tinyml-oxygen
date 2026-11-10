// Auto-generated KAN Inference Engine
// Matches PyTorch / NumPy implementation of KANBinaryClassifier

#pragma once

#ifdef ARDUINO
#include <Arduino.h>
#else
#include <cmath>
#include <cstring>
#include <cstdint>
#endif
#if defined(OXY_MODEL_P5) && defined(OXY_MODEL_V2)
#error "OXY_MODEL_P5 and OXY_MODEL_V2 are separate models; define one"
#endif
#if defined(OXY_MODEL_P5)
// P5 (job F2, ADR-0008): 5 beat inputs, kan_model_weights_p5.h. Never the default.
#include "kan_model_weights_p5.h"
#elif defined(OXY_MODEL_V2)
#include "kan_model_weights_v2.h"
#else
#include "kan_model_weights.h"
#endif
#include <math.h>

// Mean-Perturbation reference: the value a standardized input is set to in each XAI pass.
// Shipped images: 0, the training mean. -DOXY_XAI_REF_LABEL0: KAN_XAI_REF_Z[i], the label-0
// median from the model's reference header (scripts/make_xai_reference_header.py). Same
// passes, ranking, top-3 and debounce; only the constant differs. Without the flag these
// macros expand to exactly the old tokens.
#ifdef OXY_XAI_REF_LABEL0
#if defined(OXY_MODEL_P5)
#include "kan_xai_reference_p5.h"
#elif defined(OXY_MODEL_V2)
#include "kan_xai_reference_v2.h"
#else
#error "OXY_XAI_REF_LABEL0: no label-0 reference header for this model"
#endif
#define KAN_XAI_REFERENCE(i) (KAN_XAI_REF_Z[(i)])
#define KAN_XAI_REF_IDENTITY " XAI_REF=LABEL0"
#else
#define KAN_XAI_REFERENCE(i) 0.0f
#define KAN_XAI_REF_IDENTITY ""
#endif

// Feature names for XAI reporting, in model input order
#if defined(OXY_MODEL_P5)
const char* const KAN_FEATURE_NAMES[KAN_INPUT_DIM] = {
    "r_beat", "pi_beat", "hr_bpm", "sqi_template", "sqi_ibi_ratio"
};
#else
const char* const KAN_FEATURE_NAMES[KAN_INPUT_DIM] = {
    "mean", "std", "minimum", "maximum", "value_range", "rms", "skewness", "kurtosis",
    "energy", "derivative_std", "dominant_frequency_hz", "bandpower_0_5_2_hz",
    "bandpower_2_5_hz", "bandpower_ratio_0_5_2_to_2_5"
#ifdef OXY_MODEL_V2
    , "ratio_of_ratios", "ir_perfusion_index"
#endif
};
#endif

// Standardize features using StandardScaler parameters
void kan_standardize(const float* raw_features, float* scaled_features) {
    for (int i = 0; i < KAN_INPUT_DIM; ++i) {
        scaled_features[i] = (raw_features[i] - SCALER_MEAN[i]) / (SCALER_STD[i] + 1e-10f);
    }
}

// Forward pass for KAN Layer 1: 14 -> 12
void kan_layer1_forward(const float* scaled_features, float* output_12) {
    for (int j = 0; j < KAN_HIDDEN_DIM; ++j) {
        // 1. Linear combination: u_j = sum(W_jk * x_k) + b_j
        float u_j = KAN1_B[j];
        for (int k = 0; k < KAN_INPUT_DIM; ++k) {
            u_j += KAN1_W[j][k] * scaled_features[k];
        }
        
        // 2. Spline evaluation: out_j = sum(basis_k * theta_jk)
        float spline_out = 0.0f;
        
        // Clamp input to spline range [-3.0, 3.0]
        float u_clamped = u_j;
        if (u_clamped < -3.0f) u_clamped = -3.0f;
        if (u_clamped > 3.0f) u_clamped = 3.0f;
        
        for (int k = 0; k < KAN_KNOTS; ++k) {
            float dist = fabsf((u_clamped - KAN1_CENTERS[k]) / (KAN1_STEP + 1e-10f));
            float basis_val = 0.0f;
            
            if (dist < 1.0f) {
                basis_val = (2.0f / 3.0f) - dist * dist + 0.5f * dist * dist * dist;
            } else if (dist < 2.0f) {
                float term = 2.0f - dist;
                basis_val = (term * term * term) / 6.0f;
            }
            // If dist >= 2.0f, basis_val remains 0.0f
            
            spline_out += basis_val * KAN1_THETA[j][k];
        }
        output_12[j] = spline_out;
    }
}

// Forward pass for KAN Layer 2: 12 -> 1
float kan_layer2_forward(const float* input_12) {
    // Hidden dim to 1 output
    float u_0 = KAN2_B[0];
    for (int k = 0; k < KAN_HIDDEN_DIM; ++k) {
        u_0 += KAN2_W[0][k] * input_12[k];
    }
    
    float spline_out = 0.0f;
    
    // Clamp input to spline range [-3.0, 3.0]
    float u_clamped = u_0;
    if (u_clamped < -3.0f) u_clamped = -3.0f;
    if (u_clamped > 3.0f) u_clamped = 3.0f;
    
    for (int k = 0; k < KAN_KNOTS; ++k) {
        float dist = fabsf((u_clamped - KAN2_CENTERS[k]) / (KAN2_STEP + 1e-10f));
        float basis_val = 0.0f;
        
        if (dist < 1.0f) {
            basis_val = (2.0f / 3.0f) - dist * dist + 0.5f * dist * dist * dist;
        } else if (dist < 2.0f) {
            float term = 2.0f - dist;
            basis_val = (term * term * term) / 6.0f;
        }
        // If dist >= 2.0f, basis_val remains 0.0f
        
        spline_out += basis_val * KAN2_THETA[0][k];
    }
    return spline_out;
}

// Full forward pass on standardized features, returning logit
float kan_predict_logit(const float* scaled_features) {
    float hidden[KAN_HIDDEN_DIM];
    kan_layer1_forward(scaled_features, hidden);
    return kan_layer2_forward(hidden);
}

// Helper to compute sigmoid probability from logit
float kan_sigmoid(float logit) {
    return 1.0f / (1.0f + expf(-logit));
}

// Predict probability directly from raw features
float kan_predict_probability(const float* raw_features) {
    float scaled[KAN_INPUT_DIM];
    kan_standardize(raw_features, scaled);
    float logit = kan_predict_logit(scaled);
    return kan_sigmoid(logit);
}

// Structure to hold XAI attribution results
struct KanXaiAttribution {
    const char* feature_name;
    float attribution_score;
};

// Simple bubble sort to sort attributions descending
void kan_sort_attributions(KanXaiAttribution* attrs, int count) {
    for (int i = 0; i < count - 1; ++i) {
        for (int j = 0; j < count - i - 1; ++j) {
            if (fabsf(attrs[j].attribution_score) < fabsf(attrs[j+1].attribution_score)) {
                KanXaiAttribution temp = attrs[j];
                attrs[j] = attrs[j+1];
                attrs[j+1] = temp;
            }
        }
    }
}

// Computes KAN predictions and feature attributions using the mean-perturbation method.
// Returns risk probability (0 to 1). Fills `top_features` with top-3 features and their raw attribution scores.
float kan_predict_with_xai(const float* raw_features, KanXaiAttribution* top_features_3) {
    float scaled[KAN_INPUT_DIM];
    kan_standardize(raw_features, scaled);
    
    // 1. Get baseline logit and probability
    float baseline_logit = kan_predict_logit(scaled);
    float baseline_prob = kan_sigmoid(baseline_logit);
    
    // 2. Perturbation loop: zero out one feature at a time (setting to dataset mean)
    KanXaiAttribution all_attributions[KAN_INPUT_DIM];
    float perturbed_scaled[KAN_INPUT_DIM];
    
    for (int i = 0; i < KAN_INPUT_DIM; ++i) {
        // Copy scaled features
        memcpy(perturbed_scaled, scaled, sizeof(scaled));
        
        // Zero out feature i (replaces it with training mean; label-0 median under OXY_XAI_REF_LABEL0)
        perturbed_scaled[i] = KAN_XAI_REFERENCE(i);
        
        // Forward pass
        float perturbed_logit = kan_predict_logit(perturbed_scaled);
        float perturbed_prob = kan_sigmoid(perturbed_logit);
        
        // Attribution = baseline - perturbed probability
        // If probability drops when feature is zeroed, attribution is positive (feature drove risk up)
        // If probability rises when feature is zeroed, attribution is negative (feature drove risk down)
        all_attributions[i].feature_name = KAN_FEATURE_NAMES[i];
        all_attributions[i].attribution_score = baseline_prob - perturbed_prob;
    }
    
    // 3. Sort by absolute attribution score to find top drivers
    kan_sort_attributions(all_attributions, KAN_INPUT_DIM);
    
    // 4. Copy top 3
    for (int i = 0; i < 3; ++i) {
        top_features_3[i] = all_attributions[i];
    }
    
    return baseline_prob;
}
