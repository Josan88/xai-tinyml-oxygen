#ifdef CONTEXT_PARITY_EKAN_BUILD
#ifndef OXY_MODEL_V2
#error "CONTEXT_PARITY_EKAN_BUILD requires OXY_MODEL_V2"
#endif
#include <Arduino.h>
#include <cstdlib>
#include "oxygen_risk_pipeline.h"
#include "edge_kan_inference.h"

// Edge-KAN board parity (docs/edge-kan-board-preregistration.md). Diagnostic only.
// The shipped v2 r2 pipeline runs unchanged on replayed IR/red blocks; on each decision the
// edge KAN (ekan16) is evaluated on the same raw_features. Protocol as context_parity_replay_v2.cpp
// (H, R, B <500 FP32 hex words>), without A: RESULT carries the v2 words, then
// edge P, the 16 edge deltas and the 3 edge top-3 indices.
static_assert(EKAN_INPUT_DIM == KAN_INPUT_DIM, "edge KAN and v2 pipeline input counts differ");
static const unsigned BLOCK_PAIRS = 250;
static const unsigned BLOCK_WORDS = 2 * BLOCK_PAIRS;
static OxygenRiskPipeline pipeline;
static float block_ir[BLOCK_PAIRS];
static float block_red[BLOCK_PAIRS];
static char token[9];
static unsigned token_size = 0, count = 0;
static char command = 0;
static bool invalid = false;

static void emit_float(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    Serial.print(' ');
    Serial.print(bits, HEX);
}

static void emit_decision(const RiskDecision& d) {
    EkanXaiAttribution edge_top[3];
    float edge_deltas[EKAN_INPUT_DIM];
    const float edge_p = ekan_predict_with_xai(d.raw_features, edge_top, edge_deltas);
    Serial.print("RESULT ");
    Serial.print(d.window_index); Serial.print(' ');
    Serial.print(static_cast<unsigned long>(d.total_samples_ingested)); Serial.print(' ');
    Serial.print(static_cast<unsigned long>(d.window_start_sample)); Serial.print(' ');
    Serial.print(static_cast<unsigned long>(d.window_end_sample)); Serial.print(' ');
    Serial.print(d.decision_delay_samples);
    emit_float(d.risk_probability);
    for (float feature : d.raw_features) emit_float(feature);
    Serial.print(d.is_candidate_risk ? " 1" : " 0");
    Serial.print(d.is_confirmed_risk ? " 1 " : " 0 ");
    Serial.print(static_cast<unsigned>(d.consecutive_risk_windows));
    for (const auto& attribution : d.top_attributions) {
        Serial.print(' '); Serial.print(attribution.feature_index);
        emit_float(attribution.importance_delta);
    }
    Serial.print(d.two_wavelength_valid ? " 1" : " 0");
    emit_float(edge_p);
    for (float delta : edge_deltas) emit_float(delta);
    for (const auto& attribution : edge_top) {
        Serial.print(' '); Serial.print(attribution.feature_index);
    }
    Serial.println();
}

static void accept_token() {
    if (!token_size) return;
    if (command != 'B' || token_size != 8 || count >= BLOCK_WORDS) invalid = true;
    if (!invalid) {
        token[8] = '\0';
        const uint32_t bits = strtoul(token, nullptr, 16);
        float value;
        memcpy(&value, &bits, sizeof(bits));
        if (!isfinite(value)) invalid = true;
        else {
            if (count % 2 == 0) block_ir[count / 2] = value;
            else block_red[count / 2] = value;
            ++count;
        }
    }
    token_size = 0;
}

static void finish_line() {
    accept_token();
    if (invalid || (command == 'B' && count != BLOCK_WORDS) ||
        (command != 'B' && command != 'H' && command != 'R')) {
        Serial.println("ERR invalid_command");
    } else if (command == 'H') {
        Serial.println("CONTEXT_PARITY_EKAN WINDOW=500 CONTEXT=250 STRIDE=250 N=2 THRESHOLD=0.50 INPUTS=16 PAIRS=IR,RED MODEL=EKAN16"
#ifdef EKAN_XAI_REF_LABEL0
                       " XAI_REF=LABEL0"
#endif
                       );
    } else if (command == 'R') {
        pipeline.reset();
        Serial.println("RESET 0");
    } else {
        bool emitted = false;
        for (unsigned i = 0; i < BLOCK_PAIRS; ++i) {
            const auto decision = pipeline.push_sample(block_ir[i], block_red[i]);
            if (decision.has_value()) {
                emit_decision(*decision);
                emitted = true;
            }
        }
        if (!emitted) {
            Serial.print("BUFFER ");
            Serial.println(static_cast<unsigned long>(pipeline.get_total_samples()));
        }
    }
    command = 0;
    invalid = false;
    token_size = count = 0;
}

void setup() {
    Serial.begin(115200);
}

void loop() {
    while (Serial.available()) {
        const char c = static_cast<char>(Serial.read());
        if (c == '\n') finish_line();
        else if (!command) {
            if (c != ' ' && c != '\r') command = c;
        } else if (c == ' ' || c == '\r') accept_token();
        else if ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
            if (token_size < 8) token[token_size++] = c;
            else invalid = true;
        } else invalid = true;
    }
}
#endif
