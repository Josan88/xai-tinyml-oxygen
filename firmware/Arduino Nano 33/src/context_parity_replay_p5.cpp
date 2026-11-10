#ifdef CONTEXT_PARITY_P5_BUILD
#ifndef OXY_MODEL_P5
#error "CONTEXT_PARITY_P5_BUILD requires OXY_MODEL_P5"
#endif
#include <Arduino.h>
#include <cstdlib>
#include "oxygen_risk_pipeline.h"

// P5 (job F2, ADR-0008) of context_parity_replay_v2.cpp. H identifies; R resets;
// B <500 FP32 hex words> ingests one 250-sample stride as IR/red pairs, one pair per
// sensor record: ir0 red0 ir1 red1 ... ir249 red249. Each decision prints one RESULT line:
//   window total start end delay probability f0..f4 candidate confirmed streak
//   (index delta) x3 beat_valid n_beats s_hat delta0..delta4
// 28 words; floats as FP32 hex. The five deltas are recomputed outside the pipeline exactly as
// its XAI loop computes them (reference KAN_XAI_REF_Z under -DOXY_XAI_REF_LABEL0).
// Stage a complete block before changing pipeline state.
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
    Serial.print(d.beat_valid ? " 1 " : " 0 ");
    Serial.print(static_cast<unsigned>(d.n_beats));
    emit_float(d.s_hat);
    float scaled[KAN_INPUT_DIM];
    kan_standardize(d.raw_features, scaled);
    for (int i = 0; i < KAN_INPUT_DIM; ++i) {
        const float saved = scaled[i];
        scaled[i] = KAN_XAI_REFERENCE(i);
        emit_float(d.risk_probability - kan_sigmoid(kan_predict_logit(scaled)));
        scaled[i] = saved;
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
        // -DOXY_XAI_REF_LABEL0 appends " XAI_REF=LABEL0" (kan_inference.h).
        Serial.println("CONTEXT_PARITY_P5 WINDOW=500 CONTEXT=250 STRIDE=250 N=2 THRESHOLD=0.50 INPUTS=5 PAIRS=IR,RED"
                       KAN_XAI_REF_IDENTITY);
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
    // H gives an on-demand handshake even if startup USB output was missed.
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
