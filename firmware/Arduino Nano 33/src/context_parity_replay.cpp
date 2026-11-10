#ifdef CONTEXT_PARITY_BUILD
#include <Arduino.h>
#include <cstdlib>
#include "oxygen_risk_pipeline.h"

// H identifies; R resets; B <250 FP32 hex words> ingests a stride.
// Stage a complete block before changing pipeline state.
static OxygenRiskPipeline pipeline;
static float block[250];
static char token[9];
static unsigned token_size = 0, count = 0;
static char command = 0;
static bool invalid = false;
// Diagnostic-only observation of all mean-perturbation deltas. The deployed
// pipeline and its top-three payload remain unchanged.
static float last_deltas[KAN_INPUT_DIM];
static uint32_t last_window = 0;
static bool have_deltas = false;

static void emit_float(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    Serial.print(' ');
    Serial.print(bits, HEX);
}

static void emit_decision(const RiskDecision& d) {
    float scaled[KAN_INPUT_DIM];
    kan_standardize(d.raw_features, scaled);
    for (int i = 0; i < KAN_INPUT_DIM; ++i) {
        const float saved = scaled[i];
        scaled[i] = KAN_XAI_REFERENCE(i);
        last_deltas[i] = d.risk_probability - kan_sigmoid(kan_predict_logit(scaled));
        scaled[i] = saved;
    }
    last_window = d.window_index;
    have_deltas = true;
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
    Serial.println();
}

static void accept_token() {
    if (!token_size) return;
    if (command != 'B' || token_size != 8 || count >= 250) invalid = true;
    if (!invalid) {
        token[8] = '\0';
        const uint32_t bits = strtoul(token, nullptr, 16);
        memcpy(&block[count], &bits, sizeof(bits));
        if (!isfinite(block[count])) invalid = true;
        else ++count;
    }
    token_size = 0;
}

static void finish_line() {
    accept_token();
    if (invalid || (command == 'B' && count != 250) ||
        (command != 'B' && command != 'H' && command != 'R' && command != 'A')) {
        Serial.println("ERR invalid_command");
    } else if (command == 'H') {
        Serial.println("CONTEXT_PARITY_V1 WINDOW=500 CONTEXT=250 STRIDE=250 N=2 THRESHOLD=0.50");
    } else if (command == 'R') {
        pipeline.reset();
        have_deltas = false;
        Serial.println("RESET 0");
    } else if (command == 'A') {
        if (!have_deltas) Serial.println("ERR no_decision");
        else {
            Serial.print("ATTR14 "); Serial.print(last_window);
            for (float delta : last_deltas) emit_float(delta);
            Serial.println();
        }
    } else {
        bool emitted = false;
        for (float value : block) {
            const auto decision = pipeline.push_sample(value);
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
