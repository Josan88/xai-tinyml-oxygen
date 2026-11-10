#ifdef PARITY_REPLAY_BUILD
#include <Arduino.h>
#include <cstdlib>
#include "oxygen_risk_pipeline.h"

// Diagnostic protocol: one line of 500 eight-digit FP32 hex words per window.
// Each window starts with a reset; this tests arithmetic, not consecutive debounce.
static BasicOxygenRiskPipeline<0> pipeline;
static RiskDecision decision;
static char token[9];
static unsigned token_size = 0;
static unsigned sample_count = 0;
static bool invalid = false;

static void emit_float(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    Serial.print(' ');
    Serial.print(bits, HEX);
}

static void accept_token() {
    if (!token_size) return;
    if (token_size != 8 || sample_count >= PPG_WINDOW_SIZE) {
        invalid = true;
    } else if (!invalid) {
        token[8] = '\0';
        const uint32_t bits = strtoul(token, nullptr, 16);
        float value;
        memcpy(&value, &bits, sizeof(value));
        if (!isfinite(value)) {
            invalid = true;
        } else {
            const auto result = pipeline.push_sample(value);
            if (result.has_value()) decision = *result;
            ++sample_count;
        }
    }
    token_size = 0;
}

static void finish_window() {
    accept_token();
    if (invalid || sample_count != PPG_WINDOW_SIZE) {
        Serial.println("ERR invalid_window");
    } else {
        Serial.print("RESULT");
        emit_float(decision.risk_probability);
        for (float value : decision.raw_features) emit_float(value);
        float scaled[KAN_INPUT_DIM];
        kan_standardize(decision.raw_features, scaled);
        for (float value : scaled) emit_float(value);
        for (int i = 0; i < KAN_INPUT_DIM; ++i) {
            const float saved = scaled[i];
            scaled[i] = KAN_XAI_REFERENCE(i);
            emit_float(decision.risk_probability - kan_sigmoid(kan_predict_logit(scaled)));
            scaled[i] = saved;
        }
        Serial.print(decision.is_candidate_risk ? " 1" : " 0");
        Serial.println(decision.is_confirmed_risk ? " 1" : " 0");
    }
    pipeline.reset();
    sample_count = token_size = 0;
    invalid = false;
}

void setup() {
    Serial.begin(115200);
    while (!Serial) delay(10);
    Serial.println("PARITY_READY_V1");
}

void loop() {
    while (Serial.available()) {
        const char c = static_cast<char>(Serial.read());
        if (c == '\n') finish_window();
        else if (c == ' ' || c == '\r') accept_token();
        else if ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
            if (token_size < 8) token[token_size++] = c;
            else invalid = true;
        } else invalid = true;
    }
}
#endif
