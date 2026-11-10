#ifdef CONTEXT_TIMING_V2_BUILD
#ifndef OXY_MODEL_V2
#error "CONTEXT_TIMING_V2_BUILD requires OXY_MODEL_V2"
#endif
#include <Arduino.h>
#include <cstdlib>
#include <malloc.h>
#include "cmsis_os2.h"
#include "rtx_os.h"
#include "timing_pipeline_v2.h"

// v2 (ADR-0006) of context_timing_replay.cpp. B carries 500 FP32 hex words:
// 250 IR/red pairs, one pair per sensor record (ir0 red0 ... ir249 red249).
static const unsigned BLOCK_PAIRS = 250;
static const unsigned BLOCK_WORDS = 2 * BLOCK_PAIRS;
static TimedOxygenRiskPipeline pipeline;
static float block_ir[BLOCK_PAIRS];
static float block_red[BLOCK_PAIRS];
static char token[9];
static unsigned token_size = 0, count = 0;
static char command = 0;
static bool invalid = false, timing_ready = false;

// Nano 33 BLE D2 is P1.11 in the installed Arduino variant. Direct GPIO
// avoids lazy DigitalInOut allocation. The pin brackets both counter reads;
// CYCCNT excludes the GPIO stores. Neither measurement subtracts overhead.
struct BoardClock {
    uint32_t begin() {
        NRF_P1->OUTSET = 1UL << 11;
        __DSB(); __ISB();
        return DWT->CYCCNT;
    }
    uint32_t end() {
        __DSB(); __ISB();
        const uint32_t stop = DWT->CYCCNT;
        NRF_P1->OUTCLR = 1UL << 11;
        return stop;
    }
};
static BoardClock cycle_clock;

static void emit_float(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    Serial.print(' '); Serial.print(bits, HEX);
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
    Serial.print(d.two_wavelength_valid ? " 1" : " 0");
    Serial.println();
}

static void emit_memory() {
    // Current Arduino loop thread only. Watermark is since thread creation,
    // including startup/USB/reporting; pipeline reset does not reset it.
    const auto thread = osThreadGetId();
    const uint32_t reserved = osThreadGetStackSize(thread);
    int32_t peak_used = -1;
    if (reserved && (osRtxConfig.flags & osRtxConfigStackWatermark)) {
        const uint32_t free_watermark = osThreadGetStackSpace(thread);
        if (free_watermark <= reserved) peak_used = reserved - free_watermark;
    }
    const struct mallinfo heap = mallinfo();
    Serial.print("MEMORY "); Serial.print(reserved); Serial.print(' ');
    Serial.print(peak_used); Serial.print(' ');
    Serial.print(heap.arena); Serial.print(' ');
    Serial.print(heap.uordblks); Serial.print(' ');
    Serial.println(heap.fordblks);
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
        (command != 'B' && command != 'H' && command != 'R' && command != 'C' && command != 'M')) {
        Serial.println("ERR invalid_command");
    } else if (!timing_ready) {
        Serial.println("ERR timing_unavailable");
    } else if (command == 'H') {
        Serial.println("TIMING_REPLAY_V2 WINDOW=500 CONTEXT=250 STRIDE=250 N=2 THRESHOLD=0.50 CLOCK_HZ=64000000 GPIO=D2 ATTR16=0 INPUTS=16 PAIRS=IR,RED");
    } else if (command == 'R') {
        pipeline.reset();
        Serial.println("RESET 0");
    } else if (command == 'M') {
        emit_memory();
    } else if (command == 'C') {
        const uint32_t wall_start = millis();
        const uint32_t start = cycle_clock.begin();
        const uint32_t cycles = cycle_clock.end() - start;
        const uint32_t elapsed_ms = millis() - wall_start;
        Serial.print("CAL "); Serial.print(cycles); Serial.print(' '); Serial.println(elapsed_ms);
    } else {
        // Parsing and all output are outside the pulse. Interrupts remain enabled.
        const uint32_t wall_start = millis();
        const auto result = pipeline.push_block(block_ir, block_red, cycle_clock);
        const uint32_t elapsed_ms = millis() - wall_start;
        if (!result.cycles || elapsed_ms >= 60000) {
            Serial.println("ERR invalid_timing");
        } else {
            Serial.print("TIMED "); Serial.print(result.cycles); Serial.print(' ');
            Serial.print(elapsed_ms); Serial.print(' ');
            if (result.decision) emit_decision(*result.decision);
            else {
                Serial.print("BUFFER ");
                Serial.println(static_cast<unsigned long>(pipeline.get_total_samples()));
            }
        }
    }
    command = 0;
    invalid = false;
    token_size = count = 0;
}

void setup() {
    Serial.begin(115200);
    NRF_P1->OUTCLR = 1UL << 11;
    NRF_P1->DIRSET = 1UL << 11;
    if (SystemCoreClock != 64000000 || (DWT->CTRL & DWT_CTRL_NOCYCCNT_Msk)) return;
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    __DSB(); __ISB();
    const uint32_t before = DWT->CYCCNT;
    for (int i = 0; i < 16; ++i) __NOP();
    timing_ready = DWT->CYCCNT != before;
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
