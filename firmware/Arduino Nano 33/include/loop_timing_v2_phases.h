#pragma once
// Second loop-timing diagnostic (env nano33ble_v2_looptiming2, -DLOOP_TIMING_V2_PHASES).
// Included only from loop_timing_v2.h with the flag set. Adds, per decision:
//   * the decision block split into its sub-phases (both clocks, as in LT:), and
//   * a 4 kHz sampling interrupt (TIMER4, priority 0) active only during the decision
//     block, which records what the CPU was running at each tick: the Arduino loop
//     thread, another RTOS thread (by id), or another interrupt handler (by exception
//     number). This separates "the code got slower" from "the CPU was elsewhere".
//
// The frozen pipeline headers are not edited (their digests are recorded by the parity
// tooling). The decision is computed by lt2_push_sample()/lt2_evaluate(), which mirror
// push_sample()/evaluate_window()/evaluate_kan_and_xai() line for line and call the same
// compiled functions (ppg_bandpass_filter, window_mean, extract_features_from_filtered,
// compute_two_wavelength_features, kan_standardize, kan_predict_logit, kan_sigmoid).
// Private members are reached through the explicit-instantiation access rule
// ([temp.explicit]: access checking does not apply to names in explicit instantiations).
#ifdef LOOP_TIMING_V2_PHASES
#ifndef LOOP_TIMING_V2
#error "LOOP_TIMING_V2_PHASES extends LOOP_TIMING_V2"
#endif
#include "cmsis_os2.h"
#include "rtx_os.h"

// ---- access to the frozen pipeline's private members --------------------------------
template <typename Tag, typename Tag::type M>
struct Lt2Rob { friend typename Tag::type lt2_get(Tag) { return M; } };
#define LT2_ROB(Name, Type, Member)                              \
  struct Name { typedef Type type; friend type lt2_get(Name); }; \
  template struct Lt2Rob<Name, &OxygenRiskPipeline::Member>;

typedef OxygenRiskPipeline Lt2P;
typedef float (Lt2P::*Lt2Ring)[Lt2P::BUFFER_SAMPLES];
typedef float (Lt2P::*Lt2Window)[PPG_WINDOW_SIZE];
typedef float (Lt2P::*Lt2Temp)[Lt2P::BUFFER_SAMPLES + 2 * PPG_FILTER_PAD];
typedef float (Lt2P::*Lt2Feat)[KAN_INPUT_DIM];
typedef float Lt2P::*Lt2Float;
typedef bool Lt2P::*Lt2Bool;
typedef int Lt2P::*Lt2Int;
typedef uint64_t Lt2P::*Lt2U64;
typedef uint32_t Lt2P::*Lt2U32;
typedef uint8_t Lt2P::*Lt2U8;
typedef float (*Lt2MeanFn)(const float*);
typedef void (Lt2P::*Lt2ExtractFn)(const float*, float*);
typedef void (Lt2P::*Lt2TwoWlFn)(float*);

LT2_ROB(Lt2RawRing, Lt2Ring, raw_ring_buffer)
LT2_ROB(Lt2RedRing, Lt2Ring, red_ring_buffer)
LT2_ROB(Lt2Aligned, Lt2Ring, aligned_raw)
LT2_ROB(Lt2Filtered, Lt2Window, filtered_buffer)
LT2_ROB(Lt2RedFiltered, Lt2Window, red_filtered_buffer)
LT2_ROB(Lt2TempBuf, Lt2Temp, filter_temp_buf)
LT2_ROB(Lt2Features, Lt2Feat, raw_features)
LT2_ROB(Lt2IrMean, Lt2Float, ir_raw_mean)
LT2_ROB(Lt2RedMean, Lt2Float, red_raw_mean)
LT2_ROB(Lt2TwoWlValid, Lt2Bool, two_wavelength_valid)
LT2_ROB(Lt2WriteIdx, Lt2Int, write_idx)
LT2_ROB(Lt2Total, Lt2U64, total_samples_ingested)
LT2_ROB(Lt2Since, Lt2U32, samples_since_last_window)
LT2_ROB(Lt2WindowIdx, Lt2U32, window_index)
LT2_ROB(Lt2Consecutive, Lt2U8, consecutive_risk_windows)
LT2_ROB(Lt2Primed, Lt2Bool, is_buffer_primed)
LT2_ROB(Lt2Threshold, Lt2Float, decision_threshold)
LT2_ROB(Lt2Required, Lt2U8, required_consecutive_windows)
LT2_ROB(Lt2WindowMean, Lt2MeanFn, window_mean)
LT2_ROB(Lt2Extract, Lt2ExtractFn, extract_features_from_filtered)
LT2_ROB(Lt2TwoWl, Lt2TwoWlFn, compute_two_wavelength_features)
#define LT2_M(p, Tag) ((p).*lt2_get(Tag()))

// ---- sub-phase record ------------------------------------------------------------------
enum Lt2Phase {
  LT2_INGEST, LT2_ALIGN_IR, LT2_FILT_IR, LT2_MEAN_IR, LT2_ALIGN_RED, LT2_FILT_RED, LT2_MEAN_RED,
  LT2_FEAT14, LT2_FEAT2, LT2_KAN_BASE, LT2_KAN_PERT, LT2_SORT, LT2_TAIL, LT2_PHASES
};
static const char* const LT2_PHASE_KEYS[LT2_PHASES] = {
  "ingest", "align_ir", "filt_ir", "mean_ir", "align_red", "filt_red", "mean_red",
  "feat14", "feat2", "kan_base", "kan_pert", "sort", "tail"
};
static LtDur lt2_cur[LT2_PHASES];   // being filled by the current decision
static LtDur lt2_dec[LT2_PHASES];   // copy for the latest decision (printed on LT2:)

// ---- sampling interrupt ---------------------------------------------------------------
struct Lt2Tally { uint32_t key, n; };
static const int LT2_OTH = 4, LT2_ISR = 6;
static const uint32_t LT2_SAMPLE_US = 250;  // 4 kHz
static bool lt2_smp_ok = false;
static osThreadId_t lt2_main_thread = nullptr;
static volatile bool lt2_smp_on = false;
static volatile uint32_t lt2_n, lt2_main, lt2_oth_n, lt2_isr_n, lt2_full, lt2_cost_c;
static volatile Lt2Tally lt2_oth[LT2_OTH], lt2_isr[LT2_ISR];
struct Lt2Sample {  // copy for the latest decision
  uint32_t n, main, oth, isr, full, cost_c;
  Lt2Tally oth_t[LT2_OTH], isr_t[LT2_ISR];
};
static Lt2Sample lt2_smp;

static void lt2_tally(volatile Lt2Tally* t, int size, uint32_t key) {
  for (int i = 0; i < size; ++i) {
    if (t[i].n && t[i].key == key) { t[i].n++; return; }
    if (!t[i].n) { t[i].key = key; t[i].n = 1; return; }
  }
  lt2_full++;
}

// exc_return: LR at exception entry. frame: the interrupted context's stacked frame.
extern "C" void lt2_sampler_body(uint32_t exc_return, const uint32_t* frame) {
  const uint32_t c0 = DWT->CYCCNT;
  NRF_TIMER4->EVENTS_COMPARE[0] = 0;
  (void)NRF_TIMER4->EVENTS_COMPARE[0];  // complete the write before returning
  if (lt2_smp_on) {
    lt2_n++;
    if ((exc_return & 0x8u) == 0) {      // returning to Handler mode: another ISR was running
      lt2_isr_n++;
      lt2_tally(lt2_isr, LT2_ISR, frame[7] & 0x1FFu);  // stacked xPSR exception number
    } else {
      const osThreadId_t id = osThreadGetId();  // RTX: callable from ISRs, reads run.curr
      if (id == lt2_main_thread) lt2_main++;
      else { lt2_oth_n++; lt2_tally(lt2_oth, LT2_OTH, (uint32_t)id); }
    }
  }
  lt2_cost_c += DWT->CYCCNT - c0;
}

// Hands LR (EXC_RETURN) and the interrupted frame (MSP or PSP) to the body; the body's
// return loads EXC_RETURN into PC, which performs the exception return.
extern "C" __attribute__((naked)) void lt2_sampler_isr(void) {
  __asm volatile(
    "tst lr, #4        \n"
    "ite eq            \n"
    "mrseq r1, msp     \n"
    "mrsne r1, psp     \n"
    "mov r0, lr        \n"
    "b lt2_sampler_body\n");
}

static void lt2_setup() {
  lt2_main_thread = osThreadGetId();
  // The vector table must be the RAM copy mbed installs (nrf_reloc_vector_table).
  if (SCB->VTOR != 0x20000000u || !lt_ready) {
    Serial.println("LT2_INFO: sampler=0 reason=vtor_or_cyccnt");
    return;
  }
  NRF_TIMER4->TASKS_STOP = 1;
  NRF_TIMER4->TASKS_CLEAR = 1;
  NRF_TIMER4->MODE = TIMER_MODE_MODE_Timer;
  NRF_TIMER4->BITMODE = TIMER_BITMODE_BITMODE_32Bit;
  NRF_TIMER4->PRESCALER = 4;  // 16 MHz / 2^4 = 1 MHz
  NRF_TIMER4->CC[0] = LT2_SAMPLE_US;
  NRF_TIMER4->SHORTS = TIMER_SHORTS_COMPARE0_CLEAR_Msk;
  NRF_TIMER4->EVENTS_COMPARE[0] = 0;
  NRF_TIMER4->INTENSET = TIMER_INTENSET_COMPARE0_Msk;
  NVIC_SetVector(TIMER4_IRQn, (uint32_t)&lt2_sampler_isr);
  NVIC_SetPriority(TIMER4_IRQn, 0);
  NVIC_ClearPendingIRQ(TIMER4_IRQn);
  NVIC_EnableIRQ(TIMER4_IRQn);
  lt2_smp_ok = true;
  Serial.print("LT2_INFO: sampler=1 timer=TIMER4 irq_priority=0 sample_us=");
  Serial.print(LT2_SAMPLE_US);
  Serial.print(" main_thread=");
  Serial.println((unsigned long)(uint32_t)lt2_main_thread);
}

static void lt2_sampler_start() {
  lt2_n = lt2_main = lt2_oth_n = lt2_isr_n = lt2_full = lt2_cost_c = 0;
  for (int i = 0; i < LT2_OTH; ++i) lt2_oth[i].n = 0;
  for (int i = 0; i < LT2_ISR; ++i) lt2_isr[i].n = 0;
  if (!lt2_smp_ok) return;
  lt2_smp_on = true;
  NRF_TIMER4->TASKS_CLEAR = 1;
  NRF_TIMER4->TASKS_START = 1;
}

static void lt2_sampler_stop() {
  if (lt2_smp_ok) {
    NRF_TIMER4->TASKS_STOP = 1;
    lt2_smp_on = false;
    NRF_TIMER4->EVENTS_COMPARE[0] = 0;
    NVIC_ClearPendingIRQ(TIMER4_IRQn);
  }
  lt2_smp.n = lt2_n; lt2_smp.main = lt2_main; lt2_smp.oth = lt2_oth_n; lt2_smp.isr = lt2_isr_n;
  lt2_smp.full = lt2_full; lt2_smp.cost_c = lt2_cost_c;
  for (int i = 0; i < LT2_OTH; ++i) { lt2_smp.oth_t[i].key = lt2_oth[i].key; lt2_smp.oth_t[i].n = lt2_oth[i].n; }
  for (int i = 0; i < LT2_ISR; ++i) { lt2_smp.isr_t[i].key = lt2_isr[i].key; lt2_smp.isr_t[i].n = lt2_isr[i].n; }
}

// ---- mirrored decision path ------------------------------------------------------------
// evaluate_kan_and_xai(), split into baseline / perturbation / sort. Same calls, same order.
__attribute__((noinline))
static float lt2_kan_and_xai(const float* features, FeatureAttribution top_3_out[3]) {
  LtStamp a = lt_now();
  float scaled[KAN_INPUT_DIM];
  kan_standardize(features, scaled);
  float baseline_logit = kan_predict_logit(scaled);
  float baseline_prob = kan_sigmoid(baseline_logit);
  LtStamp b = lt_now();
  lt2_cur[LT2_KAN_BASE].add(a, b);

  FeatureAttribution all_attrs[KAN_INPUT_DIM];
  float perturbed_scaled[KAN_INPUT_DIM];
  for (int i = 0; i < KAN_INPUT_DIM; ++i) {
    std::memcpy(perturbed_scaled, scaled, KAN_INPUT_DIM * sizeof(float));
    perturbed_scaled[i] = KAN_XAI_REFERENCE(i);
    float perturbed_logit = kan_predict_logit(perturbed_scaled);
    float perturbed_prob = kan_sigmoid(perturbed_logit);
    all_attrs[i].feature_index = static_cast<uint8_t>(i);
    all_attrs[i].importance_delta = baseline_prob - perturbed_prob;
    all_attrs[i].feature_name = KAN_FEATURE_NAMES[i];
  }
  a = lt_now();
  lt2_cur[LT2_KAN_PERT].add(b, a);

  for (int i = 0; i < KAN_INPUT_DIM - 1; ++i) {
    for (int j = 0; j < KAN_INPUT_DIM - 1 - i; ++j) {
      if (std::fabs(all_attrs[j].importance_delta) < std::fabs(all_attrs[j + 1].importance_delta)) {
        FeatureAttribution temp = all_attrs[j];
        all_attrs[j] = all_attrs[j + 1];
        all_attrs[j + 1] = temp;
      }
    }
  }
  for (int i = 0; i < 3; ++i) top_3_out[i] = all_attrs[i];
  lt2_cur[LT2_SORT].add(a, lt_now());
  return baseline_prob;
}

// evaluate_window() including prepare_filtered_window(), split into sub-phases.
__attribute__((noinline))
static RiskDecision lt2_evaluate(Lt2P& p) {
  float* const aligned = LT2_M(p, Lt2Aligned);
  const uint64_t total = LT2_M(p, Lt2Total);
  const int write_idx = LT2_M(p, Lt2WriteIdx);

  LtStamp a = lt_now();
  const int count = total < Lt2P::BUFFER_SAMPLES ? static_cast<int>(total) : Lt2P::BUFFER_SAMPLES;
  int read_idx = total < Lt2P::BUFFER_SAMPLES ? 0 : write_idx;
  for (int i = 0; i < count; ++i) {
    aligned[i] = LT2_M(p, Lt2RawRing)[read_idx];
    read_idx = (read_idx + 1) % Lt2P::BUFFER_SAMPLES;
  }
  LtStamp b = lt_now(); lt2_cur[LT2_ALIGN_IR].add(a, b);
  const int target_offset = count - Lt2P::FIRST_DECISION_SAMPLES;
  ppg_bandpass_filter(aligned, LT2_M(p, Lt2Filtered), LT2_M(p, Lt2TempBuf), count, target_offset);
  a = lt_now(); lt2_cur[LT2_FILT_IR].add(b, a);
  LT2_M(p, Lt2IrMean) = lt2_get(Lt2WindowMean())(aligned + target_offset);
  b = lt_now(); lt2_cur[LT2_MEAN_IR].add(a, b);
  read_idx = total < Lt2P::BUFFER_SAMPLES ? 0 : write_idx;
  for (int i = 0; i < count; ++i) {
    aligned[i] = LT2_M(p, Lt2RedRing)[read_idx];
    read_idx = (read_idx + 1) % Lt2P::BUFFER_SAMPLES;
  }
  a = lt_now(); lt2_cur[LT2_ALIGN_RED].add(b, a);
  ppg_bandpass_filter(aligned, LT2_M(p, Lt2RedFiltered), LT2_M(p, Lt2TempBuf), count, target_offset);
  b = lt_now(); lt2_cur[LT2_FILT_RED].add(a, b);
  LT2_M(p, Lt2RedMean) = lt2_get(Lt2WindowMean())(aligned + target_offset);
  a = lt_now(); lt2_cur[LT2_MEAN_RED].add(b, a);

  float* const raw_features = LT2_M(p, Lt2Features);
  (p.*lt2_get(Lt2Extract()))(LT2_M(p, Lt2Filtered), raw_features);
  b = lt_now(); lt2_cur[LT2_FEAT14].add(a, b);
  (p.*lt2_get(Lt2TwoWl()))(raw_features);
  a = lt_now(); lt2_cur[LT2_FEAT2].add(b, a);

  FeatureAttribution top_3[3];
  float risk_prob = lt2_kan_and_xai(raw_features, top_3);

  a = lt_now();
  uint8_t& consecutive = LT2_M(p, Lt2Consecutive);
  bool is_candidate = (risk_prob >= LT2_M(p, Lt2Threshold));
  if (is_candidate) {
    if (consecutive < 255) consecutive++;
  } else {
    consecutive = 0;
  }
  bool is_confirmed = (consecutive >= LT2_M(p, Lt2Required));

  RiskDecision decision;
  decision.window_index = LT2_M(p, Lt2WindowIdx)++;
  decision.total_samples_ingested = total;
  decision.window_end_sample = total - Lt2P::CONTEXT_SAMPLES;
  decision.window_start_sample = decision.window_end_sample - PPG_WINDOW_SIZE;
  decision.decision_delay_samples = Lt2P::CONTEXT_SAMPLES;
  decision.risk_probability = risk_prob;
  decision.is_candidate_risk = is_candidate;
  decision.is_confirmed_risk = is_confirmed;
  decision.consecutive_risk_windows = consecutive;
  for (int i = 0; i < 3; ++i) decision.top_attributions[i] = top_3[i];
  for (int i = 0; i < KAN_INPUT_DIM; ++i) decision.raw_features[i] = raw_features[i];
  decision.two_wavelength_valid = LT2_M(p, Lt2TwoWlValid);
  lt2_cur[LT2_TAIL].add(a, lt_now());
  return decision;
}

// push_sample(ir, red) mirror. PPG_STEP_SIZE is private; 250 is its value.
static std::optional<RiskDecision> lt2_push_sample(Lt2P& p, float ir_val, float red_val) {
  const LtStamp a = lt_now();
  int& write_idx = LT2_M(p, Lt2WriteIdx);
  LT2_M(p, Lt2RedRing)[write_idx] = red_val;
  LT2_M(p, Lt2RawRing)[write_idx] = ir_val;
  write_idx = (write_idx + 1) % Lt2P::BUFFER_SAMPLES;
  uint64_t& total = LT2_M(p, Lt2Total);
  uint32_t& since = LT2_M(p, Lt2Since);
  bool& primed = LT2_M(p, Lt2Primed);
  total++;
  since++;

  bool due = false;
  if (!primed) {
    if (total >= Lt2P::FIRST_DECISION_SAMPLES) { primed = true; since = 0; due = true; }
  } else if (since >= 250) {
    since = 0;
    due = true;
  }
  if (!due) return std::nullopt;

  memset(lt2_cur, 0, sizeof(lt2_cur));
  lt2_sampler_start();
  lt2_cur[LT2_INGEST].add(a, lt_now());
  const RiskDecision decision = lt2_evaluate(p);
  lt2_sampler_stop();
  memcpy(lt2_dec, lt2_cur, sizeof(lt2_dec));
  return decision;
}

// ---- LT2: line ------------------------------------------------------------------------
static int lt2_append(int n, const char* fmt, ...) __attribute__((format(printf, 2, 3)));
static int lt2_append(int n, const char* fmt, ...) {
  if (n < 0 || n >= (int)sizeof(lt_line)) return n;
  va_list ap;
  va_start(ap, fmt);
  const int k = vsnprintf(lt_line + n, sizeof(lt_line) - n, fmt, ap);
  va_end(ap);
  return k < 0 ? -1 : n + k;
}

// Printed right after the LT: line of the same decision (same n), inside the timed print.
static void lt2_emit() {
  int n = lt2_append(0, "LT2: n=%lu win=%lu", (unsigned long)lt_seq, (unsigned long)lt_dec_win);
  uint32_t sum_u = 0, sum_c = 0;
  for (int i = 0; i < LT2_PHASES; ++i) {
    n = lt2_append(n, " %s_us=%lu %s_c=%lu", LT2_PHASE_KEYS[i], (unsigned long)lt2_dec[i].u,
                   LT2_PHASE_KEYS[i], (unsigned long)lt2_dec[i].c);
    sum_u += lt2_dec[i].u;
    sum_c += lt2_dec[i].c;
  }
  n = lt2_append(n, " phases_us=%lu phases_c=%lu smp_ok=%d smp_us=%lu smp_n=%lu smp_main=%lu"
                 " smp_oth=%lu smp_isr=%lu smp_full=%lu smp_cost_c=%lu",
                 (unsigned long)sum_u, (unsigned long)sum_c, lt2_smp_ok ? 1 : 0,
                 (unsigned long)LT2_SAMPLE_US, (unsigned long)lt2_smp.n, (unsigned long)lt2_smp.main,
                 (unsigned long)lt2_smp.oth, (unsigned long)lt2_smp.isr, (unsigned long)lt2_smp.full,
                 (unsigned long)lt2_smp.cost_c);
  for (int i = 0; i < LT2_OTH; ++i) {
    const Lt2Tally& t = lt2_smp.oth_t[i];
    if (!t.n) continue;
    // Thread names contain no spaces in mbed ("main", "rtx_idle", "rtx_timer",
    // "application_unnamed_thread"); a space would still be replaced so the line stays k=v.
    char name[33] = "unknown";
    const char* src = osThreadGetName((osThreadId_t)t.key);
    if (src) {
      int j = 0;
      for (; src[j] && j < 32; ++j) name[j] = (src[j] == ' ' || src[j] == '=') ? '_' : src[j];
      name[j] = '\0';
    }
    n = lt2_append(n, " oth%d_id=%lu oth%d_n=%lu oth%d_name=%s", i + 1, (unsigned long)t.key,
                   i + 1, (unsigned long)t.n, i + 1, name);
  }
  for (int i = 0; i < LT2_ISR; ++i) {
    const Lt2Tally& t = lt2_smp.isr_t[i];
    if (!t.n) continue;
    n = lt2_append(n, " isr%d_exc=%lu isr%d_n=%lu", i + 1, (unsigned long)t.key, i + 1, (unsigned long)t.n);
  }
  // Loop-thread stack: reserved size and lowest free space since thread creation
  // (RTX watermark; -1 when the RTX build has no watermark support).
  const osThreadId_t self = osThreadGetId();
  const uint32_t stk_size = osThreadGetStackSize(self);
  long stk_free = -1;
  if (stk_size && (osRtxConfig.flags & osRtxConfigStackWatermark)) stk_free = (long)osThreadGetStackSpace(self);
  n = lt2_append(n, " stk_size=%lu stk_free=%ld\r\n", (unsigned long)stk_size, stk_free);
  if (n > 0) {
    const size_t len = (size_t)n < sizeof(lt_line) ? (size_t)n : sizeof(lt_line) - 1;
    Serial.write(lt_line, len);
  }
}
#endif
