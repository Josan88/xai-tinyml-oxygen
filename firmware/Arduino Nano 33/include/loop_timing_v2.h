#pragma once
// Diagnostic instrumentation for the live v2 loop (env nano33ble_v2_looptiming).
// Included only with -DLOOP_TIMING_V2. It times the existing loop and prints
// LT: / LT_IDLE: / LT_INFO: lines; it does not change what the loop does.
//
// Every stamp reads DWT CYCCNT (64 MHz core clock) and then micros(). Both
// clocks are reported as raw integers; neither is rescaled to the other and no
// bracket overhead is subtracted. CYCCNT wraps every 67.1 s, so cycle fields of
// spans longer than that (long no-finger gaps) are not meaningful; micros()
// fields are.
//
// Register 0x05 (FIFO_OVF) is not read. Loss is estimated from wall time:
// expected records = drain-to-drain elapsed x 50 /s, minus records drained.
#ifdef LOOP_TIMING_V2
#ifndef OXY_MODEL_V2
#error "LOOP_TIMING_V2 instruments the live v2 image; use env nano33ble_v2_looptiming"
#endif
#include <Arduino.h>
#include <stdio.h>

struct LtStamp { uint32_t c; uint32_t u; };

static inline LtStamp lt_now() {
  LtStamp s;
  s.c = DWT->CYCCNT;
  s.u = micros();
  return s;
}

// Sum of bracketed durations within one loop pass.
struct LtDur {
  uint32_t c, u;
  void add(const LtStamp& a, const LtStamp& b) { c += b.c - a.c; u += b.u - a.u; }
};

// One loop() pass.
struct LtPass {
  LtStamp start;
  LtDur central, drain, dec, push, hrp, hre, st, feat, ble;  // st includes feat
  uint32_t gap_c, gap_u;  // previous drain start -> this drain start
  uint32_t pass_c, pass_u;
  uint8_t frames, wr, rd, pend, dpos, st_n, ble_n;
  bool decided;
};

// Counters over a span of drains: (span start drain, latest drain].
struct LtSpan {
  LtStamp from, to;
  uint32_t drained, passes, maxpass_u, maxgap_u;
  uint32_t p600, p640, g600, g640, e600, shortrd, nf, fin;
};

static bool     lt_ready = false;        // CYCCNT running at 64 MHz
static uint32_t lt_seq = 0;
static LtPass   lt_pass;
static LtPass   lt_dec_pass;             // snapshot of the pass that returned a decision
static LtSpan   lt_dec_span;             // since the previous decision's drain
static LtSpan   lt_dec_snap;             // closed at the decision's drain
static LtSpan   lt_idle_span;            // since the previous LT/LT_IDLE line
static LtStamp  lt_last_drain;
static bool     lt_have_drain = false;
static LtStamp  lt_decision_at;          // push_sample return of the latest decision
static bool     lt_have_decision = false;
static uint32_t lt_ddt_c = 0, lt_ddt_u = 0;
static uint32_t lt_dec_win = 0, lt_dec_tot = 0;
static bool     lt_report_due = false;   // emit LT at the end of the pass after the decision
static LtStamp  lt_last_emit;
static LtDur    lt_print = {0, 0};       // cost of the previous LT/LT_IDLE print
static uint8_t  lt_raw_wr = 0, lt_raw_rd = 0, lt_raw_pend = 0;  // set by drainSensorFifo
static char     lt_line[1400];         // worst case (all 10-digit values) is under 1300

static const uint32_t LT_IDLE_EVERY_US      = 5000000;  // LT_IDLE cadence
static const uint32_t LT_IDLE_NO_DECISION_US = 7000000; // > one decision interval (5.0-5.7 s)

#ifdef LOOP_TIMING_V2_PHASES
#include <stdarg.h>
#include "loop_timing_v2_phases.h"  // LT2: decision sub-phases and the 4 kHz context sampler
#endif

static void lt_span_reset(LtSpan& s, const LtStamp& from) {
  memset(&s, 0, sizeof(s));
  s.from = from;
  s.to = from;
}

static void lt_setup() {
  if (SystemCoreClock == 64000000 && !(DWT->CTRL & DWT_CTRL_NOCYCCNT_Msk)) {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    __DSB(); __ISB();
    const uint32_t before = DWT->CYCCNT;
    for (int i = 0; i < 16; ++i) __NOP();
    lt_ready = DWT->CYCCNT != before;
  }
  lt_last_emit = lt_now();
  lt_span_reset(lt_dec_span, lt_last_emit);
  lt_span_reset(lt_idle_span, lt_last_emit);
  Serial.print("LT_INFO: build=LOOP_TIMING_V2 clock_hz=");
  Serial.print((unsigned long)SystemCoreClock);
  Serial.print(" cyccnt=");
  Serial.print(lt_ready ? 1 : 0);
  Serial.println(" fifo_depth=32 records_per_s=50 idle_every_us=5000000 fifo_ovf_reg=unused");
#ifdef LOOP_TIMING_V2_PHASES
  lt2_setup();
#endif
}

static void lt_pass_begin() {
  memset(&lt_pass, 0, sizeof(lt_pass));
  lt_pass.start = lt_now();
}

static void lt_on_drain(const LtStamp& a, const LtStamp& b, uint8_t got) {
  lt_pass.drain.add(a, b);
  lt_pass.frames = got;
  lt_pass.wr = lt_raw_wr;
  lt_pass.rd = lt_raw_rd;
  lt_pass.pend = lt_raw_pend;
  if (lt_have_drain) {
    lt_pass.gap_c = a.c - lt_last_drain.c;
    lt_pass.gap_u = a.u - lt_last_drain.u;
  } else {
    lt_span_reset(lt_dec_span, a);
    lt_span_reset(lt_idle_span, a);
  }
  lt_last_drain = a;
  lt_have_drain = true;

  LtSpan* spans[2] = {&lt_dec_span, &lt_idle_span};
  for (LtSpan* s : spans) {
    s->to = a;
    s->drained += got;
    if (lt_pass.gap_u > s->maxgap_u) s->maxgap_u = lt_pass.gap_u;
    if (lt_pass.gap_u > 600000) s->g600++;
    if (lt_pass.gap_u > 640000) s->g640++;
    if (lt_pass.gap_u > 600000 && lt_raw_pend == 0) s->e600++;
    if (got < lt_raw_pend) s->shortrd++;
  }
}

static void lt_on_push(const LtStamp& a, const LtStamp& b, bool decided, uint8_t f) {
  if (decided) {
    lt_pass.dec.add(a, b);
    lt_pass.decided = true;
    lt_pass.dpos = f;
    if (lt_have_decision) {
      lt_ddt_c = b.c - lt_decision_at.c;
      lt_ddt_u = b.u - lt_decision_at.u;
    } else {
      lt_ddt_c = lt_ddt_u = 0;
    }
    lt_decision_at = b;
    lt_have_decision = true;
  } else {
    lt_pass.push.add(a, b);
  }
  lt_dec_span.fin++;
  lt_idle_span.fin++;
}

static void lt_on_no_finger() {
  lt_dec_span.nf++;
  lt_idle_span.nf++;
}

// expected milli-records over a drain-to-drain span at 50 records/s
static uint32_t lt_exp_mrec(const LtSpan& s) { return (s.to.u - s.from.u) / 20; }

static long lt_loss_mrec(const LtSpan& s) {
  return (long)lt_exp_mrec(s) - (long)s.drained * 1000L;
}

static void lt_emit_decision(const LtPass& post) {
  const LtPass& d = lt_dec_pass;
  const LtSpan& s = lt_dec_snap;
  const long other_u = (long)d.pass_u - (long)(d.central.u + d.drain.u + d.dec.u + d.push.u +
                                              d.hrp.u + d.hre.u + d.st.u + d.ble.u);
  const long other_c = (long)d.pass_c - (long)(d.central.c + d.drain.c + d.dec.c + d.push.c +
                                              d.hrp.c + d.hre.c + d.st.c + d.ble.c);
  const int n = snprintf(lt_line, sizeof(lt_line),
    "LT: n=%lu win=%lu tot=%lu conn=%d dpos=%u frames=%u wr=%u rd=%u pend=%u"
    " pass_us=%lu pass_c=%lu central_us=%lu central_c=%lu drain_us=%lu drain_c=%lu"
    " dec_us=%lu dec_c=%lu push_us=%lu push_c=%lu hrp_us=%lu hrp_c=%lu hre_us=%lu hre_c=%lu"
    " st_us=%lu st_c=%lu st_n=%u feat_us=%lu feat_c=%lu ble_us=%lu ble_c=%lu ble_n=%u"
    " other_us=%ld other_c=%ld"
    " post_gap_us=%lu post_gap_c=%lu post_wr=%u post_rd=%u post_pend=%u post_got=%u"
    " post_pass_us=%lu post_pass_c=%lu"
    " ddt_us=%lu ddt_c=%lu el_us=%lu el_c=%lu drained=%lu exp_mrec=%lu loss_mrec=%ld"
    " passes=%lu maxpass_us=%lu maxgap_us=%lu p600=%lu p640=%lu g600=%lu g640=%lu e600=%lu"
    " short=%lu nf=%lu ltp_us=%lu ltp_c=%lu\r\n",
    (unsigned long)++lt_seq, (unsigned long)lt_dec_win, (unsigned long)lt_dec_tot,
    bleConnected ? 1 : 0, d.dpos, d.frames, d.wr, d.rd, d.pend,
    (unsigned long)d.pass_u, (unsigned long)d.pass_c,
    (unsigned long)d.central.u, (unsigned long)d.central.c,
    (unsigned long)d.drain.u, (unsigned long)d.drain.c,
    (unsigned long)d.dec.u, (unsigned long)d.dec.c,
    (unsigned long)d.push.u, (unsigned long)d.push.c,
    (unsigned long)d.hrp.u, (unsigned long)d.hrp.c,
    (unsigned long)d.hre.u, (unsigned long)d.hre.c,
    (unsigned long)(d.st.u - d.feat.u), (unsigned long)(d.st.c - d.feat.c), d.st_n,
    (unsigned long)d.feat.u, (unsigned long)d.feat.c,
    (unsigned long)d.ble.u, (unsigned long)d.ble.c, d.ble_n,
    other_u, other_c,
    (unsigned long)post.gap_u, (unsigned long)post.gap_c,
    post.wr, post.rd, post.pend, post.frames,
    (unsigned long)post.pass_u, (unsigned long)post.pass_c,
    (unsigned long)lt_ddt_u, (unsigned long)lt_ddt_c,
    (unsigned long)(s.to.u - s.from.u), (unsigned long)(s.to.c - s.from.c),
    (unsigned long)s.drained, (unsigned long)lt_exp_mrec(s), lt_loss_mrec(s),
    (unsigned long)s.passes, (unsigned long)s.maxpass_u, (unsigned long)s.maxgap_u,
    (unsigned long)s.p600, (unsigned long)s.p640, (unsigned long)s.g600,
    (unsigned long)s.g640, (unsigned long)s.e600, (unsigned long)s.shortrd,
    (unsigned long)s.nf,
    (unsigned long)lt_print.u, (unsigned long)lt_print.c);
  if (n > 0) Serial.write(lt_line, (size_t)n < sizeof(lt_line) ? (size_t)n : sizeof(lt_line) - 1);
}

static void lt_emit_idle() {
  const LtSpan& s = lt_idle_span;
  const int n = snprintf(lt_line, sizeof(lt_line),
    "LT_IDLE: n=%lu el_us=%lu el_c=%lu passes=%lu maxpass_us=%lu maxgap_us=%lu"
    " drained=%lu exp_mrec=%lu loss_mrec=%ld p600=%lu p640=%lu g600=%lu g640=%lu e600=%lu"
    " short=%lu nf=%lu fin=%lu conn=%d ltp_us=%lu ltp_c=%lu\r\n",
    (unsigned long)++lt_seq, (unsigned long)(s.to.u - s.from.u),
    (unsigned long)(s.to.c - s.from.c), (unsigned long)s.passes,
    (unsigned long)s.maxpass_u, (unsigned long)s.maxgap_u,
    (unsigned long)s.drained, (unsigned long)lt_exp_mrec(s), lt_loss_mrec(s),
    (unsigned long)s.p600, (unsigned long)s.p640, (unsigned long)s.g600,
    (unsigned long)s.g640, (unsigned long)s.e600, (unsigned long)s.shortrd,
    (unsigned long)s.nf, (unsigned long)s.fin, bleConnected ? 1 : 0,
    (unsigned long)lt_print.u, (unsigned long)lt_print.c);
  if (n > 0) Serial.write(lt_line, (size_t)n < sizeof(lt_line) ? (size_t)n : sizeof(lt_line) - 1);
}

// End of loop(): close the pass, then print (outside every timed section).
static void lt_pass_end() {
  const LtStamp end = lt_now();
  lt_pass.pass_c = end.c - lt_pass.start.c;
  lt_pass.pass_u = end.u - lt_pass.start.u;

  LtSpan* spans[2] = {&lt_dec_span, &lt_idle_span};
  for (LtSpan* s : spans) {
    s->passes++;
    if (lt_pass.pass_u > s->maxpass_u) s->maxpass_u = lt_pass.pass_u;
    if (lt_pass.pass_u > 600000) s->p600++;
    if (lt_pass.pass_u > 640000) s->p640++;
  }

  bool emit_lt = false, emit_idle = false;
  if (lt_pass.decided) {
    // The decision interval closes at this pass's drain; the next pass's drain
    // (post_*) shows what the FIFO held after this pass.
    lt_dec_pass = lt_pass;
    lt_dec_snap = lt_dec_span;
    lt_dec_win = lastWindowIndex;
    lt_dec_tot = (uint32_t)lastTotalSamples;
    lt_span_reset(lt_dec_span, lt_last_drain);
    // Pass counters for this (decision) pass stay in the closed interval.
    lt_report_due = true;
  } else if (lt_report_due) {
    lt_report_due = false;
    emit_lt = true;
  } else if ((end.u - lt_last_emit.u) >= LT_IDLE_EVERY_US &&
             (!lt_have_decision || (end.u - lt_decision_at.u) >= LT_IDLE_NO_DECISION_US)) {
    emit_idle = true;
  }
  if (!emit_lt && !emit_idle) return;

  const LtStamp a = lt_now();
  if (emit_lt) lt_emit_decision(lt_pass);
  else lt_emit_idle();
#ifdef LOOP_TIMING_V2_PHASES
  if (emit_lt) lt2_emit();  // same decision, same n; its cost is inside ltp_*
#endif
  const LtStamp b = lt_now();
  lt_print.c = b.c - a.c;
  lt_print.u = b.u - a.u;
  lt_last_emit = b;
  lt_span_reset(lt_idle_span, lt_last_drain);
}
#endif
