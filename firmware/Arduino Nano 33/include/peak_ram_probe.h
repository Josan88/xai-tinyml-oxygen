#pragma once
// Whole-device peak-RAM probe for the live images (envs nano33ble_peakram and
// nano33ble_v2_f1a_peakram, -DPEAK_RAM_PROBE). Method: docs/peak-ram-design.md.
// Commands: docs/peak-ram-runbook.md. Not a shipped image.
//
// What it adds to the live sketch:
//   * a priority-101 constructor (before USB, BLE and setup()) that paints, with the
//     kernel locked: the heap above the current break and the MSP below the current MSP
//     with PR_PATTERN, and the main-thread stack below its own SP with RTX's own fill,
//     so RTX's watermark restarts there;
//   * at the end of setup() a second mark and repaint of the main stack and the MSP, so
//     both report a boot+setup peak and a loop-only peak;
//   * every 5 s a PR: line (per-region reserved/peak/headroom and the whole-device
//     high-water), and a PR_INFO: line at the end of setup() and every 60 s;
//   * forced decisions: main.cpp holds the finger gate open (isFingerPresent) and
//     builds the BLE payload with or without a central (pr_force_payload).
// The pipeline, weights, threshold and debounce are not touched. Reports are built in a
// static buffer from integers and strings only: no snprintf, no float, no heap.
//
// Sampler images (envs nano33ble_peakram_sampler and nano33ble_v2_f1a_peakram_sampler) add
//   -DPEAK_RAM_LIVE_GATE  the finger gate and the BLE payload as shipped (no forced decisions);
//   -DPEAK_RAM_SAMPLER    the instantaneous-total sampler (include/peak_ram_sampler.h).
#if (defined(PEAK_RAM_LIVE_GATE) || defined(PEAK_RAM_SAMPLER)) && !defined(PEAK_RAM_PROBE)
#error "PEAK_RAM_LIVE_GATE and PEAK_RAM_SAMPLER extend PEAK_RAM_PROBE"
#endif
#if defined(PEAK_RAM_SAMPLER) && !defined(PEAK_RAM_LIVE_GATE)
#error "the sampler images run the live finger gate (PEAK_RAM_LIVE_GATE)"
#endif
#ifdef PEAK_RAM_PROBE
#if defined(LOOP_TIMING_V2) || defined(LED1_IDENTITY_CHECK)
#error "PEAK_RAM_PROBE extends the live images only (nano33ble, nano33ble_v2_f1a)"
#endif
#if defined(OXY_MODEL_V2) && !defined(LOOP_PRIORITY_V2)
#error "the v2 peak-RAM probe extends nano33ble_v2_f1a (LOOP_PRIORITY_V2)"
#endif
#include <Arduino.h>
#include "cmsis_os2.h"
#include "rtx_os.h"
#include "platform/mbed_stats.h"

#if defined(OXY_MODEL_V2) && defined(PEAK_RAM_SAMPLER)
#define PR_IMAGE "nano33ble_v2_f1a_peakram_sampler"
#define PR_MODEL "v2"
#elif defined(OXY_MODEL_V2)
#define PR_IMAGE "nano33ble_v2_f1a_peakram"
#define PR_MODEL "v2"
#elif defined(PEAK_RAM_SAMPLER)
#define PR_IMAGE "nano33ble_peakram_sampler"
#define PR_MODEL "v1"
#else
#define PR_IMAGE "nano33ble_peakram"
#define PR_MODEL "v1"
#endif
#ifdef PEAK_RAM_LIVE_GATE
#define PR_FORCE "0"
#else
#define PR_FORCE "1"
#endif

// Linker-script symbols (variants/ARDUINO_NANO33BLE/linker_script.ld) and Mbed's _sbrk,
// under local names so no core declaration can clash.
extern uint32_t pr_ld_data_start  __asm__("__data_start__");
extern uint32_t pr_ld_data_end    __asm__("__data_end__");
extern uint32_t pr_ld_bss_start   __asm__("__bss_start__");
extern uint32_t pr_ld_bss_end     __asm__("__bss_end__");
extern uint32_t pr_ld_heap_base   __asm__("__end__");
extern uint32_t pr_ld_heap_limit  __asm__("__HeapLimit");
extern uint32_t pr_ld_stack_limit __asm__("__StackLimit");
extern uint32_t pr_ld_stack_top   __asm__("__StackTop");
extern void* pr_sbrk(int incr) __asm__("_sbrk");  // incr 0: current break, no side effect

#define PR_ADDR(sym) ((uint32_t)&(sym))

static const uint32_t PR_RAM_START   = 0x20000000u;  // nRF52840 SRAM
static const uint32_t PR_RAM_SIZE    = 0x40000u;     // 262,144 B
static const uint32_t PR_PATTERN     = 0x9E3779B9u;  // heap and MSP paint
static const uint32_t PR_PERIOD_MS   = 5000u;
static const uint32_t PR_INFO_EVERY  = 12u;          // reports, so 60 s
static const uint32_t PR_SP_MARGIN   = 256u;         // left unpainted below the main SP
static const uint32_t PR_MAX_THREADS = 16u;
#ifdef PEAK_RAM_LIVE_GATE
static const bool     pr_force_payload = false;      // main.cpp: updateBLE only with a central
#else
static const bool     pr_force_payload = true;       // main.cpp: updateBLE without a central
#endif

static bool     pr_heap_painted = false, pr_msp_painted = false, pr_main_painted = false;
static bool     pr_msp2_painted = false, pr_main2_painted = false;
static uint32_t pr_heap_paint_lo = 0;     // break at the hook (first painted heap word)
static uint32_t pr_msp_paint_hi = 0;      // MSP at the hook: [__StackLimit, this) painted
static uint32_t pr_msp2_paint_hi = 0;     // MSP at the end of setup(), second paint
static uint32_t pr_msp_low_setup = 0;     // MSP low-water from the hook to the end of setup()
static uint32_t pr_main_paint_lo = 0, pr_main_paint_hi = 0, pr_main2_paint_hi = 0;
static uint32_t pr_main_space_pre = 0;    // RTX never-touched bytes, creation to the hook
static uint32_t pr_main_space_setup = 0;  // ... hook to the end of setup()
static uint32_t pr_brk_ctor = 0, pr_brk_setup = 0, pr_brk_max = 0;
static osThreadId_t pr_main_id = nullptr;

static uint32_t pr_decisions = 0;         // decisions returned by push_sample
static uint32_t pr_live_finger = 0;       // samples the live gate would pass (red, IR > 3000)
static uint32_t pr_live_nofinger = 0;     // samples the live gate would reset on
static bool     pr_started = false;
static uint32_t pr_last_ms = 0, pr_reports = 0, pr_rpt_alloc = 0;

struct PrThreadRow { uint32_t id, base, size, space; const char* name; const char* loc; };
struct PrSpan { uint32_t lo, hi; };
static osThreadId_t pr_ids[PR_MAX_THREADS];
static PrThreadRow  pr_rows[PR_MAX_THREADS];
static PrSpan       pr_spans[PR_MAX_THREADS + 2];
static char         pr_buf[256];   // flushed to Serial whenever full
static uint32_t     pr_len = 0;

// ---- paint ---------------------------------------------------------------------------
static void pr_fill(uint32_t lo, uint32_t hi, uint32_t pattern) {
  for (volatile uint32_t* p = (volatile uint32_t*)lo; p < (volatile uint32_t*)hi; ++p) *p = pattern;
}

// Privileged thread mode on PSP: no exception is active, so nothing below MSP is live.
static bool pr_thread_mode_on_psp() {
  const uint32_t control = __get_CONTROL();
  return __get_IPSR() == 0u && (control & 2u) != 0u && (control & 1u) == 0u;
}

static bool pr_watermark() { return (osRtxConfig.flags & osRtxConfigStackWatermark) != 0u; }

// Fills [__StackLimit, MSP) with PR_PATTERN, interrupts masked. Returns the end, or 0 if
// it could not paint (then the MSP counts as fully used).
static uint32_t pr_paint_msp() {
  if (!pr_thread_mode_on_psp()) return 0;
  const uint32_t primask = __get_PRIMASK();
  __disable_irq();
  const uint32_t msp = __get_MSP() & ~3u;
  uint32_t hi = 0;
  if (msp > PR_ADDR(pr_ld_stack_limit) && msp <= PR_ADDR(pr_ld_stack_top)) {
    pr_fill(PR_ADDR(pr_ld_stack_limit), msp, PR_PATTERN);
    hi = msp;
  }
  __set_PRIMASK(primask);
  return hi;
}

// Fills [lo, PSP - PR_SP_MARGIN) of the running thread's stack with RTX's fill and
// returns the end. A leaf: nothing is called after PSP is read, and the margin covers
// this function's own frame.
__attribute__((noinline)) static uint32_t pr_fill_below_psp(uint32_t lo) {
  const uint32_t hi = (__get_PSP() & ~3u) - PR_SP_MARGIN;
  for (volatile uint32_t* p = (volatile uint32_t*)lo; p < (volatile uint32_t*)hi; ++p) {
    *p = osRtxStackFillPattern;
  }
  return hi;
}

// Records the main thread's never-touched bytes so far, then restarts RTX's watermark
// below the current SP. Returns the end of the repainted range, or 0 if none.
static uint32_t pr_paint_main(uint32_t* space_so_far) {
  osRtxThread_t* t = (osRtxThread_t*)osThreadGetId();
  *space_so_far = osThreadGetStackSpace(t);
  if (!pr_watermark() || !pr_thread_mode_on_psp()) return 0;
  const uint32_t lo = (uint32_t)t->stack_mem + 4u;  // the magic word stays
  const uint32_t hi = pr_fill_below_psp(lo);
  return hi > lo ? hi : 0;
}

static void pr_paint_at_hook() {
  pr_main_id = osThreadGetId();
  const int32_t lock = osKernelLock();  // no thread can reach _sbrk while the heap is painted
  const uint32_t brk = (uint32_t)pr_sbrk(0);
  pr_brk_ctor = pr_brk_max = brk;
  const uint32_t lo = (brk + 3u) & ~3u;
  const uint32_t limit = PR_ADDR(pr_ld_heap_limit);
  if (lo >= PR_ADDR(pr_ld_heap_base) && lo < limit) {
    pr_fill(lo, limit, PR_PATTERN);
    pr_heap_paint_lo = lo;
    pr_heap_painted = true;
  }
  pr_msp_paint_hi = pr_paint_msp();
  pr_msp_painted = pr_msp_paint_hi != 0;
  pr_main_paint_lo = (uint32_t)((osRtxThread_t*)pr_main_id)->stack_mem + 4u;
  pr_main_paint_hi = pr_paint_main(&pr_main_space_pre);
  pr_main_painted = pr_main_paint_hi != 0;
  osKernelRestoreLock(lock);
}

// Runs in mbed_start's __libc_init_array on the main thread, after osKernelStart and
// before every constructor without a priority (docs/peak-ram-design.md, section 2).
struct PrHook { PrHook() { pr_paint_at_hook(); } };
static PrHook pr_hook __attribute__((init_priority(101)));

// ---- scan ----------------------------------------------------------------------------
// Heap high-water: the highest word above the paint break that is not the pattern,
// raised to the highest break seen. An unpainted heap counts as fully used.
static uint32_t pr_heap_top() {
  const uint32_t limit = PR_ADDR(pr_ld_heap_limit);
  uint32_t top = limit;
  if (pr_heap_painted) {
    top = pr_heap_paint_lo;
    for (volatile uint32_t* p = (volatile uint32_t*)limit - 1; p >= (volatile uint32_t*)pr_heap_paint_lo; --p) {
      if (*p != PR_PATTERN) { top = (uint32_t)(p + 1); break; }
    }
  }
  const uint32_t brk = (uint32_t)pr_sbrk(0);
  if (brk > pr_brk_max) pr_brk_max = brk;
  if (pr_brk_max > top) top = pr_brk_max;  // allocated but unwritten counts as used
  return top < limit ? top : limit;
}

// MSP low-water: the lowest word in [__StackLimit, hi) that is not the pattern, or hi.
static uint32_t pr_msp_low(uint32_t hi) {
  for (volatile uint32_t* p = (volatile uint32_t*)PR_ADDR(pr_ld_stack_limit); p < (volatile uint32_t*)hi; ++p) {
    if (*p != PR_PATTERN) return (uint32_t)p;
  }
  return hi;
}

// Lifetime MSP low-water since the hook; __StackLimit (fully used) if never painted.
static uint32_t pr_msp_low_lifetime(uint32_t* loop_low) {
  const uint32_t limit = PR_ADDR(pr_ld_stack_limit);
  *loop_low = limit;
  if (!pr_msp_painted) return limit;
  if (!pr_msp2_painted) {
    const uint32_t low = pr_msp_low(pr_msp_paint_hi);
    *loop_low = low;
    return low;
  }
  *loop_low = pr_msp_low(pr_msp2_paint_hi);
  return *loop_low < pr_msp_low_setup ? *loop_low : pr_msp_low_setup;
}

static const char* pr_location(uint32_t addr) {
  if (addr >= PR_ADDR(pr_ld_bss_start) && addr < PR_ADDR(pr_ld_bss_end)) return "bss";
  if (addr >= PR_ADDR(pr_ld_heap_base) && addr < PR_ADDR(pr_ld_heap_limit)) return "heap";
  return "other";
}

// Fills pr_rows; the main thread's space is its minimum over the painted phases.
static uint32_t pr_scan_threads(uint32_t* main_now_space) {
  uint32_t n = osThreadEnumerate(pr_ids, PR_MAX_THREADS);
  if (n > PR_MAX_THREADS) n = PR_MAX_THREADS;
  *main_now_space = 0;
  for (uint32_t i = 0; i < n; ++i) {
    osRtxThread_t* t = (osRtxThread_t*)pr_ids[i];
    PrThreadRow& r = pr_rows[i];
    r.id = (uint32_t)t;
    r.base = (uint32_t)t->stack_mem;
    r.size = t->stack_size;
    r.space = pr_watermark() ? osThreadGetStackSpace(t) : 0u;  // 0: counts as fully used
    if (r.space > r.size) r.space = r.size;
    r.name = osThreadGetName(t);
    r.loc = pr_location(r.base);
    if (pr_ids[i] == pr_main_id) {
      *main_now_space = r.space;
      if (pr_main_painted && pr_main_space_pre < r.space) r.space = pr_main_space_pre;
      if (pr_main2_painted && pr_main_space_setup < r.space) r.space = pr_main_space_setup;
    }
  }
  return n;
}

// Bytes in the union of the spans (sorted and merged); *listed is their plain sum.
static uint32_t pr_union(PrSpan* s, uint32_t n, uint32_t* listed) {
  *listed = 0;
  for (uint32_t i = 0; i < n; ++i) {
    if (s[i].lo < PR_RAM_START) s[i].lo = PR_RAM_START;
    if (s[i].hi > PR_RAM_START + PR_RAM_SIZE) s[i].hi = PR_RAM_START + PR_RAM_SIZE;
    if (s[i].hi < s[i].lo) s[i].hi = s[i].lo;
    *listed += s[i].hi - s[i].lo;
  }
  for (uint32_t i = 1; i < n; ++i) {
    const PrSpan v = s[i];
    uint32_t j = i;
    while (j > 0 && s[j - 1].lo > v.lo) { s[j] = s[j - 1]; --j; }
    s[j] = v;
  }
  uint32_t total = 0, lo = 0, hi = 0;
  bool open = false;
  for (uint32_t i = 0; i < n; ++i) {
    if (s[i].hi == s[i].lo) continue;
    if (!open) { lo = s[i].lo; hi = s[i].hi; open = true; }
    else if (s[i].lo <= hi) { if (s[i].hi > hi) hi = s[i].hi; }
    else { total += hi - lo; lo = s[i].lo; hi = s[i].hi; }
  }
  if (open) total += hi - lo;
  return total;
}

// ---- report --------------------------------------------------------------------------
static void pr_flush() {
  if (pr_len) Serial.write((const uint8_t*)pr_buf, pr_len);
  pr_len = 0;
}
static void pr_put(const char* s) {
  while (*s) {
    if (pr_len >= sizeof(pr_buf)) pr_flush();
    pr_buf[pr_len++] = *s++;
  }
}
static void pr_put_c(char c) {
  const char s[2] = {c, '\0'};
  pr_put(s);
}
static void pr_put_dec(uint32_t v) {
  char s[11];
  int i = 10;
  s[i] = '\0';
  do { s[--i] = (char)('0' + v % 10u); v /= 10u; } while (v);
  pr_put(s + i);
}
static void pr_put_hex(uint32_t v) {
  char s[11] = "0x00000000";
  for (int i = 9; i >= 2; --i) { s[i] = "0123456789ABCDEF"[v & 0xFu]; v >>= 4; }
  pr_put(s);
}
static void pr_key(const char* k) { pr_put_c(' '); pr_put(k); pr_put_c('='); }
static void pr_u(const char* k, uint32_t v) { pr_key(k); pr_put_dec(v); }
static void pr_x(const char* k, uint32_t v) { pr_key(k); pr_put_hex(v); }
static void pr_span(const char* k, uint32_t lo, uint32_t hi) {  // <lo>+<bytes>
  pr_key(k); pr_put_hex(lo); pr_put_c('+'); pr_put_dec(hi > lo ? hi - lo : 0u);
}
static void pr_rph(const char* k, uint32_t reserved, uint32_t peak) {  // reserved/peak/headroom
  pr_key(k); pr_put_dec(reserved); pr_put_c('/'); pr_put_dec(peak); pr_put_c('/');
  pr_put_dec(reserved - peak);
}
// T.<name>@<id>: the RTX name with anything but [A-Za-z0-9_] as '_', or "anon".
static void pr_thread_key(const PrThreadRow& r) {
  pr_put(" T.");
  const char* s = r.name;
  if (!s || !*s) s = "anon";
  for (int i = 0; s[i] && i < 24; ++i) {
    const char c = s[i];
    const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
    pr_put_c(ok ? c : '_');
  }
  pr_put_c('@');
  pr_put_hex(r.id);
  pr_put_c('=');
}

static uint32_t pr_static_bytes() {
  return (PR_ADDR(pr_ld_data_end) - PR_ADDR(pr_ld_data_start)) +
         (PR_ADDR(pr_ld_bss_end) - PR_ADDR(pr_ld_bss_start));
}

#ifdef PEAK_RAM_SAMPLER
#include <reent.h>
#include "peak_ram_sampler.h"
#endif

static void pr_info() {
  const uint32_t fixed_hi = PR_ADDR(pr_ld_data_start);
  const uint32_t ram_hi = PR_RAM_START + PR_RAM_SIZE;
  const uint32_t heap_res = PR_ADDR(pr_ld_heap_limit) - PR_ADDR(pr_ld_heap_base);
  const uint32_t msp_res = PR_ADDR(pr_ld_stack_top) - PR_ADDR(pr_ld_stack_limit);
  const uint32_t gap = PR_RAM_SIZE - (fixed_hi - PR_RAM_START) - pr_static_bytes() - heap_res - msp_res;
  uint32_t main_now = 0;
  const uint32_t n = pr_scan_threads(&main_now);
  pr_put("PR_INFO: v=1 img=" PR_IMAGE " model=" PR_MODEL " force=" PR_FORCE " hook=ctor101");
  pr_span("ram", PR_RAM_START, ram_hi);
  pr_span("fixed", PR_RAM_START, fixed_hi);
  pr_span("data", PR_ADDR(pr_ld_data_start), PR_ADDR(pr_ld_data_end));
  pr_span("bss", PR_ADDR(pr_ld_bss_start), PR_ADDR(pr_ld_bss_end));
  pr_span("heap", PR_ADDR(pr_ld_heap_base), PR_ADDR(pr_ld_heap_limit));
  pr_span("msp", PR_ADDR(pr_ld_stack_limit), PR_ADDR(pr_ld_stack_top));
  pr_u("gap", gap);
  pr_x("pattern", PR_PATTERN);
  pr_x("rtx_fill", osRtxStackFillPattern);
  pr_u("wm", pr_watermark() ? 1u : 0u);
  pr_u("period_ms", PR_PERIOD_MS);
  pr_u("heap_painted", pr_heap_painted ? 1u : 0u);
  pr_u("msp_painted", pr_msp_painted ? 1u : 0u);
  pr_u("main_painted", pr_main_painted ? 1u : 0u);
  pr_u("setup_painted", (pr_msp2_painted && pr_main2_painted) ? 1u : 0u);
  pr_span("heap_paint", pr_heap_paint_lo, pr_heap_painted ? PR_ADDR(pr_ld_heap_limit) : pr_heap_paint_lo);
  pr_span("msp_paint", PR_ADDR(pr_ld_stack_limit), pr_msp_painted ? pr_msp_paint_hi : PR_ADDR(pr_ld_stack_limit));
  pr_span("main_paint", pr_main_paint_lo, pr_main_painted ? pr_main_paint_hi : pr_main_paint_lo);
  pr_x("brk_ctor", pr_brk_ctor);
  pr_x("brk_setup", pr_brk_setup);
  pr_u("nthr", n);
  for (uint32_t i = 0; i < n; ++i) {
    pr_thread_key(pr_rows[i]);
    pr_put_hex(pr_rows[i].base); pr_put_c('+'); pr_put_dec(pr_rows[i].size);
    pr_put_c('/'); pr_put(pr_rows[i].loc);
  }
#ifdef PEAK_RAM_SAMPLER
  prs_info();
#endif
  pr_put("\r\n");
  pr_flush();
}

static void pr_report() {
  mbed_stats_heap_t hs_before;
  mbed_stats_heap_get(&hs_before);
  const uint32_t t0 = micros();
  ++pr_reports;

  const uint32_t heap_base = PR_ADDR(pr_ld_heap_base), heap_limit = PR_ADDR(pr_ld_heap_limit);
  const uint32_t stack_limit = PR_ADDR(pr_ld_stack_limit), stack_top = PR_ADDR(pr_ld_stack_top);
  const uint32_t heap_top = pr_heap_top();
  uint32_t msp_loop_low = stack_limit;
  const uint32_t msp_low = pr_msp_low_lifetime(&msp_loop_low);
  uint32_t main_now = 0;
  const uint32_t n = pr_scan_threads(&main_now);

  uint32_t k = 0;
  pr_spans[k++] = {heap_top, heap_limit};
  pr_spans[k++] = {stack_limit, msp_low};
  for (uint32_t i = 0; i < n; ++i) pr_spans[k++] = {pr_rows[i].base, pr_rows[i].base + pr_rows[i].space};
  uint32_t listed = 0;
  const uint32_t untouched = pr_union(pr_spans, k, &listed);
  const uint32_t scan_us = micros() - t0;

  uint32_t main_size = 0;
  for (uint32_t i = 0; i < n; ++i) {
    if (pr_ids[i] == pr_main_id) main_size = pr_rows[i].size;
  }
  mbed_stats_heap_t hs;
  mbed_stats_heap_get(&hs);

  pr_put("PR: v=1");
  pr_u("t_ms", millis());
  pr_u("rep", pr_reports);
  pr_u("dec", pr_decisions);
  pr_u("lf", pr_live_finger);
  pr_u("lnf", pr_live_nofinger);
  pr_rph("heap", heap_limit - heap_base, heap_top - heap_base);
  pr_rph("msp", stack_top - stack_limit, stack_top - msp_low);
  pr_u("msp_loop", stack_top - msp_loop_low);
  pr_u("main_pre", main_size - pr_main_space_pre);
  pr_u("main_setup", main_size - pr_main_space_setup);
  pr_u("main_loop", main_size - main_now);
  pr_u("nthr", n);
  for (uint32_t i = 0; i < n; ++i) {
    pr_thread_key(pr_rows[i]);
    pr_put_dec(pr_rows[i].size); pr_put_c('/');
    pr_put_dec(pr_rows[i].size - pr_rows[i].space); pr_put_c('/');
    pr_put_dec(pr_rows[i].space); pr_put_c('/');
    pr_put(pr_rows[i].loc);
  }
  pr_x("brk", (uint32_t)pr_sbrk(0));
  pr_x("heap_hw", heap_top);
  pr_u("hs_cur", hs.current_size);
  pr_u("hs_max", hs.max_size);
  pr_u("hs_ovh", hs.overhead_size);
  pr_u("hs_fail", hs.alloc_fail_cnt);
  pr_u("static", pr_static_bytes());
  pr_u("fixed", PR_ADDR(pr_ld_data_start) - PR_RAM_START);
  pr_u("untouched", untouched);
  pr_u("ovl", listed - untouched);
  pr_u("whole_hw", PR_RAM_SIZE - untouched);
  pr_u("ram", PR_RAM_SIZE);
  pr_u("scan_us", scan_us);
  pr_u("rpt_alloc", pr_rpt_alloc);  // bytes allocated (any thread) during the previous report
#ifdef PEAK_RAM_SAMPLER
  prs_report(hs, n);
#endif
  pr_put("\r\n");
  pr_flush();

  mbed_stats_heap_t hs_after;
  mbed_stats_heap_get(&hs_after);
  pr_rpt_alloc = hs_after.total_size - hs_before.total_size;
}

// ---- hooks called from main.cpp -------------------------------------------------------
static inline void pr_count_live_gate(bool finger) {
  if (finger) ++pr_live_finger; else ++pr_live_nofinger;
}

static inline void pr_on_decision() { ++pr_decisions; }

// End of setup(): close the boot+setup phase of the MSP and main stack, repaint them for
// the loop phase, then print PR_INFO: and the first PR: line.
static void pr_setup_done() {
  pr_brk_setup = (uint32_t)pr_sbrk(0);
  if (pr_brk_setup > pr_brk_max) pr_brk_max = pr_brk_setup;
  if (pr_msp_painted) {
    const uint32_t primask = __get_PRIMASK();
    __disable_irq();  // no exception between the setup-phase scan and the repaint
    pr_msp_low_setup = pr_msp_low(pr_msp_paint_hi);
    pr_msp2_paint_hi = pr_paint_msp();
    __set_PRIMASK(primask);
    pr_msp2_painted = pr_msp2_paint_hi != 0;
  }
  if (pr_main_painted) {
    pr_main2_paint_hi = pr_paint_main(&pr_main_space_setup);
    pr_main2_painted = pr_main2_paint_hi != 0;
  } else {
    pr_main_space_setup = osThreadGetStackSpace(osThreadGetId());
  }
  pr_started = true;
  pr_last_ms = millis();
#ifdef PEAK_RAM_SAMPLER
  prs_start();
#endif
  pr_info();
  pr_report();
}

// Called at the top of every loop() pass.
static void pr_poll() {
  if (!pr_started) return;
#ifdef PEAK_RAM_SAMPLER
  prs_poll_ble();
#endif
  const uint32_t now = millis();
  if ((uint32_t)(now - pr_last_ms) < PR_PERIOD_MS) return;
  pr_last_ms = now;
  pr_report();
  if (pr_reports % PR_INFO_EVERY == 0u) pr_info();
}

#endif  // PEAK_RAM_PROBE
