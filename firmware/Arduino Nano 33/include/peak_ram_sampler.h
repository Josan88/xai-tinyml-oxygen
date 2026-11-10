#pragma once
// Instantaneous-total sampler for the peak-RAM probe (envs nano33ble_peakram_sampler and
// nano33ble_v2_f1a_peakram_sampler: -DPEAK_RAM_PROBE -DPEAK_RAM_LIVE_GATE -DPEAK_RAM_SAMPLER).
// Method and proofs: docs/peak-ram-design.md, section 8. Included by peak_ram_probe.h only,
// after its report helpers. Not a shipped image.
//
// A TIMER4 compare interrupt at 4 kHz, at the priority of PendSV/SysTick, adds up at each tick
//   s_const (fixed + static - every thread's stack size) + heap in use (break - __end__ - free
//   newlib-nano chunks) + every thread's stack from its SP to its top + the MSP in use.
// That is in_use(t) of section 7 at one instant. The largest value is samp_max, a measured lower
// bound on the peak; s_win is the largest since the previous PR: line. A tick is skipped (and
// counted) if a kernel exception is active, the newlib malloc mutex is held, the free list is
// implausible, or a thread does not match the table built at the end of setup().
//
// RTX in this core (precompiled, no source in the package; read from the ELF): svcRtxMutexAcquire
// takes a mutex only when its lock count is 0 and then sets owner_thread, lock = 1 and puts it at
// the head of the owner's mutex_list. svcRtxMutexRelease unlinks it and lowers lock, but writes
// owner_thread only when it hands the mutex to a waiter: after the last release owner_thread still
// names the old owner. "Held" is therefore lock != 0, RTX's own test (design section 8, 1 Nov note).
#ifdef PEAK_RAM_SAMPLER

static const uint32_t PRS_HZ = 4000u;
static const uint32_t PRS_PERIOD_US = 1000000u / PRS_HZ;  // TIMER4 at 1 MHz
static const uint32_t PRS_MAX_THREADS = 8u;
static const uint32_t PRS_MAX_FREE = 32u;                 // free-list walk bound per tick

// newlib-nano free chunk (nano-mallocr.c): size includes the 4 B header; list sorted by address.
struct PrsChunk { int32_t size; PrsChunk* next; };
extern PrsChunk* prs_free_list __asm__("__malloc_free_list");
extern void prs_malloc_lock(struct _reent*) __asm__("__malloc_lock");
extern void prs_malloc_unlock(struct _reent*) __asm__("__malloc_unlock");

struct PrsThread { osRtxThread_t* t; uint32_t lo, top; };
static PrsThread prs_thr[PRS_MAX_THREADS];  // built once at the end of setup(), read by the ISR
static uint32_t prs_nthr = 0;
static uint32_t prs_const = 0;
static bool prs_running = false, prs_set_ok = false;
static osRtxMutex_t* prs_hlock = nullptr;
static uint32_t prs_lvl_svc = 0, prs_lvl_pendsv = 0, prs_lvl_systick = 0, prs_lvl = 0;
static uint32_t prs_ble_conn = 0, prs_ble_disc = 0;
static bool prs_ble_last = false;
// prs_start()'s three preconditions as evaluated (PR_INFO st_*), and each step of the malloc-mutex
// lookup (PR_INFO lk_*; step: 0 = found, else the first check that failed).
static uint32_t prs_st_hlock = 0, prs_st_set = 0, prs_st_vtor = 0;
struct PrsLookup {
  uint32_t step, ipsr, mask, self, pre, head, id, own, cnt, post, rel, stale;
};
static PrsLookup prs_lk;

struct PrsState {
  uint32_t n, ok, skk, skl, skh, skt;           // ticks, samples, skips by cause
  uint32_t max, win, max_heap, max_msp, max_stk; // samp_max, s_win, breakdown at samp_max
  uint32_t cmax_heap, cmax_msp;                  // component maxima over samples
  uint32_t cyc_max, cyc_sum;
  uint32_t max_depth[PRS_MAX_THREADS], cmax_depth[PRS_MAX_THREADS];
};
static volatile PrsState prs;             // written by the ISR
static PrsState prs_snap;                 // copy for the report, TIMER4 interrupt masked
static volatile uint32_t prs_depth_now[PRS_MAX_THREADS];  // ISR scratch (no stack locals)
static volatile uint32_t prs_c0;                          // DWT->CYCCNT at entry (ditto)

// Heap in use from the allocator's own state: break - __end__ - free chunks. Returns false if the
// free list is implausible (more than PRS_MAX_FREE chunks, or a chunk outside [__end__, break)).
__attribute__((always_inline)) static inline bool prs_heap_in_use(uint32_t* out) {
  const uint32_t base = PR_ADDR(pr_ld_heap_base);
  const uint32_t brk = (uint32_t)pr_sbrk(0);
  uint32_t free_bytes = 0, n = 0;
  for (const PrsChunk* c = prs_free_list; c != nullptr; c = c->next) {
    const uint32_t a = (uint32_t)c;
    if (++n > PRS_MAX_FREE || a < base || c->size <= 0 || a + (uint32_t)c->size > brk) return false;
    free_bytes += (uint32_t)c->size;
  }
  *out = brk - base - free_bytes;
  return true;
}

// The sampling body. Runs at the level of PendSV and SysTick, so it only ever interrupts thread
// mode (directly or by tail-chaining): no kernel exception can be active or preempt it, and SVC
// is synchronous. exc_return: LR at entry; psp: PSP at entry (the interrupted thread's SP,
// hardware frame included); msp: MSP after prs_isr's push, which points at a word the sampler
// wrote. The body's own frame lies below msp and is not counted (it may hold unwritten slots).
extern "C" void prs_body(uint32_t exc_return, uint32_t psp, uint32_t msp) {
  prs_c0 = DWT->CYCCNT;
  NRF_TIMER4->EVENTS_COMPARE[0] = 0;
  (void)NRF_TIMER4->EVENTS_COMPARE[0];  // complete the write before returning
  ++prs.n;
  do {
    if ((exc_return & 4u) == 0u ||
        (SCB->SHCSR & (SCB_SHCSR_SVCALLACT_Msk | SCB_SHCSR_PENDSVACT_Msk | SCB_SHCSR_SYSTICKACT_Msk)) != 0u) {
      ++prs.skk;
      break;
    }
    if (prs_hlock->lock != 0u) { ++prs.skl; break; }  // owner_thread is stale after release
    uint32_t heap = 0;
    if (!prs_heap_in_use(&heap)) { ++prs.skh; break; }
    osRtxThread_t* const curr = osRtxInfo.thread.run.curr;
    uint32_t stacks = 0;
    bool ok = false;  // the running thread must be in the table
    uint32_t i = 0;
    for (; i < prs_nthr; ++i) {
      const PrsThread& r = prs_thr[i];
      const bool running = r.t == curr;
      const uint32_t sp = running ? psp : r.t->sp;
      if (r.t->id != osRtxIdThread || sp < r.lo || sp > r.top) break;
      ok = ok || running;
      prs_depth_now[i] = r.top - sp;
      stacks += r.top - sp;
    }
    if (i != prs_nthr || !ok) { ++prs.skt; break; }
    const uint32_t msp_in_use = PR_ADDR(pr_ld_stack_top) - msp;
    const uint32_t total = prs_const + heap + stacks + msp_in_use;
    ++prs.ok;
    if (heap > prs.cmax_heap) prs.cmax_heap = heap;
    if (msp_in_use > prs.cmax_msp) prs.cmax_msp = msp_in_use;
    for (i = 0; i < prs_nthr; ++i) {
      if (prs_depth_now[i] > prs.cmax_depth[i]) prs.cmax_depth[i] = prs_depth_now[i];
    }
    if (total > prs.win) prs.win = total;
    if (total > prs.max) {
      prs.max = total;
      prs.max_heap = heap;
      prs.max_msp = msp_in_use;
      prs.max_stk = stacks;
      for (i = 0; i < prs_nthr; ++i) prs.max_depth[i] = prs_depth_now[i];
    }
  } while (false);
  const uint32_t cyc = DWT->CYCCNT - prs_c0;
  prs.cyc_sum += cyc;
  if (cyc > prs.cyc_max) prs.cyc_max = cyc;
}

// Pushes r4 and LR (EXC_RETURN) on the MSP, reads MSP there (a written word) and PSP, calls the
// body, and returns by popping EXC_RETURN into PC (exception return, as the looptiming2 sampler,
// outputs/firmware-v2-looptiming2/). The 8 B push keeps the MSP's 8-byte alignment.
extern "C" __attribute__((naked)) void prs_isr(void) {
  __asm volatile(
    "push {r4, lr} \n"
    "mrs r2, msp   \n"
    "mrs r1, psp   \n"
    "mov r0, lr    \n"
    "bl prs_body   \n"
    "pop {r4, pc}  \n");
}

// Thread table, once, at the end of setup(): every thread exists by then (main, idle, timer,
// USBevt, the BLE thread). s_const = fixed + static - the stack sizes of all of them, because
// each stack is added back from its SP at every sample (heap stacks are inside the heap term).
static bool prs_build_table() {
  uint32_t n = osThreadEnumerate(pr_ids, PR_MAX_THREADS);
  if (n == 0u || n > PRS_MAX_THREADS) return false;
  uint32_t sizes = 0;
  for (uint32_t i = 0; i < n; ++i) {
    osRtxThread_t* t = (osRtxThread_t*)pr_ids[i];
    const uint32_t lo = (uint32_t)t->stack_mem;
    if (pr_location(lo)[0] == 'o') return false;  // neither .bss nor heap
    prs_thr[i] = {t, lo, lo + t->stack_size};
    sizes += t->stack_size;
  }
  prs_nthr = n;
  prs_const = (PR_ADDR(pr_ld_data_start) - PR_RAM_START) + pr_static_bytes() - sizes;
  return true;
}

// The table still matches the enumeration (the report's pr_scan_threads filled pr_ids).
static void prs_check_set(uint32_t n) {
  if (n != prs_nthr) { prs_set_ok = false; return; }
  for (uint32_t i = 0; i < n; ++i) {
    bool found = false;
    for (uint32_t j = 0; j < prs_nthr; ++j) found = found || prs_thr[j].t == (osRtxThread_t*)pr_ids[i];
    if (!found) prs_set_ok = false;
  }
}

// The newlib malloc mutex: take __malloc_lock once in thread context and read the new head of the
// caller's mutex_list. The head must have changed, carry the mutex id, be owned by the caller with
// lock 1 (a fresh acquire), and be off the list again after __malloc_unlock. Not tested: its
// owner_thread after the release (lk_stale; RTX leaves it set) and its lock after the release
// (lk_rel; 0 unless a waiting thread took the mutex over in that release).
static osRtxMutex_t* prs_find_malloc_mutex() {
  PrsLookup& k = prs_lk;
  k.ipsr = __get_IPSR();
  k.mask = (__get_PRIMASK() != 0u || __get_BASEPRI() != 0u) ? 1u : 0u;  // as RTX's IsIrqMasked
  osRtxThread_t* const self = (osRtxThread_t*)osThreadGetId();
  k.self = (uint32_t)self;
  if (k.ipsr != 0u || k.mask != 0u) { k.step = 1; return nullptr; }  // osMutexAcquire would refuse
  if (self == nullptr) { k.step = 2; return nullptr; }
  osRtxMutex_t* const before = self->mutex_list;
  k.pre = (uint32_t)before;
  prs_malloc_lock(_REENT);
  osRtxMutex_t* const m = self->mutex_list;
  const bool fresh = m != nullptr && m != before;
  k.head = (uint32_t)m;
  if (fresh) {
    k.id = m->id;
    k.own = m->owner_thread == self ? 1u : 0u;
    k.cnt = m->lock;
  }
  prs_malloc_unlock(_REENT);
  k.post = (uint32_t)self->mutex_list;
  if (fresh) {
    k.rel = m->lock;
    k.stale = m->owner_thread == self ? 1u : 0u;
  }
  if (!fresh) k.step = 3;                  // nothing new at the head
  else if (k.id != osRtxIdMutex) k.step = 4;
  else if (k.own != 1u) k.step = 5;
  else if (k.cnt != 1u) k.step = 6;        // not a fresh acquire
  else if (k.post != k.pre) k.step = 7;    // the release did not unlink it
  else k.step = 0;
  return k.step == 0u ? m : nullptr;
}

// End of setup(), after the repaint: find the lock, build the table, start TIMER4.
static void prs_start() {
  prs_lvl_svc = SCB->SHP[7] >> (8u - __NVIC_PRIO_BITS);
  prs_lvl_pendsv = SCB->SHP[10] >> (8u - __NVIC_PRIO_BITS);
  prs_lvl_systick = SCB->SHP[11] >> (8u - __NVIC_PRIO_BITS);
  prs_lvl = prs_lvl_pendsv < prs_lvl_systick ? prs_lvl_pendsv : prs_lvl_systick;
  prs_hlock = prs_find_malloc_mutex();
  prs_set_ok = prs_build_table();
  prs_st_hlock = prs_hlock != nullptr ? 1u : 0u;
  prs_st_set = prs_set_ok ? 1u : 0u;
  // The vector table must be the RAM copy mbed installs (nrf_reloc_vector_table).
  prs_st_vtor = SCB->VTOR == PR_RAM_START ? 1u : 0u;
  if (prs_st_hlock == 0u || prs_st_set == 0u || prs_st_vtor == 0u) return;
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
  NRF_TIMER4->TASKS_STOP = 1;
  NRF_TIMER4->TASKS_CLEAR = 1;
  NRF_TIMER4->MODE = TIMER_MODE_MODE_Timer;
  NRF_TIMER4->BITMODE = TIMER_BITMODE_BITMODE_32Bit;
  NRF_TIMER4->PRESCALER = 4;  // 16 MHz / 2^4 = 1 MHz
  NRF_TIMER4->CC[0] = PRS_PERIOD_US;
  NRF_TIMER4->SHORTS = TIMER_SHORTS_COMPARE0_CLEAR_Msk;
  NRF_TIMER4->EVENTS_COMPARE[0] = 0;
  NRF_TIMER4->INTENSET = TIMER_INTENSET_COMPARE0_Msk;
  NVIC_SetVector(TIMER4_IRQn, (uint32_t)&prs_isr);
  NVIC_SetPriority(TIMER4_IRQn, prs_lvl);
  NVIC_ClearPendingIRQ(TIMER4_IRQn);
  NVIC_EnableIRQ(TIMER4_IRQn);
  NRF_TIMER4->TASKS_START = 1;
  prs_running = true;
}

// Top of every loop() pass: count connection changes (bleConnected changes only in loop()).
static void prs_poll_ble() {
  if (bleConnected != prs_ble_last) {
    if (bleConnected) ++prs_ble_conn; else ++prs_ble_disc;
    prs_ble_last = bleConnected;
  }
}

static void prs_info() {
  pr_u("smp", prs_running ? 1u : 0u);
  pr_put(" smp_timer=TIMER4");
  pr_u("smp_hz", PRS_HZ);
  pr_u("smp_prio", prs_lvl);
  pr_u("prio_svc", prs_lvl_svc);
  pr_u("prio_pendsv", prs_lvl_pendsv);
  pr_u("prio_systick", prs_lvl_systick);
  pr_x("vtor", SCB->VTOR);
  pr_x("hlock", (uint32_t)prs_hlock);
  pr_u("s_const", prs_const);
  pr_u("s_nthr", prs_nthr);
  // Start diagnostics (appended 1 Nov 2026; no check reads them).
  pr_u("st_hlock", prs_st_hlock);
  pr_u("st_set", prs_st_set);
  pr_u("st_vtor", prs_st_vtor);
  pr_u("lk_step", prs_lk.step);
  pr_u("lk_ipsr", prs_lk.ipsr);
  pr_u("lk_mask", prs_lk.mask);
  pr_x("lk_self", prs_lk.self);
  pr_x("lk_pre", prs_lk.pre);
  pr_x("lk_head", prs_lk.head);
  pr_x("lk_id", prs_lk.id);
  pr_u("lk_own", prs_lk.own);
  pr_u("lk_cnt", prs_lk.cnt);
  pr_x("lk_post", prs_lk.post);
  pr_u("lk_rel", prs_lk.rel);
  pr_u("lk_stale", prs_lk.stale);
}

static void prs_thread_key(const PrsThread& r) {
  pr_put(" S.");
  const char* s = r.t->name;
  if (!s || !*s) s = "anon";
  for (int i = 0; s[i] && i < 24; ++i) {
    const char c = s[i];
    const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
    pr_put_c(ok ? c : '_');
  }
  pr_put_c('@');
  pr_put_hex((uint32_t)r.t);
  pr_put_c('=');
}

// Appended to every PR: line. hs: the heap statistics the line already printed.
static void prs_report(const mbed_stats_heap_t& hs, uint32_t nthr) {
  prs_check_set(nthr);
  // Heap in use from the free list, under the malloc lock, bracketed by two statistics reads:
  // hl_st=1 when no allocation or free completed since the line's hs_* fields were read.
  uint32_t hl = 0;
  prs_malloc_lock(_REENT);
  const bool walk_ok = prs_heap_in_use(&hl);
  prs_malloc_unlock(_REENT);
  mbed_stats_heap_t after;
  mbed_stats_heap_get(&after);
  const bool stable = walk_ok && after.total_size == hs.total_size && after.alloc_cnt == hs.alloc_cnt &&
                      after.current_size == hs.current_size;
  NVIC_DisableIRQ(TIMER4_IRQn);  // consistent copy; a pending tick runs after re-enabling
  prs_snap.n = prs.n; prs_snap.ok = prs.ok;
  prs_snap.skk = prs.skk; prs_snap.skl = prs.skl; prs_snap.skh = prs.skh; prs_snap.skt = prs.skt;
  prs_snap.max = prs.max; prs_snap.win = prs.win; prs.win = 0;
  prs_snap.max_heap = prs.max_heap; prs_snap.max_msp = prs.max_msp; prs_snap.max_stk = prs.max_stk;
  prs_snap.cmax_heap = prs.cmax_heap; prs_snap.cmax_msp = prs.cmax_msp;
  prs_snap.cyc_max = prs.cyc_max; prs_snap.cyc_sum = prs.cyc_sum;
  for (uint32_t i = 0; i < prs_nthr; ++i) {
    prs_snap.max_depth[i] = prs.max_depth[i];
    prs_snap.cmax_depth[i] = prs.cmax_depth[i];
  }
  if (prs_running) NVIC_EnableIRQ(TIMER4_IRQn);
  pr_u("ble", bleConnected ? 1u : 0u);
  pr_u("sub", dataChar.subscribed() ? 1u : 0u);
  pr_u("ble_conn", prs_ble_conn);
  pr_u("ble_disc", prs_ble_disc);
  pr_u("hl", hl);
  pr_u("hl_st", stable ? 1u : 0u);
  pr_u("s_n", prs_snap.n);
  pr_u("s_ok", prs_snap.ok);
  pr_u("s_skk", prs_snap.skk);
  pr_u("s_skl", prs_snap.skl);
  pr_u("s_skh", prs_snap.skh);
  pr_u("s_skt", prs_snap.skt);
  pr_u("s_set", prs_set_ok ? 1u : 0u);
  pr_u("samp_max", prs_snap.max);
  pr_u("s_win", prs_snap.win);
  pr_u("sm_heap", prs_snap.max_heap);
  pr_u("sm_msp", prs_snap.max_msp);
  pr_u("sm_stk", prs_snap.max_stk);
  pr_u("smx_heap", prs_snap.cmax_heap);
  pr_u("smx_msp", prs_snap.cmax_msp);
  pr_u("s_cyc_max", prs_snap.cyc_max);
  pr_u("s_cyc_sum", prs_snap.cyc_sum);
  for (uint32_t i = 0; i < prs_nthr; ++i) {
    prs_thread_key(prs_thr[i]);
    pr_put_dec(prs_snap.max_depth[i]); pr_put_c('/'); pr_put_dec(prs_snap.cmax_depth[i]);
  }
}

#endif  // PEAK_RAM_SAMPLER
