"""Host model of the peak-RAM sampler (firmware/Arduino Nano 33/include/peak_ram_sampler.h).
Used by tests/test_peak_ram_sampler.py on synthetic states; it never touches a board.

Each function mirrors one firmware function:
    heap_in_use       prs_heap_in_use
    s_const           prs_build_table's s_const
    Sampler.tick      prs_body
    Sampler.report    prs_report's snapshot (s_win restarts at every PR: line)
    format_fields     prs_report's output format (the fields the analyser reads)
    find_malloc_mutex prs_find_malloc_mutex (r2, 1 Nov 2026), with its PR_INFO lk_* fields
and in_use / whole_hw restate the definitions of docs/peak-ram-design.md sections 3 and 7, so the
tests can check that a sample is in_use(t) and that it cannot exceed whole_hw. Rtx models the two
RTX mutex service calls as read from the core's ELF (design section 8, 1 Nov note), and
find_malloc_mutex_d7c9d54 is the lookup the 31 Oct images ran.
"""
from dataclasses import dataclass, field

RTX_ID_THREAD = 0xF1
RTX_ID_MUTEX = 0xF5
SHCSR_KERNEL = (1 << 7) | (1 << 10) | (1 << 11)   # SVCALLACT, PENDSVACT, SYSTICKACT
EXC_RETURN_PSP = 0xFFFFFFFD                         # return to thread mode, PSP
MAX_FREE = 32                                       # PRS_MAX_FREE


def heap_in_use(heap_base, brk, free_chunks, max_free=MAX_FREE):
    """break - __end__ - free chunk bytes, or None if the free list is implausible.

    free_chunks: [(address, size)] in list order (newlib-nano keeps it sorted by address).
    """
    free_bytes = 0
    for n, (addr, size) in enumerate(free_chunks, start=1):
        if n > max_free or addr < heap_base or size <= 0 or addr + size > brk:
            return None
        free_bytes += size
    return brk - heap_base - free_bytes


def s_const(fixed, static, stack_sizes):
    """fixed + static - every thread's stack size (each stack is added back from its SP)."""
    return fixed + static - sum(stack_sizes)


@dataclass
class Thread:
    tid: int          # control-block address (identity)
    lo: int           # stack_mem
    size: int
    sp: int           # saved SP (ignored for the running thread: the sampler uses PSP)
    obj_id: int = RTX_ID_THREAD

    @property
    def top(self):
        return self.lo + self.size


@dataclass
class State:
    """What the sampler reads at one tick."""
    threads: list
    curr: int                 # tid of osRtxInfo.thread.run.curr
    psp: int
    msp: int                  # MSP after prs_isr's push
    stack_top: int
    heap_base: int
    brk: int
    free_chunks: list = field(default_factory=list)
    exc_return: int = EXC_RETURN_PSP
    shcsr: int = 0
    lock_owner: int = 0       # malloc mutex owner_thread; RTX leaves it set after release, never read
    lock_count: int = 0       # malloc mutex lock: held iff != 0


class Sampler:
    """prs_body's counters and maxima."""

    def __init__(self, table, const):
        self.table = list(table)            # [Thread] as built at the end of setup()
        self.const = const
        self.n = self.ok = self.skk = self.skl = self.skh = self.skt = 0
        self.max = self.win = self.max_heap = self.max_msp = self.max_stk = 0
        self.cmax_heap = self.cmax_msp = 0
        self.max_depth = [0] * len(self.table)
        self.cmax_depth = [0] * len(self.table)

    def tick(self, st):
        """One timer tick. Returns the total, or None for a skipped tick."""
        self.n += 1
        if (st.exc_return & 4) == 0 or (st.shcsr & SHCSR_KERNEL):
            self.skk += 1
            return None
        if st.lock_count:
            self.skl += 1
            return None
        heap = heap_in_use(st.heap_base, st.brk, st.free_chunks)
        if heap is None:
            self.skh += 1
            return None
        live = {t.tid: t for t in st.threads}
        depths, seen_curr = [], False
        for r in self.table:
            t = live.get(r.tid)
            running = r.tid == st.curr
            sp = st.psp if running else (t.sp if t else 0)
            if t is None or t.obj_id != RTX_ID_THREAD or not r.lo <= sp <= r.top:
                self.skt += 1
                return None
            seen_curr = seen_curr or running
            depths.append(r.top - sp)
        if not seen_curr:
            self.skt += 1
            return None
        msp_in_use = st.stack_top - st.msp
        stacks = sum(depths)
        total = self.const + heap + stacks + msp_in_use
        self.ok += 1
        self.cmax_heap = max(self.cmax_heap, heap)
        self.cmax_msp = max(self.cmax_msp, msp_in_use)
        self.cmax_depth = [max(a, b) for a, b in zip(self.cmax_depth, depths)]
        self.win = max(self.win, total)
        if total > self.max:
            self.max, self.max_heap, self.max_msp, self.max_stk = total, heap, msp_in_use, stacks
            self.max_depth = list(depths)
        return total

    def report(self):
        """The PR: line's snapshot; s_win restarts."""
        snap = {"s_n": self.n, "s_ok": self.ok, "s_skk": self.skk, "s_skl": self.skl, "s_skh": self.skh,
                "s_skt": self.skt, "samp_max": self.max, "s_win": self.win, "sm_heap": self.max_heap,
                "sm_msp": self.max_msp, "sm_stk": self.max_stk, "smx_heap": self.cmax_heap,
                "smx_msp": self.cmax_msp, "max_depth": list(self.max_depth), "cmax_depth": list(self.cmax_depth)}
        self.win = 0
        return snap


@dataclass
class Mutex:
    """The osRtxMutex_t fields the lookup and the sampler read."""
    addr: int
    obj_id: int = RTX_ID_MUTEX
    owner: int = 0            # owner_thread
    lock: int = 0
    waiters: list = field(default_factory=list)   # tids blocked on it, the one RTX wakes first first


class Rtx:
    """svcRtxMutexAcquire / svcRtxMutexRelease of the 4.5.0 core, as read from the ELF, reduced to
    the fields above and each thread's mutex_list (head first). osMutexAcquire and osMutexRelease
    return osErrorISR (-6) and change nothing from an exception or with interrupts masked."""

    def __init__(self):
        self.mutex_list = {}

    def head(self, tid):
        owned = self.mutex_list.get(tid, [])
        return owned[0] if owned else None

    def acquire(self, m, tid, ipsr=0, masked=False):
        if ipsr or masked:
            return -6
        if m.lock == 0:                       # free: take it, put it at the head of the owner's list
            m.owner, m.lock = tid, 1
            self.mutex_list.setdefault(tid, []).insert(0, m)
            return 0
        if m.owner == tid:                    # recursive (the malloc mutex is osMutexRecursive)
            m.lock += 1
            return 0
        raise NotImplementedError("a blocking acquire is not modelled")

    def release(self, m, tid, ipsr=0, masked=False):
        if ipsr or masked:
            return -6
        if m.lock == 0 or m.owner != tid:
            return -3                         # osErrorResource
        m.lock -= 1
        if m.lock == 0:
            self.mutex_list[tid].remove(m)
            if m.waiters:                     # hand-over: the only path that writes owner_thread
                new = m.waiters.pop(0)
                m.owner, m.lock = new, 1
                self.mutex_list.setdefault(new, []).insert(0, m)
            # no waiter: owner_thread keeps naming the old owner
        return 0


def _addr(m):
    return m.addr if m is not None else 0


def find_malloc_mutex(rtx, malloc_mutex, self_tid, ipsr=0, masked=False, while_held=None):
    """prs_find_malloc_mutex as of 1 Nov 2026: (mutex or None, the PR_INFO lk_* values without the
    prefix). __malloc_lock / __malloc_unlock act on malloc_mutex; while_held(rtx) runs between them."""
    k = {"step": 0, "ipsr": ipsr, "mask": 1 if masked else 0, "self": self_tid, "pre": 0, "head": 0, "id": 0,
         "own": 0, "cnt": 0, "post": 0, "rel": 0, "stale": 0}
    if ipsr or masked:
        k["step"] = 1
        return None, k
    if not self_tid:
        k["step"] = 2
        return None, k
    before = rtx.head(self_tid)
    k["pre"] = _addr(before)
    rtx.acquire(malloc_mutex, self_tid)
    m = rtx.head(self_tid)
    fresh = m is not None and m is not before
    k["head"] = _addr(m)
    if fresh:
        k["id"], k["own"], k["cnt"] = m.obj_id, int(m.owner == self_tid), m.lock
    if while_held:
        while_held(rtx)
    rtx.release(malloc_mutex, self_tid)
    k["post"] = _addr(rtx.head(self_tid))
    if fresh:
        k["rel"], k["stale"] = m.lock, int(m.owner == self_tid)
    if not fresh:
        k["step"] = 3
    elif k["id"] != RTX_ID_MUTEX:
        k["step"] = 4
    elif k["own"] != 1:
        k["step"] = 5
    elif k["cnt"] != 1:
        k["step"] = 6
    elif k["post"] != k["pre"]:
        k["step"] = 7
    return (m if k["step"] == 0 else None), k


def find_malloc_mutex_d7c9d54(rtx, malloc_mutex, self_tid):
    """The lookup the 31 Oct images ran (commit d7c9d54): it needs owner_thread == 0 after release."""
    before = rtx.head(self_tid)
    rtx.acquire(malloc_mutex, self_tid)
    m = rtx.head(self_tid)
    owned = m is not None and m is not before and m.obj_id == RTX_ID_MUTEX and m.owner == self_tid
    rtx.release(malloc_mutex, self_tid)
    return m if owned and m.owner == 0 and m.lock == 0 else None


def in_use(fixed, static, threads, sp_of, heap_live, msp, stack_top):
    """Design section 7: in_use(t) for one instant.

    threads: [(Thread, loc)] with loc 'bss' or 'heap'; sp_of: tid -> SP at that instant.
    heap_live: allocated chunk bytes.
    """
    bss = sum(sp_of[t.tid] - t.lo for t, loc in threads if loc == "bss")
    heap = sum(sp_of[t.tid] - t.lo for t, loc in threads if loc == "heap")
    return fixed + static - bss + heap_live - heap + (stack_top - msp)


def whole_hw(fixed, static, threads, space_of, heap_peak, msp_peak):
    """Design section 3, written in the same terms: space_of: tid -> RTX never-touched bytes."""
    bss = sum(space_of[t.tid] for t, loc in threads if loc == "bss")
    heap = sum(space_of[t.tid] for t, loc in threads if loc == "heap")
    return fixed + static - bss + heap_peak - heap + msp_peak


def format_fields(snap, ble, sub, ble_conn, ble_disc, hl, hl_st, s_set, cyc_max, cyc_sum, thread_keys):
    """The sampler tail of a PR: line, in prs_report's key order."""
    parts = [f"ble={ble}", f"sub={sub}", f"ble_conn={ble_conn}", f"ble_disc={ble_disc}", f"hl={hl}",
             f"hl_st={hl_st}"]
    for k in ("s_n", "s_ok", "s_skk", "s_skl", "s_skh", "s_skt"):
        parts.append(f"{k}={snap[k]}")
    parts.append(f"s_set={s_set}")
    for k in ("samp_max", "s_win", "sm_heap", "sm_msp", "sm_stk", "smx_heap", "smx_msp"):
        parts.append(f"{k}={snap[k]}")
    parts += [f"s_cyc_max={cyc_max}", f"s_cyc_sum={cyc_sum}"]
    for key, at_max, mx in zip(thread_keys, snap["max_depth"], snap["cmax_depth"]):
        parts.append(f"S.{key}={at_max}/{mx}")
    return " ".join(parts)
