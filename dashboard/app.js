const core = globalThis.DashboardCore;
if (!core) {
  throw new Error('dashboard_core.js must load before app.js');
}

const SERVICE_UUID = '19b10000-e8f2-537e-4f6c-d104768a1214';
const DATA_CHAR_UUID = '19b10001-e8f2-537e-4f6c-d104768a1214';

const byId = (id) => document.getElementById(id);

const connectBtn = byId('connectBtn');
const statusDot = byId('statusDot');
const statusText = byId('statusText');
const notSupported = byId('notSupported');
const replayView = byId('replayView');
const liveView = byId('liveView');

let bleDevice = null;
let pulseTimer = null;

// Shared render helpers live in dashboard_view.js.
const {
  el, clearChildren, isScore, sideOf, relationText, makeTrack, setTrack,
  makeScore, renderScore, renderDrivers,
} = globalThis.DashboardView;

// ---------------------------------------------------------------------------
// What this result means: the supervisor results guide as a legend of all five
// bands, plus a card for the current one. Text comes from resultGuideFor.

function makeGuide(root, mode) {
  const part = (role) => root.querySelector('[data-role="' + role + '"]');
  const legend = part('legend');
  const rows = {};
  core.resultGuideRows(mode).forEach((row) => {
    const li = el('li', '');
    li.dataset.level = row.level;
    li.dataset.available = row.available ? 'true' : 'false';
    const swatch = el('span', 'sw');
    swatch.setAttribute('aria-hidden', 'true');
    li.appendChild(swatch);
    const text = el('span', 'lg-text');
    text.appendChild(el('b', '', row.title));
    if (row.shows) text.appendChild(el('span', 'lg-shows', row.shows));
    li.appendChild(text);
    li.appendChild(el('span', 'lg-risk', row.available ? row.risk : 'Live only'));
    li.appendChild(el('span', 'lg-now'));
    legend.appendChild(li);
    rows[row.level] = li;
  });
  return {
    card: part('card'),
    title: part('title'),
    risk: part('risk'),
    meaning: part('meaning'),
    action: part('action'),
    clue: part('clue'),
    rows,
    level: null,
  };
}

// Driver bars, or the withheld sentence in their place when explanationGate
// says the lead contradicts the result or is 0.000. Operator page only.
// Shown bars get the definition line, the three-bar reading, and no 0.000 rows.
function renderWhy(container, state, mode, emptyText) {
  const gate = core.explanationGate(state, mode);
  if (gate.show) {
    renderDrivers(container, state.top3, emptyText, {
      plain: true,
      hideZero: true,
      note: core.driverNote(state.xaiRef),
      summary: core.driverSummary(state.top3),
    });
    return;
  }
  clearChildren(container);
  const box = el('p', 'driver-withheld', gate.text);
  box.dataset.reason = gate.reason;
  container.appendChild(box);
}

// Writes only what changed: the live card is a polite status region and
// packets arrive about twice a second.
function setText(node, text) {
  if (node.textContent !== text) node.textContent = text;
}

function renderGuide(view, g) {
  view.card.dataset.level = g.level;
  view.card.dataset.caution = g.caution ? 'true' : 'false';
  setText(view.title, g.title);
  setText(view.risk, g.risk);
  setText(view.meaning, g.meaning);
  setText(view.action, g.action);
  setText(view.clue, g.clue);
  view.action.parentNode.hidden = !g.action;
  view.clue.parentNode.hidden = !g.clue;
  if (view.level === g.level) return;
  view.level = g.level;
  Object.keys(view.rows).forEach((level) => {
    const li = view.rows[level];
    const on = level === g.level;
    if (on) li.setAttribute('aria-current', 'true');
    else li.removeAttribute('aria-current');
    li.lastChild.textContent = on ? 'Now' : '';
  });
}

// ---------------------------------------------------------------------------
// Replay: the 20 OpenOximetry rubric rows in replay_windows.js. The id is the
// rubric sample name, not this page's decision. The on-screen decision is the
// firmware score against 0.50. The rows are not consecutive windows, so replay
// never shows a Confirmed flag (replayState gets no previous candidate).

const windowList = byId('windowList');
const windowCount = byId('windowCount');
const stripWrap = byId('stripWrap');
const stripAxis = byId('stripAxis');
const orderFile = byId('orderFile');
const orderScore = byId('orderScore');
const prevBtn = byId('prevBtn');
const nextBtn = byId('nextBtn');
const selId = byId('selId');
const replayScore = makeScore(byId('replayScore'));
const replayDrivers = byId('replayDrivers');
const replayGuide = makeGuide(byId('replayGuide'), 'replay');

const replayRows = (globalThis.REPLAY_WINDOWS || []).map((row, index) => ({
  index,
  id: row.window_id,
  state: core.replayState(row, false),
}));
let replayOrder = 'file';
let selected = replayRows.length ? 0 : -1;

// Three decimals in replay: 0.497 and 0.501 both round to "0.50" but sit on
// opposite sides of the threshold. Live packets carry two decimals only.
function formatReplayScore(score) {
  return isScore(score) ? score.toFixed(3) : '—';
}

function orderedRows() {
  if (replayOrder === 'file') return replayRows;
  return replayRows.slice().sort((a, b) => (b.state.score - a.state.score) || (a.index - b.index));
}

// Strip of 20 columns, score on a 0–1 vertical scale. Each column has a stem
// from the 0.50 line to its mark: up and filled at or above, down and hollow
// below. The y axis is built from THRESHOLD so the line and labels agree.
function renderStripAxis() {
  stripWrap.style.setProperty('--t', core.THRESHOLD * 100 + '%');
  clearChildren(stripAxis);
  [1, 0.75, core.THRESHOLD, 0.25, 0].forEach((v) => {
    const tick = el('i', v === core.THRESHOLD ? 't' : '', v === 0 || v === 1 ? String(v) : v.toFixed(2));
    tick.style.bottom = v * 100 + '%';
    stripAxis.appendChild(tick);
  });
}

function placeColumn(col, score) {
  const stem = col.firstChild;
  const mark = col.lastChild;
  if (!isScore(score)) {
    col.dataset.side = 'none';
    mark.hidden = true;
    return;
  }
  const t = core.THRESHOLD * 100;
  const s = Math.min(1, Math.max(0, score)) * 100;
  col.dataset.side = sideOf(score);
  stem.style.bottom = Math.min(s, t) + '%';
  stem.style.height = Math.abs(s - t) + '%';
  mark.style.bottom = s + '%';
  mark.hidden = false;
}

function renderWindowList() {
  clearChildren(windowList);
  orderedRows().forEach((r) => {
    const col = el('div', 'col');
    col.id = 'win-' + r.index;
    col.dataset.index = String(r.index);
    col.setAttribute('role', 'option');
    col.setAttribute(
      'aria-label',
      r.id + ', score ' + formatReplayScore(r.state.score) +
        (isScore(r.state.score) ? ', ' + relationText(r.state.score) : '')
    );
    col.title = r.id + '  ' + formatReplayScore(r.state.score);
    col.appendChild(el('span', 'stem'));
    col.appendChild(el('span', 'mark'));
    placeColumn(col, r.state.score);
    col.addEventListener('click', () => {
      selectWindow(r.index);
      windowList.focus({ preventScroll: true });
    });
    windowList.appendChild(col);
  });

  const above = replayRows.filter((r) => isScore(r.state.score) && r.state.score >= core.THRESHOLD).length;
  windowCount.textContent = above + ' of ' + replayRows.length + ' at or above ' + core.THRESHOLD.toFixed(2);
}

// The strip never scrolls, so selecting does not move the page. On a phone the
// stepper sits by the score and the track is below it.
function selectWindow(index) {
  if (index < 0 || index >= replayRows.length) return;
  selected = index;
  const r = replayRows[index];

  windowList.querySelectorAll('.col').forEach((col) => {
    const on = Number(col.dataset.index) === index;
    col.setAttribute('aria-selected', on ? 'true' : 'false');
    col.classList.toggle('selected', on);
  });
  windowList.setAttribute('aria-activedescendant', 'win-' + index);

  selId.textContent = r.id;
  renderScore(replayScore, r.state.score, formatReplayScore);
  renderWhy(replayDrivers, r.state, 'replay', 'No non-zero drivers in this row.');
  renderGuide(replayGuide, core.resultGuideFor(r.state, 'replay'));

  const order = orderedRows();
  const pos = order.indexOf(r);
  prevBtn.disabled = pos <= 0;
  nextBtn.disabled = pos >= order.length - 1;
}

function stepReplay(dir) {
  const order = orderedRows();
  if (order.length === 0) return;
  const pos = order.findIndex((r) => r.index === selected);
  const next = order[Math.min(order.length - 1, Math.max(0, pos + dir))];
  if (next && next.index !== selected) selectWindow(next.index);
}

function jumpReplay(end) {
  const order = orderedRows();
  if (order.length === 0) return;
  selectWindow(order[end < 0 ? 0 : order.length - 1].index);
}

function setOrder(order) {
  replayOrder = order;
  orderFile.setAttribute('aria-pressed', order === 'file' ? 'true' : 'false');
  orderScore.setAttribute('aria-pressed', order === 'score' ? 'true' : 'false');
  renderWindowList();
  selectWindow(selected);
}

orderFile.addEventListener('click', () => setOrder('file'));
orderScore.addEventListener('click', () => setOrder('score'));
prevBtn.addEventListener('click', () => stepReplay(-1));
nextBtn.addEventListener('click', () => stepReplay(1));

windowList.addEventListener('keydown', (event) => {
  const keys = { ArrowDown: 1, ArrowRight: 1, ArrowUp: -1, ArrowLeft: -1 };
  if (event.key in keys) {
    event.preventDefault();
    stepReplay(keys[event.key]);
  } else if (event.key === 'Home' || event.key === 'End') {
    event.preventDefault();
    jumpReplay(event.key === 'Home' ? -1 : 1);
  }
});

// Left/right also step from anywhere on the replay page.
document.addEventListener('keydown', (event) => {
  if (replayView.hidden || event.defaultPrevented) return;
  if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
  if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
  const t = event.target;
  if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
  event.preventDefault();
  stepReplay(event.key === 'ArrowLeft' ? -1 : 1);
});

// ---------------------------------------------------------------------------
// Live Bluetooth.

const liveScore = makeScore(byId('liveScore'));
const heldTag = byId('heldTag');
const candidateTile = byId('candidateTile');
const confirmedTile = byId('confirmedTile');
const candidateValue = byId('candidateValue');
const confirmedValue = byId('confirmedValue');
const signalTile = byId('signalTile');
const signalValue = byId('signalValue');
const hrValue = byId('hrValue');
const spo2Value = byId('spo2Value');
const satTile = byId('satTile');
const readings = satTile.parentNode;
const satValue = byId('satValue');
byId('satLabel').textContent = core.SAT_EST_LABEL;
byId('satNote').textContent = core.SAT_EST_NOTE;
const xaiRefNote = byId('xaiRefNote');
const liveDrivers = byId('liveDrivers');
const recentBody = byId('recentBody');
const liveSentence = byId('liveSentence');
const irNow = byId('irNow');
const verdictBox = byId('verdict');
const verdictTitle = byId('verdictTitle');
const verdictDetail = byId('verdictDetail');
const warm = byId('warm');
const warmBar = byId('warmBar');
const liveGuide = makeGuide(byId('liveGuide'), 'live');

// Fill progress before the first completed window: 750 samples at 50 Hz.
// Clock-based, so it is a guide, not a sample count. Capped short of full.
const FIRST_WINDOW_MS = 15000;
let warmStart = null;
const irCanvas = byId('irTrace');

// Last completed window, held on screen while the next one fills.
let heldWindow = null;

// The model sat. est tile appears once a packet has carried field 10, so the
// shipped 10-field images look unchanged. Reset on each connection.
let satEstSeen = false;

// Completed windows only. Packets repeat the current result at ~2 Hz, so a
// result is logged once until it changes.
const MAX_RECENT = 8;
const recent = [];
let lastLogKey = '';

// Live infrared counts, one point per Bluetooth notification (~2 Hz).
// Too slow to resolve a pulse; this shows finger placement and motion only.
const IR_POINTS = 120;
const irBuffer = [];

if (!navigator.bluetooth) {
  notSupported.hidden = false;
  connectBtn.disabled = true;
}

connectBtn.addEventListener('click', async () => {
  if (isLive()) {
    bleDevice.gatt.disconnect();
    return;
  }
  if (!navigator.bluetooth) {
    setLinkStatus('Web Bluetooth not supported', false);
    return;
  }

  try {
    setLinkStatus('Scanning…', false);
    setBusy(true);

    bleDevice = await navigator.bluetooth.requestDevice({
      filters: [{ name: 'OxygenRiskDemo' }],
      optionalServices: [SERVICE_UUID],
    });

    setLinkStatus('Connecting…', false);
    bleDevice.addEventListener('gattserverdisconnected', onDisconnected);
    const server = await bleDevice.gatt.connect();
    const service = await server.getPrimaryService(SERVICE_UUID);
    const dataChar = await service.getCharacteristic(DATA_CHAR_UUID);
    await dataChar.startNotifications();
    dataChar.addEventListener('characteristicvaluechanged', onBleNotify);

    setLinkStatus('Connected to ' + bleDevice.name, true);
    resetLive();
    setMode('live');
    setBusy(false);
    connectBtn.textContent = 'Disconnect';
    connectBtn.classList.add('connected');
  } catch (error) {
    setLinkStatus(error.name === 'NotFoundError' ? 'No device selected' : 'Connection failed', false);
    if (bleDevice && bleDevice.gatt && bleDevice.gatt.connected) bleDevice.gatt.disconnect();
    setBusy(false);
    connectBtn.textContent = 'Connect';
  }
});

function isLive() {
  return !!(bleDevice && bleDevice.gatt && bleDevice.gatt.connected);
}

function setBusy(busy) {
  connectBtn.disabled = busy;
  if (busy) connectBtn.setAttribute('aria-busy', 'true');
  else connectBtn.removeAttribute('aria-busy');
}

function setMode(mode) {
  document.body.dataset.mode = mode;
  replayView.hidden = mode !== 'replay';
  liveView.hidden = mode !== 'live';
  if (mode === 'live') drawIrTrace();
}

function onBleNotify(event) {
  const raw = new TextDecoder('utf-8').decode(event.target.value).trim();
  const state = core.parseFirmwarePacket(raw);
  pulseStatusDot();
  pushIr(state);
  const result = core.holdLastWindow(heldWindow, state);
  heldWindow = result.held;
  renderLive(result.view);
}

function onDisconnected() {
  setLinkStatus('Disconnected', false);
  connectBtn.textContent = 'Connect';
  connectBtn.classList.remove('connected');
  setBusy(false);
  heldWindow = null;
  setMode('replay');
}

function setLinkStatus(text, connected) {
  statusText.textContent = text;
  statusDot.classList.toggle('connected', connected);
}

function pulseStatusDot() {
  statusDot.classList.add('pulse');
  clearTimeout(pulseTimer);
  pulseTimer = setTimeout(() => statusDot.classList.remove('pulse'), 160);
}

function resetLive() {
  heldWindow = null;
  satEstSeen = false;
  recent.length = 0;
  lastLogKey = '';
  irBuffer.length = 0;
  irNow.textContent = '—';
  warmStart = null;
  renderLive(null);
  renderRecent();
}

// Score, Candidate, and Confirmed only for a completed window. Warmup packets
// (Prob 0.00, none features) leave them as dashes and are not logged.
function renderLive(state) {
  const ready = !!(state && state.windowReady);

  // The window's own score, so the number always sits on the same side of
  // 0.50 as the Candidate tile and the verdict.
  const displayScore = ready ? state.score : null;
  renderScore(liveScore, displayScore);
  heldTag.hidden = !(ready && state.held);
  setTile(candidateTile, candidateValue, ready ? state.candidate : null);
  setTile(confirmedTile, confirmedValue, ready ? state.confirmed : null);

  const signal = state ? state.signal : '—';
  signalValue.textContent = signal;
  signalTile.dataset.tone =
    signal === 'OK' ? 'ok' : signal === 'CLIPPED' ? 'warn' : signal === 'NO FINGER' ? 'off' : 'none';
  hrValue.textContent = core.formatBpm(state ? state.bpm : null);
  spo2Value.textContent = core.formatSpo2Est(state ? state.spo2Est : null);
  if (state && state.hasSatEst) satEstSeen = true;
  satTile.hidden = !satEstSeen;
  readings.dataset.n = satEstSeen ? '4' : '3';
  satValue.textContent = core.formatSatEst(ready ? state.satEst : null);

  const refNote = ready ? core.xaiRefNote(state.xaiRef) : '';
  setText(xaiRefNote, refNote);
  xaiRefNote.hidden = !refNote;

  if (ready) renderWhy(liveDrivers, state, 'live', 'No completed window yet.');
  else renderDrivers(liveDrivers, [], 'No completed window yet.', { plain: true });
  renderVerdict(state);
  // Same score as the readout; level from the raw flags, as the tiles.
  renderGuide(liveGuide, core.resultGuideFor(ready ? Object.assign({}, state, { score: displayScore }) : state, 'live'));
  liveSentence.textContent = core.operatorExplanation(
    ready ? Object.assign({}, state, { score: displayScore }) : (state || { windowReady: false, score: null }),
    'live'
  );

  if (state && state.noFinger) {
    lastLogKey = '';
  }
  if (ready && !state.held) appendRecent(state);
}

function renderVerdict(state) {
  const v = core.verdictFor(state);
  verdictBox.dataset.level = v.level;
  verdictTitle.textContent = v.title;
  verdictDetail.textContent = v.detail;

  if (v.level === 'wait') {
    if (warmStart === null) warmStart = Date.now();
    const frac = Math.min(0.95, (Date.now() - warmStart) / FIRST_WINDOW_MS);
    warmBar.style.width = Math.round(frac * 100) + '%';
    warm.hidden = false;
  } else {
    warmStart = null;
    warm.hidden = true;
  }
}

function setTile(tile, valueEl, flag) {
  valueEl.textContent = flag === null ? '—' : core.yesNo(flag);
  tile.dataset.on = flag === null ? 'none' : flag ? 'yes' : 'no';
}

function appendRecent(state) {
  const key = core.logLine(state);
  const now = Date.now();
  if (key === lastLogKey) {
    // Same result still coming in: extend the row instead of adding one.
    if (recent[0]) {
      recent[0].last = now;
      renderRecent();
    }
    return;
  }
  lastLogKey = key;
  recent.unshift({ time: new Date(now).toLocaleTimeString([], { hour12: false }), first: now, last: now, state });
  if (recent.length > MAX_RECENT) recent.pop();
  renderRecent();
}

function renderRecent() {
  clearChildren(recentBody);
  if (recent.length === 0) {
    const tr = el('tr', 'empty');
    const td = el('td', 'muted', 'No completed windows yet.');
    td.colSpan = 6;
    tr.appendChild(td);
    recentBody.appendChild(tr);
    return;
  }
  recent.forEach((entry) => {
    const s = entry.state;
    const tr = el('tr', '');
    const timeCell = el('td', 'c-time', entry.time);
    const steady = steadyFor(entry);
    if (steady) timeCell.appendChild(el('span', 'c-steady', steady));
    tr.appendChild(timeCell);
    const trackCell = el('td', 'c-track');
    const track = makeTrack('track sm');
    setTrack(track, s.score);
    trackCell.appendChild(track);
    tr.appendChild(trackCell);
    tr.appendChild(el('td', 'c-num', core.formatScore(s.score)));
    tr.appendChild(el('td', 'c-flag' + (s.candidate ? ' yes' : ''), core.yesNo(s.candidate)));
    tr.appendChild(el('td', 'c-flag' + (s.confirmed ? ' yes' : ''), core.yesNo(s.confirmed)));
    const gate = core.explanationGate(s, 'live');
    const feat = el('td', gate.show ? 'c-feat' : 'c-feat withheld', gate.show ? (gate.lead.gloss || gate.lead.name) : 'Withheld');
    if (!gate.show) feat.title = gate.text;
    tr.appendChild(feat);
    recentBody.appendChild(tr);
  });
}

// "steady 2 min 25 s": how long the same result kept arriving. Packets come at
// about 2 Hz, so this is time on screen, not a count of windows.
function steadyFor(entry) {
  const secs = Math.round((entry.last - entry.first) / 1000);
  if (secs < 5) return '';
  const m = Math.floor(secs / 60);
  const rest = secs % 60;
  return 'steady ' + (m ? m + ' min ' : '') + rest + ' s';
}

function pushIr(state) {
  if (state.noFinger || state.ir === null || state.ir === undefined) {
    irNow.textContent = state.noFinger ? 'no finger' : '—';
    return;
  }
  irBuffer.push(state.ir);
  if (irBuffer.length > IR_POINTS) irBuffer.shift();
  irNow.textContent = state.ir.toLocaleString() + ' counts';
  drawIrTrace();
}

function drawIrTrace() {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(irCanvas.clientWidth * dpr));
  const h = Math.max(1, Math.round(irCanvas.clientHeight * dpr));
  if (irCanvas.width !== w) irCanvas.width = w;
  if (irCanvas.height !== h) irCanvas.height = h;

  const ctx = irCanvas.getContext('2d');
  ctx.clearRect(0, 0, w, h);
  if (irBuffer.length === 0) return;

  let lo = Math.min.apply(null, irBuffer);
  let hi = Math.max.apply(null, irBuffer);
  // Floor the span so a steady finger reads as a flat line, not amplified noise.
  const minSpan = Math.max(200, hi * 0.002);
  if (hi - lo < minSpan) {
    const mid = (hi + lo) / 2;
    lo = mid - minSpan / 2;
    hi = mid + minSpan / 2;
  }
  const pad = 8 * dpr;
  const xStep = (w - 2 * pad) / (IR_POINTS - 1);
  const x0 = w - pad - (irBuffer.length - 1) * xStep;
  const yFor = (v) => h - pad - ((v - lo) / (hi - lo)) * (h - 2 * pad);
  const ink = getComputedStyle(document.documentElement).getPropertyValue('--ink-2').trim() || '#c2c8cf';

  ctx.strokeStyle = ink;
  ctx.lineWidth = 2 * dpr;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  irBuffer.forEach((v, i) => {
    const x = x0 + i * xStep;
    if (i === 0) ctx.moveTo(x, yFor(v));
    else ctx.lineTo(x, yFor(v));
  });
  ctx.stroke();

  const last = irBuffer[irBuffer.length - 1];
  ctx.fillStyle = ink;
  ctx.beginPath();
  ctx.arc(x0 + (irBuffer.length - 1) * xStep, yFor(last), 4 * dpr, 0, Math.PI * 2);
  ctx.fill();
}


window.addEventListener('resize', () => {
  if (!liveView.hidden) drawIrTrace();
});

renderStripAxis();
renderWindowList();
selectWindow(selected);
resetLive();
setMode('replay');
