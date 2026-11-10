// TinyKAN dashboard field list (assumed 16 Sep 2026).
// BLE packet is spo2|status|explanation|red|ir|signal|xai1|xai2|xai3|bpm
// Field 0 is the Red/IR estimate. Live mode shows it as SpO2 est. It is not TinyKAN output.
// bpm (field 9) is optional: "-" or a missing field means unknown (null).
// Newer images append field 10, the model saturation estimate (one decimal or
// "-"), and field 11, the explainer reference tag ("label0"; absent = training
// mean). Shipped images send 10 fields and parse exactly as before.
//
// Wrapped so its helpers stay off the page's global scope: app.js is a classic
// script too, and a shared top-level name (e.g. xaiRefNote) is a SyntaxError
// that stops app.js from running. Browser code reads globalThis.DashboardCore.

(function () {
const THRESHOLD = 0.50;

// Field 10 tile. Uncalibrated for this sensor and not SpO2: never call it an
// oxygen reading. Wording lives here only.
const SAT_EST_LABEL = 'Model sat. est';
const SAT_EST_NOTE = 'uncalibrated · not SpO2';

// What each explainer reference replaces an input with. 'mean' is the shipped
// images; 'label0' is the median over label-0 windows (window-median ScalcO2 >= 90).
const XAI_REF = {
  mean: { note: '', swap: 'its training average' },
  label0: { note: 'vs a typical non-low-oxygen window', swap: 'its typical non-low-oxygen value' },
};

const FEATURE_GLOSS = {
  // Features come from the bandpass-filtered window, so the mean sits near zero.
  mean: 'filtered-window mean',
  std: 'amplitude variability',
  minimum: 'waveform extremes',
  maximum: 'waveform extremes',
  value_range: 'waveform extremes',
  rms: 'overall amplitude',
  skewness: 'waveform asymmetry',
  kurtosis: 'waveform peakedness',
  energy: 'signal energy',
  derivative_std: 'rate-of-change variability',
  dominant_frequency_hz: 'dominant frequency',
  bandpower_0_5_2_hz: 'low-frequency power',
  bandpower_2_5_hz: 'higher-frequency power',
  bandpower_ratio_0_5_2_to_2_5: 'low vs high frequency power',
  // v2 model inputs (ADR-0006). Neither is an oxygen saturation value.
  ratio_of_ratios: 'red/infrared pulse ratio',
  ir_perfusion_index: 'infrared pulse strength (perfusion index)',
  // P5 beat-level inputs. None is an oxygen saturation value.
  r_beat: 'red/infrared beat ratio',
  pi_beat: 'beat perfusion index',
  hr_bpm: 'pulse rate from detected beats',
  sqi_template: 'beat-shape consistency',
  sqi_ibi_ratio: 'beat-interval spread',
};

// What the shown feature measures. Operator page only. Not a score movement
// and not an oxygen reading. One line, no cause language.
const FEATURE_MEASURES = {
  mean: 'where the filtered window sits',
  std: 'how much the wave height varies in this window',
  minimum: 'the lowest point of the filtered wave',
  maximum: 'the highest point of the filtered wave',
  value_range: 'the gap between the lowest and highest points',
  rms: 'how large the filtered wave is, overall',
  skewness: 'whether the filtered wave leans to one side',
  kurtosis: 'how sharp the filtered wave is. A single spike raises it.',
  energy: 'the total size of the frequency content',
  derivative_std: 'how fast the filtered wave is changing',
  dominant_frequency_hz: 'the strongest frequency in this window',
  bandpower_0_5_2_hz: 'how much of the wave sits between 0.5 and 2 Hz',
  bandpower_2_5_hz: 'how much of the wave sits between 2 and 5 Hz',
  bandpower_ratio_0_5_2_to_2_5: 'low-frequency size divided by higher-frequency size',
  ratio_of_ratios: 'red wave size divided by infrared wave size',
  ir_perfusion_index: 'infrared wave size relative to its steady level',
  r_beat: 'red beat size divided by infrared beat size',
  pi_beat: 'beat size relative to the steady infrared level',
  hr_bpm: 'how fast the detected beats repeat',
  sqi_template: 'how alike the detected beats look',
  sqi_ibi_ratio: 'how evenly the beats are spaced',
};

function measuresFor(name) {
  if (!name || name === 'none') return '';
  return FEATURE_MEASURES[name] || '';
}

// Static icon for the lead row only. '' means no icon.
function leadIcon(name) {
  if (name === 'ratio_of_ratios' || name === 'r_beat') return 'ratio';
  if (name === 'kurtosis') return 'peakedness';
  return '';
}

function glossFor(name) {
  if (!name || name === 'none') return '';
  return FEATURE_GLOSS[name] || name.replace(/_/g, ' ');
}

function directionFor(delta) {
  if (!(delta > 0) && !(delta < 0)) return 'unchanged';
  return delta > 0 ? 'toward risk' : 'toward normal';
}

function parseXaiPart(part) {
  if (!part) return null;
  const colon = part.lastIndexOf(':');
  if (colon <= 0) return null;
  const name = part.slice(0, colon).trim();
  const delta = parseFloat(part.slice(colon + 1));
  if (!name || name === 'none' || !Number.isFinite(delta)) return null;
  return { name, delta, gloss: glossFor(name), direction: directionFor(delta) };
}

function parseFirmwarePacket(raw) {
  const empty = {
    noFinger: true,
    signal: 'NO FINGER',
    score: null,
    candidate: false,
    confirmed: false,
    top3: [],
    windowReady: false,
    bpm: null,
    spo2Est: null,
    satEst: null,
    hasSatEst: false,
    xaiRef: 'mean',
  };

  if (typeof raw !== 'string' || !raw.trim()) return empty;

  const parts = raw.trim().split('|');
  if (parts.length < 6) return empty;

  const red = parseCount(parts[3]);
  const ir = parseCount(parts[4]);
  const status = parts[1] || '';
  const explanation = parts[2] || '';
  const signalRaw = (parts[5] || '').trim();
  const noFinger =
    signalRaw === 'NO FINGER' ||
    status === 'No finger detected' ||
    explanation.indexOf('Place finger') !== -1;

  const top3 = [];
  for (let i = 6; i < 9; i += 1) {
    const item = parseXaiPart(parts[i]);
    if (item) top3.push(item);
  }

  let score = null;
  const probMatch = explanation.match(/Prob:\s*([0-9]+(?:\.[0-9]+)?)/i);
  if (probMatch) {
    const parsed = parseFloat(probMatch[1]);
    if (Number.isFinite(parsed)) score = parsed;
  }

  const confirmed = !noFinger && /CONFIRMED/i.test(status) && !/UNCONFIRMED/i.test(status);
  const candidate = !noFinger && (confirmed || /CANDIDATE/i.test(status));
  const windowReady = !noFinger && top3.length > 0 && score !== null;

  let signal = 'OK';
  if (noFinger) signal = 'NO FINGER';
  else if (signalRaw === 'CLIPPED') signal = 'CLIPPED';
  else if (signalRaw === 'OK') signal = 'OK';
  else if (signalRaw) signal = signalRaw;

  return {
    noFinger,
    signal,
    score: windowReady ? score : null,
    candidate,
    confirmed,
    top3,
    windowReady,
    red,
    ir,
    bpm: noFinger ? null : parseBpm(parts[9]),
    spo2Est: noFinger ? null : parseSpo2Est(parts[0]),
    // A model output for the completed window, so null until one exists.
    satEst: windowReady ? parseSatEst(parts[10]) : null,
    hasSatEst: parts.length > 10,
    xaiRef: parseXaiRef(parts[11]),
  };
}

// Firmware resets its window on any sub-threshold sample, then sends warmup
// packets (Prob 0.00, none:0.0000) until 500 new samples arrive. Keep the last
// completed window on screen through that gap; drop it on no finger.
// Returns { view, held }: view is what to render, held is the window to keep.
function holdLastWindow(held, state) {
  if (!state || state.noFinger) return { view: state, held: null };
  if (state.windowReady) return { view: state, held: state };
  if (!held) return { view: state, held: null };
  return {
    view: {
      noFinger: false,
      signal: state.signal,
      score: held.score,
      candidate: held.candidate,
      confirmed: held.confirmed,
      top3: held.top3,
      windowReady: true,
      held: true,
      red: state.red,
      ir: state.ir,
      bpm: state.bpm,
      spo2Est: state.spo2Est,
      // Window outputs: the held window's, like its score and drivers.
      satEst: held.satEst,
      hasSatEst: state.hasSatEst,
      xaiRef: held.xaiRef,
    },
    held,
  };
}

function parseCount(part) {
  if (part === undefined || part === null || part === '---' || part === '') return null;
  const n = Number(part);
  return Number.isFinite(n) ? n : null;
}

// Firmware sends 40-180 or "-"; anything else is treated as unknown.
function parseBpm(part) {
  if (part === undefined || part === null) return null;
  const text = part.trim();
  if (!/^[0-9]+$/.test(text)) return null;
  const n = Number(text);
  return n >= 40 && n <= 180 ? n : null;
}

function formatScore(score) {
  if (score === null || score === undefined || !Number.isFinite(score)) return '—';
  return score.toFixed(2);
}

function formatBpm(bpm) {
  if (bpm === null || bpm === undefined || !Number.isFinite(bpm)) return '—';
  return String(bpm);
}

// Field 0 is the firmware Red/IR estimate, clamped 0–100, one decimal.
// "---", a missing field, or no finger is unknown. Not a TinyKAN output.
function parseSpo2Est(part) {
  if (part === undefined || part === null) return null;
  const text = String(part).trim();
  if (!/^[0-9]+(?:\.[0-9]+)?$/.test(text)) return null;
  const n = Number(text);
  return n >= 0 && n <= 100 ? n : null;
}

function formatSpo2Est(value) {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  return value.toFixed(1);
}

// Field 10, the model saturation estimate: same text rules as field 0
// ("-", missing or outside 0–100 is unknown). Uncalibrated, not SpO2.
function parseSatEst(part) {
  return parseSpo2Est(part);
}

function formatSatEst(value) {
  return formatSpo2Est(value);
}

// Field 11: 'label0' or, absent or anything else, the training mean.
function parseXaiRef(part) {
  return typeof part === 'string' && part.trim() === 'label0' ? 'label0' : 'mean';
}

// Few-word note for the driver-bars heading; empty for the training mean.
function xaiRefNote(ref) {
  return (XAI_REF[ref] || XAI_REF.mean).note;
}

function yesNo(flag) {
  return flag ? 'Yes' : 'No';
}

function buildExplanation(state) {
  if (!state || state.noFinger) {
    return 'Place finger on the sensor.';
  }
  if (!state.windowReady || state.score === null) {
    return 'Waiting for the first completed window.';
  }

  const top = state.top3[0];
  const clue = top ? (top.gloss || top.name) : 'the pulse shape';
  const flagged = state.score >= THRESHOLD;
  const pull = top && top.direction === 'toward normal' ? 'pulls away from a flag' : 'pushes toward a flag';
  const line = flagged
    ? `Flagged. Main clue: ${clue}, which ${pull}. A shape match, not an oxygen level.`
    : `Not flagged. Main clue: ${clue}, which ${pull}. Not an oxygen level.`;
  return state.held ? line + ' Last window.' : line;
}

// Explanation withheld rule (operator page only; the survey keeps
// buildExplanation and the bars as they were). The lead is top3[0], the largest
// |delta| in payload order. Flag side is the one the verdict and results guide
// use: the raw Candidate flag live, score vs 0.50 in replay. Withhold when the
// lead prints as 0.000 at three decimals (reason 'zero') or pushes against the
// flag side (reason 'opposes'). One wording, used on every operator surface.
// Only the 'zero' sentence names the reference; WITHHELD_TEXT is the training mean.
const EXPLANATION_ZERO = 0.0005;
function withheldZero(ref) {
  return `Explanation withheld. Swapping any one feature for ${(XAI_REF[ref] || XAI_REF.mean).swap} ` +
    'left this score unchanged, so there is no clue to show.';
}
const WITHHELD_TEXT = {
  zero: withheldZero('mean'),
  opposes: 'Explanation withheld. The feature that moved this score most pushed it the other way from this result, so it does not explain the result.',
};

function withheldText(reason, ref) {
  return reason === 'zero' ? withheldZero(ref) : WITHHELD_TEXT[reason];
}

function shownExplanation(lead) {
  return `Biggest influence: ${lead.gloss || lead.name}, which ${lead.delta > 0 ? 'raised' : 'lowered'} the score ` +
    `by ${Math.abs(lead.delta).toFixed(3)}. This feature moved the model's score; it is not an oxygen reading or a cause.`;
}

function explanationGate(state, mode) {
  const flagged = mode === 'replay'
    ? !!state && Number.isFinite(state.score) && state.score >= THRESHOLD
    : !!(state && state.candidate);
  const lead = (state && state.top3 && state.top3[0]) || null;
  let reason = null;
  if (!lead || !(Math.abs(lead.delta) >= EXPLANATION_ZERO)) reason = 'zero';
  else if (flagged ? lead.delta < 0 : lead.delta > 0) reason = 'opposes';
  return {
    show: reason === null,
    reason,
    flagged,
    lead,
    text: reason ? withheldText(reason, state && state.xaiRef) : shownExplanation(lead),
  };
}

// "Why this score" bars, operator page only (the survey uses none of these).
// One definition line, and one reading of the shown bars together. The reading
// is arithmetic on the signs and on which deltas print as 0.000: no threshold
// beyond EXPLANATION_ZERO and no "expected" feature. The firmware sends the
// three largest |delta|, so a slot below EXPLANATION_ZERO means no other
// feature moved the score either; its name is filled by feature order.
function activeDrivers(top3) {
  return (top3 || []).filter((d) => d && Math.abs(d.delta) >= EXPLANATION_ZERO);
}

function driverNote(ref) {
  return 'Top three features: how much each one\u2019s value raised or lowered the score, ' +
    `compared with ${(XAI_REF[ref] || XAI_REF.mean).swap}.`;
}

const COUNT_WORD = ['no', 'one', 'two', 'three'];
function driverSummary(top3) {
  const active = activeDrivers(top3);
  const n = active.length;
  if (n === 0 || n > 3) return '';
  const up = active.filter((d) => d.delta > 0).length;
  const down = n - up;
  if (n === 1) return 'Only one feature moved this score.';
  if (n === 2) {
    const how = up === 2 ? 'both raised it' : down === 2 ? 'both lowered it' : 'one raised it, one lowered it';
    return `Only two features moved this score; ${how}.`;
  }
  if (up === 3) return 'All three raised the score.';
  if (down === 3) return 'All three lowered the score.';
  return `Pulled both ways: ${COUNT_WORD[up]} raised the score, ${COUNT_WORD[down]} lowered it.`;
}

// Operator-page text summary. Same no-finger and waiting lines as
// buildExplanation; a completed window gets the flag side and the gated clue.
function operatorExplanation(state, mode) {
  if (!state || state.noFinger) return buildExplanation(state);
  if (!state.windowReady || state.score === null) return buildExplanation(state);
  const gate = explanationGate(state, mode);
  const line = `${gate.flagged ? 'Flagged.' : 'Not flagged.'} ${gate.text}`;
  return state.held ? line + ' Last window.' : line;
}

// Plain-language headline for the live view. Uses the raw window flags, the
// same ones the Candidate and Confirmed tiles show. It names the model's flag,
// never an oxygen level.
function verdictFor(state) {
  if (!state || state.noFinger) {
    return { level: 'off', title: 'Place a finger on the sensor', detail: 'Cover the sensor fully and keep still.' };
  }
  if (!state.windowReady || state.score === null) {
    return { level: 'wait', title: 'Reading pulse…', detail: 'First result after about 15 s. Keep still.' };
  }
  const last = state.held ? ' (last window)' : '';
  if (state.confirmed) {
    return { level: 'alert', title: 'Alert', detail: 'Two windows in a row matched the pattern the model was trained to flag' + last + '.' };
  }
  if (state.candidate) {
    return { level: 'watch', title: 'Possible', detail: 'This window matched the flagged pattern. One more in a row raises an alert' + last + '.' };
  }
  return { level: 'clear', title: 'No alert', detail: 'The pulse pattern does not match the flagged examples' + last + '.' };
}

// "What this result means": the supervisor results guide (23 Oct 2026), one row
// per verdictFor level, wording kept. The off row names NO FINGER only: CLIPPED
// still yields results, so it is a caution on the clue, not a level of its own.
const T_TEXT = THRESHOLD.toFixed(2);
const GUIDE_LEVELS = ['off', 'wait', 'clear', 'watch', 'alert'];
const RESULT_GUIDE = {
  off: {
    band: 'grey', title: 'Place a finger on the sensor', shows: 'Signal: NO FINGER',
    meaning: 'The sensor cannot see a usable pulse.', risk: 'No result',
    action: 'Cover the sensor fully and keep still.',
  },
  wait: {
    band: 'blue', title: 'Reading pulse…', shows: 'No score yet',
    meaning: 'Collecting the first 10-second pulse window.', risk: 'No result yet',
    action: 'Keep still. First result in about 15 s, then every 5 s.',
  },
  clear: {
    band: 'green', title: 'No alert', shows: `Score below ${T_TEXT}`,
    meaning: 'The pulse pattern does not look like the low-oxygen examples the model learned from.',
    risk: 'Low', action: 'No action needed.',
  },
  watch: {
    band: 'yellow', title: 'Possible', shows: `Score ${T_TEXT} or above, once (Candidate: Yes)`,
    meaning: 'One 10-second window looks like the low-oxygen examples. A single window can be caused by movement.',
    risk: 'Watch', action: 'Keep still. The system re-checks 5 s later.',
  },
  alert: {
    band: 'red', title: 'Alert', shows: `Score ${T_TEXT} or above, twice in a row (Confirmed: Yes)`,
    meaning: 'Two windows in a row look like the low-oxygen examples.', risk: 'Possible low oxygen',
    action: 'Check finger placement, then confirm with a certified pulse oximeter. Seek medical help if it also reads low or you feel unwell.',
  },
};
// Saved OpenOximetry rows are not consecutive, so replay has no Alert: a score
// at or above 0.50 is one flagged window. Actions say what live use would do.
const REPLAY_GUIDE = {
  clear: {
    band: 'green', title: 'Not flagged', shows: `Score below ${T_TEXT}`,
    meaning: 'The pulse pattern in this saved window does not look like the low-oxygen examples the model learned from.',
    risk: 'Low', action: 'In live use: no action needed.',
  },
  watch: {
    band: 'yellow', title: 'Flagged window', shows: `Score ${T_TEXT} or above`,
    meaning: 'This saved 10-second window looks like the low-oxygen examples. A single window can be caused by movement. Saved windows are not consecutive, so there is no two-in-a-row alert here.',
    risk: 'Watch', action: 'In live use: keep still. The system re-checks 5 s later.',
  },
};
const REPLAY_NO_SCORE = {
  band: 'grey', title: 'No score', shows: '', meaning: 'This saved row has no score.',
  risk: 'No result', action: '',
};

function lineSide(score) {
  if (score > THRESHOLD) return 'above';
  if (score < THRESHOLD) return 'below';
  return 'on';
}

// One plain clue from the current window. The score is the one the readout
// shows (the window's own score). Firmware prints it to two decimals, so a
// window just under the line can read "0.50" while its flag is No.
function guideClue(state, level, replay) {
  if (level === 'off') return state ? 'No usable pulse, so no score.' : 'No data from the sensor yet.';
  const clipped = !replay && state.signal === 'CLIPPED';
  if (level === 'wait') {
    const line = 'No score yet. The 10-second window is still filling.';
    return clipped ? line + ' Signal CLIPPED: reposition the finger and keep still.' : line;
  }
  const parts = [];
  if (state.held) parts.push('Last completed window, kept while the next one fills.');
  const shown = replay ? state.score.toFixed(3) : formatScore(state.score);
  const flagged = level === 'watch' || level === 'alert';
  if (flagged === state.score >= THRESHOLD) {
    parts.push(`Score ${shown}, ${lineSide(state.score)} the ${T_TEXT} line.`);
  } else {
    parts.push(`Score ${shown} is rounded; this window on its own ` +
      `${flagged ? 'reached' : 'was just below'} the ${T_TEXT} line.`);
  }
  parts.push(explanationGate(state, replay ? 'replay' : 'live').text);
  if (clipped) parts.push('Signal CLIPPED: reposition the finger. This result may be unreliable.');
  return parts.join(' ');
}

// Live levels are verdictFor's (raw window flags, same as the Candidate and
// Confirmed tiles). Replay levels are score vs 0.50 only, never alert. Pass
// the displayed score in state.score. Never states an oxygen level.
function resultGuideFor(state, mode) {
  const replay = mode === 'replay';
  let level;
  let row;
  if (replay) {
    const scored = !!state && Number.isFinite(state.score);
    level = !scored ? 'off' : state.score >= THRESHOLD ? 'watch' : 'clear';
    row = scored ? REPLAY_GUIDE[level] : REPLAY_NO_SCORE;
  } else {
    level = verdictFor(state).level;
    row = RESULT_GUIDE[level];
  }
  const clue = replay && level === 'off' ? '' : guideClue(state, level, replay);
  const clipped = !replay && !!state && !state.noFinger && state.signal === 'CLIPPED';
  return {
    level,
    band: row.band,
    title: row.title,
    shows: row.shows,
    meaning: row.meaning,
    risk: row.risk,
    action: row.action,
    clue,
    caution: clipped,
  };
}

// The five guide rows in order, for the legend. Replay marks the rows that
// cannot occur there as unavailable and uses its own clear/watch wording.
function resultGuideRows(mode) {
  const replay = mode === 'replay';
  return GUIDE_LEVELS.map((level) => {
    const available = !replay || !!REPLAY_GUIDE[level];
    const row = replay && available ? REPLAY_GUIDE[level] : RESULT_GUIDE[level];
    return { level, band: row.band, title: row.title, shows: row.shows, risk: row.risk, available };
  });
}

function logLine(state) {
  const top = state.top3[0] ? state.top3[0].name : '—';
  return `${formatScore(state.score)}  ·  cand ${yesNo(state.candidate)}  ·  conf ${yesNo(state.confirmed)}  ·  ${top}`;
}

// One OpenOximetry rubric row -> dashboard state. Score is firmware_probability
// (the device model), never csv_probability. The file has no confirmed flag, red,
// IR, or heart rate, so confirmed is derived from the previous replayed row and the
// rest stay null. The rows are a rubric list, not consecutive windows, so the page
// does not show that confirmed flag; replay shows score vs 0.50 and drivers only.
function replayState(row, prevCandidate) {
  const p = Number(row && row.firmware_probability);
  const score = Number.isFinite(p) ? p : null;
  const top3 = [];
  for (let k = 1; k <= 3; k += 1) {
    const name = row[`top${k}_feature`];
    const delta = Number(row[`top${k}_delta`]);
    if (!name || name === 'none' || !Number.isFinite(delta) || delta === 0) continue;
    top3.push({ name, delta, gloss: glossFor(name), direction: directionFor(delta) });
  }
  const candidate = score !== null && score >= THRESHOLD;
  return {
    noFinger: false,
    signal: '—',
    score,
    candidate,
    confirmed: candidate && prevCandidate === true,
    top3,
    windowReady: score !== null,
    red: null,
    ir: null,
    bpm: null,
    replay: true,
    windowId: row.window_id,
  };
}

const api = {
  THRESHOLD,
  SAT_EST_LABEL,
  SAT_EST_NOTE,
  FEATURE_GLOSS,
  glossFor,
  measuresFor,
  leadIcon,
  directionFor,
  parseFirmwarePacket,
  holdLastWindow,
  buildExplanation,
  WITHHELD_TEXT,
  EXPLANATION_ZERO,
  explanationGate,
  operatorExplanation,
  activeDrivers,
  driverNote,
  driverSummary,
  formatScore,
  formatBpm,
  parseSpo2Est,
  formatSpo2Est,
  parseSatEst,
  formatSatEst,
  parseXaiRef,
  xaiRefNote,
  yesNo,
  logLine,
  verdictFor,
  resultGuideFor,
  resultGuideRows,
  replayState,
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = api;
}
if (typeof globalThis !== 'undefined') {
  globalThis.DashboardCore = api;
}
})();
