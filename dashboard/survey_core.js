// Survey logic shared by the browser (survey.js) and the server (server.js).
// No DOM. Question text, window assignment, validation, and scoring.
//
// Quadrants are judged on what the respondent sees: the firmware score against
// 0.50. The rubric id prefix (TP-, FP-, TN-, FN-) comes from the host CSV
// probability, so it encodes the true label only:
//   TP-, FN-  -> true label 1 (Low-Oxygen Risk)
//   FP-, TN-  -> true label 0
// Five of the 20 rows change quadrant under the firmware score (for example
// TP-01 is a firmware FN at 0.405). The page never shows the rubric id.

(function () {
const core = (typeof require === 'function')
  ? require('./dashboard_core.js')
  : globalThis.DashboardCore;

const VERSION = 'tinykan-survey-v1';
const QUADRANTS = ['TP', 'FP', 'TN', 'FN'];
const LIKERT = [1, 2, 3, 4, 5];
const TEXT_MAX = 1000;

// ---------------------------------------------------------------------------
// Questions. `id` is the stored key. Option values are stored; labels are shown.

const BACKGROUND = [
  {
    id: 'role',
    text: 'Which best describes you?',
    options: [
      ['eng_cs_student', 'Engineering or computing student'],
      ['health', 'Healthcare worker or healthcare student'],
      ['other_student', 'Student in another field'],
      ['other', 'Other'],
    ],
  },
  {
    id: 'oximeter_familiarity',
    text: 'How familiar are you with pulse oximeters (the finger clip that reads blood oxygen)?',
    scale: ['Not at all', 'Very familiar'],
  },
  {
    id: 'ml_familiarity',
    text: 'How familiar are you with AI or machine learning?',
    scale: ['Not at all', 'Very familiar'],
  },
];

// Asked once, on the first window, before any other window question.
const MEANING = {
  id: 'meaning',
  text: 'What does the Score number on this screen show?',
  options: [
    ['spo2', 'Blood oxygen level (SpO2 %)'],
    ['risk', 'How likely the window is Low-Oxygen Risk'],
    ['heart_rate', 'Heart rate'],
    ['not_sure', 'Not sure'],
  ],
  correct: 'risk',
};

// Per window. `top_driver` options are built from the window's own drivers.
const WINDOW_CHOICE = [
  {
    id: 'flagged',
    text: 'Is this window flagged as Low-Oxygen Risk?',
    options: [['yes', 'Yes'], ['no', 'No'], ['not_sure', 'Not sure']],
  },
  {
    id: 'top_driver',
    text: 'Which feature moved the score the most?',
  },
  {
    id: 'direction',
    text: 'Did that feature push the score toward risk or toward normal?',
    options: [['toward_risk', 'Toward risk'], ['toward_normal', 'Toward normal'], ['not_sure', 'Not sure']],
  },
];

const WINDOW_RATINGS = [
  { id: 'compact', text: 'The explanation is short enough to read at a glance.' },
  { id: 'matches', text: 'The explanation matches the score shown.' },
  { id: 'helps', text: 'The explanation would help me decide what to do next.' },
  { id: 'trust', text: 'I would trust this result.' },
];
const AGREE = ['Strongly disagree', 'Strongly agree'];

const WINDOW_ACTION = {
  id: 'action',
  text: 'If you were monitoring this person, what would you do next?',
  options: [
    ['escalate', 'Escalate: tell a nurse or doctor'],
    ['check_sensor', 'Check the sensor and finger placement first'],
    ['no_action', 'No action'],
    ['not_sure', 'Not sure'],
  ],
};

// System Usability Scale (Brooke, 1996), "system" -> "dashboard".
// Odd items are positive, even items negative.
const SUS = [
  'I think that I would like to use this dashboard frequently.',
  'I found the dashboard unnecessarily complex.',
  'I thought the dashboard was easy to use.',
  'I think that I would need the support of a technical person to be able to use this dashboard.',
  'I found the various functions in this dashboard were well integrated.',
  'I thought there was too much inconsistency in this dashboard.',
  'I would imagine that most people would learn to use this dashboard very quickly.',
  'I found the dashboard very cumbersome to use.',
  'I felt very confident using the dashboard.',
  'I needed to learn a lot of things before I could get going with this dashboard.',
];

// Explanation Satisfaction Scale items 1, 3, 6, 8 (Hoffman et al., 2018,
// arXiv:1812.04608), "software" -> "dashboard".
const ESS = [
  { item: 1, text: 'From the explanations, I understand how the dashboard reaches its score.' },
  { item: 3, text: 'The explanations have sufficient detail.' },
  { item: 6, text: 'The explanations are useful to my goals.' },
  { item: 8, text: 'The explanations let me judge when I should trust and not trust the dashboard.' },
];

const OPEN = [
  { id: 'confusing', text: 'What confused you most?' },
  { id: 'change', text: 'What would you add, remove, or change?' },
  { id: 'aide', text: 'Could a clinic aide use this without training? Why or why not?' },
];

// ---------------------------------------------------------------------------
// Windows.

function trueLabelOf(windowId) {
  const prefix = String(windowId || '').slice(0, 2);
  if (prefix === 'TP' || prefix === 'FN') return 1;
  if (prefix === 'FP' || prefix === 'TN') return 0;
  return null;
}

function firmwareFlag(row) {
  return Number(row.firmware_probability) >= core.THRESHOLD ? 1 : 0;
}

function quadrantOf(row) {
  const truth = trueLabelOf(row.window_id);
  if (truth === null) return null;
  const flag = firmwareFlag(row);
  if (truth === 1) return flag ? 'TP' : 'FN';
  return flag ? 'FP' : 'TN';
}

// Drivers exactly as the dashboard shows them (zero deltas dropped).
function driversOf(row) {
  return core.replayState(row, false).top3;
}

function byId(rows) {
  const map = new Map();
  rows.forEach((r) => map.set(r.window_id, r));
  return map;
}

function shuffle(list, rng) {
  const out = list.slice();
  for (let i = out.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

// One window per firmware quadrant, in random order.
function assignWindows(rows, rng = Math.random) {
  const picks = QUADRANTS.map((q) => {
    const pool = rows.filter((r) => quadrantOf(r) === q);
    if (pool.length === 0) throw new Error('no replay window in quadrant ' + q);
    return pool[Math.floor(rng() * pool.length)].window_id;
  });
  return shuffle(picks, rng);
}

function driverOptions(row, rng = Math.random) {
  const opts = shuffle(driversOf(row), rng).map((d) => [d.name, d.gloss + ' (' + d.name + ')']);
  opts.push(['not_sure', 'Not sure']);
  return opts;
}

// ---------------------------------------------------------------------------
// Validation. Returns { ok, errors, clean }. `clean` holds only known keys.

function optionValues(q) {
  return q.options.map((o) => o[0]);
}

function validateResponse(body, rows) {
  const errors = [];
  const clean = { version: VERSION };
  const windowsById = byId(rows);

  if (!body || typeof body !== 'object' || Array.isArray(body)) {
    return { ok: false, errors: ['body must be an object'], clean: null };
  }
  if (body.version !== VERSION) errors.push('version must be ' + VERSION);

  const consent = body.consent || {};
  if (consent.read !== true || consent.adult !== true || consent.agree !== true) {
    errors.push('consent: read, adult, and agree must all be true');
  }
  clean.consent = { read: true, adult: true, agree: true };

  const bg = body.background || {};
  clean.background = {};
  BACKGROUND.forEach((q) => {
    const v = bg[q.id];
    if (q.options) {
      if (!optionValues(q).includes(v)) errors.push('background.' + q.id + ' invalid');
    } else if (!LIKERT.includes(v)) {
      errors.push('background.' + q.id + ' must be 1-5');
    }
    clean.background[q.id] = v;
  });

  if (!optionValues(MEANING).includes(body.meaning)) errors.push('meaning invalid');
  clean.meaning = body.meaning;

  const wins = Array.isArray(body.windows) ? body.windows : [];
  if (wins.length !== QUADRANTS.length) errors.push('windows must have ' + QUADRANTS.length + ' entries');
  const seenIds = new Set();
  const seenQuads = new Set();
  clean.windows = wins.slice(0, QUADRANTS.length).map((w, i) => {
    const p = 'windows[' + i + '].';
    w = w || {};
    const row = windowsById.get(w.window_id);
    if (!row) {
      errors.push(p + 'window_id unknown');
      return null;
    }
    if (seenIds.has(w.window_id)) errors.push(p + 'window_id repeated');
    seenIds.add(w.window_id);
    seenQuads.add(quadrantOf(row));

    const out = { window_id: w.window_id, position: i + 1 };
    if (!optionValues(WINDOW_CHOICE[0]).includes(w.flagged)) errors.push(p + 'flagged invalid');
    out.flagged = w.flagged;

    const driverNames = driversOf(row).map((d) => d.name).concat(['not_sure']);
    if (!driverNames.includes(w.top_driver)) errors.push(p + 'top_driver invalid');
    out.top_driver = w.top_driver;

    if (!optionValues(WINDOW_CHOICE[2]).includes(w.direction)) errors.push(p + 'direction invalid');
    out.direction = w.direction;

    WINDOW_RATINGS.forEach((q) => {
      if (!LIKERT.includes(w[q.id])) errors.push(p + q.id + ' must be 1-5');
      out[q.id] = w[q.id];
    });

    if (!optionValues(WINDOW_ACTION).includes(w.action)) errors.push(p + 'action invalid');
    out.action = w.action;

    const secs = Number(w.seconds);
    out.seconds = Number.isFinite(secs) && secs >= 0 && secs < 86400 ? Math.round(secs) : null;
    return out;
  });
  if (wins.length === QUADRANTS.length && seenQuads.size !== QUADRANTS.length) {
    errors.push('windows must cover TP, FP, TN, FN once each');
  }

  ['sus', 'ess'].forEach((key) => {
    const n = key === 'sus' ? SUS.length : ESS.length;
    const list = body[key];
    if (!Array.isArray(list) || list.length !== n || !list.every((v) => LIKERT.includes(v))) {
      errors.push(key + ' must be ' + n + ' answers of 1-5');
    }
    clean[key] = Array.isArray(list) ? list.slice(0, n) : [];
  });

  const open = body.open || {};
  clean.open = {};
  OPEN.forEach((q) => {
    const v = open[q.id];
    if (v !== undefined && v !== null && typeof v !== 'string') errors.push('open.' + q.id + ' must be text');
    clean.open[q.id] = typeof v === 'string' ? v.trim().slice(0, TEXT_MAX) : '';
  });

  const dur = Number(body.duration_s);
  clean.duration_s = Number.isFinite(dur) && dur >= 0 && dur < 86400 ? Math.round(dur) : null;

  return { ok: errors.length === 0, errors, clean: errors.length ? null : clean };
}

// ---------------------------------------------------------------------------
// Scoring. Pure functions of a clean response and the replay rows.

// SUS: odd items (v - 1), even items (5 - v), sum x 2.5 -> 0..100.
function susScore(answers) {
  if (!Array.isArray(answers) || answers.length !== 10) return null;
  let sum = 0;
  for (let i = 0; i < 10; i += 1) {
    const v = answers[i];
    if (!LIKERT.includes(v)) return null;
    sum += i % 2 === 0 ? v - 1 : 5 - v;
  }
  return sum * 2.5;
}

function scoreWindow(answer, row) {
  const top = driversOf(row)[0];
  const flagged = firmwareFlag(row) === 1;
  const truthDir = top ? (top.delta > 0 ? 'toward_risk' : 'toward_normal') : null;
  return {
    window_id: row.window_id,
    quadrant: quadrantOf(row),
    true_label: trueLabelOf(row.window_id),
    firmware_probability: Number(row.firmware_probability),
    near_threshold: Math.abs(Number(row.firmware_probability) - core.THRESHOLD) < 0.02,
    flagged_correct: answer.flagged === (flagged ? 'yes' : 'no'),
    top_driver_correct: !!top && answer.top_driver === top.name,
    direction_correct: !!truthDir && answer.direction === truthDir,
  };
}

function scoreResponse(clean, rows) {
  const windowsById = byId(rows);
  return {
    meaning_correct: clean.meaning === MEANING.correct,
    sus: susScore(clean.sus),
    ess_mean: clean.ess.length ? clean.ess.reduce((a, b) => a + b, 0) / clean.ess.length : null,
    windows: clean.windows.map((w) => scoreWindow(w, windowsById.get(w.window_id))),
  };
}

const api = {
  VERSION, QUADRANTS, LIKERT, TEXT_MAX,
  BACKGROUND, MEANING, WINDOW_CHOICE, WINDOW_RATINGS, AGREE, WINDOW_ACTION,
  SUS, ESS, OPEN,
  trueLabelOf, firmwareFlag, quadrantOf, driversOf, shuffle,
  assignWindows, driverOptions, validateResponse, susScore, scoreWindow, scoreResponse,
};

if (typeof module !== 'undefined' && module.exports) module.exports = api;
if (typeof globalThis !== 'undefined') globalThis.SurveyCore = api;
})();
