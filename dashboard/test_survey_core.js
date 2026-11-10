const assert = require('assert');
const fs = require('fs');
const path = require('path');
const S = require('./survey_core.js');
const REPLAY_WINDOWS = require('./replay_windows.js');
const replay = require('./build_replay.js');

function check(name, fn) {
  fn();
  console.log('ok  ' + name);
}

// Deterministic RNG for repeatable assignment.
function seeded(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

// A complete, valid response for the given window ids.
function validBody(ids) {
  return {
    version: S.VERSION,
    consent: { read: true, adult: true, agree: true },
    background: { role: 'eng_cs_student', oximeter_familiarity: 3, ml_familiarity: 4 },
    meaning: 'risk',
    windows: ids.map((id) => {
      const row = REPLAY_WINDOWS.find((r) => r.window_id === id);
      const top = S.driversOf(row)[0];
      return {
        window_id: id,
        flagged: row.firmware_probability >= 0.5 ? 'yes' : 'no',
        top_driver: top.name,
        direction: top.delta > 0 ? 'toward_risk' : 'toward_normal',
        compact: 4, matches: 4, helps: 3, trust: 3,
        action: 'check_sensor',
        seconds: 40,
      };
    }),
    sus: [4, 2, 4, 2, 4, 2, 4, 2, 4, 2],
    ess: [4, 3, 4, 3],
    open: { confusing: '  the bars  ', change: '', aide: 'maybe' },
    duration_s: 600,
  };
}

check('true label comes from the rubric id prefix', () => {
  assert.strictEqual(S.trueLabelOf('TP-01'), 1);
  assert.strictEqual(S.trueLabelOf('FN-03'), 1);
  assert.strictEqual(S.trueLabelOf('FP-02'), 0);
  assert.strictEqual(S.trueLabelOf('TN-05'), 0);
  assert.strictEqual(S.trueLabelOf('XX-01'), null);
});

check('true label from prefix matches the rubric CSV when present', () => {
  if (!fs.existsSync(replay.SOURCE)) {
    console.log('    (skipped: ' + path.basename(replay.SOURCE) + ' not on this machine)');
    return;
  }
  replay.readRows().forEach((r) => {
    assert.strictEqual(S.trueLabelOf(r.window_id), Number(r.true_label), r.window_id);
  });
});

check('quadrant uses the firmware score, not the rubric id', () => {
  const q = (id) => S.quadrantOf(REPLAY_WINDOWS.find((r) => r.window_id === id));
  assert.strictEqual(q('TP-01'), 'FN'); // firmware 0.405
  assert.strictEqual(q('TP-02'), 'TP');
  assert.strictEqual(q('FP-01'), 'TN'); // firmware 0.405
  assert.strictEqual(q('FN-01'), 'TP'); // firmware 0.501
  assert.strictEqual(q('TN-03'), 'TN');
});

check('every firmware quadrant has at least three windows', () => {
  const counts = {};
  REPLAY_WINDOWS.forEach((r) => { counts[S.quadrantOf(r)] = (counts[S.quadrantOf(r)] || 0) + 1; });
  assert.deepStrictEqual(Object.keys(counts).sort(), ['FN', 'FP', 'TN', 'TP']);
  Object.values(counts).forEach((n) => assert.ok(n >= 3, JSON.stringify(counts)));
});

check('assignment picks one window per quadrant, in varied order', () => {
  const firsts = new Set();
  for (let seed = 1; seed <= 200; seed += 1) {
    const ids = S.assignWindows(REPLAY_WINDOWS, seeded(seed));
    assert.strictEqual(ids.length, 4);
    const quads = ids.map((id) => S.quadrantOf(REPLAY_WINDOWS.find((r) => r.window_id === id)));
    assert.deepStrictEqual(quads.slice().sort(), ['FN', 'FP', 'TN', 'TP']);
    firsts.add(quads[0]);
  }
  assert.strictEqual(firsts.size, 4, 'each quadrant should sometimes come first');
});

check('driver options are the shown drivers plus not sure', () => {
  const row = REPLAY_WINDOWS.find((r) => r.window_id === 'TP-01'); // third delta is 0
  const opts = S.driverOptions(row, seeded(7));
  assert.deepStrictEqual(opts.map((o) => o[0]).sort(), ['energy', 'minimum', 'not_sure']);
  assert.strictEqual(opts[opts.length - 1][0], 'not_sure');
  assert.ok(opts.find((o) => o[0] === 'minimum')[1].includes('(minimum)'));
});

check('valid response passes and is cleaned', () => {
  const ids = S.assignWindows(REPLAY_WINDOWS, seeded(3));
  const body = validBody(ids);
  body.extra = 'dropped';
  body.website = '';
  const res = S.validateResponse(body, REPLAY_WINDOWS);
  assert.ok(res.ok, res.errors.join('; '));
  assert.strictEqual(res.clean.extra, undefined);
  assert.strictEqual(res.clean.website, undefined);
  assert.strictEqual(res.clean.open.confusing, 'the bars');
  assert.strictEqual(res.clean.windows[2].position, 3);
});

check('missing consent, bad scale values, and wrong window sets are rejected', () => {
  const ids = S.assignWindows(REPLAY_WINDOWS, seeded(5));
  const noConsent = validBody(ids);
  noConsent.consent.adult = false;
  assert.ok(!S.validateResponse(noConsent, REPLAY_WINDOWS).ok);

  const badLikert = validBody(ids);
  badLikert.sus[3] = 6;
  assert.ok(!S.validateResponse(badLikert, REPLAY_WINDOWS).ok);

  const badDriver = validBody(ids);
  badDriver.windows[0].top_driver = 'heart_rate';
  assert.ok(!S.validateResponse(badDriver, REPLAY_WINDOWS).ok);

  const sameQuad = validBody(['TP-02', 'TP-03', 'TN-01', 'FN-02']);
  const res = S.validateResponse(sameQuad, REPLAY_WINDOWS);
  assert.ok(!res.ok);
  assert.ok(res.errors.some((e) => /cover TP, FP, TN, FN/.test(e)));

  const unknown = validBody(ids);
  unknown.windows[1].window_id = 'ZZ-99';
  assert.ok(!S.validateResponse(unknown, REPLAY_WINDOWS).ok);

  assert.ok(!S.validateResponse([], REPLAY_WINDOWS).ok);
  assert.ok(!S.validateResponse(null, REPLAY_WINDOWS).ok);
});

check('open text is capped', () => {
  const body = validBody(S.assignWindows(REPLAY_WINDOWS, seeded(9)));
  body.open.change = 'x'.repeat(5000);
  const res = S.validateResponse(body, REPLAY_WINDOWS);
  assert.ok(res.ok);
  assert.strictEqual(res.clean.open.change.length, S.TEXT_MAX);
});

check('SUS follows the standard formula', () => {
  assert.strictEqual(S.susScore([5, 1, 5, 1, 5, 1, 5, 1, 5, 1]), 100);
  assert.strictEqual(S.susScore([1, 5, 1, 5, 1, 5, 1, 5, 1, 5]), 0);
  assert.strictEqual(S.susScore([3, 3, 3, 3, 3, 3, 3, 3, 3, 3]), 50);
  assert.strictEqual(S.susScore([4, 2, 4, 2, 4, 2, 4, 2, 4, 2]), 75);
  assert.strictEqual(S.susScore([4, 2]), null);
});

check('scoring marks correct answers against the firmware score and top driver', () => {
  const ids = ['TP-02', 'FP-03', 'TN-01', 'TP-01'];
  const body = validBody(ids);
  body.windows[3].flagged = 'yes'; // TP-01 firmware 0.405: wrong
  body.windows[3].direction = 'toward_normal'; // energy +0.173: wrong
  const clean = S.validateResponse(body, REPLAY_WINDOWS).clean;
  const scored = S.scoreResponse(clean, REPLAY_WINDOWS);
  assert.strictEqual(scored.meaning_correct, true);
  assert.strictEqual(scored.sus, 75);
  assert.strictEqual(scored.ess_mean, 3.5);
  const last = scored.windows[3];
  assert.strictEqual(last.quadrant, 'FN');
  assert.strictEqual(last.true_label, 1);
  assert.strictEqual(last.flagged_correct, false);
  assert.strictEqual(last.top_driver_correct, true);
  assert.strictEqual(last.direction_correct, false);
  assert.ok(scored.windows.slice(0, 3).every((w) => w.flagged_correct && w.top_driver_correct && w.direction_correct));
});

check('near-threshold windows are marked', () => {
  const row = REPLAY_WINDOWS.find((r) => r.window_id === 'FN-01'); // 0.501
  const w = S.scoreWindow({ flagged: 'yes', top_driver: 'kurtosis', direction: 'toward_risk' }, row);
  assert.strictEqual(w.near_threshold, true);
  assert.strictEqual(w.flagged_correct, true);
});

console.log('all survey tests passed');
