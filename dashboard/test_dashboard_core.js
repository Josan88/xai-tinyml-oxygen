const assert = require('assert');
const core = require('./dashboard_core.js');
const replay = require('./build_replay.js');
const REPLAY_WINDOWS = require('./replay_windows.js');

// Firmware-format packet fixtures (test-only; the page no longer offers examples).
const PACKETS = {
  candidate: '---|CANDIDATE RISK (Unconfirmed)|Candidate risk detected (Prob: 0.62), awaiting confirmation|0|0|OK|kurtosis:0.0800|bandpower_ratio_0_5_2_to_2_5:0.0210|std:-0.0100',
  confirmed: '---|CONFIRMED LOW OXYGEN RISK|CONFIRMED RISK (Prob: 0.71) top driver: kurtosis|0|0|OK|kurtosis:0.0910|skewness:0.0240|std:-0.0110',
  nofinger: '---|No finger detected|Place finger on sensor|0|0|NO FINGER|none:0.0000|none:0.0000|none:0.0000',
};

function check(name, fn) {
  fn();
  console.log('ok  ' + name);
}

check('hides toy spo2 and reads TinyKAN fields', () => {
  const state = core.parseFirmwarePacket(
    '88.2|Normal|Normal (Prob: 0.39)|12345|14500|OK|kurtosis:0.0556|bandpower_ratio_0_5_2_to_2_5:0.0120|std:-0.0080'
  );
  assert.strictEqual(state.noFinger, false);
  assert.strictEqual(state.signal, 'OK');
  assert.strictEqual(state.score, 0.39);
  assert.strictEqual(state.candidate, false);
  assert.strictEqual(state.confirmed, false);
  assert.strictEqual(state.top3[0].name, 'kurtosis');
  assert.strictEqual(state.top3[0].gloss, 'waveform peakedness');
  assert.strictEqual(state.top3[0].direction, 'toward risk');
  assert.strictEqual(state.top3[2].direction, 'toward normal');
  const line = core.buildExplanation(state);
  assert.ok(!/88/.test(line));
  assert.ok(!/SpO2/i.test(line));
  assert.ok(line.indexOf('Not flagged.') === 0);
  assert.ok(line.indexOf('waveform peakedness, which pushes toward a flag') !== -1);
  assert.ok(line.indexOf('Not an oxygen level.') !== -1);
});

check('candidate and confirmed flags', () => {
  const cand = core.parseFirmwarePacket(PACKETS.candidate);
  assert.strictEqual(cand.candidate, true);
  assert.strictEqual(cand.confirmed, false);
  assert.strictEqual(cand.score, 0.62);
  const conf = core.parseFirmwarePacket(PACKETS.confirmed);
  assert.strictEqual(conf.candidate, true);
  assert.strictEqual(conf.confirmed, true);
  assert.strictEqual(conf.score, 0.71);
});

check('no finger clears score', () => {
  const state = core.parseFirmwarePacket(PACKETS.nofinger);
  assert.strictEqual(state.noFinger, true);
  assert.strictEqual(state.signal, 'NO FINGER');
  assert.strictEqual(state.score, null);
  assert.strictEqual(core.buildExplanation(state), 'Place finger on the sensor.');
});

check('rejects short packets', () => {
  const state = core.parseFirmwarePacket('only|two');
  assert.strictEqual(state.noFinger, true);
  assert.strictEqual(core.buildExplanation(state), 'Place finger on the sensor.');
});

check('does not treat none:0 as a feature', () => {
  const state = core.parseFirmwarePacket(
    '---|Normal|Normal (Prob: 0.00)|0|0|OK|none:0.0000|none:0.0000|none:0.0000'
  );
  assert.strictEqual(state.windowReady, false);
  assert.strictEqual(state.score, null);
  assert.strictEqual(core.buildExplanation(state), 'Waiting for the first completed window.');
});

check('exposes red and ir counts', () => {
  const state = core.parseFirmwarePacket(
    '97.4|Normal|Normal (Prob: 0.39)|81234|102345|OK|kurtosis:0.0556|bandpower_ratio_0_5_2_to_2_5:0.0120|std:-0.0080'
  );
  assert.strictEqual(state.red, 81234);
  assert.strictEqual(state.ir, 102345);
});

check('warmup packet with counts keeps score null', () => {
  const state = core.parseFirmwarePacket(
    '97.4|Normal|Normal (Prob: 0.00)|81234|102345|OK|none:0.0000|none:0.0000|none:0.0000'
  );
  assert.strictEqual(state.ir, 102345);
  assert.strictEqual(state.windowReady, false);
  assert.strictEqual(state.score, null);
});

check('boot packet has null counts', () => {
  const state = core.parseFirmwarePacket('---|Waiting...|Connect to Arduino to see live explanation|---|---|---');
  assert.strictEqual(state.red, null);
  assert.strictEqual(state.ir, null);
  assert.strictEqual(state.score, null);
});

check('holds last window through warmup, clears on no finger', () => {
  const ready = core.parseFirmwarePacket(PACKETS.candidate);
  let r = core.holdLastWindow(null, ready);
  assert.strictEqual(r.view, ready);
  assert.strictEqual(r.held, ready);

  const warmup = core.parseFirmwarePacket(
    '97.4|Normal|Normal (Prob: 0.00)|81234|102345|CLIPPED|none:0.0000|none:0.0000|none:0.0000'
  );
  r = core.holdLastWindow(r.held, warmup);
  assert.strictEqual(r.view.held, true);
  assert.strictEqual(r.view.score, 0.62);
  assert.strictEqual(r.view.candidate, true);
  assert.strictEqual(r.view.top3.length, 3);
  assert.strictEqual(r.view.signal, 'CLIPPED');
  assert.strictEqual(r.view.ir, 102345);
  assert.ok(core.buildExplanation(r.view).indexOf('Flagged.') === 0);
  assert.ok(core.buildExplanation(r.view).indexOf('Last window.') !== -1);

  r = core.holdLastWindow(r.held, core.parseFirmwarePacket(PACKETS.nofinger));
  assert.strictEqual(r.held, null);
  assert.strictEqual(r.view.score, null);

  r = core.holdLastWindow(r.held, warmup);
  assert.strictEqual(r.view, warmup);
  assert.strictEqual(core.buildExplanation(r.view), 'Waiting for the first completed window.');
});

check('reads bpm from field 9', () => {
  const state = core.parseFirmwarePacket(
    '97.4|Normal|Normal (Prob: 0.39)|81234|102345|OK|kurtosis:0.0556|bandpower_ratio_0_5_2_to_2_5:0.0120|std:-0.0080|72'
  );
  assert.strictEqual(state.bpm, 72);
  assert.strictEqual(core.formatBpm(state.bpm), '72');
  assert.strictEqual(state.score, 0.39);
  assert.strictEqual(state.top3.length, 3);
  assert.strictEqual(state.top3[2].name, 'std');
});

check('hyphen bpm is null', () => {
  const state = core.parseFirmwarePacket(
    '97.4|Normal|Normal (Prob: 0.39)|81234|102345|OK|kurtosis:0.0556|bandpower_ratio_0_5_2_to_2_5:0.0120|std:-0.0080|-'
  );
  assert.strictEqual(state.bpm, null);
  assert.strictEqual(core.formatBpm(state.bpm), '—');
  assert.strictEqual(state.score, 0.39);
  const nf = core.parseFirmwarePacket(PACKETS.nofinger + '|-');
  assert.strictEqual(nf.noFinger, true);
  assert.strictEqual(nf.bpm, null);
});

check('packet without field 9 gives null bpm, not zero', () => {
  const state = core.parseFirmwarePacket(
    '97.4|Normal|Normal (Prob: 0.39)|81234|102345|OK|kurtosis:0.0556|bandpower_ratio_0_5_2_to_2_5:0.0120|std:-0.0080'
  );
  assert.strictEqual(state.bpm, null);
  assert.strictEqual(core.formatBpm(state.bpm), '—');
  assert.strictEqual(core.parseFirmwarePacket('---|Waiting...|Connect to Arduino to see live explanation|---|---|---').bpm, null);
  assert.strictEqual(core.parseFirmwarePacket('only|two').bpm, null);
});

check('out-of-range or malformed bpm is null', () => {
  const base = '97.4|Normal|Normal (Prob: 0.39)|81234|102345|OK|kurtosis:0.0556|none:0.0000|none:0.0000|';
  ['0', '39', '181', '72.5', 'abc', ''].forEach((v) => {
    assert.strictEqual(core.parseFirmwarePacket(base + v).bpm, null, v);
  });
  assert.strictEqual(core.parseFirmwarePacket(base + '40').bpm, 40);
  assert.strictEqual(core.parseFirmwarePacket(base + '180').bpm, 180);
});

check('held window shows live bpm, not the held one', () => {
  const ready = core.parseFirmwarePacket(PACKETS.candidate + '|64');
  const warmup = core.parseFirmwarePacket(
    '97.4|Normal|Normal (Prob: 0.00)|81234|102345|OK|none:0.0000|none:0.0000|none:0.0000|71'
  );
  const r = core.holdLastWindow(ready, warmup);
  assert.strictEqual(r.view.held, true);
  assert.strictEqual(r.view.bpm, 71);
});

check('replay row uses firmware_probability, not the 0.93 csv_probability', () => {
  const rows = replay.readRows();
  const tp01 = rows.find((r) => r.window_id === 'TP-01');
  assert.ok(tp01);
  assert.ok(Math.abs(Number(tp01.csv_probability) - 0.9306) < 1e-3);
  const state = core.replayState(tp01, false);
  assert.ok(Math.abs(state.score - 0.4052) < 1e-3);
  assert.strictEqual(core.formatScore(state.score), '0.41');
  assert.strictEqual(state.candidate, false);
  assert.strictEqual(state.confirmed, false);
  const line = core.buildExplanation(state);
  assert.ok(line.indexOf('Not flagged.') === 0, line);
  assert.ok(!/0\.93/.test(line), line);
  assert.ok(line.indexOf('signal energy, which pushes toward a flag') !== -1, line);
});

check('replay skips zero deltas and has no red, IR, HR, or SpO2', () => {
  const state = core.replayState(replay.readRows()[0], false);
  assert.deepStrictEqual(state.top3.map((t) => t.name), ['energy', 'minimum']);
  assert.strictEqual(state.red, null);
  assert.strictEqual(state.ir, null);
  assert.strictEqual(state.bpm, null);
  assert.strictEqual(core.formatBpm(state.bpm), '—');
  assert.strictEqual(state.replay, true);
  assert.strictEqual(state.windowId, 'TP-01');
  assert.ok(!/spo2/i.test(JSON.stringify(state)));
});

check('replay confirmed needs this row and the previous row as candidates', () => {
  const row = { window_id: 'X', firmware_probability: 0.6, top1_feature: 'kurtosis', top1_delta: 0.05 };
  assert.strictEqual(core.replayState(row, false).confirmed, false);
  assert.strictEqual(core.replayState(row, true).candidate, true);
  assert.strictEqual(core.replayState(row, true).confirmed, true);
  const below = Object.assign({}, row, { firmware_probability: 0.49 });
  assert.strictEqual(core.replayState(below, true).candidate, false);
  assert.strictEqual(core.replayState(below, true).confirmed, false);
});

check('replay_windows.js matches the 20-row CSV and drops the hash and csv fields', () => {
  const rows = replay.readRows();
  assert.strictEqual(rows.length, 20);
  assert.strictEqual(REPLAY_WINDOWS.length, 20);
  rows.forEach((r, i) => {
    const w = REPLAY_WINDOWS[i];
    assert.strictEqual(w.window_id, r.window_id);
    assert.strictEqual(w.firmware_probability, Number(r.firmware_probability));
    assert.strictEqual(w.top1_feature, r.top1_feature);
    assert.strictEqual(w.top3_delta, Number(r.top3_delta));
    ['recording_id', 'csv_probability', 'explanation', 'true_label'].forEach((k) => {
      assert.ok(!(k in w), k);
    });
  });
});

// The two v2 inputs are pulse-amplitude ratios and their glosses say "pulse" by design;
// they get their own check below. So do the five P5 beat inputs (hr_bpm is a pulse
// rate). Every v1 entry keeps the original rule.
const V2_GLOSS_KEYS = ['ratio_of_ratios', 'ir_perfusion_index'];
const P5_GLOSS = {
  r_beat: 'red/infrared beat ratio',
  pi_beat: 'beat perfusion index',
  hr_bpm: 'pulse rate from detected beats',
  sqi_template: 'beat-shape consistency',
  sqi_ibi_ratio: 'beat-interval spread',
};

check('no glossary entry implies heart rate or SpO2', () => {
  const own = V2_GLOSS_KEYS.concat(Object.keys(P5_GLOSS));
  Object.keys(core.FEATURE_GLOSS).filter((k) => own.indexOf(k) === -1).forEach((k) => {
    assert.ok(!/pulse|heart|bpm|spo2|oxygen/i.test(core.FEATURE_GLOSS[k]), k);
  });
  assert.strictEqual(Object.keys(core.FEATURE_GLOSS).length, 21);
});

check('P5 glosses name the beat inputs without calling them oxygen saturation', () => {
  Object.keys(P5_GLOSS).forEach((k) => {
    assert.strictEqual(core.glossFor(k), P5_GLOSS[k], k);
    assert.ok(!/spo2|oxygen|saturation|sao2|scalc/i.test(core.FEATURE_GLOSS[k]), k);
  });
  // v1 glosses the tests and survey rely on are still there.
  assert.strictEqual(core.glossFor('energy'), 'signal energy');
  assert.strictEqual(core.glossFor('kurtosis'), 'waveform peakedness');
});

check('v2 glosses name the red/IR inputs without calling them oxygen saturation', () => {
  assert.strictEqual(core.glossFor('ratio_of_ratios'), 'red/infrared pulse ratio');
  assert.strictEqual(core.glossFor('ir_perfusion_index'), 'infrared pulse strength (perfusion index)');
  V2_GLOSS_KEYS.forEach((k) => {
    assert.ok(!/heart|bpm|spo2|oxygen|saturation|sao2/i.test(core.FEATURE_GLOSS[k]), k);
  });
});

check('v2 packet with ratio_of_ratios as top driver', () => {
  const state = core.parseFirmwarePacket(
    '96.4|CONFIRMED LOW OXYGEN RISK|CONFIRMED RISK (Prob: 0.89) top driver: ratio_of_ratios|81234|102345|OK|ratio_of_ratios:0.4120|ir_perfusion_index:-0.0310|kurtosis:0.0200|72'
  );
  assert.strictEqual(state.confirmed, true);
  assert.strictEqual(state.score, 0.89);
  assert.deepStrictEqual(state.top3.map((t) => t.name), ['ratio_of_ratios', 'ir_perfusion_index', 'kurtosis']);
  assert.strictEqual(state.top3[0].gloss, 'red/infrared pulse ratio');
  assert.strictEqual(state.top3[1].gloss, 'infrared pulse strength (perfusion index)');
  assert.strictEqual(state.top3[1].direction, 'toward normal');
  assert.strictEqual(state.bpm, 72);
  const line = core.buildExplanation(state);
  assert.strictEqual(line, 'Flagged. Main clue: red/infrared pulse ratio, which pushes toward a flag. A shape match, not an oxygen level.');
  assert.ok(!/SpO2|96/.test(line), line);
});

check('v2 packet with ir_perfusion_index pulling away from a flag', () => {
  const state = core.parseFirmwarePacket(
    '97.0|Normal|Normal (Prob: 0.12)|81234|102345|OK|ir_perfusion_index:-0.0410|ratio_of_ratios:-0.0300|std:0.0010'
  );
  assert.strictEqual(state.candidate, false);
  const line = core.buildExplanation(state);
  assert.strictEqual(line, 'Not flagged. Main clue: infrared pulse strength (perfusion index), which pulls away from a flag. Not an oxygen level.');
});

check('live packet exposes the Red/IR estimate without putting it in the sentence', () => {
  const state = core.parseFirmwarePacket(
    '88.2|Normal|Normal (Prob: 0.39)|12345|14500|OK|kurtosis:0.0556|bandpower_ratio_0_5_2_to_2_5:0.0120|std:-0.0080'
  );
  assert.strictEqual(state.spo2Est, 88.2);
  assert.strictEqual(core.formatSpo2Est(state.spo2Est), '88.2');
  const line = core.buildExplanation(state);
  assert.ok(!/88/.test(line));
  assert.ok(!/SpO2/i.test(line));
  assert.strictEqual(core.parseSpo2Est('---'), null);
  assert.strictEqual(core.parseSpo2Est('101'), null);
  assert.strictEqual(core.parseFirmwarePacket(PACKETS.nofinger).spo2Est, null);
  const ready = core.parseFirmwarePacket(PACKETS.candidate.replace('---|', '96.1|') + '|64');
  const warmup = core.parseFirmwarePacket(
    '97.4|Normal|Normal (Prob: 0.00)|81234|102345|OK|none:0.0000|none:0.0000|none:0.0000|71'
  );
  const held = core.holdLastWindow(ready, warmup);
  assert.strictEqual(held.view.spo2Est, 97.4);
  assert.strictEqual(held.view.score, 0.62);
  const replayRow = core.replayState(replay.readRows()[0], false);
  assert.ok(!/spo2/i.test(JSON.stringify(replayRow)));
});

check('plain verdict follows the raw window flags and never names oxygen', () => {
  assert.strictEqual(core.verdictFor(null).level, 'off');
  assert.strictEqual(core.verdictFor(core.parseFirmwarePacket(PACKETS.nofinger)).level, 'off');
  const warmup = core.parseFirmwarePacket(
    '97.4|Normal|Normal (Prob: 0.00)|81234|102345|OK|none:0.0000|none:0.0000|none:0.0000|71'
  );
  assert.strictEqual(core.verdictFor(warmup).level, 'wait');
  const normal = core.parseFirmwarePacket(
    '88.2|Normal|Normal (Prob: 0.39)|12345|14500|OK|kurtosis:0.0556|bandpower_ratio_0_5_2_to_2_5:0.0120|std:-0.0080'
  );
  assert.strictEqual(core.verdictFor(normal).title, 'No alert');
  assert.strictEqual(core.verdictFor(core.parseFirmwarePacket(PACKETS.candidate)).title, 'Possible');
  assert.strictEqual(core.verdictFor(core.parseFirmwarePacket(PACKETS.confirmed)).title, 'Alert');
  const held = core.holdLastWindow(core.parseFirmwarePacket(PACKETS.candidate), warmup).view;
  assert.ok(/last window/.test(core.verdictFor(held).detail));
  [normal, PACKETS.candidate, PACKETS.confirmed].forEach((p) => {
    const v = core.verdictFor(typeof p === 'string' ? core.parseFirmwarePacket(p) : p);
    assert.ok(!/oxygen|spo2/i.test(v.title + v.detail));
  });
});

// ---------------------------------------------------------------------------
// Result guide ("What this result means").

const NORMAL_PKT = '88.2|Normal|Normal (Prob: 0.39)|12345|14500|OK|kurtosis:0.0556|bandpower_ratio_0_5_2_to_2_5:0.0120|std:-0.0080';
const WARMUP_PKT = '97.4|Normal|Normal (Prob: 0.00)|81234|102345|OK|none:0.0000|none:0.0000|none:0.0000|71';
const V2_CONF_PKT = '96.4|CONFIRMED LOW OXYGEN RISK|CONFIRMED RISK (Prob: 0.89) top driver: ratio_of_ratios|81234|102345|OK|ratio_of_ratios:0.4120|ir_perfusion_index:-0.0310|kurtosis:0.0200|72';
const V2_LOW_PKT = '97.0|Normal|Normal (Prob: 0.12)|81234|102345|OK|ir_perfusion_index:-0.0410|ratio_of_ratios:-0.0300|std:0.0010';

const SHOWN_TAIL = "This feature moved the model's score; it is not an oxygen reading or a cause.";
const WITHHELD_OPPOSES = 'Explanation withheld. The feature that moved this score most pushed it the other way from this result, so it does not explain the result.';
const WITHHELD_ZERO = 'Explanation withheld. Swapping any one feature for its training average left this score unchanged, so there is no clue to show.';

function guideText(g) {
  return [g.title, g.shows, g.meaning, g.risk, g.action, g.clue].join(' ');
}

check('result guide: live levels follow verdictFor, bands follow the guide', () => {
  const cases = [
    [null, 'off', 'grey'],
    [core.parseFirmwarePacket(PACKETS.nofinger), 'off', 'grey'],
    [core.parseFirmwarePacket(WARMUP_PKT), 'wait', 'blue'],
    [core.parseFirmwarePacket(NORMAL_PKT), 'clear', 'green'],
    [core.parseFirmwarePacket(PACKETS.candidate), 'watch', 'yellow'],
    [core.parseFirmwarePacket(PACKETS.confirmed), 'alert', 'red'],
  ];
  cases.forEach(([state, level, band]) => {
    const g = core.resultGuideFor(state, 'live');
    assert.strictEqual(g.level, level, level);
    assert.strictEqual(g.level, core.verdictFor(state).level, level);
    assert.strictEqual(g.band, band, level);
  });
  assert.strictEqual(core.resultGuideFor(null, 'live').clue, 'No data from the sensor yet.');
  assert.strictEqual(core.resultGuideFor(core.parseFirmwarePacket(PACKETS.nofinger), 'live').clue, 'No usable pulse, so no score.');
  assert.strictEqual(core.resultGuideFor(core.parseFirmwarePacket(WARMUP_PKT), 'live').clue, 'No score yet. The 10-second window is still filling.');
});

check('result guide: live wording is the supervisor results guide, verbatim', () => {
  const g = (p) => core.resultGuideFor(p === null ? null : core.parseFirmwarePacket(p), 'live');
  const off = g(PACKETS.nofinger);
  assert.strictEqual(off.title, 'Place a finger on the sensor');
  assert.strictEqual(off.meaning, 'The sensor cannot see a usable pulse.');
  assert.strictEqual(off.risk, 'No result');
  assert.strictEqual(off.action, 'Cover the sensor fully and keep still.');
  const wait = g(WARMUP_PKT);
  assert.strictEqual(wait.title, 'Reading pulse…');
  assert.strictEqual(wait.meaning, 'Collecting the first 10-second pulse window.');
  assert.strictEqual(wait.risk, 'No result yet');
  assert.strictEqual(wait.action, 'Keep still. First result in about 15 s, then every 5 s.');
  const clear = g(NORMAL_PKT);
  assert.strictEqual(clear.title, 'No alert');
  assert.strictEqual(clear.meaning, 'The pulse pattern does not look like the low-oxygen examples the model learned from.');
  assert.strictEqual(clear.risk, 'Low');
  assert.strictEqual(clear.action, 'No action needed.');
  const watch = g(PACKETS.candidate);
  assert.strictEqual(watch.title, 'Possible');
  assert.strictEqual(watch.meaning, 'One 10-second window looks like the low-oxygen examples. A single window can be caused by movement.');
  assert.strictEqual(watch.risk, 'Watch');
  assert.strictEqual(watch.action, 'Keep still. The system re-checks 5 s later.');
  const alert = g(PACKETS.confirmed);
  assert.strictEqual(alert.title, 'Alert');
  assert.strictEqual(alert.meaning, 'Two windows in a row look like the low-oxygen examples.');
  assert.strictEqual(alert.risk, 'Possible low oxygen');
  assert.strictEqual(alert.action, 'Check finger placement, then confirm with a certified pulse oximeter. Seek medical help if it also reads low or you feel unwell.');
});

check('result guide: clue names the score vs 0.50 and the top driver direction', () => {
  const clue = (p) => core.resultGuideFor(core.parseFirmwarePacket(p), 'live').clue;
  assert.strictEqual(clue(V2_CONF_PKT), 'Score 0.89, above the 0.50 line. Biggest influence: red/infrared pulse ratio, which raised the score by 0.412. ' + SHOWN_TAIL);
  assert.strictEqual(clue(V2_LOW_PKT), 'Score 0.12, below the 0.50 line. Biggest influence: infrared pulse strength (perfusion index), which lowered the score by 0.041. ' + SHOWN_TAIL);
  // Not flagged, lead raised the score: withheld (opposes).
  assert.strictEqual(clue(NORMAL_PKT), 'Score 0.39, below the 0.50 line. ' + WITHHELD_OPPOSES);
  const onLine = core.parseFirmwarePacket(PACKETS.candidate.replace('Prob: 0.62', 'Prob: 0.50'));
  assert.ok(core.resultGuideFor(onLine, 'live').clue.indexOf('Score 0.50, on the 0.50 line.') === 0);
});

check('result guide: a rounded 0.50 with flag No says it was just below the line', () => {
  // Firmware prints two decimals: P 0.497 arrives as "0.50" with Candidate No.
  const down = Object.assign({}, core.parseFirmwarePacket(NORMAL_PKT), { score: 0.50 });
  const d = core.resultGuideFor(down, 'live');
  assert.strictEqual(d.level, 'clear');
  assert.ok(d.clue.indexOf('Score 0.50 is rounded; this window on its own was just below the 0.50 line.') === 0, d.clue);
});

check('result guide: held window is named as the last window', () => {
  const warmup = core.parseFirmwarePacket(WARMUP_PKT);
  const held = core.holdLastWindow(core.parseFirmwarePacket(PACKETS.candidate), warmup).view;
  const g = core.resultGuideFor(held, 'live');
  assert.strictEqual(g.level, 'watch');
  assert.ok(g.clue.indexOf('Last completed window, kept while the next one fills. Score 0.62') === 0, g.clue);
  const fresh = core.resultGuideFor(core.parseFirmwarePacket(PACKETS.candidate), 'live');
  assert.ok(!/last/i.test(fresh.clue), fresh.clue);
});

check('result guide: CLIPPED adds a caution, without changing the level', () => {
  const clippedWarmup = core.parseFirmwarePacket(WARMUP_PKT.replace('|OK|', '|CLIPPED|'));
  const held = core.holdLastWindow(core.parseFirmwarePacket(PACKETS.candidate), clippedWarmup).view;
  const h = core.resultGuideFor(held, 'live');
  assert.strictEqual(h.level, 'watch');
  assert.strictEqual(h.caution, true);
  assert.ok(/Signal CLIPPED: reposition the finger\. This result may be unreliable\.$/.test(h.clue), h.clue);

  const fresh = core.parseFirmwarePacket(NORMAL_PKT.replace('|OK|', '|CLIPPED|'));
  const f = core.resultGuideFor(fresh, 'live');
  assert.strictEqual(f.level, 'clear');
  assert.strictEqual(f.caution, true);
  assert.ok(/may be unreliable/.test(f.clue), f.clue);

  const w = core.resultGuideFor(clippedWarmup, 'live');
  assert.strictEqual(w.level, 'wait');
  assert.strictEqual(w.caution, true);
  assert.ok(/reposition the finger/.test(w.clue) && !/unreliable/.test(w.clue), w.clue);

  const ok = core.resultGuideFor(core.parseFirmwarePacket(NORMAL_PKT), 'live');
  assert.strictEqual(ok.caution, false);
  assert.ok(!/CLIPPED|reposition/.test(ok.clue), ok.clue);
});

check('result guide: replay never returns alert, hides the window id', () => {
  REPLAY_WINDOWS.forEach((row) => {
    [false, true].forEach((prev) => {
      const state = core.replayState(row, prev);
      const g = core.resultGuideFor(state, 'replay');
      assert.ok(g.level === 'clear' || g.level === 'watch', row.window_id + ' ' + g.level);
      assert.strictEqual(g.level, state.score >= core.THRESHOLD ? 'watch' : 'clear', row.window_id);
      assert.strictEqual(g.band, g.level === 'watch' ? 'yellow' : 'green');
      assert.ok(g.action.indexOf('In live use: ') === 0, g.action);
      assert.strictEqual(g.caution, false);
      const text = JSON.stringify(g);
      assert.ok(text.indexOf(row.window_id) === -1, row.window_id);
      assert.ok(!/\b(TP|TN|FP|FN)\b|true.?label|true positive|false negative/i.test(text), text);
    });
  });
  const confirmedRow = core.replayState({ window_id: 'X', firmware_probability: 0.8, top1_feature: 'kurtosis', top1_delta: 0.05 }, true);
  assert.strictEqual(confirmedRow.confirmed, true);
  assert.strictEqual(core.resultGuideFor(confirmedRow, 'replay').level, 'watch');
  assert.strictEqual(core.resultGuideFor(confirmedRow, 'replay').title, 'Flagged window');
  // Three decimals, the same as the replay readout.
  const tp01 = core.resultGuideFor(core.replayState(replay.readRows()[0], false), 'replay');
  assert.strictEqual(tp01.title, 'Not flagged');
  // Not flagged (0.405), lead energy raised the score: withheld (opposes).
  assert.strictEqual(tp01.clue, 'Score 0.405, below the 0.50 line. ' + WITHHELD_OPPOSES);
});

check('result guide: no percentage, no oxygen level, no SpO2 value, no new score bands', () => {
  const warmup = core.parseFirmwarePacket(WARMUP_PKT);
  const clipped = core.parseFirmwarePacket(WARMUP_PKT.replace('|OK|', '|CLIPPED|'));
  const live = [
    null, PACKETS.nofinger, WARMUP_PKT, NORMAL_PKT, PACKETS.candidate, PACKETS.confirmed, V2_CONF_PKT, V2_LOW_PKT,
    NORMAL_PKT.replace('|OK|', '|CLIPPED|'),
  ].map((p) => (p === null ? null : core.parseFirmwarePacket(p)));
  live.push(core.holdLastWindow(core.parseFirmwarePacket(V2_CONF_PKT), warmup).view);
  live.push(core.holdLastWindow(core.parseFirmwarePacket(PACKETS.confirmed), clipped).view);
  live.push(Object.assign({}, core.parseFirmwarePacket(PACKETS.candidate), { score: 0.33 }));
  const outputs = live.map((s) => [s, core.resultGuideFor(s, 'live'), (x) => core.formatScore(x), 'live']);
  REPLAY_WINDOWS.forEach((row) => {
    const s = core.replayState(row, true);
    outputs.push([s, core.resultGuideFor(s, 'replay'), (x) => x.toFixed(3), 'replay']);
  });
  const banned = /%|spo2|sao2|saturation|oxygen level|blood[- ]?oxygen|hypox|confiden|\bshap\b|\blime\b|you are|your oxygen/i;
  outputs.forEach(([state, g, fmt, mode]) => {
    const text = guideText(g);
    assert.ok(!banned.test(text), text);
    if (state && state.spo2Est !== null && state.spo2Est !== undefined) {
      assert.ok(text.indexOf(core.formatSpo2Est(state.spo2Est)) === -1, text);
    }
    // Clue numbers: the shown score, 0.50, the 10-second window, and, only when
    // the explanation is shown, the lead |delta| at three decimals. Nothing else.
    const allowed = ['0.50', '10'];
    if (state && Number.isFinite(state.score) && state.windowReady) allowed.push(fmt(state.score));
    if (state && state.windowReady) {
      const gate = core.explanationGate(state, mode);
      if (gate.show) allowed.push(Math.abs(gate.lead.delta).toFixed(3));
    }
    (g.clue.match(/\d+(?:\.\d+)?/g) || []).forEach((n) => assert.ok(allowed.indexOf(n) !== -1, n + ' in ' + g.clue));
    (text.match(/\d+(?:\.\d+)?/g) || []).forEach((n) => assert.ok(allowed.concat(['5', '15']).indexOf(n) !== -1, n + ' in ' + text));
  });
});

check('result guide rows: five bands in order; replay greys out the live-only rows', () => {
  const live = core.resultGuideRows('live');
  assert.deepStrictEqual(live.map((r) => r.level), ['off', 'wait', 'clear', 'watch', 'alert']);
  assert.deepStrictEqual(live.map((r) => r.band), ['grey', 'blue', 'green', 'yellow', 'red']);
  assert.ok(live.every((r) => r.available));
  assert.deepStrictEqual(live.map((r) => r.risk), ['No result', 'No result yet', 'Low', 'Watch', 'Possible low oxygen']);
  const rep = core.resultGuideRows('replay');
  assert.deepStrictEqual(rep.map((r) => r.available), [false, false, true, true, false]);
  assert.deepStrictEqual(rep.filter((r) => r.available).map((r) => r.title), ['Not flagged', 'Flagged window']);
  // The lit legend row always carries the panel's own title.
  [null, NORMAL_PKT, PACKETS.candidate, PACKETS.confirmed].forEach((p) => {
    const g = core.resultGuideFor(p && core.parseFirmwarePacket(p), 'live');
    const row = live.find((r) => r.level === g.level);
    assert.strictEqual(row.title, g.title);
    assert.strictEqual(row.band, g.band);
  });
  const rg = core.resultGuideFor(core.replayState(REPLAY_WINDOWS[0], false), 'replay');
  assert.strictEqual(rep.find((r) => r.level === rg.level).title, rg.title);
});

// ---------------------------------------------------------------------------
// Explanation withheld rule (explanationGate), operator page only.

function xstate(score, candidate, deltas) {
  const names = ['kurtosis', 'ratio_of_ratios', 'std'];
  return {
    noFinger: false, signal: 'OK', score, candidate, confirmed: false, windowReady: true,
    top3: deltas.map((d, i) => ({ name: names[i], delta: d, gloss: core.glossFor(names[i]), direction: core.directionFor(d) })),
  };
}

check('gate: flagged with a positive lead is shown, with gloss, direction and magnitude', () => {
  ['live', 'replay'].forEach((mode) => {
    const g = core.explanationGate(xstate(0.71, true, [0.0912, -0.02, 0.01]), mode);
    assert.strictEqual(g.show, true, mode);
    assert.strictEqual(g.reason, null, mode);
    assert.strictEqual(g.flagged, true, mode);
    assert.strictEqual(g.text, 'Biggest influence: waveform peakedness, which raised the score by 0.091. ' + SHOWN_TAIL);
  });
});

check('gate: flagged with a negative lead is withheld (opposes)', () => {
  ['live', 'replay'].forEach((mode) => {
    const g = core.explanationGate(xstate(0.89, true, [-0.0120, 0, 0]), mode);
    assert.strictEqual(g.show, false, mode);
    assert.strictEqual(g.reason, 'opposes', mode);
    assert.strictEqual(g.text, WITHHELD_OPPOSES);
  });
  // The saturated Header-aligned sit: Candidate, ratio_of_ratios negative, two zero slots.
  const sit = core.parseFirmwarePacket(
    '96.9|CANDIDATE RISK (Unconfirmed)|Candidate risk detected (Prob: 0.89), awaiting confirmation|81234|102345|OK|ratio_of_ratios:-0.0120|mean:0.0000|std:0.0000|72'
  );
  assert.strictEqual(core.explanationGate(sit, 'live').reason, 'opposes');
  assert.strictEqual(core.operatorExplanation(sit, 'live'), 'Flagged. ' + WITHHELD_OPPOSES);
});

check('gate: not flagged with a negative lead is shown', () => {
  ['live', 'replay'].forEach((mode) => {
    const g = core.explanationGate(xstate(0.12, false, [-0.041, -0.03, 0.001]), mode);
    assert.strictEqual(g.show, true, mode);
    assert.strictEqual(g.flagged, false, mode);
    assert.strictEqual(g.text, 'Biggest influence: waveform peakedness, which lowered the score by 0.041. ' + SHOWN_TAIL);
  });
});

check('gate: not flagged with a positive lead is withheld (opposes)', () => {
  ['live', 'replay'].forEach((mode) => {
    const g = core.explanationGate(xstate(0.39, false, [0.0556, 0.012, -0.008]), mode);
    assert.strictEqual(g.show, false, mode);
    assert.strictEqual(g.reason, 'opposes', mode);
  });
  const tp01 = core.explanationGate(core.replayState(REPLAY_WINDOWS[0], false), 'replay');
  assert.strictEqual(tp01.reason, 'opposes');
});

check('gate: lead |delta| 0.0004 is withheld (zero); 0.0005 prints 0.001 and is judged by side', () => {
  [0.0004, -0.0004, 0].forEach((d) => {
    ['live', 'replay'].forEach((mode) => {
      [[0.62, true], [0.30, false]].forEach(([score, cand]) => {
        const g = core.explanationGate(xstate(score, cand, [d, 0.0001, 0]), mode);
        assert.strictEqual(g.show, false, d + ' ' + mode);
        assert.strictEqual(g.reason, 'zero', d + ' ' + mode);
        assert.strictEqual(g.text, WITHHELD_ZERO);
      });
    });
  });
  assert.strictEqual(core.explanationGate(xstate(0.62, true, [0.0005, 0, 0]), 'live').show, true);
  assert.strictEqual(core.explanationGate(xstate(0.62, true, [-0.0005, 0, 0]), 'live').reason, 'opposes');
  assert.strictEqual(core.explanationGate(xstate(0.62, true, [NaN, 0, 0]), 'live').reason, 'zero');
});

check('gate: empty top3 is withheld (zero)', () => {
  ['live', 'replay'].forEach((mode) => {
    const g = core.explanationGate(xstate(0.62, true, []), mode);
    assert.strictEqual(g.reason, 'zero', mode);
    assert.strictEqual(g.lead, null, mode);
  });
  const allZero = core.replayState({
    window_id: 'X', firmware_probability: 0.7,
    top1_feature: 'mean', top1_delta: 0, top2_feature: 'std', top2_delta: 0, top3_feature: 'minimum', top3_delta: 0,
  }, false);
  assert.strictEqual(allZero.top3.length, 0);
  assert.strictEqual(core.explanationGate(allZero, 'replay').reason, 'zero');
  assert.strictEqual(core.resultGuideFor(allZero, 'replay').clue, 'Score 0.700, above the 0.50 line. ' + WITHHELD_ZERO);
});

check('gate: live follows the Candidate flag, not the rounded score', () => {
  // P 0.497 arrives as "0.50" with Candidate No. Lead raised the score.
  const rounded = core.parseFirmwarePacket(NORMAL_PKT.replace('Prob: 0.39', 'Prob: 0.50'));
  assert.strictEqual(rounded.candidate, false);
  assert.strictEqual(rounded.score, 0.50);
  const live = core.explanationGate(rounded, 'live');
  assert.strictEqual(live.flagged, false);
  assert.strictEqual(live.reason, 'opposes');
  assert.strictEqual(core.explanationGate(rounded, 'replay').show, true);
  assert.strictEqual(core.operatorExplanation(rounded, 'live'), 'Not flagged. ' + WITHHELD_OPPOSES);
  assert.ok(core.resultGuideFor(rounded, 'live').clue.endsWith(WITHHELD_OPPOSES));
  // Candidate Yes with a displayed score under the line: still the flagged side.
  const cand = xstate(0.49, true, [0.08, 0.02, -0.01]);
  assert.strictEqual(core.explanationGate(cand, 'live').show, true);
  assert.strictEqual(core.explanationGate(cand, 'replay').reason, 'opposes');
});

check('operator sentence: no-finger and waiting lines unchanged, held window marked, survey sentence untouched', () => {
  assert.strictEqual(core.operatorExplanation(core.parseFirmwarePacket(PACKETS.nofinger), 'live'), 'Place finger on the sensor.');
  assert.strictEqual(core.operatorExplanation(null, 'live'), 'Place finger on the sensor.');
  assert.strictEqual(core.operatorExplanation(core.parseFirmwarePacket(WARMUP_PKT), 'live'), 'Waiting for the first completed window.');
  const held = core.holdLastWindow(core.parseFirmwarePacket(V2_CONF_PKT), core.parseFirmwarePacket(WARMUP_PKT)).view;
  assert.strictEqual(core.operatorExplanation(held, 'live'),
    'Flagged. Biggest influence: red/infrared pulse ratio, which raised the score by 0.412. ' + SHOWN_TAIL + ' Last window.');
  // buildExplanation (the survey Summary) does not apply the rule.
  assert.strictEqual(core.buildExplanation(core.parseFirmwarePacket(NORMAL_PKT)),
    'Not flagged. Main clue: waveform peakedness, which pushes toward a flag. Not an oxygen level.');
});

check('gate sentences: no percentage, no SpO2 value, never call a feature a cause', () => {
  const states = [
    NORMAL_PKT, PACKETS.candidate, PACKETS.confirmed, V2_CONF_PKT, V2_LOW_PKT,
    '96.9|CANDIDATE RISK (Unconfirmed)|Candidate risk detected (Prob: 0.89), awaiting confirmation|81234|102345|OK|ratio_of_ratios:-0.0120|mean:0.0000|std:0.0000|72',
    '97.0|Normal|Normal (Prob: 0.20)|81234|102345|OK|std:0.0003|mean:0.0000|kurtosis:0.0000',
  ].map((p) => [core.parseFirmwarePacket(p), 'live']);
  REPLAY_WINDOWS.forEach((row) => states.push([core.replayState(row, false), 'replay']));
  const reasons = new Set();
  states.forEach(([s, mode]) => {
    const g = core.explanationGate(s, mode);
    reasons.add(g.reason);
    [g.text, core.operatorExplanation(s, mode)].forEach((text) => {
      assert.ok(!/\d\s*%|%\s*\d/.test(text), text);
      assert.ok(!/spo2|sao2|saturation|oxygen level|blood[- ]?oxygen|hypox|your oxygen/i.test(text), text);
      if (s.spo2Est !== null && s.spo2Est !== undefined) {
        assert.ok(text.indexOf(core.formatSpo2Est(s.spo2Est)) === -1, text);
        assert.ok(text.indexOf(String(Math.round(s.spo2Est))) === -1, text);
      }
      assert.strictEqual((text.match(/cause/gi) || []).length, (text.match(/not an oxygen reading or a cause/g) || []).length, text);
      if (!g.show) assert.ok(!/\d/.test(text), text);
    });
  });
  assert.deepStrictEqual([...reasons].sort(), ['opposes', 'zero', null].sort());
  Object.values(core.WITHHELD_TEXT).forEach((t) => assert.ok(!/\d|%|oxygen/i.test(t), t));
});

// ---------------------------------------------------------------------------
// Fields 10 (model saturation estimate) and 11 (explainer reference tag).

const P5_PKT = '97.1|CANDIDATE RISK (Unconfirmed)|Candidate risk detected (Prob: 0.62), awaiting confirmation|81234|102345|OK|r_beat:0.0810|pi_beat:-0.0200|sqi_ibi_ratio:0.0100|72|84.6|label0';
const P5_DASH_PKT = '97.1|Normal|Normal (Prob: 0.31)|81234|102345|OK|pi_beat:-0.0450|hr_bpm:-0.0120|sqi_template:0.0030|72|-|label0';
const V2_REF0_PKT = '96.4|CONFIRMED LOW OXYGEN RISK|CONFIRMED RISK (Prob: 0.89) top driver: ratio_of_ratios|81234|102345|OK|ratio_of_ratios:0.4120|ir_perfusion_index:-0.0310|kurtosis:0.0200|72|-|label0';
const P5_WARMUP_PKT = '97.4|Normal|Normal (Prob: 0.00)|81234|102345|OK|none:0.0000|none:0.0000|none:0.0000|71|-|label0';

check('10-field packet (shipped images): no sat. est, training-mean reference', () => {
  [V2_CONF_PKT, NORMAL_PKT, PACKETS.candidate, PACKETS.nofinger, 'only|two'].forEach((p) => {
    const s = core.parseFirmwarePacket(p);
    assert.strictEqual(s.satEst, null, p);
    assert.strictEqual(s.hasSatEst, false, p);
    assert.strictEqual(s.xaiRef, 'mean', p);
  });
  assert.strictEqual(core.xaiRefNote('mean'), '');
});

check('12-field P5 packet: sat. est, label-0 reference, P5 drivers', () => {
  const s = core.parseFirmwarePacket(P5_PKT);
  assert.strictEqual(s.score, 0.62);
  assert.strictEqual(s.candidate, true);
  assert.strictEqual(s.bpm, 72);
  assert.strictEqual(s.spo2Est, 97.1);
  assert.strictEqual(s.satEst, 84.6);
  assert.strictEqual(core.formatSatEst(s.satEst), '84.6');
  assert.strictEqual(s.hasSatEst, true);
  assert.strictEqual(s.xaiRef, 'label0');
  assert.deepStrictEqual(s.top3.map((t) => t.gloss), ['red/infrared beat ratio', 'beat perfusion index', 'beat-interval spread']);
  const d = core.parseFirmwarePacket(P5_DASH_PKT);
  assert.strictEqual(d.satEst, null);
  assert.strictEqual(core.formatSatEst(d.satEst), '—');
  assert.strictEqual(d.hasSatEst, true);
  assert.strictEqual(d.score, 0.31);
});

check('12-field v2-ref0 packet: dash sat. est, label-0 reference', () => {
  const s = core.parseFirmwarePacket(V2_REF0_PKT);
  assert.strictEqual(s.satEst, null);
  assert.strictEqual(s.hasSatEst, true);
  assert.strictEqual(s.xaiRef, 'label0');
  assert.strictEqual(s.confirmed, true);
  assert.strictEqual(s.score, 0.89);
  assert.strictEqual(s.top3[0].name, 'ratio_of_ratios');
});

check('sat. est: dash, malformed, out of range, no finger, or no window is null', () => {
  const base = P5_PKT.replace('|84.6|label0', '|');
  ['-', '---', '', 'abc', '-1', '100.1', '101'].forEach((v) => {
    assert.strictEqual(core.parseFirmwarePacket(base + v + '|label0').satEst, null, v);
  });
  assert.strictEqual(core.parseFirmwarePacket(base + '0|label0').satEst, 0);
  assert.strictEqual(core.parseFirmwarePacket(base + '100|label0').satEst, 100);
  assert.strictEqual(core.parseFirmwarePacket(base + '84.6').satEst, 84.6);
  assert.strictEqual(core.parseFirmwarePacket(base + '84.6').xaiRef, 'mean');
  assert.strictEqual(core.parseFirmwarePacket(base + '84.6|other').xaiRef, 'mean');
  const nf = core.parseFirmwarePacket(PACKETS.nofinger + '|-|84.6|label0');
  assert.strictEqual(nf.satEst, null);
  assert.strictEqual(nf.hasSatEst, true);
  const warm = core.parseFirmwarePacket(P5_WARMUP_PKT.replace('|-|', '|84.6|'));
  assert.strictEqual(warm.windowReady, false);
  assert.strictEqual(warm.satEst, null);
});

check('held window keeps its sat. est and reference; no finger clears them', () => {
  let r = core.holdLastWindow(null, core.parseFirmwarePacket(P5_PKT));
  r = core.holdLastWindow(r.held, core.parseFirmwarePacket(P5_WARMUP_PKT));
  assert.strictEqual(r.view.held, true);
  assert.strictEqual(r.view.windowReady, true);
  assert.strictEqual(r.view.satEst, 84.6);
  assert.strictEqual(r.view.hasSatEst, true);
  assert.strictEqual(r.view.xaiRef, 'label0');
  assert.strictEqual(r.view.bpm, 71);
  assert.strictEqual(r.view.spo2Est, 97.4);
  r = core.holdLastWindow(r.held, core.parseFirmwarePacket(PACKETS.nofinger + '|-|-|label0'));
  assert.strictEqual(r.held, null);
  assert.strictEqual(r.view.satEst, null);
  r = core.holdLastWindow(r.held, core.parseFirmwarePacket(P5_WARMUP_PKT));
  assert.strictEqual(r.view.windowReady, false);
  assert.strictEqual(r.view.satEst, null);
  // A shipped image held through warmup still has no sat. est.
  const v1 = core.holdLastWindow(core.parseFirmwarePacket(PACKETS.candidate), core.parseFirmwarePacket(WARMUP_PKT)).view;
  assert.strictEqual(v1.satEst, null);
  assert.strictEqual(v1.hasSatEst, false);
  assert.strictEqual(v1.xaiRef, 'mean');
});

check('sat. est tile wording: one place, never SpO2 or oxygen; kept out of every sentence', () => {
  assert.strictEqual(core.SAT_EST_LABEL, 'Model sat. est');
  assert.strictEqual(core.SAT_EST_NOTE, 'uncalibrated · not SpO2');
  assert.ok(!/spo2|oxygen|blood/i.test(core.SAT_EST_LABEL));
  const warmup = core.parseFirmwarePacket(P5_WARMUP_PKT);
  const states = [core.parseFirmwarePacket(P5_PKT), core.holdLastWindow(core.parseFirmwarePacket(P5_PKT), warmup).view];
  states.forEach((s) => {
    const g = core.resultGuideFor(s, 'live');
    [core.buildExplanation(s), core.operatorExplanation(s, 'live'), guideText(g), core.logLine(s),
      JSON.stringify(core.verdictFor(s))].forEach((text) => {
      assert.ok(text.indexOf('84.6') === -1 && !/\b85\b/.test(text), text);
      assert.ok(!/sat\. est|saturation|scalc/i.test(text), text);
    });
  });
  REPLAY_WINDOWS.forEach((row) => assert.ok(!('satEst' in core.replayState(row, false)), row.window_id));
});

check('reference wording: label-0 changes the heading note and the zero sentence only', () => {
  assert.strictEqual(core.xaiRefNote('label0'), 'vs a typical non-low-oxygen window');
  assert.strictEqual(core.xaiRefNote(undefined), '');
  const withRef = (s, ref) => Object.assign({}, s, { xaiRef: ref });
  const zeroLabel0 = 'Explanation withheld. Swapping any one feature for its typical non-low-oxygen value left this score unchanged, so there is no clue to show.';
  ['live', 'replay'].forEach((mode) => {
    const zero = xstate(0.62, true, [0.0004, 0, 0]);
    assert.strictEqual(core.explanationGate(withRef(zero, 'mean'), mode).text, WITHHELD_ZERO);
    assert.strictEqual(core.explanationGate(zero, mode).text, WITHHELD_ZERO);
    assert.strictEqual(core.explanationGate(withRef(zero, 'label0'), mode).text, zeroLabel0);
    assert.strictEqual(core.explanationGate(withRef(zero, 'label0'), mode).reason, 'zero');
    // The withheld rule and the other sentences do not depend on the reference.
    [xstate(0.89, true, [-0.012, 0, 0]), xstate(0.71, true, [0.0912, -0.02, 0.01]), xstate(0.39, false, [0.0556, 0, 0])]
      .forEach((s) => {
        const a = core.explanationGate(withRef(s, 'mean'), mode);
        const b = core.explanationGate(withRef(s, 'label0'), mode);
        assert.deepStrictEqual([b.show, b.reason, b.text], [a.show, a.reason, a.text]);
      });
  });
  assert.ok(!/\d|%|spo2|saturation|oxygen level|your oxygen/i.test(zeroLabel0));
  // A real P5 packet: shown sentence has its current shape.
  assert.strictEqual(core.operatorExplanation(core.parseFirmwarePacket(P5_PKT), 'live'),
    'Flagged. Biggest influence: red/infrared beat ratio, which raised the score by 0.081. ' + SHOWN_TAIL);
  assert.strictEqual(core.operatorExplanation(core.parseFirmwarePacket(P5_DASH_PKT), 'live'),
    'Not flagged. Biggest influence: beat perfusion index, which lowered the score by 0.045. ' + SHOWN_TAIL);
});

// "Why this score": definition line and the reading of the shown bars together.

function drivers(deltas) {
  const names = ['kurtosis', 'maximum', 'dominant_frequency_hz'];
  return deltas.map((d, i) => ({ name: names[i], delta: d, gloss: core.glossFor(names[i]), direction: core.directionFor(d) }));
}

check('bars: definition line names the comparison point, no number, no oxygen', () => {
  assert.strictEqual(core.driverNote('mean'),
    'Top three features: how much each one\u2019s value raised or lowered the score, compared with its training average.');
  assert.strictEqual(core.driverNote(undefined), core.driverNote('mean'));
  assert.strictEqual(core.driverNote('label0'),
    'Top three features: how much each one\u2019s value raised or lowered the score, compared with its typical non-low-oxygen value.');
  ['mean', 'label0'].forEach((ref) => {
    assert.ok(!/\d|%|spo2|saturation|oxygen level|your oxygen|cause/i.test(core.driverNote(ref)), ref);
  });
});

check('bars: reading of three non-zero bars is counted from the signs', () => {
  assert.strictEqual(core.driverSummary(drivers([0.048, 0.041, 0.027])), 'All three raised the score.');
  assert.strictEqual(core.driverSummary(drivers([-0.041, -0.03, -0.002])), 'All three lowered the score.');
  assert.strictEqual(core.driverSummary(drivers([0.070, 0.038, -0.036])), 'Pulled both ways: two raised the score, one lowered it.');
  assert.strictEqual(core.driverSummary(drivers([-0.070, 0.038, -0.036])), 'Pulled both ways: one raised the score, two lowered it.');
});

check('bars: a slot that prints as 0.000 means no other feature moved the score', () => {
  assert.strictEqual(core.driverSummary(drivers([0.412, 0, 0])), 'Only one feature moved this score.');
  assert.strictEqual(core.driverSummary(drivers([0.412, 0.0004, -0.0001])), 'Only one feature moved this score.');
  assert.strictEqual(core.driverSummary(drivers([0.09, 0.02, 0])), 'Only two features moved this score; both raised it.');
  assert.strictEqual(core.driverSummary(drivers([-0.09, -0.02, 0])), 'Only two features moved this score; both lowered it.');
  assert.strictEqual(core.driverSummary(drivers([0.09, -0.02, 0.0004])), 'Only two features moved this score; one raised it, one lowered it.');
  // Same cut as the withheld rule: 0.0005 prints as 0.001 and counts.
  assert.strictEqual(core.EXPLANATION_ZERO, 0.0005);
  assert.strictEqual(core.driverSummary(drivers([0.09, 0.0005, 0])), 'Only two features moved this score; both raised it.');
  assert.deepStrictEqual(core.activeDrivers(drivers([0.09, 0.0004, -0.02])).map((d) => d.delta), [0.09, -0.02]);
});

check('bars: no reading when nothing moved the score or there are no bars', () => {
  assert.strictEqual(core.driverSummary([]), '');
  assert.strictEqual(core.driverSummary(undefined), '');
  assert.strictEqual(core.driverSummary(drivers([0.0004, 0, 0])), '');
  assert.strictEqual(core.driverSummary(drivers([NaN, 0, 0])), '');
});

check('bars: real replay rows and the v2 live packet read as expected', () => {
  const row = (id) => core.replayState(REPLAY_WINDOWS.find((r) => r.window_id === id), false);
  assert.strictEqual(core.driverSummary(row('TP-02').top3), 'All three raised the score.');
  assert.strictEqual(core.driverSummary(row('TP-04').top3), 'Pulled both ways: two raised the score, one lowered it.');
  const v2 = core.parseFirmwarePacket(V2_CONF_PKT);
  assert.strictEqual(core.driverSummary(v2.top3), 'Pulled both ways: two raised the score, one lowered it.');
  // A live packet keeps its 0.000 slots; activeDrivers drops them for the bars.
  const oneOnly = core.parseFirmwarePacket(V2_CONF_PKT.replace('ir_perfusion_index:-0.0310|kurtosis:0.0200', 'mean:0.0000|std:0.0000'));
  assert.strictEqual(oneOnly.top3.length, 3);
  assert.strictEqual(core.activeDrivers(oneOnly.top3).length, 1);
  assert.strictEqual(core.driverSummary(oneOnly.top3), 'Only one feature moved this score.');
});

check('bars: no reading names oxygen, a cause, an alert, or an expected feature', () => {
  const sets = [[0.048, 0.041, 0.027], [-0.041, -0.03, -0.002], [0.07, 0.038, -0.036], [-0.07, 0.038, -0.036],
    [0.412, 0, 0], [0.09, 0.02, 0], [-0.09, -0.02, 0], [0.09, -0.02, 0]];
  sets.forEach((s) => {
    const text = core.driverSummary(drivers(s));
    assert.ok(text.length > 0, String(s));
    assert.ok(!/\d|%|oxygen|spo2|saturation|cause|alert|expected|confiden|trust/i.test(text), text);
  });
});

// Operator-page caption: what the shown feature measures. Not a score movement,
// and not an oxygen reading. The survey does not call these.

check('measures line names what each glossed feature measures', () => {
  assert.strictEqual(core.measuresFor('ratio_of_ratios'), 'red wave size divided by infrared wave size');
  assert.strictEqual(core.measuresFor('kurtosis'), 'how sharp the filtered wave is. A single spike raises it.');
  assert.strictEqual(core.measuresFor('std'), 'how much the wave height varies in this window');
  assert.strictEqual(core.measuresFor('rms'), 'how large the filtered wave is, overall');
  Object.keys(core.FEATURE_GLOSS).forEach((k) => {
    const line = core.measuresFor(k);
    assert.ok(line && line.length <= 80, k + ' ' + line);
    assert.ok(!/oxygen|saturation|cause|spo2|sao2|scalc/i.test(line), k + ' ' + line);
  });
  assert.strictEqual(core.measuresFor('nope'), '');
  assert.strictEqual(core.measuresFor(''), '');
  assert.strictEqual(core.measuresFor('none'), '');
});

check('lead icon only for the ratio and for peakedness', () => {
  assert.strictEqual(core.leadIcon('ratio_of_ratios'), 'ratio');
  assert.strictEqual(core.leadIcon('r_beat'), 'ratio');
  assert.strictEqual(core.leadIcon('kurtosis'), 'peakedness');
  assert.strictEqual(core.leadIcon('std'), '');
  assert.strictEqual(core.leadIcon('rms'), '');
  assert.strictEqual(core.leadIcon('ir_perfusion_index'), '');
  assert.strictEqual(core.leadIcon(''), '');
});

console.log('all tests passed');
