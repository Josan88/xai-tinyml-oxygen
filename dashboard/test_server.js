// Starts the server on a free port with a temp data dir and exercises it.
//   node dashboard/test_server.js
const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { createServer, config, SUPERVISOR_ETHICS_NOTE } = require('./server.js');
const { summarize, load } = require('./summarize_survey.js');
const S = require('./survey_core.js');
const REPLAY_WINDOWS = require('./replay_windows.js');

function validBody() {
  const ids = S.assignWindows(REPLAY_WINDOWS);
  return {
    version: S.VERSION,
    consent: { read: true, adult: true, agree: true },
    background: { role: 'health', oximeter_familiarity: 5, ml_familiarity: 1 },
    meaning: 'spo2',
    windows: ids.map((id) => {
      const top = S.driversOf(REPLAY_WINDOWS.find((r) => r.window_id === id))[0];
      return {
        window_id: id, flagged: 'not_sure', top_driver: top.name, direction: 'not_sure',
        compact: 5, matches: 4, helps: 2, trust: 1, action: 'escalate', seconds: 30,
      };
    }),
    sus: [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    ess: [2, 2, 2, 2],
    open: { confusing: 'score vs oxygen', change: '', aide: '' },
    duration_s: 420,
    website: '',
  };
}

async function main() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'survey-test-'));
  const cfg = { ...config({}), port: 0, dataDir: dir };
  const server = createServer(cfg);
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const base = 'http://127.0.0.1:' + server.address().port;
  const post = (body, headers = { 'Content-Type': 'application/json' }) =>
    fetch(base + '/api/survey', { method: 'POST', headers, body: typeof body === 'string' ? body : JSON.stringify(body) });

  try {
    // Live mode needs exactly one ethics basis: a committee reference or the supervisor's advice.
    assert.throws(() => config({ SURVEY_MODE: 'live' }), /SURVEY_ETHICS_REF/);
    assert.strictEqual(config({ SURVEY_MODE: 'live', SURVEY_ETHICS_REF: 'X1' }).mode, 'live');
    assert.strictEqual(config({ SURVEY_MODE: 'live', SURVEY_ETHICS_REF: 'X1' }).ethicsBasis, 'reference');
    assert.strictEqual(config({ SURVEY_MODE: 'live', SURVEY_ETHICS_REF: 'X1' }).ethicsNote, '');
    assert.throws(() => config({ SURVEY_MODE: 'prod' }), /pilot or live/);
    const sup = config({ SURVEY_MODE: 'live', SURVEY_ETHICS_BASIS: 'supervisor' });
    assert.strictEqual(sup.mode, 'live');
    assert.strictEqual(sup.ethicsRef, '');
    assert.strictEqual(sup.ethicsBasis, 'supervisor');
    assert.strictEqual(sup.ethicsNote, SUPERVISOR_ETHICS_NOTE);
    assert.ok(/18 September 2026/.test(SUPERVISOR_ETHICS_NOTE) && /not been\s+reviewed by an ethics committee/.test(SUPERVISOR_ETHICS_NOTE));
    assert.throws(() => config({ SURVEY_MODE: 'live', SURVEY_ETHICS_BASIS: 'self' }), /must be supervisor/);
    assert.throws(() => config({ SURVEY_MODE: 'live', SURVEY_ETHICS_BASIS: 'supervisor', SURVEY_ETHICS_REF: 'X1' }), /not both/);
    assert.strictEqual(config({}).ethicsBasis, '');
    console.log('ok  live mode requires an ethics reference or the supervisor basis, not both');

    for (const p of ['/', '/survey', '/survey.js', '/survey_core.js', '/dashboard_view.js', '/styles.css']) {
      const res = await fetch(base + p);
      assert.strictEqual(res.status, 200, p);
      assert.ok(res.headers.get('content-security-policy').includes("script-src 'self'"));
      assert.strictEqual(res.headers.get('x-content-type-options'), 'nosniff');
    }
    console.log('ok  serves dashboard and survey files with security headers');

    for (const p of ['/server.js', '/test_server.js', '/build_replay.js', '/../data/survey/pilot.jsonl', '/%2e%2e/README.md', '/README.md']) {
      const res = await fetch(base + p);
      assert.strictEqual(res.status, 404, p);
    }
    console.log('ok  does not serve server, tests, data, or paths outside the list');

    const cfgRes = await (await fetch(base + '/api/survey/config')).json();
    assert.deepStrictEqual(cfgRes, { version: S.VERSION, mode: 'pilot', ethics_ref: '', ethics_note: '' });
    console.log('ok  config reports pilot mode');

    let res = await post(validBody());
    assert.strictEqual(res.status, 201);
    const saved = await res.json();
    assert.ok(saved.ok && /^[0-9a-f-]{36}$/.test(saved.response_id));
    const lines = fs.readFileSync(path.join(dir, 'pilot.jsonl'), 'utf8').trim().split('\n');
    assert.strictEqual(lines.length, 1);
    const rec = JSON.parse(lines[0]);
    assert.strictEqual(rec.mode, 'pilot');
    assert.strictEqual(rec.ethics_basis, null);
    assert.strictEqual(rec.response_id, saved.response_id);
    assert.strictEqual(rec.scored.sus, 50);
    assert.strictEqual(rec.scored.meaning_correct, false);
    assert.strictEqual(rec.website, undefined);
    const text = JSON.stringify(rec);
    assert.ok(!/127\.0\.0\.1|::1|user-agent|node/i.test(text), 'no IP or user agent stored');
    assert.ok(!fs.existsSync(path.join(dir, 'responses.jsonl')), 'pilot never writes the live file');
    console.log('ok  valid pilot response saved to pilot.jsonl without IP or user agent');

    const bad = validBody();
    bad.consent.agree = false;
    res = await post(bad);
    assert.strictEqual(res.status, 400);
    assert.ok((await res.json()).errors.length > 0);
    res = await post('{not json');
    assert.strictEqual(res.status, 400);
    res = await post(validBody(), { 'Content-Type': 'text/plain' });
    assert.strictEqual(res.status, 415);
    res = await post(JSON.stringify({ pad: 'x'.repeat(40 * 1024) }));
    assert.strictEqual(res.status, 413);
    res = await fetch(base + '/api/survey');
    assert.strictEqual(res.status, 405);
    console.log('ok  rejects invalid, non-JSON, oversized, and GET submissions');

    const bot = validBody();
    bot.website = 'http://spam.example';
    res = await post(bot);
    assert.strictEqual(res.status, 201);
    assert.strictEqual(fs.readFileSync(path.join(dir, 'pilot.jsonl'), 'utf8').trim().split('\n').length, 1);
    console.log('ok  honeypot submissions look accepted but are not stored');

    // 6 POSTs so far from this client; limit is 20 per hour.
    const statuses = [];
    for (let i = 0; i < 16; i += 1) {
      statuses.push((await post(validBody(), { 'Content-Type': 'application/json', 'CF-Connecting-IP': '203.0.113.9' })).status);
    }
    assert.ok(statuses.slice(0, 16).every((s) => s === 201));
    for (let i = 0; i < 4; i += 1) {
      statuses.push((await post(validBody(), { 'Content-Type': 'application/json', 'CF-Connecting-IP': '203.0.113.9' })).status);
    }
    statuses.push((await post(validBody(), { 'Content-Type': 'application/json', 'CF-Connecting-IP': '203.0.113.9' })).status);
    assert.strictEqual(statuses[statuses.length - 1], 429);
    const other = await post(validBody(), { 'Content-Type': 'application/json', 'CF-Connecting-IP': '198.51.100.4' });
    assert.strictEqual(other.status, 201);
    console.log('ok  rate limit is per visitor (CF-Connecting-IP)');

    const all = fs.readFileSync(path.join(dir, 'pilot.jsonl'), 'utf8').trim().split('\n').map((l) => JSON.parse(l));
    assert.strictEqual(new Set(all.map((r) => r.response_id)).size, all.length);
    console.log('ok  ' + all.length + ' lines, all valid JSON with unique ids');

    const md = summarize(load(path.join(dir, 'pilot.jsonl')));
    assert.ok(md.includes('Responses: **' + all.length + '**'));
    assert.ok(md.includes('SUS: mean **50.0**'));
    assert.ok(md.includes('| Healthcare worker or healthcare student | ' + all.length + ' |'));
    assert.ok(summarize([]).includes('Responses: **0**'));
    console.log('ok  summary reads the saved file');

    // Live on the supervisor basis: the note reaches the page and each record says which basis it ran under.
    const liveDir = path.join(dir, 'live');
    const live = createServer({ ...config({ SURVEY_MODE: 'live', SURVEY_ETHICS_BASIS: 'supervisor' }), dataDir: liveDir });
    await new Promise((resolve) => live.listen(0, '127.0.0.1', resolve));
    try {
      const liveBase = 'http://127.0.0.1:' + live.address().port;
      const liveCfg = await (await fetch(liveBase + '/api/survey/config')).json();
      assert.deepStrictEqual(liveCfg, { version: S.VERSION, mode: 'live', ethics_ref: '', ethics_note: SUPERVISOR_ETHICS_NOTE });
      const r = await fetch(liveBase + '/api/survey', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(validBody()),
      });
      assert.strictEqual(r.status, 201);
      const liveRec = JSON.parse(fs.readFileSync(path.join(liveDir, 'responses.jsonl'), 'utf8').trim());
      assert.strictEqual(liveRec.mode, 'live');
      assert.strictEqual(liveRec.ethics_basis, 'supervisor');
      assert.ok(!fs.existsSync(path.join(liveDir, 'pilot.jsonl')), 'live never writes the pilot file');
      console.log('ok  live supervisor basis: note served, record tagged, responses.jsonl only');
    } finally {
      live.close();
    }
  } finally {
    server.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
  console.log('all server tests passed');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
