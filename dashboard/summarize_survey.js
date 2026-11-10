// Summarises survey responses as Markdown.
//   node dashboard/summarize_survey.js                 data/survey/responses.jsonl
//   node dashboard/summarize_survey.js --pilot         data/survey/pilot.jsonl
//   node dashboard/summarize_survey.js path/to/file.jsonl
//
// Quadrants are firmware-score quadrants (what respondents saw), not rubric ids.
// Small samples: report counts next to every percentage and do not over-read them.

const fs = require('fs');
const path = require('path');
const S = require('./survey_core.js');

function load(file) {
  if (!fs.existsSync(file)) throw new Error('no file: ' + file);
  return fs.readFileSync(file, 'utf8')
    .split('\n')
    .filter((l) => l.trim())
    .map((l, i) => {
      try { return JSON.parse(l); } catch (err) { throw new Error('line ' + (i + 1) + ' is not JSON'); }
    });
}

const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);
function sd(xs) {
  if (xs.length < 2) return null;
  const m = mean(xs);
  return Math.sqrt(xs.reduce((a, x) => a + (x - m) ** 2, 0) / (xs.length - 1));
}
function median(xs) {
  if (!xs.length) return null;
  const s = xs.slice().sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}
const f = (x, d = 2) => (x === null || x === undefined ? '—' : x.toFixed(d));
const pct = (k, n) => (n ? k + '/' + n + ' (' + Math.round((100 * k) / n) + '%)' : '—');

function counts(values, options) {
  const out = {};
  options.forEach(([v]) => { out[v] = 0; });
  values.forEach((v) => { out[v] = (out[v] || 0) + 1; });
  return out;
}

function table(header, rows) {
  const line = (cells) => '| ' + cells.join(' | ') + ' |';
  return [line(header), line(header.map(() => '---')), ...rows.map(line)].join('\n');
}

function summarize(records) {
  const out = [];
  const n = records.length;
  out.push('# TinyKAN dashboard survey summary', '');
  out.push('Responses: **' + n + '**. Survey version: ' + S.VERSION + '.');
  if (n === 0) return out.join('\n');
  const modes = [...new Set(records.map((r) => r.mode))];
  out.push('Mode: ' + modes.join(', ') + '. Durations: median ' + f(median(records.map((r) => r.duration_s).filter((x) => x !== null)) / 60, 1) + ' min.');
  out.push('');

  out.push('## Background', '');
  const role = counts(records.map((r) => r.background.role), S.BACKGROUND[0].options);
  out.push(table(['Role', 'n'], S.BACKGROUND[0].options.map(([v, label]) => [label, String(role[v])])));
  out.push('');
  out.push('Pulse-oximeter familiarity mean ' + f(mean(records.map((r) => r.background.oximeter_familiarity))) +
    ' / 5; AI/ML familiarity mean ' + f(mean(records.map((r) => r.background.ml_familiarity))) + ' / 5.', '');

  out.push('## What the score means', '');
  const meaning = counts(records.map((r) => r.meaning), S.MEANING.options);
  out.push(table(['Answer', 'n'], S.MEANING.options.map(([v, label]) => [label + (v === S.MEANING.correct ? ' ✓' : ''), pct(meaning[v], n)])));
  out.push('');

  // Flatten window answers with their scoring.
  const wins = [];
  records.forEach((r) => {
    r.windows.forEach((w, i) => wins.push({ ...w, ...r.scored.windows[i], role: r.background.role }));
  });

  out.push('## Reading each window', '');
  const groups = [['All', wins], ...S.QUADRANTS.map((q) => [q, wins.filter((w) => w.quadrant === q)]),
    ['Near 0.50 (±0.02)', wins.filter((w) => w.near_threshold)]];
  out.push(table(['Windows', 'n', 'Flag read correctly', 'Top driver correct', 'Direction correct'],
    groups.map(([name, ws]) => [name, String(ws.length),
      pct(ws.filter((w) => w.flagged_correct).length, ws.length),
      pct(ws.filter((w) => w.top_driver_correct).length, ws.length),
      pct(ws.filter((w) => w.direction_correct).length, ws.length)])));
  out.push('');

  out.push('## Ratings per window (1–5, mean)', '');
  out.push(table(['Windows', 'n', ...S.WINDOW_RATINGS.map((q) => q.id)],
    groups.slice(0, 5).map(([name, ws]) => [name, String(ws.length),
      ...S.WINDOW_RATINGS.map((q) => f(mean(ws.map((w) => w[q.id]))))])));
  out.push('');
  const trustFp = mean(wins.filter((w) => w.quadrant === 'FP').map((w) => w.trust));
  const trustTp = mean(wins.filter((w) => w.quadrant === 'TP').map((w) => w.trust));
  out.push('Trust on FP minus trust on TP: ' + f(trustFp === null || trustTp === null ? null : trustFp - trustTp) +
    '. Near zero means people trust wrong flags as much as right ones.', '');

  out.push('## Next action per window', '');
  const SHORT = { escalate: 'Escalate', check_sensor: 'Check sensor', no_action: 'No action', not_sure: 'Not sure' };
  out.push(table(['Windows', ...S.WINDOW_ACTION.options.map(([v]) => SHORT[v] || v)],
    groups.slice(0, 5).map(([name, ws]) => {
      const c = counts(ws.map((w) => w.action), S.WINDOW_ACTION.options);
      return [name, ...S.WINDOW_ACTION.options.map(([v]) => pct(c[v], ws.length))];
    })));
  out.push('');

  out.push('## Questionnaires', '');
  const sus = records.map((r) => r.scored.sus).filter((x) => x !== null);
  out.push('SUS: mean **' + f(sus.length ? mean(sus) : null, 1) + '**, SD ' + f(sd(sus), 1) + ', median ' + f(median(sus), 1) +
    ' (n = ' + sus.length + '). 68 is the commonly cited average.', '');
  out.push(table(['ESS item', 'Mean (1–5)'],
    S.ESS.map((q, i) => ['(' + q.item + ') ' + q.text, f(mean(records.map((r) => r.ess[i])))])));
  out.push('');

  out.push('## Open answers', '');
  S.OPEN.forEach((q) => {
    const answers = records.map((r) => r.open[q.id]).filter((t) => t);
    out.push('**' + q.text + '** (' + answers.length + ')', '');
    answers.forEach((t) => out.push('- ' + t.replace(/\s+/g, ' ')));
    out.push('');
  });
  return out.join('\n');
}

if (require.main === module) {
  const args = process.argv.slice(2);
  const dataDir = path.join(__dirname, '..', 'data', 'survey');
  const file = args.find((a) => !a.startsWith('--')) ||
    path.join(dataDir, args.includes('--pilot') ? 'pilot.jsonl' : 'responses.jsonl');
  try {
    console.log(summarize(load(file)));
  } catch (err) {
    console.error(err.message);
    process.exit(1);
  }
}

module.exports = { summarize, load };
