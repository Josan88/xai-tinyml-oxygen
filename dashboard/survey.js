// Survey page. Builds steps from survey_core.js, renders replay windows with the
// same score and driver components as the dashboard, and posts one JSON response.

(function () {
const core = globalThis.DashboardCore;
const view = globalThis.DashboardView;
const S = globalThis.SurveyCore;
const ROWS = globalThis.REPLAY_WINDOWS || [];
if (!core || !view || !S) throw new Error('survey.js needs dashboard_core, dashboard_view, and survey_core');

const { el } = view;
const byId = (id) => document.getElementById(id);
const rowById = new Map(ROWS.map((r) => [r.window_id, r]));
const DONE_KEY = 'tinykan-survey-submitted';

const generated = byId('generated');
const stepText = byId('stepText');
const formError = byId('formError');
const backBtn = byId('backBtn');
const nextBtn = byId('nextBtn');
const stepNav = byId('stepNav');

const assigned = S.assignWindows(ROWS);
const startedAt = Date.now();
let current = 0;
let enteredAt = Date.now();
let submitting = false;
let submitted = false;
let serverOk = false;

// ---------------------------------------------------------------------------
// Question builders. Every fieldset carries data-name and data-kind so the
// reader and the required check do not depend on layout.

function legend(fieldset, number, text) {
  const lg = el('legend');
  if (number) lg.appendChild(el('span', 'num', number + '.'));
  lg.appendChild(document.createTextNode(text));
  fieldset.appendChild(lg);
}

function choiceQ(name, number, text, options) {
  const fs = el('fieldset', 'q');
  fs.dataset.name = name;
  fs.dataset.kind = 'choice';
  fs.dataset.required = '';
  legend(fs, number, text);
  const list = el('div', 'choices');
  options.forEach(([value, label]) => {
    const lab = el('label', 'choice');
    const input = el('input');
    input.type = 'radio';
    input.name = name;
    input.value = value;
    lab.appendChild(input);
    lab.appendChild(el('span', '', label));
    list.appendChild(lab);
  });
  fs.appendChild(list);
  return fs;
}

function likertQ(name, number, text, ends) {
  const fs = el('fieldset', 'q');
  fs.dataset.name = name;
  fs.dataset.kind = 'likert';
  fs.dataset.required = '';
  legend(fs, number, text);
  const wrap = el('div', 'likert');
  const row = el('div', 'likert-row');
  S.LIKERT.forEach((v) => {
    const lab = el('label');
    const input = el('input');
    input.type = 'radio';
    input.name = name;
    input.value = String(v);
    let aria = v + ' of 5';
    if (v === 1) aria += ', ' + ends[0];
    if (v === 5) aria += ', ' + ends[1];
    input.setAttribute('aria-label', aria);
    lab.appendChild(input);
    lab.appendChild(el('span', '', String(v)));
    row.appendChild(lab);
  });
  const endRow = el('div', 'likert-ends');
  endRow.setAttribute('aria-hidden', 'true');
  endRow.appendChild(el('span', '', ends[0]));
  endRow.appendChild(el('span', '', ends[1]));
  wrap.appendChild(row);
  wrap.appendChild(endRow);
  fs.appendChild(wrap);
  return fs;
}

function textQ(name, number, text) {
  const fs = el('fieldset', 'q');
  fs.dataset.name = name;
  fs.dataset.kind = 'text';
  legend(fs, number, text);
  fs.querySelector('legend').appendChild(el('span', 'optional', '(optional)'));
  const area = el('textarea');
  area.name = name;
  area.maxLength = S.TEXT_MAX;
  area.rows = 3;
  area.setAttribute('aria-label', text);
  fs.appendChild(area);
  return fs;
}

function stepSection(name, title, lead) {
  const sec = el('section', 'step panel');
  sec.dataset.step = name;
  sec.hidden = true;
  const h = el('h2', 'step-title', title);
  h.id = 'title-' + name;
  sec.setAttribute('aria-labelledby', h.id);
  sec.appendChild(h);
  if (lead) sec.appendChild(el('p', 'step-lead', lead));
  return sec;
}

// ---------------------------------------------------------------------------
// Steps.

function buildBackground() {
  const sec = stepSection('background', 'About you', 'No names or contact details. These three answers help us group results.');
  S.BACKGROUND.forEach((q, i) => {
    sec.appendChild(q.options
      ? choiceQ('bg_' + q.id, i + 1, q.text, q.options)
      : likertQ('bg_' + q.id, i + 1, q.text, q.scale));
  });
  return sec;
}

function windowCard(row, position) {
  const state = core.replayState(row, false);
  const card = el('div', 'window-card');

  const head = el('div', 'panel-head');
  head.appendChild(el('h3', 'label', 'Window ' + position + ' of ' + assigned.length));
  head.appendChild(el('span', 'tag', 'Recorded · not live'));
  card.appendChild(head);

  const score = el('div', 'score');
  score.appendChild(el('div', 'label', 'Score'));
  const scoreRow = el('div', 'score-row');
  const num = el('span', 'score-num');
  num.dataset.role = 'num';
  const rel = el('span', 'score-rel');
  rel.dataset.role = 'rel';
  scoreRow.appendChild(num);
  scoreRow.appendChild(rel);
  score.appendChild(scoreRow);
  const gauge = el('div', 'gauge');
  gauge.dataset.role = 'gauge';
  score.appendChild(gauge);
  card.appendChild(score);
  // Three decimals, as in replay: 0.497 and 0.501 sit on opposite sides of 0.50.
  view.renderScore(view.makeScore(score), state.score, (s) => (view.isScore(s) ? s.toFixed(3) : '—'));

  const block = el('div', 'drivers-block');
  block.appendChild(el('div', 'label', 'Top drivers'));
  const drivers = el('div', 'drivers');
  drivers.setAttribute('role', 'list');
  block.appendChild(drivers);
  card.appendChild(block);
  view.renderDrivers(drivers, state.top3, 'No non-zero drivers in this window.');

  const summary = el('div', 'summary');
  summary.appendChild(el('div', 'label', 'Summary'));
  summary.appendChild(el('p', '', core.buildExplanation(state)));
  card.appendChild(summary);
  return card;
}

function buildWindow(k) {
  const id = assigned[k];
  const row = rowById.get(id);
  const sec = stepSection('w' + k, 'Window ' + (k + 1) + ' of ' + assigned.length,
    k === 0
      ? 'This is one recorded 10-second window as the dashboard shows it. Read it, then answer below.'
      : 'Another recorded window. Answer the same questions for this one.');
  sec.dataset.windowId = id;
  sec.appendChild(windowCard(row, k + 1));

  let n = 1;
  const p = 'w' + k + '_';
  if (k === 0) sec.appendChild(choiceQ('meaning', n++, S.MEANING.text, S.MEANING.options));
  const [flagged, driver, direction] = S.WINDOW_CHOICE;
  sec.appendChild(choiceQ(p + 'flagged', n++, flagged.text, flagged.options));
  sec.appendChild(choiceQ(p + 'top_driver', n++, driver.text, S.driverOptions(row)));
  sec.appendChild(choiceQ(p + 'direction', n++, direction.text, direction.options));
  S.WINDOW_RATINGS.forEach((q) => sec.appendChild(likertQ(p + q.id, n++, q.text, S.AGREE)));
  sec.appendChild(choiceQ(p + 'action', n++, S.WINDOW_ACTION.text, S.WINDOW_ACTION.options));
  return sec;
}

function buildSus() {
  const sec = stepSection('sus', 'The dashboard overall',
    'Think about the screens you just used. Rate each statement from 1 (strongly disagree) to 5 (strongly agree).');
  S.SUS.forEach((text, i) => sec.appendChild(likertQ('sus_' + i, i + 1, text, S.AGREE)));
  return sec;
}

function buildEss() {
  const sec = stepSection('ess', 'The explanations',
    'Think about the explanations: the Summary line and the Top drivers bars.');
  S.ESS.forEach((q, i) => sec.appendChild(likertQ('ess_' + i, i + 1, q.text, S.AGREE)));
  return sec;
}

function buildOpen() {
  const sec = stepSection('open', 'Last few questions', 'All optional. Short answers are fine.');
  S.OPEN.forEach((q, i) => sec.appendChild(textQ('open_' + q.id, i + 1, q.text)));
  // Honeypot. People never see it; a filled value means a bot.
  const hp = el('div', 'hp');
  hp.setAttribute('aria-hidden', 'true');
  const lab = el('label', '', 'Website');
  const input = el('input');
  input.type = 'text';
  input.name = 'website';
  input.tabIndex = -1;
  input.autocomplete = 'off';
  lab.appendChild(input);
  hp.appendChild(lab);
  sec.appendChild(hp);
  return sec;
}

const steps = [document.querySelector('[data-step="info"]')];
steps.push(buildBackground());
assigned.forEach((_, k) => steps.push(buildWindow(k)));
steps.push(buildSus(), buildEss(), buildOpen());
steps.slice(1).forEach((s) => generated.appendChild(s));
const doneStep = document.querySelector('[data-step="done"]');
const seconds = steps.map(() => 0);

// ---------------------------------------------------------------------------
// Reading answers.

function picked(name) {
  const input = document.querySelector('input[name="' + name + '"]:checked');
  return input ? input.value : null;
}

function pickedInt(name) {
  const v = picked(name);
  return v === null ? null : Number(v);
}

function textOf(name) {
  const node = document.querySelector('[name="' + name + '"]');
  return node ? node.value : '';
}

function checked(name) {
  const node = document.querySelector('input[name="' + name + '"]');
  return !!(node && node.checked);
}

function collect() {
  const background = {};
  S.BACKGROUND.forEach((q) => {
    background[q.id] = q.options ? picked('bg_' + q.id) : pickedInt('bg_' + q.id);
  });
  const windows = assigned.map((id, k) => {
    const p = 'w' + k + '_';
    const w = {
      window_id: id,
      flagged: picked(p + 'flagged'),
      top_driver: picked(p + 'top_driver'),
      direction: picked(p + 'direction'),
      action: picked(p + 'action'),
      seconds: seconds[steps.findIndex((s) => s.dataset.step === 'w' + k)],
    };
    S.WINDOW_RATINGS.forEach((q) => { w[q.id] = pickedInt(p + q.id); });
    return w;
  });
  const open = {};
  S.OPEN.forEach((q) => { open[q.id] = textOf('open_' + q.id); });
  return {
    version: S.VERSION,
    consent: { read: checked('consent_read'), adult: checked('consent_adult'), agree: checked('consent_agree') },
    background,
    meaning: picked('meaning'),
    windows,
    sus: S.SUS.map((_, i) => pickedInt('sus_' + i)),
    ess: S.ESS.map((_, i) => pickedInt('ess_' + i)),
    open,
    duration_s: (Date.now() - startedAt) / 1000,
    website: textOf('website'),
  };
}

// ---------------------------------------------------------------------------
// Required check for the current step.

function answered(fs) {
  if (fs.classList.contains('consent')) {
    return Array.from(fs.querySelectorAll('input[type="checkbox"]')).every((c) => c.checked);
  }
  return !!fs.querySelector('input:checked');
}

function checkStep(sec) {
  const missing = Array.from(sec.querySelectorAll('fieldset[data-required]')).filter((fs) => !answered(fs));
  sec.querySelectorAll('fieldset.missing').forEach((fs) => fs.classList.remove('missing'));
  missing.forEach((fs) => fs.classList.add('missing'));
  if (missing.length === 0) {
    formError.hidden = true;
    return true;
  }
  formError.textContent = sec.dataset.step === 'info'
    ? 'Tick all three consent boxes to start.'
    : 'Answer the highlighted question' + (missing.length > 1 ? 's' : '') + ' (' + missing.length + ') to continue.';
  formError.hidden = false;
  const first = missing[0].querySelector('input');
  if (first) first.focus();
  missing[0].scrollIntoView({ block: 'center' });
  return false;
}

// Clear the highlight as soon as a question is answered.
document.addEventListener('change', (event) => {
  const fs = event.target.closest && event.target.closest('fieldset.missing');
  if (fs && answered(fs)) fs.classList.remove('missing');
});

// ---------------------------------------------------------------------------
// Navigation.

function show(index) {
  seconds[current] += (Date.now() - enteredAt) / 1000;
  enteredAt = Date.now();
  current = index;
  steps.forEach((s, i) => { s.hidden = i !== index; });
  doneStep.hidden = true;
  formError.hidden = true;
  backBtn.hidden = index === 0;
  nextBtn.textContent = index === 0 ? 'Start' : index === steps.length - 1 ? 'Submit' : 'Next';
  nextBtn.disabled = !serverOk;
  stepText.textContent = 'Step ' + (index + 1) + ' of ' + steps.length;
  window.scrollTo(0, 0);
  const title = steps[index].querySelector('.step-title');
  title.tabIndex = -1;
  title.focus({ preventScroll: true });
}

async function submit() {
  if (submitting) return;
  submitting = true;
  nextBtn.disabled = true;
  nextBtn.textContent = 'Saving…';
  seconds[current] += (Date.now() - enteredAt) / 1000;
  enteredAt = Date.now();
  try {
    const res = await fetch('/api/survey', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(collect()),
    });
    const data = await res.json().catch(() => ({}));
    if (res.status !== 201 || !data.ok) {
      const detail = data.errors ? data.errors.join('; ') : data.error || 'status ' + res.status;
      throw new Error(detail);
    }
    submitted = true;
    try { localStorage.setItem(DONE_KEY, new Date().toISOString()); } catch (err) { /* private mode */ }
    steps.forEach((s) => { s.hidden = true; });
    doneStep.hidden = false;
    stepNav.hidden = true;
    stepText.textContent = 'Done';
    formError.hidden = true;
    window.scrollTo(0, 0);
  } catch (err) {
    formError.textContent = 'Could not save your answers (' + err.message + '). Please press Submit again.';
    formError.hidden = false;
    nextBtn.disabled = false;
    nextBtn.textContent = 'Submit';
  } finally {
    submitting = false;
  }
}

nextBtn.addEventListener('click', () => {
  if (!checkStep(steps[current])) return;
  if (current === steps.length - 1) submit();
  else show(current + 1);
});
backBtn.addEventListener('click', () => { if (current > 0) show(current - 1); });

window.addEventListener('beforeunload', (event) => {
  if (current > 0 && !submitted) {
    event.preventDefault();
    event.returnValue = '';
  }
});

// ---------------------------------------------------------------------------
// Server config: pilot or live, and the ethics reference.

try {
  if (localStorage.getItem(DONE_KEY)) byId('doneBefore').hidden = false;
} catch (err) { /* private mode */ }

fetch('/api/survey/config', { cache: 'no-store' })
  .then((res) => (res.ok ? res.json() : Promise.reject(new Error('status ' + res.status))))
  .then((cfg) => {
    if (cfg.version !== S.VERSION) throw new Error('version mismatch');
    serverOk = true;
    nextBtn.disabled = false;
    if (cfg.mode !== 'live') {
      byId('pilotBanner').hidden = false;
      byId('pilotTag').hidden = false;
    }
    const ethicsText = cfg.ethics_note || (cfg.ethics_ref ? 'Ethics reference: ' + cfg.ethics_ref : '');
    if (ethicsText) {
      const line = byId('ethicsLine');
      line.textContent = ethicsText;
      line.hidden = false;
    }
  })
  .catch(() => {
    byId('serverBanner').hidden = false;
    nextBtn.disabled = true;
  });

show(0);
})();
