// Shared render helpers for the dashboard (app.js) and the survey (survey.js).
// Score track, score readout, and signed driver bars. Browser only.

(function () {
const core = globalThis.DashboardCore;
if (!core) throw new Error('dashboard_core.js must load before dashboard_view.js');

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function clearChildren(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

// ---------------------------------------------------------------------------
// Shared pieces: score track (score vs 0.50) and signed driver bars.
// Above or at 0.50: stem rises to the right, dot filled. Below: stem to the
// left, dot hollow. Color repeats what position and shape already say.

function isScore(score) {
  return typeof score === 'number' && Number.isFinite(score);
}

function sideOf(score) {
  return score >= core.THRESHOLD ? 'above' : 'below';
}

function relationText(score) {
  const t = core.THRESHOLD.toFixed(2);
  return score >= core.THRESHOLD ? 'at or above ' + t : 'below ' + t;
}

function makeTrack(className) {
  const track = el('span', className);
  track.style.setProperty('--t', core.THRESHOLD * 100 + '%');
  track.appendChild(el('span', 'stem'));
  track.appendChild(el('span', 'mark'));
  return track;
}

function setTrack(track, score) {
  const stem = track.firstChild;
  const mark = track.lastChild;
  if (!isScore(score)) {
    track.dataset.side = 'none';
    stem.style.width = '0';
    mark.hidden = true;
    return;
  }
  const t = core.THRESHOLD * 100;
  const s = Math.min(1, Math.max(0, score)) * 100;
  track.dataset.side = sideOf(score);
  stem.style.left = Math.min(s, t) + '%';
  stem.style.width = Math.abs(s - t) + '%';
  mark.style.left = s + '%';
  mark.hidden = false;
}

// Score readout: number, relation to 0.50, and a 0–1 gauge with ticks.
function makeScore(root) {
  const gaugeHost = root.querySelector('[data-role="gauge"]');
  const track = makeTrack('track lg');
  const ticks = el('div', 'gauge-ticks');
  ticks.setAttribute('aria-hidden', 'true');
  [0, 0.25, core.THRESHOLD, 0.75, 1].forEach((v) => {
    const tick = el('i', v === core.THRESHOLD ? 't' : '', v === 0 || v === 1 ? String(v) : v.toFixed(2));
    tick.style.left = v * 100 + '%';
    ticks.appendChild(tick);
  });
  gaugeHost.appendChild(track);
  gaugeHost.appendChild(ticks);
  return {
    root,
    num: root.querySelector('[data-role="num"]'),
    rel: root.querySelector('[data-role="rel"]'),
    track,
  };
}

function renderScore(view, score, format) {
  view.num.textContent = (format || core.formatScore)(score);
  clearChildren(view.rel);
  if (isScore(score)) {
    const glyph = el('span', 'glyph', score >= core.THRESHOLD ? '▲' : '▼');
    glyph.setAttribute('aria-hidden', 'true');
    view.rel.appendChild(glyph);
    view.rel.appendChild(document.createTextNode(relationText(score)));
  }
  view.root.dataset.side = isScore(score) ? sideOf(score) : 'none';
  setTrack(view.track, score);
}

// Mean-perturbation deltas on one symmetric scale. ±0.30 covers every replay
// row, so bars keep their length when you switch windows.
const DRIVER_SPAN = 0.30;

function signed(delta) {
  const text = Math.abs(delta).toFixed(3);
  if (delta > 0) return '+' + text;
  if (delta < 0) return '−' + text;
  return text;
}

// opts.plain: lead with the plain gloss, say "raises score" / "lowers score",
// and add a one-line "measures" caption. The lead row also gets a static icon
// when that feature is the red/infrared ratio or waveform peakedness. The icon
// is not drawn on the infrared trace.
// opts.hideZero: drop slots that print as 0.000 (their names are filled by
// feature order, not a clue). opts.note / opts.summary: a definition line and
// a reading of the bars together, above the bars. Operator page only.
// The survey calls this without opts, so its stimulus is unchanged.
function capitalize(text) {
  return text ? text.charAt(0).toUpperCase() + text.slice(1) : text;
}

// Wavelength colors, not the score colors. A red stroke here is the red LED.
const LEAD_ICON = {
  ratio: '<svg viewBox="0 0 72 32" width="72" height="32" focusable="false">' +
    '<text x="0" y="11" fill="#d95926" font-size="8" font-family="sans-serif">R</text>' +
    '<path d="M14 8h8l3-5 3 10 3-5h16" fill="none" stroke="#d95926" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>' +
    '<text x="0" y="25" fill="#8a929c" font-size="8" font-family="sans-serif">IR</text>' +
    '<path d="M14 22h8l3-4 3 8 3-4h16" fill="none" stroke="#8a929c" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>' +
    '</svg>',
  peakedness: '<svg viewBox="0 0 72 32" width="72" height="32" focusable="false">' +
    '<path d="M2 18c5-9 9-9 14 0s9 9 14 0" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>' +
    '<path d="M40 22h8l4-16 4 16h12" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>' +
    '</svg>',
};

function leadMark(kind) {
  const mark = el('span', 'driver-icon');
  mark.dataset.icon = kind;
  mark.setAttribute('aria-hidden', 'true');
  mark.innerHTML = LEAD_ICON[kind];
  return mark;
}

function renderDrivers(container, top3, emptyText, opts) {
  const plain = !!(opts && opts.plain);
  const items = opts && opts.hideZero ? core.activeDrivers(top3) : (top3 || []);
  clearChildren(container);
  if (items.length === 0) {
    container.appendChild(el('div', 'muted', emptyText));
    return;
  }
  if (opts && opts.note) container.appendChild(el('p', 'driver-note', opts.note));
  if (opts && opts.summary) container.appendChild(el('p', 'driver-summary', opts.summary));
  const span = Math.max(DRIVER_SPAN, ...items.map((item) => Math.abs(item.delta)));

  items.forEach((item, index) => {
    const kind = item.delta > 0 ? 'risk' : item.delta < 0 ? 'normal' : 'flat';
    const measures = plain ? core.measuresFor(item.name) : '';
    const icon = plain && index === 0 ? core.leadIcon(item.name) : '';
    const row = el('div', 'driver ' + kind);
    row.setAttribute('role', 'listitem');
    row.setAttribute('aria-label', item.name + ', ' + item.gloss +
      (measures ? ', ' + measures : '') + ', ' + signed(item.delta) + ', ' + item.direction);

    const label = el('div', 'driver-label');
    if (icon) label.appendChild(leadMark(icon));
    if (plain) {
      label.appendChild(el('span', 'driver-plain', capitalize(item.gloss || item.name)));
      label.appendChild(el('code', 'driver-code', item.name));
      if (measures) label.appendChild(el('span', 'driver-measures', measures));
    } else {
      label.appendChild(el('code', 'driver-name', item.name));
      label.appendChild(el('span', 'driver-gloss', item.gloss));
    }

    const track = el('div', 'driver-track');
    const bar = el('span', 'driver-bar');
    const half = Math.min(50, (Math.abs(item.delta) / span) * 50);
    bar.style.width = half + '%';
    bar.style.left = (item.delta < 0 ? 50 - half : 50) + '%';
    track.appendChild(bar);

    const value = el('div', 'driver-value');
    value.appendChild(el('span', 'driver-delta', signed(item.delta)));
    const dirText = plain && kind !== 'flat' ? (kind === 'risk' ? 'raises score' : 'lowers score') : item.direction;
    value.appendChild(el('span', 'driver-dir', dirText));

    row.appendChild(label);
    row.appendChild(track);
    row.appendChild(value);
    container.appendChild(row);
  });

  const axis = el('div', 'driver-axis');
  axis.setAttribute('aria-hidden', 'true');
  axis.appendChild(el('span', ''));
  const scale = el('span', 'driver-scale');
  const ticks = el('span', 'driver-ticks');
  ticks.appendChild(el('i', '', '−' + span.toFixed(2)));
  ticks.appendChild(el('i', '', '0'));
  ticks.appendChild(el('i', '', '+' + span.toFixed(2)));
  const sides = el('span', 'driver-sides');
  sides.appendChild(el('i', '', plain ? '← lowers score' : '← toward normal'));
  sides.appendChild(el('i', '', plain ? 'raises score →' : 'toward risk →'));
  scale.appendChild(ticks);
  scale.appendChild(sides);
  axis.appendChild(scale);
  axis.appendChild(el('span', ''));
  container.appendChild(axis);
}

globalThis.DashboardView = {
  el, clearChildren, isScore, sideOf, relationText, makeTrack, setTrack,
  makeScore, renderScore, signed, renderDrivers, DRIVER_SPAN,
};
})();
