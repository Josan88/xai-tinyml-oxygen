// Operator-page captions and lead icons. The survey calls renderDrivers with
// no opts, so that path must not grow a measures line or an icon.
const assert = require('assert');

function el(tag) {
  return {
    tag,
    className: '',
    textContent: '',
    innerHTML: '',
    style: {},
    dataset: {},
    attrs: {},
    children: [],
    setAttribute(name, value) { this.attrs[name] = value; },
    appendChild(child) { this.children.push(child); return child; },
    removeChild(child) {
      const i = this.children.indexOf(child);
      if (i >= 0) this.children.splice(i, 1);
      return child;
    },
    get firstChild() { return this.children[0] || null; },
  };
}

global.document = { createElement: el };
require('./dashboard_core.js');
require('./dashboard_view.js');
const view = global.DashboardView;

function row(name, delta) {
  return { name, delta, gloss: global.DashboardCore.glossFor(name), direction: global.DashboardCore.directionFor(delta) };
}

function classes(node) {
  return (node.className || '').split(/\s+/).filter(Boolean);
}

function walk(node, hit) {
  if (!node || typeof node !== 'object') return;
  if (classes(node).some(hit)) return node;
  for (const child of node.children || []) {
    const found = walk(child, hit);
    if (found) return found;
  }
  return null;
}

function all(node, className) {
  const out = [];
  (function visit(n) {
    if (!n || typeof n !== 'object') return;
    if (classes(n).indexOf(className) !== -1) out.push(n);
    (n.children || []).forEach(visit);
  })(node);
  return out;
}

function check(name, fn) {
  fn();
  console.log('ok  ' + name);
}

const top3 = [
  row('ratio_of_ratios', -0.814),
  row('std', -0.019),
  row('kurtosis', -0.004),
];

check('operator bars carry a measures line, and only the lead gets an icon', () => {
  const box = el('div');
  view.renderDrivers(box, top3, 'empty', { plain: true, hideZero: true });
  const measures = all(box, 'driver-measures').map((n) => n.textContent);
  assert.deepStrictEqual(measures, [
    'red wave size divided by infrared wave size',
    'how much the wave height varies in this window',
    'how sharp the filtered wave is. A single spike raises it.',
  ]);
  const icons = all(box, 'driver-icon');
  assert.strictEqual(icons.length, 1);
  assert.strictEqual(icons[0].dataset.icon, 'ratio');
  assert.strictEqual(icons[0].attrs['aria-hidden'], 'true');
  assert.ok(icons[0].innerHTML.indexOf('<svg') !== -1);
  // The second row is kurtosis and still gets no icon. Lead only.
  const rows = all(box, 'driver');
  assert.strictEqual(all(rows[1], 'driver-icon').length, 0);
  assert.strictEqual(all(rows[2], 'driver-icon').length, 0);
});

check('a peakedness lead gets the peakedness icon, not the ratio icon', () => {
  const box = el('div');
  view.renderDrivers(box, [row('kurtosis', 0.08), row('std', 0.01)], 'empty', { plain: true });
  const icons = all(box, 'driver-icon');
  assert.strictEqual(icons.length, 1);
  assert.strictEqual(icons[0].dataset.icon, 'peakedness');
});

check('a non-shape lead gets a measures line and no icon', () => {
  const box = el('div');
  view.renderDrivers(box, [row('std', -0.02), row('rms', -0.01)], 'empty', { plain: true });
  assert.strictEqual(all(box, 'driver-measures').length, 2);
  assert.strictEqual(all(box, 'driver-icon').length, 0);
});

check('survey render has no measures line and no icon', () => {
  const box = el('div');
  view.renderDrivers(box, top3, 'No non-zero drivers in this window.');
  assert.strictEqual(all(box, 'driver-measures').length, 0);
  assert.strictEqual(all(box, 'driver-icon').length, 0);
  assert.ok(walk(box, (c) => c === 'driver-gloss'));
});

console.log('all tests passed');
