// Static server for the dashboard plus the survey endpoint. Node only, no deps.
//
//   node dashboard/server.js                      pilot mode, 127.0.0.1:8000
//   SURVEY_MODE=live SURVEY_ETHICS_REF="..." node dashboard/server.js
//   SURVEY_MODE=live SURVEY_ETHICS_BASIS=supervisor node dashboard/server.js
//
// Env:
//   PORT                 default 8000
//   HOST                 default 127.0.0.1 (cloudflared on the same machine)
//   SURVEY_MODE          pilot (default) or live
//   SURVEY_ETHICS_REF    approval or exemption reference; shown on the information sheet
//   SURVEY_ETHICS_BASIS  "supervisor": no reference; the sheet shows SUPERVISOR_ETHICS_NOTE
//                        Live mode needs exactly one of SURVEY_ETHICS_REF / SURVEY_ETHICS_BASIS.
//   SURVEY_DATA_DIR      default <repo>/data/survey (git-ignored)
//
// Pilot responses go to pilot.jsonl and live ones to responses.jsonl, so test
// runs never mix with real data. No IP address, user agent, or cookie is stored.

const http = require('http');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const survey = require('./survey_core.js');
const REPLAY_WINDOWS = require('./replay_windows.js');

const ROOT = __dirname;
const MAX_BODY = 32 * 1024;
const RATE_LIMIT = 20;              // POSTs per client per hour
const RATE_WINDOW_MS = 60 * 60 * 1000;

// Only these files are served. Server, tests, and build scripts are not.
const STATIC = {
  '/': 'index.html',
  '/index.html': 'index.html',
  '/survey': 'survey.html',
  '/survey/': 'survey.html',
  '/survey.html': 'survey.html',
  '/styles.css': 'styles.css',
  '/survey.css': 'survey.css',
  '/app.js': 'app.js',
  '/dashboard_core.js': 'dashboard_core.js',
  '/dashboard_view.js': 'dashboard_view.js',
  '/replay_windows.js': 'replay_windows.js',
  '/survey_core.js': 'survey_core.js',
  '/survey.js': 'survey.js',
};

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
};

const SECURITY_HEADERS = {
  'Content-Security-Policy':
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; " +
    "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
  'X-Content-Type-Options': 'nosniff',
  'Referrer-Policy': 'no-referrer',
  'X-Frame-Options': 'DENY',
};

// Basis used when there is no committee reference: the supervisor's advice in the
// Week 3 meeting (Teams, 18 Sep 2026; vault "Supervisor meeting - Week 3"). It says
// plainly that no committee reviewed the survey, so it cannot be read as an approval.
const SUPERVISOR_ETHICS_NOTE =
  'Ethics: the supervisor, Dr Sheena Punai Philimon, advised on 18 September 2026 that ' +
  'this anonymous survey of fellow students does not need ethics approval. It has not been ' +
  'reviewed by an ethics committee, so there is no approval number.';

function config(env = process.env) {
  const mode = env.SURVEY_MODE === 'live' ? 'live' : 'pilot';
  const ethicsRef = (env.SURVEY_ETHICS_REF || '').trim();
  const basis = (env.SURVEY_ETHICS_BASIS || '').trim();
  if (env.SURVEY_MODE && !['live', 'pilot'].includes(env.SURVEY_MODE)) {
    throw new Error('SURVEY_MODE must be pilot or live');
  }
  if (basis && basis !== 'supervisor') {
    throw new Error('SURVEY_ETHICS_BASIS must be supervisor');
  }
  if (basis && ethicsRef) {
    throw new Error('set SURVEY_ETHICS_REF or SURVEY_ETHICS_BASIS, not both');
  }
  if (mode === 'live' && !ethicsRef && !basis) {
    throw new Error('SURVEY_MODE=live needs SURVEY_ETHICS_REF (ethics approval or exemption reference) ' +
      'or SURVEY_ETHICS_BASIS=supervisor');
  }
  return {
    port: Number(env.PORT || 8000),
    host: env.HOST || '127.0.0.1',
    mode,
    ethicsRef,
    ethicsBasis: ethicsRef ? 'reference' : basis,
    ethicsNote: basis === 'supervisor' ? SUPERVISOR_ETHICS_NOTE : '',
    dataDir: env.SURVEY_DATA_DIR || path.join(ROOT, '..', 'data', 'survey'),
  };
}

function send(res, status, body, headers = {}) {
  res.writeHead(status, { ...SECURITY_HEADERS, ...headers });
  res.end(body);
}

function sendJson(res, status, obj) {
  send(res, status, JSON.stringify(obj), {
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store',
  });
}

function clientKey(req) {
  // Behind Cloudflare the socket is cloudflared; the visitor is in this header.
  // Used for the in-memory rate limit only, never written to disk.
  return String(req.headers['cf-connecting-ip'] || req.socket.remoteAddress || '');
}

function createServer(cfg) {
  const hits = new Map();
  let writing = Promise.resolve();
  const file = path.join(cfg.dataDir, cfg.mode === 'live' ? 'responses.jsonl' : 'pilot.jsonl');

  function limited(key) {
    const now = Date.now();
    const recent = (hits.get(key) || []).filter((t) => now - t < RATE_WINDOW_MS);
    recent.push(now);
    hits.set(key, recent);
    return recent.length > RATE_LIMIT;
  }

  function append(record) {
    const line = JSON.stringify(record) + '\n';
    // Serialise writes. A failed write must not poison the chain for the next one.
    const job = writing.catch(() => {}).then(async () => {
      await fs.promises.mkdir(cfg.dataDir, { recursive: true });
      await fs.promises.appendFile(file, line, 'utf8');
    });
    writing = job;
    return job;
  }

  function readBody(req) {
    return new Promise((resolve, reject) => {
      let size = 0;
      const chunks = [];
      // Past the cap, keep draining but drop the bytes so the 413 still reaches the client.
      req.on('data', (chunk) => {
        size += chunk.length;
        if (size <= MAX_BODY) chunks.push(chunk);
      });
      req.on('end', () => {
        if (size > MAX_BODY) reject(Object.assign(new Error('too large'), { status: 413 }));
        else resolve(Buffer.concat(chunks).toString('utf8'));
      });
      req.on('error', reject);
    });
  }

  async function handleSubmit(req, res) {
    if (!/^application\/json\b/i.test(req.headers['content-type'] || '')) {
      return sendJson(res, 415, { ok: false, error: 'send application/json' });
    }
    if (Number(req.headers['content-length'] || 0) > MAX_BODY) {
      res.setHeader('Connection', 'close');
      return sendJson(res, 413, { ok: false, error: 'too large' });
    }
    if (limited(clientKey(req))) {
      return sendJson(res, 429, { ok: false, error: 'too many submissions, try again later' });
    }
    let body;
    try {
      body = JSON.parse(await readBody(req));
    } catch (err) {
      return sendJson(res, err.status || 400, { ok: false, error: err.status ? 'too large' : 'invalid JSON' });
    }
    // Honeypot: a hidden field people never fill. Pretend success, store nothing.
    if (body && typeof body.website === 'string' && body.website.trim() !== '') {
      return sendJson(res, 201, { ok: true });
    }
    const result = survey.validateResponse(body, REPLAY_WINDOWS);
    if (!result.ok) return sendJson(res, 400, { ok: false, errors: result.errors });

    const record = {
      response_id: crypto.randomUUID(),
      received_at: new Date().toISOString(),
      mode: cfg.mode,
      ethics_basis: cfg.ethicsBasis || null,
      ...result.clean,
      scored: survey.scoreResponse(result.clean, REPLAY_WINDOWS),
    };
    try {
      await append(record);
    } catch (err) {
      console.error('write failed:', err.message);
      return sendJson(res, 500, { ok: false, error: 'could not save, please try again' });
    }
    return sendJson(res, 201, { ok: true, response_id: record.response_id });
  }

  function handleStatic(req, res, pathname) {
    const name = STATIC[pathname];
    if (!name) return send(res, 404, 'Not found', { 'Content-Type': 'text/plain; charset=utf-8' });
    fs.readFile(path.join(ROOT, name), (err, data) => {
      if (err) return send(res, 500, 'Server error', { 'Content-Type': 'text/plain; charset=utf-8' });
      const headers = {
        'Content-Type': TYPES[path.extname(name)] || 'application/octet-stream',
        'Cache-Control': 'no-cache',
      };
      if (req.method === 'HEAD') return send(res, 200, undefined, headers);
      return send(res, 200, data, headers);
    });
  }

  return http.createServer((req, res) => {
    let pathname;
    try {
      pathname = new URL(req.url, 'http://x').pathname;
    } catch (err) {
      return send(res, 400, 'Bad request');
    }

    if (pathname === '/api/survey/config' && req.method === 'GET') {
      return sendJson(res, 200, {
        version: survey.VERSION,
        mode: cfg.mode,
        ethics_ref: cfg.ethicsRef,
        ethics_note: cfg.ethicsNote || '',
      });
    }
    if (pathname === '/api/survey') {
      if (req.method !== 'POST') return sendJson(res, 405, { ok: false, error: 'POST only' });
      return handleSubmit(req, res).catch((err) => {
        console.error(err);
        if (!res.headersSent) sendJson(res, 500, { ok: false, error: 'server error' });
      });
    }
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      return send(res, 405, 'Method not allowed', { Allow: 'GET, HEAD' });
    }
    return handleStatic(req, res, pathname);
  });
}

if (require.main === module) {
  let cfg;
  try {
    cfg = config();
  } catch (err) {
    console.error(err.message);
    process.exit(1);
  }
  createServer(cfg).listen(cfg.port, cfg.host, () => {
    console.log('dashboard  http://' + cfg.host + ':' + cfg.port + '/');
    console.log('survey     http://' + cfg.host + ':' + cfg.port + '/survey');
    console.log('mode       ' + cfg.mode + (cfg.mode === 'pilot' ? ' (responses -> pilot.jsonl)' : ' (responses -> responses.jsonl)'));
    console.log('ethics     ' + (cfg.ethicsRef ? 'reference ' + cfg.ethicsRef : cfg.ethicsBasis || 'none (pilot)'));
    console.log('data       ' + cfg.dataDir);
  });
}

module.exports = { createServer, config, SUPERVISOR_ETHICS_NOTE };
