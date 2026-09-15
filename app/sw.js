/* Tango service worker — cache-first shell + reconciled audio cache.
 * VERSION is replaced by the deploy workflow with the commit sha; fine as-is locally.
 */
'use strict';

const VERSION = 'tango-__BUILD__';
const SHELL = `${VERSION}-shell`;
const AUDIO = 'tango-audio';
const INDEX_URL = './audio/_index.json';
const MANIFEST_URL = './audio/manifest.json';
const BATCH_SIZE = 6;

const SHELL_URLS = [
  './',
  './index.html',
  './manifest.json',
  './data.json',
  './icons/icon-192.png',
  './icons/icon-512.png',
  './icons/apple-touch-icon.png',
  './audio/manifest.json'
];
// Only the page itself is mandatory; everything else may be missing (e.g. audio manifest before generation).
const REQUIRED = new Set(['./index.html']);

// ---------------------------------------------------------------- install / activate

self.addEventListener('install', (event) => {
  event.waitUntil(installShell());
});

async function installShell() {
  const cache = await caches.open(SHELL);
  for (const url of SHELL_URLS) {
    try {
      const res = await fetch(new Request(url, { cache: 'reload' }));
      if (!res || !res.ok) throw new Error('HTTP ' + (res ? res.status : '?'));
      await cache.put(url, res);
    } catch (err) {
      if (REQUIRED.has(url)) throw err;
    }
  }
}

self.addEventListener('activate', (event) => {
  event.waitUntil(
    (async () => {
      const keys = await caches.keys();
      await Promise.all(
        keys
          .filter((k) => k.startsWith('tango-') && k !== SHELL && k !== AUDIO)
          .map((k) => caches.delete(k))
      );
      await self.clients.claim();
    })().then(() => {
      // Not awaited on purpose: activation must not wait for the audio download.
      syncAudio();
    })
  );
});

// ---------------------------------------------------------------- fetch

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;
  let url;
  try {
    url = new URL(req.url);
  } catch (e) {
    return;
  }
  if (url.origin !== self.location.origin) return;

  if (req.mode === 'navigate') {
    event.respondWith(serveShellPage(req));
    return;
  }
  if (/\/audio\/[^/]+\.m4a$/.test(url.pathname)) {
    event.respondWith(serveAudio(url));
    return;
  }
  event.respondWith(serveShell(req));
});

async function serveShellPage(req) {
  try {
    const cached =
      (await caches.match('./index.html', { cacheName: SHELL })) ||
      (await caches.match('./', { cacheName: SHELL }));
    if (cached) return cached;
  } catch (e) {
    /* fall through */
  }
  try {
    const res = await fetch(req);
    if (res && res.ok) {
      caches.open(SHELL).then((c) => c.put('./index.html', res.clone())).catch(() => {});
    }
    return res;
  } catch (e) {
    return offlinePage();
  }
}

async function serveAudio(url) {
  // Strip query + Range header by using the plain URL as the cache key and fetch request.
  const key = url.origin + url.pathname;
  let cache = null;
  try {
    cache = await caches.open(AUDIO);
    const hit = await cache.match(key);
    if (hit) return hit;
  } catch (e) {
    /* cache unavailable – go to network */
  }
  try {
    const res = await fetch(key);
    if (res && res.status === 200 && cache) {
      cache.put(key, res.clone()).catch(() => {});
    }
    return res;
  } catch (e) {
    return offlineResponse();
  }
}

async function serveShell(req) {
  let cache = null;
  try {
    cache = await caches.open(SHELL);
    const hit = await cache.match(req, { ignoreSearch: true });
    if (hit) return hit;
  } catch (e) {
    /* ignore */
  }
  try {
    const res = await fetch(req);
    if (res && res.ok && res.type === 'basic' && cache) {
      cache.put(req, res.clone()).catch(() => {});
    }
    return res;
  } catch (e) {
    return offlineResponse();
  }
}

function offlineResponse() {
  return new Response('offline', { status: 503, statusText: 'Offline', headers: { 'Content-Type': 'text/plain' } });
}

// Navigation while offline before the shell was ever cached: a readable page instead of a browser error.
function offlinePage() {
  const html = '<!doctype html><html lang="de"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">' +
    '<title>Tango</title><body style="margin:0;background:#000;color:#f3ede4;font:17px/1.4 -apple-system,system-ui,sans-serif;padding:2em">' +
    '<h1 style="font-size:28px;margin:0 0 .5em">Tango</h1><p>Offline – die App wurde noch nicht geladen. Bitte mit Verbindung erneut öffnen.</p></body></html>';
  return new Response(html, { status: 503, statusText: 'Offline', headers: { 'Content-Type': 'text/html; charset=utf-8' } });
}

// ---------------------------------------------------------------- audio sync

const progress = { done: 0, total: 0, failed: 0, running: false, synced: false };
let syncPromise = null;

function syncAudio() {
  if (syncPromise) return syncPromise;
  syncPromise = doSyncAudio()
    .catch((err) => {
      progress.running = false;
      broadcast({ type: 'AUDIO_DONE', total: progress.total, failed: progress.failed, error: String(err && err.message || err) });
    })
    .finally(() => {
      syncPromise = null;
    });
  return syncPromise;
}

async function loadManifest() {
  try {
    const res = await fetch(MANIFEST_URL, { cache: 'no-store' });
    if (res && res.ok) {
      const copy = res.clone();
      const manifest = await res.json();
      // Keep the shell copy fresh so the app sees the same manifest.
      caches.open(SHELL).then((c) => c.put(MANIFEST_URL, copy)).catch(() => {});
      return manifest;
    }
  } catch (e) {
    /* offline or missing – try the cached copy */
  }
  try {
    const cached = await caches.match(MANIFEST_URL, { cacheName: SHELL });
    if (cached) return await cached.json();
  } catch (e) {
    /* ignore */
  }
  return null;
}

async function readIndex(cache) {
  try {
    const res = await cache.match(INDEX_URL);
    if (!res) return {};
    const obj = await res.json();
    return obj && typeof obj === 'object' ? obj : {};
  } catch (e) {
    return {};
  }
}

async function writeIndex(cache, index) {
  try {
    await cache.put(
      INDEX_URL,
      new Response(JSON.stringify(index), { headers: { 'Content-Type': 'application/json' } })
    );
  } catch (e) {
    /* ignore */
  }
}

function nameOf(url) {
  return url.split('?')[0].split('/').pop();
}

async function doSyncAudio() {
  progress.running = true;
  const manifest = await loadManifest();
  if (!manifest || !Array.isArray(manifest.files)) {
    // No manifest reachable (not generated yet, or offline with no cached copy): leave the audio cache
    // untouched — an unknown manifest must never be treated as "delete everything".
    progress.total = 0;
    progress.done = 0;
    progress.failed = 0;
    progress.running = false;
    progress.synced = true;
    broadcast({ type: 'AUDIO_DONE', total: 0, failed: 0, done: 0, missingManifest: true });
    return;
  }
  const files = manifest.files.filter((f) => f && typeof f.name === 'string' && /\.m4a$/.test(f.name));
  const wanted = new Map(files.map((f) => [f.name, Number(f.size) || 0]));

  const cache = await caches.open(AUDIO);
  const index = await readIndex(cache);
  const present = new Set();

  // Reconcile what is already cached against the manifest.
  const keys = await cache.keys();
  for (const reqKey of keys) {
    const name = nameOf(reqKey.url);
    if (name === '_index.json') continue;
    const want = wanted.get(name);
    let size = Number(index[name]);
    if (!size) {
      // Entry cached by the fetch handler (not by us): measure it once.
      try {
        const res = await cache.match(reqKey);
        size = res ? Number(res.headers.get('content-length')) || (await res.blob()).size : 0;
      } catch (e) {
        size = 0;
      }
    }
    if (want === undefined || (want > 0 && size > 0 && size !== want)) {
      await cache.delete(reqKey);
      delete index[name];
    } else {
      index[name] = size || want;
      present.add(name);
    }
  }
  // Drop index entries whose file is gone.
  for (const name of Object.keys(index)) {
    if (!present.has(name)) delete index[name];
  }

  const missing = files.filter((f) => !present.has(f.name));
  progress.total = files.length;
  progress.done = present.size;
  progress.failed = 0;
  await writeIndex(cache, index);
  broadcast({ type: 'AUDIO_PROGRESS', done: progress.done, total: progress.total, failed: progress.failed });

  for (let i = 0; i < missing.length; i += BATCH_SIZE) {
    const batch = missing.slice(i, i + BATCH_SIZE);
    const results = await Promise.allSettled(
      batch.map(async (f) => {
        const url = './audio/' + f.name;
        const res = await fetch(url, { cache: 'no-store' });
        if (!res || !res.ok) throw new Error('HTTP ' + (res ? res.status : '?'));
        await cache.put(url, res);
        index[f.name] = Number(f.size) || 0;
      })
    );
    results.forEach((r) => {
      if (r.status === 'fulfilled') progress.done++;
      else progress.failed++;
    });
    await writeIndex(cache, index);
    broadcast({ type: 'AUDIO_PROGRESS', done: progress.done, total: progress.total, failed: progress.failed });
  }

  progress.running = false;
  progress.synced = true;
  broadcast({ type: 'AUDIO_DONE', total: progress.total, failed: progress.failed, done: progress.done });
}

function broadcast(msg) {
  self.clients
    .matchAll({ includeUncontrolled: true, type: 'window' })
    .then((clients) => clients.forEach((c) => c.postMessage(msg)))
    .catch(() => {});
}

function reply(event, msg) {
  try {
    if (event.source && typeof event.source.postMessage === 'function') {
      event.source.postMessage(msg);
    } else {
      broadcast(msg);
    }
  } catch (e) {
    /* ignore */
  }
}

// ---------------------------------------------------------------- messages

self.addEventListener('message', (event) => {
  const data = event.data || {};
  const type = typeof data === 'string' ? data : data.type;
  if (type === 'SKIP_WAITING') {
    self.skipWaiting();
  } else if (type === 'SYNC_AUDIO') {
    event.waitUntil(syncAudio());
  } else if (type === 'AUDIO_STATUS') {
    reply(event, {
      type: progress.running ? 'AUDIO_PROGRESS' : 'AUDIO_STATUS',
      done: progress.done,
      total: progress.total,
      failed: progress.failed,
      running: progress.running,
      synced: progress.synced
    });
  } else if (type === 'GET_VERSION') {
    reply(event, { type: 'VERSION', version: VERSION });
  }
});
