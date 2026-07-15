#!/usr/bin/env node
/**
 * IndexNow ping — notifies participating search engines (Bing, Yandex, Seznam,
 * Naver, and via the shared IndexNow endpoint) that key pages changed, so they
 * re-crawl without waiting for the periodic sitemap sweep.
 *
 * Wired into `npm run build:deploy` (the Dockerfile's build step) so it fires
 * once per production deploy — NOT into plain `npm run build`, which developers
 * run locally and which must never emit outbound pings.
 *
 * Ownership is proven by serving the key verbatim at
 * https://draftproof.app/<KEY>.txt (see public/<KEY>.txt). The KEY is PUBLIC by
 * design (anyone can read the key file) — it is not a secret, so it lives here
 * as plain config, not an env secret.
 *
 * Best-effort: any failure (network, non-2xx, timeout) is logged and the script
 * still exits 0. A missed ping must never fail a deploy — the sitemap <lastmod>
 * bump is the durable re-crawl signal; this is just the fast-path accelerant.
 *
 * Usage:
 *   node scripts/indexnow-ping.mjs                 # ping default URLs
 *   node scripts/indexnow-ping.mjs --dry-run       # print payload, send nothing
 *   node scripts/indexnow-ping.mjs https://draftproof.app/whats-new  # custom URLs
 */

const HOST = 'draftproof.app';
const KEY = 'a463ff72bcfc623df98e93293032cfd8';
const ENDPOINT = 'https://api.indexnow.org/indexnow';

// Pages that carry the current announcement (en + zh homepage). Add more URLs as
// args when a specific page changes.
const DEFAULT_URLS = [`https://${HOST}/`, `https://${HOST}/zh`];

async function main() {
  const args = process.argv.slice(2);
  const dryRun = args.includes('--dry-run');
  const urlList = args.filter((a) => a.startsWith('http'));
  const urls = urlList.length ? urlList : DEFAULT_URLS;

  const payload = {
    host: HOST,
    key: KEY,
    keyLocation: `https://${HOST}/${KEY}.txt`,
    urlList: urls,
  };

  if (dryRun) {
    console.log('[indexnow] dry-run — would POST to', ENDPOINT);
    console.log(JSON.stringify(payload, null, 2));
    return;
  }

  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 10_000);
    const res = await fetch(ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json; charset=utf-8' },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    clearTimeout(timer);
    // IndexNow returns 200 (accepted) or 202 (accepted, pending). Anything else
    // is logged but non-fatal.
    console.log(`[indexnow] submitted ${urls.length} url(s) — HTTP ${res.status}`);
  } catch (err) {
    console.warn('[indexnow] ping failed (non-fatal):', err?.message || err);
  }
}

main();
