// Body prerender — runs AFTER prerender-seo.mjs has written the head-prerendered
// HTML for each route. For every public route, render the App tree to an HTML
// string and inject it into <div id="root"></div> so crawlers receive real
// content without executing the JS bundle.
//
// This is a write-only prerender: the client still mounts via createRoot (no
// hydration), so a per-route render failure is non-fatal — on error we keep the
// existing head-only file (the prior behavior) and continue.
import { readFile, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { PRERENDER_PATHS } from '../src/seoMetadata.js';
import { render } from '../dist/server/entry-server.js';

const rootDir = dirname(dirname(fileURLToPath(import.meta.url)));
const distDir = join(rootDir, 'dist');

const ROOT_PLACEHOLDER = '<div id="root"></div>';

function outputFor(pathname) {
  // Mirror prerender-seo.mjs: '/' writes to dist/index.html, everything else to
  // dist/<path>/index.html.
  return pathname === '/'
    ? join(distDir, 'index.html')
    : join(distDir, pathname.slice(1), 'index.html');
}

let injected = 0;
let skipped = 0;

for (const pathname of PRERENDER_PATHS) {
  const outputPath = outputFor(pathname);
  const locale = pathname === '/zh' || pathname.startsWith('/zh/') ? 'zh' : 'en';

  try {
    const file = await readFile(outputPath, 'utf8');
    if (!file.includes(ROOT_PLACEHOLDER)) {
      console.warn(`[prerender-render] no root placeholder in ${outputPath} — skipping`);
      skipped += 1;
      continue;
    }
    const html = await render({ url: pathname, locale });
    if (!html) {
      console.warn(`[prerender-render] empty render for ${pathname} — skipping`);
      skipped += 1;
      continue;
    }
    const next = file.replace(ROOT_PLACEHOLDER, `<div id="root">${html}</div>`);
    await writeFile(outputPath, next);
    injected += 1;
  } catch (error) {
    // Non-fatal: keep the head-only file (prior behavior) and move on.
    console.warn(`[prerender-render] failed for ${pathname} (${locale}): ${error.message}`);
    skipped += 1;
  }
}

console.log(`Prerendered body HTML for ${injected} of ${PRERENDER_PATHS.length} routes (${skipped} skipped/failed).`);
