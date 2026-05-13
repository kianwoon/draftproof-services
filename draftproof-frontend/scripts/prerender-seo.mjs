import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  DEFAULT_IMAGE,
  PRERENDER_PATHS,
  buildSchema,
  defaultTranslate,
  getCanonicalUrl,
  getRobots,
  getSeoMeta,
} from '../src/seoMetadata.js';

const rootDir = dirname(dirname(fileURLToPath(import.meta.url)));
const distDir = join(rootDir, 'dist');
const templatePath = join(distDir, 'index.html');

const template = await readFile(templatePath, 'utf8');

for (const pathname of PRERENDER_PATHS) {
  const html = renderRoute(template, pathname);
  const outputPath = pathname === '/'
    ? templatePath
    : join(distDir, pathname.slice(1), 'index.html');

  await mkdir(dirname(outputPath), { recursive: true });
  await writeFile(outputPath, html);
}

console.log(`Prerendered SEO metadata for ${PRERENDER_PATHS.length} routes.`);

function renderRoute(html, pathname) {
  const meta = getSeoMeta(pathname, (key) => defaultTranslate(key, 'en'));
  const canonicalUrl = getCanonicalUrl(meta);
  const schema = buildSchema(meta, canonicalUrl, (key) => defaultTranslate(key, 'en'));

  return [
    [/<title>.*?<\/title>/s, `<title>${escapeHtml(meta.title)}</title>`],
    [/<meta name="description"[^>]*>/, metaTag('description', meta.description)],
    [/<meta name="robots"[^>]*>/, metaTag('robots', getRobots(meta))],
    [/<link rel="canonical"[^>]*>/, `<link rel="canonical" href="${escapeAttribute(canonicalUrl)}" />`],
    [/<meta property="og:title"[^>]*>/, propertyTag('og:title', meta.title)],
    [/<meta property="og:description"[^>]*>/, propertyTag('og:description', meta.description)],
    [/<meta property="og:url"[^>]*>/, propertyTag('og:url', canonicalUrl)],
    [/<meta property="og:image"[^>]*>/, propertyTag('og:image', DEFAULT_IMAGE)],
    [/<meta property="og:image:alt"[^>]*>/, propertyTag('og:image:alt', defaultTranslate('seo.imageAlt', 'en'))],
    [/<meta name="twitter:title"[^>]*>/, metaTag('twitter:title', meta.title)],
    [/<meta name="twitter:description"[^>]*>/, metaTag('twitter:description', meta.description)],
    [/<meta name="twitter:image"[^>]*>/, metaTag('twitter:image', DEFAULT_IMAGE)],
    [
      /<script type="application\/ld\+json" data-seo-jsonld="true">.*?<\/script>/s,
      jsonLdTag(schema),
    ],
  ].reduce((currentHtml, [pattern, replacement]) => replaceRequired(currentHtml, pattern, replacement), html);
}

function replaceRequired(html, pattern, replacement) {
  if (!pattern.test(html)) {
    throw new Error(`Unable to find SEO template target: ${pattern}`);
  }
  return html.replace(pattern, replacement);
}

function metaTag(name, content) {
  return `<meta name="${escapeAttribute(name)}" content="${escapeAttribute(content)}" />`;
}

function propertyTag(property, content) {
  return `<meta property="${escapeAttribute(property)}" content="${escapeAttribute(content)}" />`;
}

function jsonLdTag(schema) {
  const json = JSON.stringify(schema, null, 2)
    .replace(/</g, '\\u003c')
    .split('\n')
    .map((line) => `      ${line}`)
    .join('\n');
  return `<script type="application/ld+json" data-seo-jsonld="true">\n${json}\n    </script>`;
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function escapeAttribute(value) {
  return escapeHtml(value).replace(/"/g, '&quot;');
}
