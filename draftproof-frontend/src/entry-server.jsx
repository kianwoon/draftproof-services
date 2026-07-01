// SSR entry — renders the App tree to an HTML string at build time so crawlers
// receive real content without executing the JS bundle. Used only by
// scripts/prerender-render.mjs (a build-time, write-only prerender). The client
// still mounts via createRoot in main.jsx (no hydration), so this output is
// "write-only for SEO" — React remounts fresh on the client.
//
// Importantly this must NOT touch the browser: no CSS import (throws in Node),
// no StrictMode (avoids SSR double-invoke), and it primes its OWN i18n instance
// from the pure data module (./i18n/resources) — never ./i18n/index.js (which
// wires LanguageDetector + localStorage + document).
import { renderToString } from 'react-dom/server';
import { StaticRouter } from 'react-router-dom/server';
import i18next from 'i18next';
import { initReactI18next, I18nextProvider } from 'react-i18next';
import App from './App';
import { resources } from './i18n/resources';

/**
 * Render a single route to an HTML string.
 * @param {{ url: string, locale: 'en' | 'zh' }} opts
 * @returns {Promise<string>} prerendered inner HTML for <div id="root">
 */
export async function render({ url, locale = 'en' }) {
  const i18n = i18next.createInstance();
  await i18n
    .use(initReactI18next)
    .init({
      resources,
      lng: locale,
      fallbackLng: 'en',
      supportedLngs: ['en', 'zh'],
      nonExplicitSupportedLngs: true,
      interpolation: { escapeValue: false },
      react: { useSuspense: false },
    });

  return renderToString(
    <I18nextProvider i18n={i18n}>
      <StaticRouter location={url}>
        <App />
      </StaticRouter>
    </I18nextProvider>,
  );
}
