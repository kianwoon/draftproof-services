import { enTranslation } from './en.js';
import { zhTranslation } from './zh.js';

// Pure, side-effect-free data (no `document`/`localStorage`). Lives here — not in
// i18n/index.js — so that components on the SSR path (e.g. LanguageSwitcher) can
// import it WITHOUT pulling the browser-only i18n init into the server bundle.
export const supportedLanguages = [
  { code: 'en', labelKey: 'nav.english' },
  { code: 'zh', labelKey: 'nav.chinese' },
];

export const resources = {
  en: { translation: enTranslation },
  zh: { translation: zhTranslation },
};
