import i18n from 'i18next';
import LanguageDetector from 'i18next-browser-languagedetector';
import { initReactI18next } from 'react-i18next';
import { resources, supportedLanguages } from './resources';

// Re-exported for backwards compatibility (some modules import { supportedLanguages } from '../i18n').
export { supportedLanguages };

i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources,
    fallbackLng: 'en',
    supportedLngs: supportedLanguages.map((language) => language.code),
    nonExplicitSupportedLngs: true,
    interpolation: {
      escapeValue: false,
    },
    detection: {
      order: ['localStorage', 'navigator', 'htmlTag'],
      caches: ['localStorage'],
      lookupLocalStorage: 'draftproof_language',
    },
    react: {
      useSuspense: false,
    },
  });

i18n.on('languageChanged', (lng) => {
  document.documentElement.lang = lng?.startsWith('zh') ? 'zh-CN' : 'en';
});

document.documentElement.lang = i18n.resolvedLanguage?.startsWith('zh') ? 'zh-CN' : 'en';

export default i18n;
