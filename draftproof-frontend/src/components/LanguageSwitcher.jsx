import { useTranslation } from 'react-i18next';
import { supportedLanguages } from '../i18n';

export default function LanguageSwitcher({ compact = false }) {
  const { i18n, t } = useTranslation();
  const activeLanguage = i18n.resolvedLanguage?.startsWith('zh') || i18n.language?.startsWith('zh') ? 'zh' : 'en';

  return (
    <label className={`language-switcher${compact ? ' language-switcher-compact' : ''}`}>
      <span>{t('nav.language')}</span>
      <select
        value={activeLanguage}
        aria-label={t('nav.language')}
        onChange={(event) => i18n.changeLanguage(event.target.value)}
      >
        {supportedLanguages.map((language) => (
          <option key={language.code} value={language.code}>
            {t(language.labelKey)}
          </option>
        ))}
      </select>
    </label>
  );
}
