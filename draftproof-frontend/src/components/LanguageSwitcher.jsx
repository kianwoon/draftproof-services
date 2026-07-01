import { useTranslation } from 'react-i18next';
import { useLocation, useNavigate } from 'react-router-dom';
import { supportedLanguages } from '../i18n/resources';
import {
  isLocalizablePublicPath,
  localizePath,
  stripLocaleFromPathname,
} from '../localeRouting';

export default function LanguageSwitcher({ compact = false }) {
  const { i18n, t } = useTranslation();
  const location = useLocation();
  const navigate = useNavigate();
  const activeLanguage = i18n.resolvedLanguage?.startsWith('zh') || i18n.language?.startsWith('zh') ? 'zh' : 'en';
  const currentPath = `${location.pathname}${location.search}${location.hash}`;

  const handleLanguageChange = (language) => {
    i18n.changeLanguage(language);
    if (isLocalizablePublicPath(location.pathname)) {
      navigate(localizePath(stripLocaleFromPathname(currentPath), language), { replace: true });
    }
  };

  return (
    <label className={`language-switcher${compact ? ' language-switcher-compact' : ''}`}>
      <span>{t('nav.language')}</span>
      <select
        value={activeLanguage}
        aria-label={t('nav.language')}
        onChange={(event) => handleLanguageChange(event.target.value)}
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
