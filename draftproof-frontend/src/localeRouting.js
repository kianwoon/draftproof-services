export const DEFAULT_LOCALE = 'en';
export const SEO_LOCALES = ['en', 'zh'];
export const LOCALE_PREFIXES = ['zh'];
export const LOCALIZABLE_PUBLIC_PATHS = ['/', '/why', '/features', '/technology', '/rewrite', '/content-checker', '/turnitin-ai-score', '/academic-integrity-ai', '/ai-declaration', '/reduce-ai-detection', '/turnitin-vs-ai-detectors', '/turnitin-alternatives', '/check-essay-before-turnitin', '/turnitin-flagged-my-essay-ai', '/draftproof-vs-turnitin', '/pricing', '/faq', '/privacy', '/security', '/signin'];

export function getLocaleFromPathname(pathname = '/') {
  return String(pathname).split('/')[1] === 'zh' ? 'zh' : DEFAULT_LOCALE;
}

export function stripLocaleFromPathname(pathname = '/') {
  const normalized = normalizePath(pathname);
  const [base, suffix = ''] = normalized.split(/([?#].*)/, 2);
  if (base === '/zh') return `/${suffix}`;
  if (base.startsWith('/zh/')) return `${base.slice(3) || '/'}${suffix}`;
  return normalized;
}

export function localizePath(path, locale = DEFAULT_LOCALE) {
  const normalized = normalizePath(path);
  if (locale !== 'zh') return normalized;
  const [base, suffix = ''] = normalized.split(/([?#].*)/, 2);
  if (base === '/') return `/zh${suffix}`;
  return `/zh${base}${suffix}`;
}

export function isLocalizablePublicPath(pathname = '/') {
  return LOCALIZABLE_PUBLIC_PATHS.includes(stripLocaleFromPathname(pathname));
}

export function normalizePath(path = '/') {
  const [pathname, suffix = ''] = String(path).split(/([?#].*)/, 2);
  const normalized = pathname.startsWith('/') ? pathname : `/${pathname}`;
  return `${normalized.replace(/\/+$/, '') || '/'}${suffix}`;
}
