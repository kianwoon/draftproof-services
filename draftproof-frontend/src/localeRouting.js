export const DEFAULT_LOCALE = 'en';
export const SEO_LOCALES = ['en', 'zh'];
export const LOCALE_PREFIXES = ['zh'];
export const LOCALIZABLE_PUBLIC_PATHS = ['/', '/why', '/features', '/content-checker', '/pricing', '/faq', '/privacy', '/security', '/signin'];

export function getLocaleFromPathname(pathname = '/') {
  return String(pathname).split('/')[1] === 'zh' ? 'zh' : DEFAULT_LOCALE;
}

export function stripLocaleFromPathname(pathname = '/') {
  const normalized = normalizePath(pathname);
  if (normalized === '/zh') return '/';
  if (normalized.startsWith('/zh/')) return normalized.slice(3) || '/';
  return normalized;
}

export function localizePath(path, locale = DEFAULT_LOCALE) {
  const normalized = normalizePath(path);
  if (locale !== 'zh') return normalized;
  if (normalized === '/') return '/zh';
  return `/zh${normalized}`;
}

export function isLocalizablePublicPath(pathname = '/') {
  return LOCALIZABLE_PUBLIC_PATHS.includes(stripLocaleFromPathname(pathname));
}

export function normalizePath(path = '/') {
  const [pathname, suffix = ''] = String(path).split(/([?#].*)/, 2);
  const normalized = pathname.startsWith('/') ? pathname : `/${pathname}`;
  return `${normalized.replace(/\/+$/, '') || '/'}${suffix}`;
}
