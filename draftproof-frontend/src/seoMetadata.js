import { resources } from './i18n/resources.js';
import {
  DEFAULT_LOCALE,
  SEO_LOCALES,
  getLocaleFromPathname,
  localizePath,
  stripLocaleFromPathname,
} from './localeRouting.js';

const configuredSiteUrl =
  (typeof import.meta !== 'undefined' && import.meta.env?.VITE_SITE_URL)
  || (typeof process !== 'undefined' && process.env?.VITE_SITE_URL)
  || 'https://draftproof.app';

export const SITE_URL = normalizeSiteUrl(configuredSiteUrl);
export const DEFAULT_IMAGE = `${SITE_URL}/og-image.png`;

export const PAGE_META = {
  '/': {
    titleKey: 'seo.defaultTitle',
    descriptionKey: 'seo.defaultDescription',
    canonical: '/',
    schemaType: 'SoftwareApplication',
    freshness: { type: 'reviewed', date: '2026-05-21' },
  },
  '/why': {
    titleKey: 'seo.whyTitle',
    descriptionKey: 'seo.whyDescription',
    canonical: '/why',
    schemaType: 'AboutPage',
    freshness: { type: 'reviewed', date: '2026-05-21' },
  },
  '/essay-checker': {
    titleKey: 'seo.essayCheckerTitle',
    descriptionKey: 'seo.essayCheckerDescription',
    canonical: '/essay-checker',
    schemaType: 'WebPage',
    freshness: { type: 'reviewed', date: '2026-05-21' },
  },
  '/pricing': {
    titleKey: 'seo.pricingTitle',
    descriptionKey: 'seo.pricingDescription',
    canonical: '/pricing',
    schemaType: 'WebPage',
    freshness: { type: 'updated', date: '2026-05-21' },
  },
  '/faq': {
    titleKey: 'seo.faqTitle',
    descriptionKey: 'seo.faqDescription',
    canonical: '/faq',
    schemaType: 'FAQPage',
    freshness: { type: 'reviewed', date: '2026-05-21' },
  },
  '/privacy': {
    titleKey: 'seo.privacyTitle',
    descriptionKey: 'seo.privacyDescription',
    canonical: '/privacy',
    schemaType: 'PrivacyPolicy',
    freshness: { type: 'updated', date: '2026-05-21' },
  },
  '/security': {
    titleKey: 'seo.securityTitle',
    descriptionKey: 'seo.securityDescription',
    canonical: '/security',
    schemaType: 'WebPage',
    freshness: { type: 'updated', date: '2026-05-21' },
  },
  '/signin': {
    titleKey: 'seo.signInTitle',
    descriptionKey: 'seo.signInDescription',
    canonical: '/signin',
    robots: 'noindex, nofollow',
    schemaType: 'WebPage',
  },
};

export const PRIVATE_PREFIXES = ['/dashboard', '/scan', '/reports', '/report/', '/rewrite/', '/buy', '/history', '/auth/callback'];
export const PRERENDER_PATHS = Object.keys(PAGE_META).flatMap((pathname) => (
  SEO_LOCALES.map((locale) => localizePath(pathname, locale))
));

export function defaultTranslate(key, language = DEFAULT_LOCALE) {
  return getResourceValue(key, language) || key;
}

export function getSeoMeta(pathname, translate = defaultTranslate) {
  const locale = getLocaleFromPathname(pathname);
  const basePathname = stripLocaleFromPathname(pathname);
  const metaConfig = PAGE_META[basePathname] || privateMeta(basePathname);
  const canonical = localizePath(metaConfig.canonical, locale);
  return {
    ...metaConfig,
    basePathname,
    canonical,
    locale,
    alternates: getAlternates(metaConfig.canonical),
    title: translate(metaConfig.titleKey),
    description: translate(metaConfig.descriptionKey),
  };
}

export function getCanonicalUrl(meta) {
  return `${SITE_URL}${meta.canonical}`;
}

export function getRobots(meta) {
  return meta.robots || 'index, follow, max-image-preview:large';
}

export function buildSchema(meta, url, translate = defaultTranslate) {
  if (meta.schemaType === 'SoftwareApplication') {
    return withFreshness(meta, {
      '@context': 'https://schema.org',
      '@type': 'SoftwareApplication',
      name: 'DraftProof',
      applicationCategory: 'EducationalApplication',
      operatingSystem: 'Web',
      url,
      description: meta.description,
      offers: {
        '@type': 'Offer',
        price: '0.90',
        priceCurrency: 'USD',
        description: translate('seo.offerDescription'),
      },
    });
  }

  if (meta.schemaType === 'FAQPage') {
    return withFreshness(meta, {
      '@context': 'https://schema.org',
      '@type': 'FAQPage',
      name: meta.title,
      url,
      description: meta.description,
      mainEntity: getFaqEntities(meta.locale),
      isPartOf: {
        '@type': 'WebSite',
        name: 'DraftProof',
        url: SITE_URL,
      },
    });
  }

  return withFreshness(meta, {
    '@context': 'https://schema.org',
    '@type': meta.schemaType,
    name: meta.title,
    url,
    description: meta.description,
    isPartOf: {
      '@type': 'WebSite',
      name: 'DraftProof',
      url: SITE_URL,
    },
  });
}

export function getAlternateUrls(meta) {
  return Object.fromEntries(
    Object.entries(meta.alternates || {}).map(([locale, pathname]) => [
      locale === DEFAULT_LOCALE ? 'en' : 'zh-CN',
      `${SITE_URL}${pathname}`,
    ])
  );
}

export function getHtmlLang(locale = DEFAULT_LOCALE) {
  return locale === 'zh' ? 'zh-CN' : 'en';
}

export function getFreshnessLabelKey(meta) {
  return meta.freshness?.type === 'reviewed' ? 'common.lastReviewed' : 'common.lastUpdated';
}

export function formatFreshnessDate(date, locale = 'en') {
  const [year, month, day] = String(date).split('-').map(Number);
  if (!year || !month || !day) return date;
  const normalizedLocale = String(locale).toLowerCase().startsWith('zh') ? 'zh-CN' : 'en-US';
  return new Intl.DateTimeFormat(normalizedLocale, {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  }).format(new Date(year, month - 1, day));
}

function withFreshness(meta, schema) {
  if (!meta.freshness?.date) return schema;
  return {
    ...schema,
    dateModified: meta.freshness.date,
  };
}

function privateMeta(pathname) {
  const isPrivate = PRIVATE_PREFIXES.some((prefix) => pathname === prefix || pathname.startsWith(prefix));
  if (isPrivate) {
    return {
      titleKey: 'seo.privateTitle',
      descriptionKey: 'seo.privateDescription',
      canonical: pathname,
      robots: 'noindex, nofollow',
      schemaType: 'WebPage',
    };
  }
  return PAGE_META['/'];
}

function getAlternates(canonical) {
  return {
    en: localizePath(canonical, 'en'),
    zh: localizePath(canonical, 'zh'),
  };
}

function getFaqEntities(locale = DEFAULT_LOCALE) {
  const groups = getResourceValue('faqPage.groups', locale);
  if (!Array.isArray(groups)) return [];
  return groups.flatMap((group) => (
    (group.items || []).map((item) => ({
      '@type': 'Question',
      name: item.q,
      acceptedAnswer: {
        '@type': 'Answer',
        text: item.a,
      },
    }))
  ));
}

function getResourceValue(key, language = DEFAULT_LOCALE) {
  return key.split('.').reduce((value, part) => value?.[part], resources[language]?.translation);
}

function normalizeSiteUrl(url) {
  return String(url).replace(/\/+$/, '');
}
