import { resources } from './i18n/resources.js';
import {
  DEFAULT_LOCALE,
  SEO_LOCALES,
  getLocaleFromPathname,
  localizePath,
  stripLocaleFromPathname,
} from './localeRouting.js';
import { TOKEN_CURRENCY_CODE, TOKEN_PRICE_SGD, formatSgdAmount } from './pricingConfig.js';

const configuredSiteUrl =
  (typeof import.meta !== 'undefined' && import.meta.env?.VITE_SITE_URL)
  || (typeof process !== 'undefined' && process.env?.VITE_SITE_URL)
  || 'https://draftproof.app';

export const SITE_URL = normalizeSiteUrl(configuredSiteUrl);
// GA4 measurement ID. No Docker build-arg wiring exists for VITE_ vars in prod
// (see root Dockerfile — `npm run build` runs with no --build-arg), so the real
// ID is hardcoded as the fallback, same pattern as SITE_URL above. Not a secret:
// a GA4 measurement ID is public in every page's rendered HTML.
export const GA_MEASUREMENT_ID =
  (typeof import.meta !== 'undefined' && import.meta.env?.VITE_GA_MEASUREMENT_ID)
  || (typeof process !== 'undefined' && process.env?.VITE_GA_MEASUREMENT_ID)
  || 'G-8NXE7ESYTQ';
// ?v bumped whenever og-image.png is regenerated so social scrapers (which
// cache by URL) re-fetch the updated card instead of serving a stale image.
export const DEFAULT_IMAGE = `${SITE_URL}/og-image.png?v=2`;
export const SITE_NAME = 'DraftProof';
export const SEO_REVIEW_DATE = '2026-06-05';
// Homepage reviewed when the fine-tune v1 detector announcement was added to the
// hero (2026-07-14/15). Drives the homepage sitemap <lastmod>, JSON-LD dateModified,
// and the visible "last reviewed" footer — bump this whenever homepage content
// meaningfully changes so Google/Bing re-crawl and re-index the new copy.
export const HOME_REVIEW_DATE = '2026-07-15';
// Content-checker + FAQ copy extended to surface the Critical Thinking module.
export const CRITICAL_THINKING_REVIEW_DATE = '2026-06-17';
// SEO keyword landing pages (Turnitin score, academic integrity, AI declaration, reduce detection).
export const SEO_LANDING_REVIEW_DATE = '2026-06-24';

export const PAGE_META = {
  '/': {
    titleKey: 'seo.defaultTitle',
    descriptionKey: 'seo.defaultDescription',
    socialDescriptionKey: 'seo.defaultSocialDescription',
    canonical: '/',
    schemaType: 'SoftwareApplication',
    freshness: { type: 'reviewed', date: HOME_REVIEW_DATE },
  },
  '/why': {
    titleKey: 'seo.whyTitle',
    descriptionKey: 'seo.whyDescription',
    socialDescriptionKey: 'seo.whySocialDescription',
    canonical: '/why',
    schemaType: 'AboutPage',
    freshness: { type: 'reviewed', date: SEO_REVIEW_DATE },
  },
  '/features': {
    titleKey: 'seo.featuresTitle',
    descriptionKey: 'seo.featuresDescription',
    socialDescriptionKey: 'seo.featuresSocialDescription',
    canonical: '/features',
    schemaType: 'WebPage',
    freshness: { type: 'reviewed', date: '2026-06-19' },
  },
  '/technology': {
    titleKey: 'seo.technologyTitle',
    descriptionKey: 'seo.technologyDescription',
    socialDescriptionKey: 'seo.technologySocialDescription',
    canonical: '/technology',
    schemaType: 'AboutPage',
    freshness: { type: 'reviewed', date: '2026-07-05' },
  },
  '/rewrite': {
    titleKey: 'seo.rewriteTitle',
    descriptionKey: 'seo.rewriteDescription',
    socialDescriptionKey: 'seo.rewriteSocialDescription',
    canonical: '/rewrite',
    schemaType: 'WebPage',
    freshness: { type: 'reviewed', date: '2026-07-06' },
  },
  '/content-checker': {
    titleKey: 'seo.essayCheckerTitle',
    descriptionKey: 'seo.essayCheckerDescription',
    socialDescriptionKey: 'seo.essayCheckerSocialDescription',
    canonical: '/content-checker',
    schemaType: 'WebPage',
    freshness: { type: 'reviewed', date: CRITICAL_THINKING_REVIEW_DATE },
  },
  '/turnitin-ai-score': {
    titleKey: 'seo.turnitinScoreTitle',
    descriptionKey: 'seo.turnitinScoreDescription',
    socialDescriptionKey: 'seo.turnitinScoreSocialDescription',
    canonical: '/turnitin-ai-score',
    schemaType: 'WebPage',
    freshness: { type: 'reviewed', date: SEO_LANDING_REVIEW_DATE },
  },
  '/academic-integrity-ai': {
    titleKey: 'seo.academicIntegrityTitle',
    descriptionKey: 'seo.academicIntegrityDescription',
    socialDescriptionKey: 'seo.academicIntegritySocialDescription',
    canonical: '/academic-integrity-ai',
    schemaType: 'WebPage',
    freshness: { type: 'reviewed', date: SEO_LANDING_REVIEW_DATE },
  },
  '/ai-declaration': {
    titleKey: 'seo.aiDeclarationTitle',
    descriptionKey: 'seo.aiDeclarationDescription',
    socialDescriptionKey: 'seo.aiDeclarationSocialDescription',
    canonical: '/ai-declaration',
    schemaType: 'WebPage',
    freshness: { type: 'reviewed', date: SEO_LANDING_REVIEW_DATE },
  },
  '/reduce-ai-detection': {
    titleKey: 'seo.reduceDetectionTitle',
    descriptionKey: 'seo.reduceDetectionDescription',
    socialDescriptionKey: 'seo.reduceDetectionSocialDescription',
    canonical: '/reduce-ai-detection',
    schemaType: 'WebPage',
    freshness: { type: 'reviewed', date: SEO_LANDING_REVIEW_DATE },
  },
  '/turnitin-vs-ai-detectors': {
    titleKey: 'seo.turnitinVsDetectorsTitle',
    descriptionKey: 'seo.turnitinVsDetectorsDescription',
    socialDescriptionKey: 'seo.turnitinVsDetectorsSocialDescription',
    canonical: '/turnitin-vs-ai-detectors',
    schemaType: 'WebPage',
    freshness: { type: 'reviewed', date: '2026-07-10' },
  },
  '/turnitin-alternatives': {
    titleKey: 'seo.turnitinAlternativesTitle',
    descriptionKey: 'seo.turnitinAlternativesDescription',
    socialDescriptionKey: 'seo.turnitinAlternativesSocialDescription',
    canonical: '/turnitin-alternatives',
    schemaType: 'WebPage',
    freshness: { type: 'reviewed', date: '2026-07-10' },
  },
  '/pricing': {
    titleKey: 'seo.pricingTitle',
    descriptionKey: 'seo.pricingDescription',
    socialDescriptionKey: 'seo.pricingSocialDescription',
    canonical: '/pricing',
    schemaType: 'WebPage',
    freshness: { type: 'updated', date: SEO_REVIEW_DATE },
  },
  '/faq': {
    titleKey: 'seo.faqTitle',
    descriptionKey: 'seo.faqDescription',
    canonical: '/faq',
    schemaType: 'WebPage',
    freshness: { type: 'reviewed', date: CRITICAL_THINKING_REVIEW_DATE },
  },
  '/privacy': {
    titleKey: 'seo.privacyTitle',
    descriptionKey: 'seo.privacyDescription',
    canonical: '/privacy',
    schemaType: 'PrivacyPolicy',
    freshness: { type: 'updated', date: SEO_REVIEW_DATE },
  },
  '/terms': {
    titleKey: 'seo.termsTitle',
    descriptionKey: 'seo.termsDescription',
    canonical: '/terms',
    schemaType: 'WebPage',
    freshness: { type: 'updated', date: SEO_REVIEW_DATE },
  },
  '/support': {
    titleKey: 'seo.supportTitle',
    descriptionKey: 'seo.supportDescription',
    canonical: '/support',
    schemaType: 'WebPage',
  },
  '/eula': {
    titleKey: 'seo.eulaTitle',
    descriptionKey: 'seo.eulaDescription',
    canonical: '/eula',
    schemaType: 'WebPage',
    freshness: { type: 'updated', date: SEO_REVIEW_DATE },
  },
  '/security': {
    titleKey: 'seo.securityTitle',
    descriptionKey: 'seo.securityDescription',
    canonical: '/security',
    schemaType: 'WebPage',
    freshness: { type: 'updated', date: SEO_REVIEW_DATE },
  },
  '/signin': {
    titleKey: 'seo.signInTitle',
    descriptionKey: 'seo.signInDescription',
    canonical: '/signin',
    robots: 'noindex, nofollow',
    schemaType: 'WebPage',
  },
  '/404': {
    titleKey: 'seo.notFoundTitle',
    descriptionKey: 'seo.notFoundDescription',
    canonical: '/404',
    robots: 'noindex, nofollow',
    schemaType: 'WebPage',
  },
};

export const PRIVATE_PREFIXES = ['/dashboard', '/scan', '/reports', '/report/', '/rewrite/', '/buy', '/history', '/api-keys', '/auth/callback'];
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
    socialDescription: metaConfig.socialDescriptionKey
      ? translate(metaConfig.socialDescriptionKey)
      : translate(metaConfig.descriptionKey),
  };
}

export function getCanonicalUrl(meta) {
  return `${SITE_URL}${meta.canonical}`;
}

export function getRobots(meta) {
  return meta.robots || 'index, follow, max-image-preview:large';
}

export function buildSchema(meta, url, translate = defaultTranslate) {
  const language = getHtmlLang(meta.locale);
  const entityId = `${SITE_URL}/#organization`;
  const websiteId = `${SITE_URL}/#website`;

  if (meta.schemaType === 'SoftwareApplication') {
    return graphSchema([
      organizationSchema(entityId),
      websiteSchema(websiteId, entityId, language),
      withFreshness(meta, {
        '@type': 'SoftwareApplication',
        '@id': `${url}#software`,
        name: SITE_NAME,
        applicationCategory: 'EducationalApplication',
        operatingSystem: 'Web',
        url,
        image: DEFAULT_IMAGE,
        inLanguage: language,
        description: meta.description,
        publisher: { '@id': entityId },
        isPartOf: { '@id': websiteId },
        offers: {
          '@type': 'Offer',
          price: formatSgdAmount(TOKEN_PRICE_SGD),
          priceCurrency: TOKEN_CURRENCY_CODE,
          description: translate('seo.offerDescription'),
          url: `${SITE_URL}/pricing`,
        },
      }),
    ]);
  }

  const basePageSchema = withFreshness(meta, {
    '@type': meta.schemaType,
    '@id': `${url}#webpage`,
    name: meta.title,
    url,
    image: DEFAULT_IMAGE,
    inLanguage: language,
    description: meta.description,
    publisher: { '@id': entityId },
    isPartOf: { '@id': websiteId },
  });

  return graphSchema([
    organizationSchema(entityId),
    websiteSchema(websiteId, entityId, language),
    basePageSchema,
  ]);
}

function graphSchema(items) {
  return {
    '@context': 'https://schema.org',
    '@graph': items,
  };
}

function organizationSchema(id) {
  return {
    '@type': 'Organization',
    '@id': id,
    name: SITE_NAME,
    url: SITE_URL,
    logo: `${SITE_URL}/favicon.png`,
  };
}

function websiteSchema(id, publisherId, language) {
  return {
    '@type': 'WebSite',
    '@id': id,
    name: SITE_NAME,
    url: SITE_URL,
    inLanguage: language,
    publisher: { '@id': publisherId },
  };
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
  return {
    ...PAGE_META['/404'],
    canonical: pathname,
  };
}

function getAlternates(canonical) {
  return {
    en: localizePath(canonical, 'en'),
    zh: localizePath(canonical, 'zh'),
  };
}

function getResourceValue(key, language = DEFAULT_LOCALE) {
  return key.split('.').reduce((value, part) => value?.[part], resources[language]?.translation);
}

function normalizeSiteUrl(url) {
  return String(url).replace(/\/+$/, '');
}
