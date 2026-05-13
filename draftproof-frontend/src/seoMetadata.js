import { resources } from './i18n/resources.js';

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
  },
  '/why': {
    titleKey: 'seo.whyTitle',
    descriptionKey: 'seo.whyDescription',
    canonical: '/why',
    schemaType: 'AboutPage',
  },
  '/pricing': {
    titleKey: 'seo.pricingTitle',
    descriptionKey: 'seo.pricingDescription',
    canonical: '/pricing',
    schemaType: 'WebPage',
  },
  '/privacy': {
    titleKey: 'seo.privacyTitle',
    descriptionKey: 'seo.privacyDescription',
    canonical: '/privacy',
    schemaType: 'PrivacyPolicy',
  },
  '/security': {
    titleKey: 'seo.securityTitle',
    descriptionKey: 'seo.securityDescription',
    canonical: '/security',
    schemaType: 'WebPage',
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
export const PRERENDER_PATHS = Object.keys(PAGE_META);

export function defaultTranslate(key, language = 'en') {
  return key.split('.').reduce((value, part) => value?.[part], resources[language]?.translation) || key;
}

export function getSeoMeta(pathname, translate = defaultTranslate) {
  const metaConfig = PAGE_META[pathname] || privateMeta(pathname);
  return {
    ...metaConfig,
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
    return {
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
    };
  }

  return {
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

function normalizeSiteUrl(url) {
  return String(url).replace(/\/+$/, '');
}
