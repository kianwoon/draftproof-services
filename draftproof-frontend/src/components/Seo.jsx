import { useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';

const SITE_URL = 'https://draftproof.app';
const DEFAULT_IMAGE = `${SITE_URL}/og-image.png`;

const PAGE_META = {
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

const PRIVATE_PREFIXES = ['/dashboard', '/scan', '/reports', '/report/', '/rewrite/', '/buy', '/history', '/auth/callback'];

export default function Seo() {
  const { pathname } = useLocation();
  const { t, i18n } = useTranslation();

  useEffect(() => {
    const metaConfig = PAGE_META[pathname] || privateMeta(pathname);
    const meta = {
      ...metaConfig,
      title: t(metaConfig.titleKey),
      description: t(metaConfig.descriptionKey),
    };
    const canonicalUrl = `${SITE_URL}${meta.canonical}`;

    document.title = meta.title;
    setMeta('description', meta.description);
    setMeta('robots', meta.robots || 'index, follow, max-image-preview:large');
    setCanonical(canonicalUrl);

    setProperty('og:site_name', 'DraftProof');
    setProperty('og:type', 'website');
    setProperty('og:title', meta.title);
    setProperty('og:description', meta.description);
    setProperty('og:url', canonicalUrl);
    setProperty('og:image', DEFAULT_IMAGE);
    setProperty('og:image:width', '1200');
    setProperty('og:image:height', '630');
    setProperty('og:image:alt', t('seo.imageAlt'));

    setMeta('twitter:card', 'summary_large_image');
    setMeta('twitter:title', meta.title);
    setMeta('twitter:description', meta.description);
    setMeta('twitter:image', DEFAULT_IMAGE);

    setJsonLd(buildSchema(meta, canonicalUrl, t));
  }, [pathname, i18n.resolvedLanguage, t]);

  return null;
}

function privateMeta(pathname) {
  const isPrivate = PRIVATE_PREFIXES.some((prefix) => pathname === prefix || pathname.startsWith(prefix));
  if (isPrivate) {
    return {
      title: 'DraftProof App',
      titleKey: 'seo.privateTitle',
      descriptionKey: 'seo.privateDescription',
      canonical: pathname,
      robots: 'noindex, nofollow',
      schemaType: 'WebPage',
    };
  }
  return PAGE_META['/'];
}

function setMeta(name, content) {
  let tag = document.querySelector(`meta[name="${name}"]`);
  if (!tag) {
    tag = document.createElement('meta');
    tag.setAttribute('name', name);
    document.head.appendChild(tag);
  }
  tag.setAttribute('content', content);
}

function setProperty(property, content) {
  let tag = document.querySelector(`meta[property="${property}"]`);
  if (!tag) {
    tag = document.createElement('meta');
    tag.setAttribute('property', property);
    document.head.appendChild(tag);
  }
  tag.setAttribute('content', content);
}

function setCanonical(href) {
  let tag = document.querySelector('link[rel="canonical"]');
  if (!tag) {
    tag = document.createElement('link');
    tag.setAttribute('rel', 'canonical');
    document.head.appendChild(tag);
  }
  tag.setAttribute('href', href);
}

function setJsonLd(schema) {
  let tag = document.querySelector('script[data-seo-jsonld="true"]');
  if (!tag) {
    tag = document.createElement('script');
    tag.type = 'application/ld+json';
    tag.dataset.seoJsonld = 'true';
    document.head.appendChild(tag);
  }
  tag.textContent = JSON.stringify(schema);
}

function buildSchema(meta, url, t) {
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
        description: t('seo.offerDescription'),
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
