import { useEffect } from 'react';
import { useLocation } from 'react-router-dom';

const SITE_URL = 'https://draftproof.app';
const DEFAULT_IMAGE = `${SITE_URL}/og-image.png`;

const PAGE_META = {
  '/': {
    title: 'DraftProof | Pre-Submission Writing Integrity Review',
    description: 'DraftProof helps students, educators, and researchers review writing before submission by checking citation gaps, source grounding, generic phrasing, and review-only authorship signals.',
    canonical: '/',
    schemaType: 'SoftwareApplication',
  },
  '/why': {
    title: 'Why DraftProof Exists | Writing Integrity Review',
    description: 'Learn why DraftProof focuses on writing integrity, source grounding, citation support, and actionable review signals instead of AI verdicts.',
    canonical: '/why',
    schemaType: 'AboutPage',
  },
  '/pricing': {
    title: 'DraftProof Pricing | Writing Integrity Review Tokens',
    description: 'Simple DraftProof pricing for pre-submission writing integrity reviews. Pay as you go with tokens for scans and guided revisions.',
    canonical: '/pricing',
    schemaType: 'WebPage',
  },
  '/privacy': {
    title: 'Privacy Policy | DraftProof',
    description: 'How DraftProof handles account data, documents, reports, payments, cookies, storage, and deletion requests.',
    canonical: '/privacy',
    schemaType: 'PrivacyPolicy',
  },
  '/security': {
    title: 'Security | DraftProof',
    description: 'How DraftProof protects academic documents with encrypted storage, OAuth sign-in, secure payments, and user-controlled deletion.',
    canonical: '/security',
    schemaType: 'WebPage',
  },
  '/signin': {
    title: 'Sign In | DraftProof',
    description: 'Sign in to DraftProof with Google or Microsoft to review drafts before submission.',
    canonical: '/signin',
    robots: 'noindex, nofollow',
    schemaType: 'WebPage',
  },
};

const PRIVATE_PREFIXES = ['/dashboard', '/scan', '/reports', '/report/', '/rewrite/', '/buy', '/history', '/auth/callback'];

export default function Seo() {
  const { pathname } = useLocation();

  useEffect(() => {
    const meta = PAGE_META[pathname] || privateMeta(pathname);
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
    setProperty('og:image:alt', 'DraftProof writing integrity review dashboard preview');

    setMeta('twitter:card', 'summary_large_image');
    setMeta('twitter:title', meta.title);
    setMeta('twitter:description', meta.description);
    setMeta('twitter:image', DEFAULT_IMAGE);

    setJsonLd(buildSchema(meta, canonicalUrl));
  }, [pathname]);

  return null;
}

function privateMeta(pathname) {
  const isPrivate = PRIVATE_PREFIXES.some((prefix) => pathname === prefix || pathname.startsWith(prefix));
  if (isPrivate) {
    return {
      title: 'DraftProof App',
      description: 'DraftProof account workspace for writing integrity reviews.',
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

function buildSchema(meta, url) {
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
        description: 'Pre-submission writing integrity review per 1,000 words',
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
