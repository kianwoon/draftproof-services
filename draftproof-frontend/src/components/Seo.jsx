import { useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import {
  DEFAULT_IMAGE,
  buildSchema,
  getCanonicalUrl,
  getRobots,
  getSeoMeta,
} from '../seoMetadata';

export default function Seo() {
  const { pathname } = useLocation();
  const { t, i18n } = useTranslation();

  useEffect(() => {
    const meta = getSeoMeta(pathname, t);
    const canonicalUrl = getCanonicalUrl(meta);

    document.title = meta.title;
    setMeta('description', meta.description);
    setMeta('robots', getRobots(meta));
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
