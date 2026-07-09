import { useMemo, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import CodeTexture from '../components/CodeTexture';
import PageFreshness from '../components/PageFreshness';
import { getLocaleFromPathname, localizePath } from '../localeRouting';

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function highlight(text, query) {
  if (!query) return text;
  const parts = String(text).split(new RegExp(`(${escapeRegExp(query)})`, 'ig'));
  return parts.map((part, index) => (
    part.toLowerCase() === query.toLowerCase()
      ? <mark key={index}>{part}</mark>
      : part
  ));
}

export default function FAQ() {
  const { t } = useTranslation();
  const location = useLocation();
  const locale = getLocaleFromPathname(location.pathname);
  const publicPath = (path) => localizePath(path, locale);
  const groups = t('faqPage.groups', { returnObjects: true });
  const relatedLinks = t('faqPage.related', { returnObjects: true });
  const totalQuestions = groups.reduce((count, group) => count + group.items.length, 0);
  const featuredGroups = groups.slice(0, 3);

  const [query, setQuery] = useState('');
  const term = query.trim();
  const searching = term.length > 0;

  const filteredGroups = useMemo(() => {
    if (!searching) return groups;
    const needle = term.toLowerCase();
    return groups
      .map((group) => {
        // Include the category label so natural searches like "privacy" or
        // "billing" surface the whole group even when no Q/A repeats the word.
        const groupText = `${group.kicker} ${group.title}`.toLowerCase();
        return {
          ...group,
          items: group.items.filter((item) => (
            `${groupText} ${item.q} ${item.a}`.toLowerCase().includes(needle)
          )),
        };
      })
      .filter((group) => group.items.length > 0);
  }, [groups, searching, term]);

  const matchCount = filteredGroups.reduce((count, group) => count + group.items.length, 0);
  const navGroups = searching ? filteredGroups : groups;

  return (
    <main className="faq-shell">
      <div className="container">
        <section className="faq-hero app-hero app-hero-dark">
          <CodeTexture id="faqHero" />
          <div>
            <p className="eyebrow">{t('faqPage.eyebrow')}</p>
            <h1>{t('faqPage.title')}</h1>
            <p className="lead">{t('faqPage.lead')}</p>
          </div>
          <div className="app-hero-stat">
            <span>{t('faqPage.heroStatLabel')}</span>
            <strong>{t('faqPage.heroStatValue')}</strong>
            <small>{t('faqPage.heroStatDetail')}</small>
          </div>
        </section>

        <section className="faq-overview" aria-label={t('faqPage.overviewLabel')}>
          <article>
            <span>{t('faqPage.questionCountLabel')}</span>
            <strong>{totalQuestions}</strong>
            <p>{t('faqPage.questionCountDetail')}</p>
          </article>
          {featuredGroups.map((group) => (
            <a href={`#${group.id}`} key={group.id}>
              <span>{group.kicker}</span>
              <strong>{group.title}</strong>
            </a>
          ))}
        </section>

        <section className="faq-search" aria-label={t('faqPage.searchLabel')}>
          <label className="faq-search-field">
            <svg viewBox="0 0 20 20" aria-hidden="true" focusable="false">
              <path
                d="M9 3a6 6 0 1 0 3.7 10.7l3.3 3.3 1.4-1.4-3.3-3.3A6 6 0 0 0 9 3Zm0 2a4 4 0 1 1 0 8 4 4 0 0 1 0-8Z"
                fill="currentColor"
              />
            </svg>
            <input
              type="search"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder={t('faqPage.searchPlaceholder')}
              aria-label={t('faqPage.searchLabel')}
              autoComplete="off"
            />
            {searching && (
              <button
                type="button"
                className="faq-search-clear"
                onClick={() => setQuery('')}
                aria-label={t('faqPage.searchClear')}
              >
                &times;
              </button>
            )}
          </label>
          <p className="faq-search-status" role="status" aria-live="polite">
            {searching
              ? t('faqPage.searchSummary', { matches: matchCount, total: totalQuestions })
              : ''}
          </p>
        </section>

        <section className="faq-intro">
          <div>
            <h2>{t('faqPage.helpTitle')}</h2>
            <p>{t('faqPage.helpBody')}</p>
          </div>
          <div className="faq-intro-actions">
            <Link to="/signin?next=/scan" className="btn btn-primary">{t('faqPage.startScan')}</Link>
            <Link to={publicPath('/#report')} className="btn btn-secondary">{t('faqPage.viewSample')}</Link>
          </div>
        </section>

        <div className="faq-layout">
          <aside className="faq-nav" aria-label={t('faqPage.navLabel')}>
            {navGroups.map((group) => (
              <a key={group.id} href={`#${group.id}`}>
                <span>{group.kicker}</span>
                {group.title}
              </a>
            ))}
          </aside>

          <div className="faq-groups">
            {filteredGroups.map((group, groupIndex) => (
              <section className="faq-group" id={group.id} key={group.id}>
                <div className="faq-group-head">
                  <div>
                    <span>{group.kicker}</span>
                    <h2>{group.title}</h2>
                  </div>
                  <strong>{String(groupIndex + 1).padStart(2, '0')}</strong>
                </div>
                <div className="faq-question-list">
                  {group.items.map((item) => (
                    <details className="faq-question" key={item.q} open={searching || undefined}>
                      <summary>{highlight(item.q, term)}</summary>
                      <p>{highlight(item.a, term)}</p>
                    </details>
                  ))}
                </div>
              </section>
            ))}

            {searching && filteredGroups.length === 0 && (
              <div className="faq-empty">
                <strong>{t('faqPage.searchNoResults', { query: term })}</strong>
                <p>{t('faqPage.searchNoResultsHint')}</p>
                <button type="button" className="btn btn-secondary" onClick={() => setQuery('')}>
                  {t('faqPage.searchClear')}
                </button>
              </div>
            )}
          </div>
        </div>

        {Array.isArray(relatedLinks) && relatedLinks.length > 0 && (
          <section className="content-checker-section seo-related">
            <p className="eyebrow">{t('faqPage.relatedEyebrow')}</p>
            <h2>{t('faqPage.relatedTitle')}</h2>
            <div className="seo-related-grid">
              {relatedLinks.map((link) => (
                <Link key={link.to} to={publicPath(link.to)} className="seo-related-link">
                  <strong>{link.label}</strong>
                  <span>{link.body}</span>
                </Link>
              ))}
            </div>
          </section>
        )}

        <section className="faq-cta">
          <h2>{t('faqPage.ctaTitle')}</h2>
          <p>{t('faqPage.ctaBody')}</p>
          <div className="hero-actions">
            <Link to="/signin?next=/scan" className="btn btn-primary">{t('faqPage.startScan')}</Link>
            <Link to={publicPath('/pricing')} className="btn btn-secondary">{t('faqPage.viewPricing')}</Link>
          </div>
        </section>

        <PageFreshness path="/faq" />
      </div>
    </main>
  );
}
