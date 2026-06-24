// allow-hardcode: presentational React component — CSS class names + JSX structure, not a scoring/matching oracle. All human-facing text comes from i18n.
import { Link, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import CodeTexture from '../components/CodeTexture';
import PageFreshness from '../components/PageFreshness';
import DeclarationGenerator from '../components/DeclarationGenerator';
import { getLocaleFromPathname, localizePath } from '../localeRouting';

/**
 * Data-driven SEO landing page. Renders the same section flow as the
 * /content-checker page (EssayChecker), but driven entirely by an i18n
 * namespace so each keyword page is just a content file. See
 * docs/superpowers/specs/2026-06-24-seo-landing-pages-design.md
 */
export default function SeoLandingPage({ ns, path, generator = false }) {
  const { t } = useTranslation();
  const location = useLocation();
  const locale = getLocaleFromPathname(location.pathname);
  const publicPath = (p) => localizePath(p, locale);
  const k = (key) => `${ns}.${key}`;
  const asArray = (value) => (Array.isArray(value) ? value : []);

  const heroStat = t(k('heroStat'), { returnObjects: true });
  const intro = asArray(t(k('intro'), { returnObjects: true }));
  const sections = asArray(t(k('sections'), { returnObjects: true }));
  const links = asArray(t(k('links'), { returnObjects: true }));

  return (
    <main className="content-checker-shell seo-article-shell">
      <div className="container">
        <section className="app-hero app-hero-dark content-checker-hero">
          <CodeTexture id={`${ns}Hero`} />
          <div>
            <p className="eyebrow">{t(k('eyebrow'))}</p>
            <h1>{t(k('title'))}</h1>
            <p className="lead">{t(k('lead'))}</p>
            <div className="hero-actions">
              <Link to={publicPath('/signin?next=/scan')} className="btn btn-primary">{t(k('ctaPrimaryLabel'))}</Link>
              <Link to={publicPath('/content-checker')} className="btn btn-ghost">{t(k('ctaSecondaryLabel'))}</Link>
            </div>
          </div>
          {heroStat && heroStat.value && (
            <div className="app-hero-stat">
              <span>{heroStat.label}</span>
              <strong>{heroStat.value}</strong>
              <small>{heroStat.detail}</small>
            </div>
          )}
        </section>

        {generator && <DeclarationGenerator />}

        {intro.length > 0 && (
          <section className="content-checker-intro">
            {intro.map((item) => (
              <div key={item.title}>
                <h2>{item.title}</h2>
                <p>{item.body}</p>
              </div>
            ))}
          </section>
        )}

        {sections.map((section, si) => (
          <section className="content-checker-section" key={section.title || si}>
            {section.eyebrow && <p className="eyebrow">{section.eyebrow}</p>}
            <h2>{section.title}</h2>
            {section.lead && <p className="seo-section-lead">{section.lead}</p>}
            <div className={section.type === 'templates' ? 'seo-template-grid' : 'content-checker-grid'}>
              {asArray(section.items).map((item, ii) => {
                if (section.type === 'templates') {
                  return (
                    <article key={item.title || ii} className="seo-template-card">
                      <h3>{item.title}</h3>
                      <pre className="seo-template-text">{item.body}</pre>
                      {item.note && <p className="seo-template-note">{item.note}</p>}
                    </article>
                  );
                }
                return (
                  <article key={item.title || ii}>
                    {section.type === 'steps' && <span>{String(ii + 1).padStart(2, '0')}</span>}
                    <h3>{item.title}</h3>
                    <p>{item.body}</p>
                  </article>
                );
              })}
            </div>
          </section>
        ))}

        {links.length > 0 && (
          <section className="content-checker-section seo-related">
            <p className="eyebrow">{t(k('linksEyebrow'))}</p>
            <h2>{t(k('linksTitle'))}</h2>
            <div className="seo-related-grid">
              {links.map((link) => (
                <Link key={link.to} to={publicPath(link.to)} className="seo-related-link">
                  <strong>{link.label}</strong>
                  <span>{link.body}</span>
                </Link>
              ))}
            </div>
          </section>
        )}

        <section className="content-checker-cta">
          <h2>{t(k('ctaTitle'))}</h2>
          <p>{t(k('ctaBody'))}</p>
          <Link to={publicPath('/signin?next=/scan')} className="btn btn-primary">{t(k('ctaPrimaryLabel'))}</Link>
        </section>

        <PageFreshness path={path} />
      </div>
    </main>
  );
}
