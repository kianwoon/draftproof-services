import { Link, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import CodeTexture from '../components/CodeTexture';
import PageFreshness from '../components/PageFreshness';
import { getLocaleFromPathname, localizePath } from '../localeRouting';

export default function Why() {
  const { t } = useTranslation();
  const location = useLocation();
  const locale = getLocaleFromPathname(location.pathname);
  const publicPath = (path) => localizePath(path, locale);
  const sections = t('whyPage.sections', { returnObjects: true });
  const beliefs = t('whyPage.beliefs', { returnObjects: true });

  return (
    <main className="why-shell">
      <div className="container">
        <section className="why-hero app-hero app-hero-dark">
          <CodeTexture id="whyHero" />
          <div>
            <p className="eyebrow">{t('whyPage.eyebrow')}</p>
            <h1>{t('whyPage.title')}</h1>
            <p className="lead">{t('whyPage.lead1')}</p>
            <p className="lead">{t('whyPage.lead2')}</p>
            <p className="why-punch">{t('whyPage.punch')}</p>
          </div>
        </section>

        {sections.map((section, index) => (
          <section className="why-section" key={section.title}>
            <span className="why-num">{String(index + 1).padStart(2, '0')}</span>
            <h2>{section.title}</h2>
            {section.paragraphs?.map((paragraph) => (
              <p className={paragraph.startsWith('"') || paragraph.startsWith('“') ? 'why-quote' : undefined} key={paragraph}>
                {paragraph}
              </p>
            ))}
            {section.list && (
              <ul className="why-list">
                {section.list.map((item) => <li key={item}>{item}</li>)}
              </ul>
            )}
            {section.highlight && <p className="why-highlight">{section.highlight}</p>}
            {section.punch && <p className="why-punch">{section.punch}</p>}
          </section>
        ))}

        <section className="why-section why-beliefs">
          <span className="why-num">{String(sections.length + 1).padStart(2, '0')}</span>
          <h2>{t('whyPage.beliefsTitle')}</h2>
          <p>{t('whyPage.beliefsIntro')}</p>
          <ul className="why-beliefs-list">
            {beliefs.map((belief) => <li key={belief}>{belief}</li>)}
          </ul>
          <p className="why-highlight">{t('whyPage.beliefsHighlight')}</p>
        </section>

        <section className="why-cta">
          <h2>{t('whyPage.ctaTitle')}</h2>
          <p>{t('whyPage.ctaBody')}</p>
          <p className="why-highlight">{t('whyPage.ctaHighlight')}</p>
          <div className="hero-actions" style={{ justifyContent: 'center' }}>
            <Link to="/signin?next=/scan" className="btn btn-primary">{t('whyPage.ctaRun')}</Link>
            <Link to={publicPath('/technology')} className="btn btn-secondary">{t('whyPage.ctaHow')}</Link>
          </div>
        </section>

        <PageFreshness path="/why" />
      </div>
    </main>
  );
}
