import { Link, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import CodeTexture from '../components/CodeTexture';
import PageFreshness from '../components/PageFreshness';
import { getLocaleFromPathname, localizePath } from '../localeRouting';

export default function Technology() {
  const { t } = useTranslation();
  const location = useLocation();
  const locale = getLocaleFromPathname(location.pathname);
  const publicPath = (path) => localizePath(path, locale);
  const pillars = t('technologyPage.pillars', { returnObjects: true });

  return (
    <main className="why-shell">
      <div className="container">
        <section className="why-hero app-hero app-hero-dark">
          <CodeTexture id="technologyHero" />
          <div>
            <p className="eyebrow">{t('technologyPage.eyebrow')}</p>
            <h1>{t('technologyPage.title')}</h1>
            <p className="lead">{t('technologyPage.lead')}</p>
          </div>
        </section>

        {pillars.map((pillar, index) => (
          <section className="why-section" key={pillar.title} aria-label={t('technologyPage.pillarsLabel')}>
            <span className="why-num">{String(index + 1).padStart(2, '0')}</span>
            <h2>{pillar.title}</h2>
            <p>{pillar.body}</p>
            <p className="why-highlight">{pillar.whyItMatters}</p>
          </section>
        ))}

        <section className="why-cta">
          <h2>{t('technologyPage.ctaTitle')}</h2>
          <p>{t('technologyPage.ctaBody')}</p>
          <div className="hero-actions" style={{ justifyContent: 'center' }}>
            <Link to="/signin?next=/scan" className="btn btn-primary">{t('technologyPage.ctaRun')}</Link>
            <Link to={publicPath('/#report')} className="btn btn-secondary">{t('technologyPage.ctaHow')}</Link>
          </div>
        </section>

        <PageFreshness path="/technology" />
      </div>
    </main>
  );
}
