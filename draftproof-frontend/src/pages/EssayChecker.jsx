import { Link, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import CodeTexture from '../components/CodeTexture';
import PageFreshness from '../components/PageFreshness';
import { getLocaleFromPathname, localizePath } from '../localeRouting';

export default function EssayChecker() {
  const { t } = useTranslation();
  const location = useLocation();
  const locale = getLocaleFromPathname(location.pathname);
  const publicPath = (path) => localizePath(path, locale);
  const checks = t('essayChecker.checks', { returnObjects: true });
  const steps = t('essayChecker.steps', { returnObjects: true });
  const related = t('essayChecker.related', { returnObjects: true });

  return (
    <main className="content-checker-shell">
      <div className="container">
        <section className="app-hero app-hero-dark content-checker-hero">
          <CodeTexture id="essayCheckerHero" />
          <div>
            <p className="eyebrow">{t('essayChecker.eyebrow')}</p>
            <h1>{t('essayChecker.title')}</h1>
            <p className="lead">{t('essayChecker.lead')}</p>
            <div className="hero-actions">
              <Link to={publicPath('/signin?next=/scan')} className="btn btn-primary">{t('essayChecker.startReview')}</Link>
              <Link to={publicPath('/#report')} className="btn btn-ghost">{t('essayChecker.viewSample')}</Link>
            </div>
          </div>
          <div className="app-hero-stat">
            <span>{t('essayChecker.heroStatLabel')}</span>
            <strong>{t('essayChecker.heroStatValue')}</strong>
            <small>{t('essayChecker.heroStatDetail')}</small>
          </div>
        </section>

        <section className="content-checker-intro">
          <div>
            <h2>{t('essayChecker.problemTitle')}</h2>
            <p>{t('essayChecker.problemBody')}</p>
          </div>
          <div>
            <h2>{t('essayChecker.positionTitle')}</h2>
            <p>{t('essayChecker.positionBody')}</p>
          </div>
        </section>

        <section className="content-checker-section">
          <p className="eyebrow">{t('essayChecker.checksEyebrow')}</p>
          <h2>{t('essayChecker.checksTitle')}</h2>
          <div className="content-checker-grid">
            {checks.map((check) => (
              <article key={check.title}>
                <h3>{check.title}</h3>
                <p>{check.body}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="content-checker-section content-checker-steps">
          <p className="eyebrow">{t('essayChecker.stepsEyebrow')}</p>
          <h2>{t('essayChecker.stepsTitle')}</h2>
          <div className="content-checker-grid">
            {steps.map((step, index) => (
              <article key={step.title}>
                <span>{String(index + 1).padStart(2, '0')}</span>
                <h3>{step.title}</h3>
                <p>{step.body}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="content-checker-cta">
          <h2>{t('essayChecker.ctaTitle')}</h2>
          <p>{t('essayChecker.ctaBody')}</p>
          <Link to={publicPath('/signin?next=/scan')} className="btn btn-primary">{t('essayChecker.startReview')}</Link>
        </section>

        {Array.isArray(related) && related.length > 0 && (
          <section className="content-checker-section content-checker-related">
            <p className="eyebrow">{t('essayChecker.relatedEyebrow')}</p>
            <h2>{t('essayChecker.relatedTitle')}</h2>
            <div className="content-checker-grid">
              {related.map((link) => (
                <article key={link.to}>
                  <Link to={publicPath(link.to)}>
                    <h3>{link.label}</h3>
                    <p>{link.body}</p>
                  </Link>
                </article>
              ))}
            </div>
          </section>
        )}

        <PageFreshness path="/content-checker" />
      </div>
    </main>
  );
}
