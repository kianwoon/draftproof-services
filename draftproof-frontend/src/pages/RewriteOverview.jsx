import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import CodeTexture from '../components/CodeTexture';
import PageFreshness from '../components/PageFreshness';
import { RewriteBeforeAfter } from '../components/RewriteBeforeAfter';

export default function RewriteOverview() {
  const { t } = useTranslation();
  const examples = t('rewriteOverview.examples', { returnObjects: true });
  const beforeLabel = t('rewriteOverview.beforeLabel');
  const afterLabel = t('rewriteOverview.afterLabel');
  const valueCards = t('featuresPage.rewriteCards', { returnObjects: true });

  return (
    <main className="why-shell">
      <div className="container">
        <section className="why-hero app-hero app-hero-dark">
          <CodeTexture id="rewriteHero" />
          <div>
            <p className="eyebrow">{t('rewriteOverview.eyebrow')}</p>
            <h1>{t('rewriteOverview.title')}</h1>
            <p className="lead">{t('rewriteOverview.lead')}</p>
          </div>
        </section>

        <section className="why-section" aria-label={t('rewriteOverview.demoHeading')}>
          <h2>{t('rewriteOverview.demoHeading')}</h2>
          {(Array.isArray(examples) ? examples : []).map((ex) => (
            <div key={ex.fixType} style={{ marginBottom: '1.25rem' }}>
              <p className="why-highlight">{ex.fixType}</p>
              <RewriteBeforeAfter
                before={ex.before}
                after={ex.after}
                marker={ex.marker}
                beforeLabel={beforeLabel}
                afterLabel={afterLabel}
              />
            </div>
          ))}
        </section>

        <section className="why-section">
          <h2>{t('rewriteFraming.title')}</h2>
          <p>{t('rewriteFraming.isCopy')}</p>
          <p className="why-quote">{t('rewriteFraming.isntCopy')}</p>
          <p className="why-punch">{t('rewriteFraming.action')}</p>
        </section>

        <section className="why-section">
          <div className="why-card-grid">
            {(Array.isArray(valueCards) ? valueCards : []).map((card) => (
              <article className="why-card" key={card.title}>
                <h3>{card.title}</h3>
                <p>{card.body}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="why-cta">
          <h2>{t('rewriteOverview.ctaTitle')}</h2>
          <p>{t('rewriteOverview.ctaBody')}</p>
          <div className="hero-actions" style={{ justifyContent: 'center' }}>
            <Link to="/signin?next=/scan" className="btn btn-primary">{t('rewriteOverview.ctaButton')}</Link>
          </div>
        </section>

        <PageFreshness path="/rewrite" />
      </div>
    </main>
  );
}
