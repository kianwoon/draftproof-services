import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useAuth } from '../context/AuthContext';
import CodeTexture from '../components/CodeTexture';
import PageFreshness from '../components/PageFreshness';

export default function Pricing() {
  const { user } = useAuth();
  const { t } = useTranslation();
  const scanFeatures = t('pricing.scanFeatures', { returnObjects: true });
  const rewriteFeatures = t('pricing.rewriteFeatures', { returnObjects: true });
  const faqs = t('pricing.faqs', { returnObjects: true });

  return (
    <main className="pricing-shell">
      <div className="container">
        <section className="pricing-hero app-hero app-hero-dark">
          <CodeTexture id="pricingHero" />
          <div>
          <p className="eyebrow">{t('pricing.eyebrow')}</p>
          <h1>{t('pricing.title')}</h1>
          <p className="pricing-lead">{t('pricing.lead')}</p>
          </div>
          <div className="app-hero-stat">
            <span>{t('pricing.baseRate')}</span>
            <strong>$0.90</strong>
            <small>{t('pricing.perWords')}</small>
          </div>
        </section>

        <div className="pricing-grid">
        <div className="pricing-card">
          <div className="pricing-card-header">
            <h2>{t('pricing.scanTitle')}</h2>
            <div className="pricing-amount">
              <span className="pricing-currency">$</span>
              <span className="pricing-value">0.90</span>
              <span className="pricing-unit">{t('pricing.scanUnit')}</span>
            </div>
          </div>

          <ul className="pricing-features">
            {scanFeatures.map((feature) => (
              <li key={feature}>
                <span className="check" aria-hidden="true">&#10003;</span>
                <span>{feature}</span>
              </li>
            ))}
          </ul>

          <Link
            to={user ? '/scan' : '/signin'}
            className="btn btn-primary pricing-cta"
          >
            {user ? t('pricing.startScan') : t('pricing.signInStart')}
          </Link>
        </div>

        <div className="pricing-card pricing-card--rewrite">
          <div className="pricing-card-header">
            <h2>{t('pricing.rewriteTitle')}</h2>
            <div className="pricing-amount">
              <span className="pricing-currency">$</span>
              <span className="pricing-value">4.50</span>
              <span className="pricing-unit">{t('pricing.rewriteUnit')}</span>
            </div>
          </div>

          <ul className="pricing-features">
            {rewriteFeatures.map((feature) => (
              <li key={feature}>
                <span className="check" aria-hidden="true">&#10003;</span>
                <span>{feature}</span>
              </li>
            ))}
          </ul>

          <Link
            to={user ? '/scan' : '/signin'}
            className="btn btn-primary pricing-cta"
          >
            {user ? t('pricing.startWithScan') : t('pricing.signInStart')}
          </Link>
        </div>
        </div>

        <section className="pricing-faq">
          <h2>{t('pricing.faqTitle')}</h2>
          {faqs.map((faq) => (
            <div className="faq-item" key={faq.q}>
              <h3>{faq.q}</h3>
              <p>{faq.a}</p>
            </div>
          ))}
        </section>

        <PageFreshness path="/pricing" />
      </div>
    </main>
  );
}
