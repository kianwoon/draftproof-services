import { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import CodeTexture from '../components/CodeTexture';
import PageFreshness from '../components/PageFreshness';
import { getLocaleFromPathname, localizePath } from '../localeRouting';

function CellValue({ value }) {
  if (value === 'yes') return <span className="feat-yes" aria-label="yes">✓</span>;
  if (value === 'no') return <span className="feat-no" aria-label="no">✗</span>;
  return <span className="feat-partial" aria-label="partial">partial</span>;
}

export default function Features() {
  const { t } = useTranslation();
  const location = useLocation();
  const locale = getLocaleFromPathname(location.pathname);
  const publicPath = (path) => localizePath(path, locale);

  const competitors = t('featuresPage.competitors', { returnObjects: true });
  const rows = t('featuresPage.rows', { returnObjects: true });
  const tabKeys = ['scan', 'rewrite'];
  const [activeTab, setActiveTab] = useState('scan');
  const cardsByTab = {
    scan: t('featuresPage.scanCards', { returnObjects: true }),
    rewrite: t('featuresPage.rewriteCards', { returnObjects: true }),
  };
  const activeCards = Array.isArray(cardsByTab[activeTab]) ? cardsByTab[activeTab] : [];

  return (
    <main className="feat-shell">
      <div className="container">
        <section className="app-hero app-hero-dark">
          <CodeTexture id="featHero" />
          <div>
            <p className="eyebrow">{t('featuresPage.eyebrow')}</p>
            <h1>{t('featuresPage.title')}</h1>
            <p className="lead">{t('featuresPage.lead')}</p>
          </div>
        </section>

        <section style={{ marginTop: '2.5rem' }}>
          <p className="feat-section-label">{t('featuresPage.tableLabel')}</p>
          <div className="feat-table-wrap">
            <table className="feat-table">
              <thead>
                <tr>
                  <th />
                  {competitors.map((name, i) => (
                    <th key={name} className={i === 0 ? 'feat-th-dp' : undefined}>
                      {name}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.label}>
                    <td>{row.label}</td>
                    {row.values.map((val, i) => (
                      <td key={i} className={i === 0 ? 'feat-td-dp' : undefined}>
                        <CellValue value={val} />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section>
          <p className="feat-section-label">{t('featuresPage.cardsLabel')}</p>
          <div className="feat-tabs" role="tablist" aria-label={t('featuresPage.cardsLabel')}>
            {tabKeys.map((key) => (
              <button
                key={key}
                type="button"
                role="tab"
                id={`feat-tab-${key}`}
                aria-selected={activeTab === key}
                aria-controls="feat-tab-panel"
                className={`feat-tab${activeTab === key ? ' is-active' : ''}`}
                onClick={() => setActiveTab(key)}
              >
                {t(`featuresPage.tabs.${key}.label`)}
              </button>
            ))}
          </div>
          <p className="feat-tab-desc">{t(`featuresPage.tabs.${activeTab}.desc`)}</p>
          <div
            className="feat-cards"
            id="feat-tab-panel"
            role="tabpanel"
            aria-labelledby={`feat-tab-${activeTab}`}
          >
            {activeCards.map((card) => (
              <div className="feat-card" key={card.title}>
                <div className="feat-card-icon" aria-hidden="true">
                  <i className={`ti ${card.icon}`} />
                </div>
                <h3>{card.title}</h3>
                <p>{card.body}</p>
              </div>
            ))}
          </div>
          {activeTab === 'rewrite' && (
            <p style={{ textAlign: 'center', marginTop: '1.25rem', fontSize: '0.875rem' }}>
              <Link to={publicPath('/rewrite')}>{t('featuresPage.rewriteLearnMore')}</Link>
            </p>
          )}
        </section>

        <section className="why-cta">
          <h2>{t('featuresPage.ctaTitle')}</h2>
          <p>{t('featuresPage.ctaBody')}</p>
          <div className="hero-actions" style={{ justifyContent: 'center' }}>
            <Link to={publicPath('/signin')} className="btn btn-primary">
              {t('featuresPage.ctaButton')}
            </Link>
          </div>
        </section>

        <PageFreshness path="/features" />
      </div>
    </main>
  );
}
