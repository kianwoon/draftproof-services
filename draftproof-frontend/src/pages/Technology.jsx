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
            {pillar.diagram && <SignalFusionDiagram data={pillar.diagram} />}
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

function SignalFusionDiagram({ data }) {
  if (!data) return null;
  const chips = Array.isArray(data.chips) ? data.chips : [];

  return (
    <div className="fusion-diagram">
      <svg viewBox="0 0 460 170" width="100%" height="190" role="img" aria-label={`${data.signal1} + ${data.signal2} → ${data.fusedLabel} → ${data.bandLabel}`}>
        <rect x="10" y="14" width="150" height="48" rx="8" fill="none" stroke="var(--navy-900)" strokeWidth="1.5" />
        <text x="85" y="34" fontSize="12" fill="var(--ink)" textAnchor="middle" fontWeight="600">{data.signal1}</text>

        <rect x="10" y="104" width="150" height="48" rx="8" fill="none" stroke="var(--navy-900)" strokeWidth="1.5" />
        <text x="85" y="124" fontSize="12" fill="var(--ink)" textAnchor="middle" fontWeight="600">{data.signal2}</text>

        <line x1="160" y1="38" x2="220" y2="85" stroke="rgba(13, 27, 42, .25)" strokeWidth="1.5" />
        <line x1="160" y1="128" x2="220" y2="85" stroke="rgba(13, 27, 42, .25)" strokeWidth="1.5" />

        <circle cx="260" cy="85" r="46" fill="none" stroke="var(--gold)" strokeWidth="2" />
        <text x="260" y="82" fontSize="12" fill="var(--ink)" textAnchor="middle" fontWeight="600">{data.fusedLabel}</text>

        <line x1="306" y1="85" x2="352" y2="85" stroke="rgba(13, 27, 42, .25)" strokeWidth="1.5" markerEnd="url(#fusionArrow)" />

        <rect x="352" y="60" width="98" height="50" rx="8" fill="rgba(22, 163, 74, .12)" stroke="rgba(22, 163, 74, .35)" strokeWidth="1.5" />
        <circle cx="374" cy="85" r="6" fill="#15803d" />
        <text x="410" y="81" fontSize="12" fill="#15803d" textAnchor="middle" fontWeight="600">{data.bandLabel}</text>
        <text x="410" y="95" fontSize="9" fill="var(--muted)" textAnchor="middle">{data.bandCaption}</text>

        <defs>
          <marker id="fusionArrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
            <path d="M0,0 L8,4 L0,8 Z" fill="rgba(13, 27, 42, .45)" />
          </marker>
        </defs>
      </svg>

      <div className="fusion-diagram-chips">
        {chips.map((chip) => (
          <span className="fusion-diagram-chip" key={chip}>{chip}</span>
        ))}
      </div>
    </div>
  );
}
