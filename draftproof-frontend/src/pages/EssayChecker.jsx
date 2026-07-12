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
            {checks.map((check, index) => (
              <article key={check.title}>
                <CheckIcon index={index} />
                <h3>{check.title}</h3>
                <p>{check.body}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="content-checker-section content-checker-steps">
          <p className="eyebrow">{t('essayChecker.stepsEyebrow')}</p>
          <h2>{t('essayChecker.stepsTitle')}</h2>
          <ChecksFlow steps={steps} />
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

// Minimal line-icon set for the "what it checks" grid, indexed positionally to
// essayChecker.checks (same order in every locale) so translation files never
// need an icon key. Falls back to the last icon if a locale ever adds more
// than 6 checks, rather than rendering undefined.
const CHECK_ICON_PATHS = [
  // Citation gaps — a broken link
  <g key="citation">
    <path d="M8.5 15.5l-1.6 1.6a2.5 2.5 0 0 1-3.5-3.5l2.6-2.6" />
    <path d="M15.5 8.5l1.6-1.6a2.5 2.5 0 0 0-3.5-3.5l-2.6 2.6" />
    <path d="M9.5 14.5l5-5" strokeDasharray="1.6 2.2" />
  </g>,
  // Source grounding — overlapping circles
  <g key="grounding">
    <circle cx="9" cy="12" r="5" />
    <circle cx="15" cy="12" r="5" />
  </g>,
  // AI-like writing signals — sparkle
  <path key="ai-signal" d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z" />,
  // Similarity risk — overlapping layers
  <g key="similarity">
    <rect x="4" y="4" width="12" height="12" rx="2" />
    <rect x="8" y="8" width="12" height="12" rx="2" />
  </g>,
  // Author reasoning — pencil
  <g key="reasoning">
    <path d="M4 20l4-1 10-10-3-3L5 16z" />
    <path d="M14 6l3 3" />
  </g>,
  // PDF record — file with a checkmark
  <g key="pdf">
    <path d="M6 3h9l3 3v15H6z" />
    <path d="M9 13l2 2 4-4" />
  </g>,
];

function CheckIcon({ index }) {
  const path = CHECK_ICON_PATHS[index] ?? CHECK_ICON_PATHS[CHECK_ICON_PATHS.length - 1];
  return (
    <span className="content-checker-check-icon" aria-hidden="true">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
        {path}
      </svg>
    </span>
  );
}

// Icons for the 3-stage flow strip, indexed positionally to essayChecker.steps.
const FLOW_ICON_PATHS = [
  // Scan — magnifying glass
  <g key="scan">
    <circle cx="10" cy="10" r="6" />
    <path d="M20 20l-5.5-5.5" />
  </g>,
  // Report — a checklist
  <g key="report">
    <rect x="6" y="4" width="12" height="16" rx="2" />
    <path d="M9 9h6M9 13h6M9 17h3" />
  </g>,
  // Revise — pencil
  <g key="revise">
    <path d="M4 20l4-1 10-10-3-3L5 16z" />
    <path d="M14 6l3 3" />
  </g>,
];

// A quick-scan visual summary of the 3 steps below it (Scan → Report → Revise),
// rendered as plain HTML/CSS rather than SVG text so labels wrap naturally in
// every locale — the detailed 3-card grid underneath keeps all step body copy
// unchanged; this strip only adds a glanceable diagram on top of it.
function ChecksFlow({ steps }) {
  if (!Array.isArray(steps) || steps.length === 0) return null;

  return (
    <div className="content-checker-flow" role="list" aria-label="Workflow steps">
      {steps.map((step, index) => (
        <div className="content-checker-flow-item" key={step.title}>
          <div className="content-checker-flow-node" role="listitem">
            <span className="content-checker-flow-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
                {FLOW_ICON_PATHS[index] ?? FLOW_ICON_PATHS[FLOW_ICON_PATHS.length - 1]}
              </svg>
            </span>
            <span className="content-checker-flow-label">{step.title}</span>
          </div>
          {index < steps.length - 1 && (
            <span className="content-checker-flow-arrow" aria-hidden="true">&#8594;</span>
          )}
        </div>
      ))}
    </div>
  );
}
