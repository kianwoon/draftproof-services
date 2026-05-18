import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import CodeTexture from '../components/CodeTexture';

export default function Landing() {
  const { t } = useTranslation();
  const checks = t('landing.checks', { returnObjects: true });
  const whyCards = t('landing.whyCards', { returnObjects: true });
  const contentStrategies = t('landing.contentStrategies', { returnObjects: true });
  const beliefs = t('landing.beliefs', { returnObjects: true });

  return (
    <main className="landing-page">
      <section id="hero" className="landing-hero">
        <CodeTexture id="landingHero" className="hero-code-field" />
        <div className="section-inner landing-hero-grid">
          <div className="hero-copy">
            <p className="brand-pill">{t('landing.heroPill')}</p>
            <h1>
              {t('landing.heroTitle').replace(t('landing.heroTitleHighlight'), '')}<span>{t('landing.heroTitleHighlight')}</span>
            </h1>
            <p className="lead">{t('landing.heroLead')}</p>

            <div className="hero-actions" id="check">
              <Link to="/scan" className="btn btn-ghost">{t('landing.runCheck')}</Link>
              <a href="#report" className="btn btn-ghost">{t('landing.viewSample')}</a>
            </div>

            <div className="trust-note">
              <span className="mini-shield" aria-hidden="true" />
              <span>{t('landing.trustNote')}</span>
            </div>
          </div>

          <aside className="review-panel" aria-label={t('landing.quickSummary')}>
            <div className="review-panel-top">
              <p className="card-kicker">{t('landing.quickSummary')}</p>
              <span className="live-dot">{t('landing.livePreview')}</span>
            </div>
            <h2>{t('landing.preSubmissionReview')}</h2>
            <p>{t('landing.runningCheck')}</p>

            <div className="review-grid">
              <Metric label={t('landing.reviewTier')} value={t('landing.medium')} tone="warning" width="50%" />
              <Metric label={t('landing.grounding')} value={t('landing.strong')} tone="positive" width="82%" />
              <Metric label={t('landing.citationGaps')} value={t('landing.foundCount', { count: 2 })} tone="warning" width="35%" />
              <Metric label={t('landing.sourceIntegrity')} value={t('landing.verified')} tone="positive" width="92%" />
            </div>

            <div className="primary-fix">
              <div>
                <span>{t('landing.primaryFix')}</span>
                <strong>{t('landing.oneCitation')}</strong>
              </div>
              <em>{t('landing.actionable')}</em>
            </div>
          </aside>
        </div>
      </section>

      <section className="trust-bar" aria-label={t('landing.audienceDetails')}>
        <div className="section-inner trust-bar-inner">
          <span>{t('landing.builtFor')}</span>
          <strong>{t('landing.students')}</strong>
          <strong>{t('landing.researchers')}</strong>
          <strong>{t('landing.educators')}</strong>
          <strong>{t('landing.policyWriters')}</strong>
          <span>{t('landing.tokenRate')}</span>
          <span>{t('landing.pdfReport')}</span>
          <span>{t('landing.citationGrounding')}</span>
          <span>{t('landing.contentAwareRewrite')}</span>
        </div>
      </section>

      <section id="report" className="landing-section sample-report-section">
        <div className="section-inner sample-report-layout">
          <div className="sample-report-copy">
            <p className="eyebrow">{t('landing.sampleEyebrow')}</p>
            <h2>{t('landing.sampleTitle')}</h2>
            <p>{t('landing.sampleBody')}</p>
            <div className="sample-report-points">
              <span>{t('landing.samplePoint1')}</span>
              <span>{t('landing.samplePoint2')}</span>
              <span>{t('landing.samplePoint3')}</span>
            </div>
            <Link to="/scan" className="btn btn-primary">{t('landing.runOwnScan')}</Link>
          </div>

          <SampleReportPreview />
        </div>
      </section>

      <section id="product" className="landing-section">
        <div className="section-inner">
          <p className="eyebrow">{t('landing.whyEyebrow')}</p>
          <h2>{t('landing.whyTitle')}</h2>
          <div className="why-card-grid">
            {whyCards.map((card, index) => (
              <article className="why-card" key={card.title}>
                <span>{String(index + 1).padStart(2, '0')}</span>
                <h3>{card.title}</h3>
                <p>{card.body}</p>
                <small>{card.note}</small>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section id="strategies" className="landing-section strategy-section">
        <div className="section-inner">
          <div className="strategy-heading">
            <div>
              <p className="eyebrow">{t('landing.strategyEyebrow')}</p>
              <h2>{t('landing.strategyTitle')}</h2>
            </div>
            <p>{t('landing.strategyBody')}</p>
          </div>

          <div className="strategy-grid" aria-label={t('landing.contentAwareRewrite')}>
            {contentStrategies.map((item) => (
              <article className="strategy-card" key={item.type}>
                <span>{item.type}</span>
                <h3>{item.strategy}</h3>
                <p>{item.detail}</p>
              </article>
            ))}
          </div>

          <div className="strategy-proof">
            <strong>{t('landing.strategyProofStrong')}</strong>
            <span>{t('landing.strategyProofBody')}</span>
          </div>
        </div>
      </section>

      <section id="engine" className="landing-section checks-section">
        <CodeTexture id="landingChecks" />
        <div className="section-inner">
          <p className="eyebrow">{t('landing.engineEyebrow')}</p>
          <h2>{t('landing.engineTitle')}</h2>
          <p className="section-lead">{t('landing.engineLead')}</p>
          <div className="check-line">
            {checks.map((check, index) => (
              <article className="check-step" key={check.title}>
                <span>{index + 1}</span>
                <h3>{check.title}</h3>
                <p>{check.body}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section id="trust" className="landing-section beliefs-section">
        <div className="section-inner">
          <p className="eyebrow">{t('landing.beliefsEyebrow')}</p>
          <h2>{t('landing.beliefsTitle')}</h2>
          <div className="belief-row-grid">
            {beliefs.map((belief) => (
              <div className="belief-row" key={belief}>
                <span aria-hidden="true">×</span>
                {belief}
              </div>
            ))}
            <div className="belief-row belief-row-positive">
              <span aria-hidden="true">✓</span>
              {t('landing.positiveBelief')}
            </div>
          </div>
        </div>
      </section>

      <section id="cta" className="landing-cta">
        <CodeTexture id="landingCta" />
        <div className="section-inner">
          <p className="brand-pill">{t('landing.ctaPill')}</p>
          <h2>{t('landing.ctaTitle')}</h2>
          <p>{t('landing.ctaBody')}</p>
          <Link to="/scan" className="btn btn-ghost">{t('landing.ctaButton')}</Link>
          <small>{t('landing.ctaSmall')}</small>
        </div>
      </section>

      <footer className="landing-footer">
        <div className="section-inner landing-footer-inner">
          <div>
            <Link to="/" className="footer-wordmark">DraftProof</Link>
            <p>{t('footer.disclaimer')}</p>
          </div>
          <nav aria-label={t('footer.product')}>
            <a href="#product">{t('footer.product')}</a>
            <a href="#engine">{t('footer.howItWorks')}</a>
            <a href="#report">{t('footer.sampleReport')}</a>
            <Link to="/pricing">{t('footer.pricing')}</Link>
            <Link to="/privacy">{t('footer.privacy')}</Link>
            <Link to="/security">{t('footer.security')}</Link>
          </nav>
        </div>
      </footer>
    </main>
  );
}

function SampleReportPreview() {
  const { t } = useTranslation();
  const sampleReportStats = t('landing.sampleStats', { returnObjects: true });
  const sampleReportNotes = t('landing.sampleReportNotes', { returnObjects: true });

  return (
    <article className="sample-report-preview" aria-label={t('landing.reportPreviewLabel')}>
      <div className="sample-report-stats">
        {sampleReportStats.map((stat) => (
          <div className={`sample-report-stat${stat.tone ? ` is-${stat.tone}` : ''}`} key={stat.label}>
            <strong>{stat.value}</strong>
            <span>{stat.label}</span>
          </div>
        ))}
      </div>

      <div className="sample-report-pattern">
        <div>
          <span>{t('landing.transformationPattern')}</span>
          <h3>{t('landing.humanUncertain')}</h3>
          <div className="sample-report-tags">
            <em>{t('landing.lowConfidence')}</em>
            <em>{t('landing.notVerdict')}</em>
          </div>
        </div>
        <div className="sample-authorship-badge">
          <span>{t('landing.authorshipRating')}</span>
          <strong>{t('landing.good')}</strong>
          <small>{t('landing.calibratedRisk')}</small>
        </div>
      </div>

      <div className="sample-contribution">
        <span>{t('landing.estimatedContribution')}</span>
        <p>{t('landing.contributionBody')}</p>
        <div className="sample-contribution-bars">
          <SampleSignalBar label={t('landing.humanContribution')} value={100} tone="human" />
          <SampleSignalBar label={t('landing.aiTransformation')} value={0} tone="ai" />
        </div>
      </div>

      <div className="sample-report-notes">
        {sampleReportNotes.map((note) => (
          <span key={note}>{note}</span>
        ))}
      </div>
    </article>
  );
}

function SampleSignalBar({ label, value, tone }) {
  return (
    <div className="sample-signal-row">
      <div className="sample-signal-row-label">
        <span>{label}</span>
        <strong>{value}%</strong>
      </div>
      <div className="sample-signal-track">
        <i className={`is-${tone}`} style={{ width: `${value}%` }} />
      </div>
    </div>
  );
}

function Metric({ label, value, tone, width }) {
  return (
    <div className="review-metric">
      <span>{label}</span>
      <strong className={tone === 'positive' ? 'tier-low' : 'tier-medium'}>{value}</strong>
      <div className={`review-bar ${tone}`}>
        <i style={{ width }} />
      </div>
    </div>
  );
}
