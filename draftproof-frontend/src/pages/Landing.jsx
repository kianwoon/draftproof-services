import { useEffect, useMemo, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import CodeTexture from '../components/CodeTexture';
import PageFreshness from '../components/PageFreshness';
import { getLocaleFromPathname, localizePath } from '../localeRouting';

export default function Landing() {
  const { t } = useTranslation();
  const location = useLocation();
  const locale = getLocaleFromPathname(location.pathname);
  const publicPath = (path) => localizePath(path, locale);
  const checks = t('landing.checks', { returnObjects: true });
  const whyCards = t('landing.whyCards', { returnObjects: true });
  const helpCards = t('landing.helpCards', { returnObjects: true });
  const contentStrategies = t('landing.contentStrategies', { returnObjects: true });
  const beliefs = t('landing.beliefs', { returnObjects: true });
  const reportValueCards = t('landing.reportValueCards', { returnObjects: true });
  const humanWrittenSignals = t('landing.humanWrittenSignals', { returnObjects: true });
  const humanizerSignals = t('landing.humanizerSignals', { returnObjects: true });

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
              <Link to="/signin?next=/scan" className="btn btn-ghost">{t('landing.runCheck')}</Link>
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

      <section className="landing-section human-written-section">
        <div className="section-inner human-written-layout">
          <div>
            <p className="eyebrow">{t('landing.humanizerEyebrow')}</p>
            <h2>{t('landing.humanizerTitle')}</h2>
            <p>{t('landing.humanizerBody1')}</p>
            <p>{t('landing.humanizerBody2')}</p>
            <p className="sample-reference-note">
              <a href={t('landing.humanizerSourceUrl')} target="_blank" rel="noopener noreferrer">
                {t('landing.humanizerSourceLabel')}
              </a>
            </p>
          </div>

          <div className="human-written-panel">
            <ul className="signal-list" aria-label={t('landing.humanizerSignalsLabel')}>
              {humanizerSignals.map((signal) => (
                <li key={signal}>{signal}</li>
              ))}
            </ul>
            <div className="human-written-guardrails">
              <span>{t('landing.humanizerGuardrail1')}</span>
              <span>{t('landing.humanizerGuardrail2')}</span>
              <strong>{t('landing.humanizerPunch')}</strong>
            </div>
          </div>
        </div>
      </section>

      <section className="landing-section human-written-section">
        <div className="section-inner human-written-layout">
          <div>
            <p className="eyebrow">{t('landing.humanWrittenEyebrow')}</p>
            <h2>{t('landing.humanWrittenTitle')}</h2>
            <p>{t('landing.humanWrittenBody1')}</p>
            <p>{t('landing.humanWrittenBody2')}</p>
          </div>

          <div className="human-written-panel">
            <ul className="signal-list" aria-label={t('landing.humanWrittenSignalsLabel')}>
              {humanWrittenSignals.map((signal) => (
                <li key={signal}>{signal}</li>
              ))}
            </ul>
            <div className="human-written-guardrails">
              <span>{t('landing.humanWrittenGuardrail1')}</span>
              <span>{t('landing.humanWrittenGuardrail2')}</span>
              <strong>{t('landing.humanWrittenPunch')}</strong>
            </div>
          </div>
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
            <div className="sample-report-value-grid" aria-label={t('landing.reportValueLabel')}>
              {reportValueCards.map((card) => (
                <article key={card.title}>
                  <strong>{card.title}</strong>
                  <p>{card.body}</p>
                </article>
              ))}
            </div>
            <Link to={publicPath('/essay-checker')} className="btn btn-primary">{t('landing.runOwnScan')}</Link>
          </div>

          <SampleReportPreview />
        </div>
      </section>

      <section id="help" className="landing-section help-section">
        <div className="section-inner">
          <div className="help-heading">
            <div>
              <p className="eyebrow">{t('landing.helpEyebrow')}</p>
              <h2>{t('landing.helpTitle')}</h2>
            </div>
            <p>{t('landing.helpLead')}</p>
          </div>

          <div className="help-grid">
            {helpCards.map((card, index) => (
              <article className="help-card" key={card.title}>
                <span>{String(index + 1).padStart(2, '0')}</span>
                <h3>{card.title}</h3>
                <p>{card.body}</p>
              </article>
            ))}
          </div>
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
          <Link to="/signin?next=/scan" className="btn btn-ghost">{t('landing.ctaButton')}</Link>
          <small>{t('landing.ctaSmall')}</small>
        </div>
      </section>

      <footer className="landing-footer">
        <div className="section-inner landing-footer-inner">
          <div>
            <Link to={publicPath('/')} className="footer-wordmark">DraftProof</Link>
            <p>{t('footer.disclaimer')}</p>
            <PageFreshness path="/" className="footer-freshness" />
          </div>
          <nav aria-label={t('footer.product')}>
            <a href="#product">{t('footer.product')}</a>
            <a href="#engine">{t('footer.howItWorks')}</a>
            <a href="#report">{t('footer.sampleReport')}</a>
            <Link to={publicPath('/essay-checker')}>{t('footer.essayChecker')}</Link>
            <Link to={publicPath('/pricing')}>{t('footer.pricing')}</Link>
            <Link to={publicPath('/faq')}>{t('footer.faq')}</Link>
            <Link to={publicPath('/privacy')}>{t('footer.privacy')}</Link>
            <Link to={publicPath('/security')}>{t('footer.security')}</Link>
            <a href={`mailto:${t('footer.supportEmail')}`}>{t('footer.supportEmail')}</a>
          </nav>
        </div>
      </footer>
    </main>
  );
}

function SampleReportPreview() {
  const { t } = useTranslation();
  const [activeSection, setActiveSection] = useState('aiSignal');
  const [isAutoPaused, setIsAutoPaused] = useState(false);
  const [isHoverPaused, setIsHoverPaused] = useState(false);
  const sampleReportNotes = t('landing.sampleReportNotes', { returnObjects: true });
  const sampleScoreSignals = t('landing.sampleScoreSignals', { returnObjects: true });
  const sampleActionItems = t('landing.sampleActionItems', { returnObjects: true });
  const previewTabs = t('landing.reportPreviewTabs', { returnObjects: true });
  const previewTabIds = useMemo(() => previewTabs.map((tab) => tab.id), [previewTabs]);
  const currentTab = previewTabs.find((tab) => tab.id === activeSection) || previewTabs[0];
  const isPreviewPaused = isAutoPaused || isHoverPaused;

  useEffect(() => {
    if (isPreviewPaused || previewTabIds.length < 2) return undefined;
    if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) return undefined;

    const timer = window.setTimeout(() => {
      setActiveSection((current) => {
        const currentIndex = previewTabIds.indexOf(current);
        return previewTabIds[(currentIndex + 1) % previewTabIds.length] || previewTabIds[0];
      });
    }, 4500);

    return () => window.clearTimeout(timer);
  }, [activeSection, isPreviewPaused, previewTabIds]);

  const selectPreviewTab = (tabId) => {
    setActiveSection(tabId);
    setIsAutoPaused(true);
  };

  return (
    <article
      className={`sample-report-preview${isPreviewPaused ? ' is-paused' : ''}`}
      aria-label={t('landing.reportPreviewLabel')}
      onMouseEnter={() => setIsHoverPaused(true)}
      onMouseLeave={() => setIsHoverPaused(false)}
      onFocusCapture={() => setIsAutoPaused(true)}
    >
      <div className="sample-preview-tabs" role="tablist" aria-label={t('landing.reportPreviewTabsLabel')}>
        {previewTabs.map((tab) => (
          <button
            type="button"
            key={tab.id}
            role="tab"
            aria-selected={activeSection === tab.id}
            className={activeSection === tab.id ? 'is-active' : ''}
            onClick={() => selectPreviewTab(tab.id)}
          >
            <span>{tab.label}</span>
            <em>{tab.summary}</em>
          </button>
        ))}
      </div>

      <div className="sample-preview-panel" role="tabpanel" aria-label={currentTab?.label}>
        {activeSection === 'aiSignal' && (
          <>
            <div className="sample-report-pattern">
              <div className="sample-report-pattern-main">
                <div className="sample-transformation-icon" aria-hidden="true">
                  <svg width="30" height="30" viewBox="0 0 30 30" fill="none">
                    <path d="M6 8.5h12.5M6 15h18M6 21.5h10" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round"/>
                    <path d="M21 7l3 3-3 3M18 18l-3 3 3 3" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </div>
                <div>
                  <span>{t('landing.transformationPattern')}</span>
                  <h3>{t('landing.humanUncertain')}</h3>
                  <div className="sample-report-tags">
                    <em>{t('landing.lowConfidence')}</em>
                    <em>{t('landing.notVerdict')}</em>
                  </div>
                </div>
              </div>
              <div className="sample-authorship-badge">
                <span>{t('landing.aiSignal')}</span>
                <strong>{t('landing.lowAiSignal')}</strong>
                <small>{t('landing.calibratedTopk')}</small>
              </div>
            </div>

            <div className="sample-report-chart">
              <div className="sample-original-scan">
                <div className="sample-original-head">
                  <div>
                    <span>{t('landing.originalScan')}</span>
                    <strong>{t('landing.humanUncertain')}</strong>
                  </div>
                  <em>{t('landing.originalScanScore')}</em>
                </div>

                <div className="sample-contribution">
                  <span>{t('landing.estimatedContribution')}</span>
                  <p>{t('landing.contributionBody')}</p>
                  <div className="sample-report-tags">
                    <em>{t('landing.calibratedAiRisk')}</em>
                    <em>{t('landing.humanAnchorDiscount')}</em>
                    <em>{t('landing.calibrationConfidence')}</em>
                    <em>{t('landing.reportingSuppression')}</em>
                  </div>
                  <div className="sample-contribution-bars">
                    <SampleSignalBar label={t('landing.humanContribution')} value={100} tone="human" />
                    <SampleSignalBar label={t('landing.aiTransformation')} value={0} tone="ai" />
                  </div>
                </div>
              </div>

              <div className="sample-report-notes">
                {sampleReportNotes.map((note) => (
                  <span key={note}>{note}</span>
                ))}
              </div>
              <p className="sample-reference-note">{t('landing.turnitinReference')}</p>
            </div>
          </>
        )}

        {activeSection === 'scoreProfile' && (
          <div className="sample-section-card">
            <div className="sample-section-card-head">
              <span>{t('landing.scoreProfile')}</span>
              <h3>{t('landing.whyScoreMoved')}</h3>
              <p>{t('landing.scoreProfileBody')}</p>
            </div>
            <div className="sample-score-profile-grid">
              {sampleScoreSignals.map((signal) => (
                <div className={`sample-score-signal is-${signal.tone}`} key={signal.label}>
                  <span>{signal.label}</span>
                  <strong>{signal.value}</strong>
                  <em>{signal.detail}</em>
                </div>
              ))}
            </div>
            <div className="sample-profile-bars">
              <SampleSignalBar label={t('landing.aiStyleSignal')} value={18} tone="ai" />
              <SampleSignalBar label={t('landing.sourceGroundingSignal')} value={64} tone="quality" />
              <SampleSignalBar label={t('landing.humanAnchorSignal')} value={82} tone="human" />
            </div>
          </div>
        )}

        {activeSection === 'actionPlan' && (
          <div className="sample-section-card">
            <div className="sample-section-card-head">
              <span>{t('landing.actionPlan')}</span>
              <h3>{t('landing.actionPlanTitle')}</h3>
              <p>{t('landing.actionPlanBody')}</p>
            </div>
            <div className="sample-action-list">
              {sampleActionItems.map((item, index) => (
                <article className={`sample-action-item is-${item.tone}`} key={item.title}>
                  <span>{String(index + 1).padStart(2, '0')}</span>
                  <div>
                    <strong>{item.title}</strong>
                    <p>{item.body}</p>
                  </div>
                </article>
              ))}
            </div>
          </div>
        )}

        {activeSection === 'findings' && (
          <div className="sample-finding-card">
            <div className="sample-finding-header">
              <div>
                <span className="sample-finding-id">{t('landing.findingsSampleId')}</span>
                <h3 className="sample-finding-type">{t('landing.findingsSampleType')}</h3>
              </div>
              <span className="sample-finding-num">#4</span>
            </div>
            <div className="sample-finding-body">
              <blockquote className="sample-finding-paragraph">
                {t('landing.findingsSampleParagraph')}
              </blockquote>
              <p className="sample-finding-description">{t('landing.findingsSampleDescription')}</p>
              <div className="sample-finding-strength-row">
                <span>{t('landing.findingsSignalStrength')}</span>
                <strong>59%</strong>
              </div>
              <div className="sample-signal-track">
                <i className="is-ai" style={{ width: '59%' }} />
              </div>
              <div className="sample-finding-chips">
                <em>{t('landing.findingsSampleChip1')}</em>
                <em>{t('landing.findingsSampleChip2')}</em>
                <em>{t('landing.findingsSampleChip3')}</em>
              </div>
              <div className="sample-finding-also">
                <span>{t('landing.findingsAlsoDetected')}</span>
                <em>{t('landing.findingsSampleAlso')}</em>
              </div>
              <div className="sample-finding-subsection">
                <span>{t('landing.findingsMainIssue')}</span>
                <p>{t('landing.findingsSampleMainIssue')}</p>
              </div>
              <div className="sample-finding-subsection">
                <span>{t('landing.findingsRewriteHint')}</span>
                <p>{t('landing.findingsSampleRewriteHint')}</p>
              </div>
            </div>
          </div>
        )}
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
