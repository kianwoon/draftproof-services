import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import CodeTexture from '../components/CodeTexture';
import PageFreshness from '../components/PageFreshness';
import MergedAuthorshipRisk from './report/MergedAuthorshipRisk';
import { RewriteBeforeAfter } from '../components/RewriteBeforeAfter';
import { getLocaleFromPathname, localizePath } from '../localeRouting';

// Representative Submission-risk shape for the marketing sample report. Mirrors the
// real ai_risk_badge.submission_risk object; values are an illustrative example, not
// a scoring oracle (the band's labels come from i18n report.submissionRisk.*).
const SAMPLE_SUBMISSION_RISK = {
  overall: { level: 'high', main_reason_code: 'ownership' },
  axes: {
    text_pattern: { level: 'medium', display_score: 41 },
    ownership: { level: 'high' },
    citation: { level: 'medium' },
    defence_readiness: { level: 'high' },
    policy_declaration: { level: 'unknown' },
  },
};

// Illustrative V7 authorship-clarity breakdown for the marketing sample report.
// Mirrors the real ai_risk_badge.authorship_breakdown + tier_authority shape
// (poc/detect_v7/breakdown_composer.py, poc/report/builder.py). Values are a
// fixed illustrative example, not a scoring oracle.
const SAMPLE_AUTHORSHIP_BREAKDOWN = {
  document_breakdown_raw: {
    student_owned: 0.62,
    ai_assisted_polished: 0.24,
    ai_paraphrased: 0.09,
    ai_generated_like: 0.05,
  },
  document_breakdown_bands: {
    student_owned: 'Strong',
    ai_assisted_polished: 'Some',
    ai_paraphrased: 'Little',
    ai_generated_like: 'None',
  },
};

const SAMPLE_TIER_AUTHORITY = {
  fused_score: 34,
  composite_score: 29,
  proportion: 0.18,
};

export default function Landing() {
  const { t } = useTranslation();
  const location = useLocation();
  const locale = getLocaleFromPathname(location.pathname);
  const publicPath = (path) => localizePath(path, locale);
  const whyCards = t('landing.whyCards', { returnObjects: true });
  const helpCards = t('landing.helpCards', { returnObjects: true });
  const contentStrategies = t('landing.contentStrategies', { returnObjects: true });
  const competitors = t('featuresPage.competitors', { returnObjects: true });
  const comparisonRows = t('featuresPage.rows', { returnObjects: true });
  const anchorCards = t('landing.anchorCards', { returnObjects: true });
  const anchorWorkflow = t('landing.anchorWorkflow', { returnObjects: true });
  const heroReviewSteps = t('landing.heroReviewSteps', { returnObjects: true });
  const heroTitlesRaw = t('landing.heroTitles', { returnObjects: true });
  const heroTitles = Array.isArray(heroTitlesRaw) && heroTitlesRaw.length
    ? heroTitlesRaw
    : [{ text: t('landing.heroTitle'), highlight: t('landing.heroTitleHighlight') }];
  const heroTitleCarousel = useLandingCarousel(heroTitles.length, 4500);
  const canHover = typeof window !== 'undefined' && window.matchMedia?.('(hover: hover)').matches;
  const useCases = t('landing.useCases', { returnObjects: true });
  const heroVideoRef = useRef(null);
  const heroSectionRef = useRef(null);
  const ctaSectionRef = useRef(null);
  // Gate the always-on hero video + code-texture drift on real viewport visibility —
  // otherwise both keep decoding/animating for as long as the tab stays open, even
  // scrolled far past the fold. Mirrors the carousels' existing visibility gating.
  const isHeroVisible = useIsVisible(heroSectionRef, 0);
  const isCtaVisible = useIsVisible(ctaSectionRef, 0);

  // Mobile browsers (esp. iOS Safari) only autoplay a video they consider
  // muted + inline. React's `muted` JSX attribute does NOT reliably set the
  // underlying DOM property, so force it here and kick off playback explicitly;
  // skip when the user prefers reduced motion (CSS hides the video → webp poster),
  // and pause playback outright once the hero scrolls out of view.
  useEffect(() => {
    const video = heroVideoRef.current;
    if (!video) return undefined;
    if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) return undefined;
    if (!isHeroVisible) {
      video.pause();
      return undefined;
    }
    video.muted = true;
    video.setAttribute('muted', '');
    const attemptPlay = () => {
      const played = video.play();
      if (played?.catch) played.catch(() => {});
    };
    attemptPlay();
    // Retry once the tab/clip is actually ready in case the first call was too early.
    video.addEventListener('loadeddata', attemptPlay, { once: true });
    return () => video.removeEventListener('loadeddata', attemptPlay);
  }, [isHeroVisible]);

  return (
    <main className="landing-page">
      <section
        id="hero"
        className={`landing-hero${isHeroVisible ? '' : ' is-offscreen'}`}
        ref={(node) => {
          heroSectionRef.current = node;
          // The hero title carousel has no section of its own — it shares this one,
          // so its visibility gate (useLandingCarousel's useIsVisible) needs the
          // same DOM node or it never fires and the headline rotates forever.
          heroTitleCarousel.sectionRef.current = node;
        }}
      >
        <video
          ref={heroVideoRef}
          className="hero-bg-video"
          autoPlay
          muted
          loop
          playsInline
          preload="auto"
          poster="/landing/hero_bg.webp"
          aria-hidden="true"
        >
          <source src="/landing/hero_background.mp4" type="video/mp4" />
        </video>
        <CodeTexture id="landingHero" className="hero-code-field" />
        <div className="section-inner landing-hero-grid">
          <div className="hero-copy">
            <p className="brand-pill">{t('landing.heroPill')}</p>
            <h1
              className="hero-rotating-title"
              onMouseEnter={canHover ? heroTitleCarousel.pause : undefined}
              onMouseLeave={canHover ? heroTitleCarousel.resume : undefined}
            >
              {heroTitles.map((item, index) => {
                const text = String(item?.text ?? '');
                const hl = String(item?.highlight ?? '');
                const at = hl ? text.indexOf(hl) : -1;
                const before = at >= 0 ? text.slice(0, at) : text;
                const after = at >= 0 ? text.slice(at + hl.length) : '';
                const isActive = heroTitleCarousel.activeSlide === index;
                return (
                  <span
                    key={text || index}
                    className={`hero-headline${isActive ? ' is-active' : ''}`}
                    aria-hidden={isActive ? undefined : 'true'}
                  >
                    {before}{at >= 0 ? <span className="hero-hl">{hl}</span> : null}{after}
                  </span>
                );
              })}
            </h1>
            {heroTitles.length > 1 ? (
              <div className="hero-headline-dots" role="tablist" aria-label={t('landing.heroTitlesLabel')}>
                {heroTitles.map((item, index) => (
                  <button
                    type="button"
                    key={`dot-${item?.text ?? index}`}
                    role="tab"
                    aria-selected={heroTitleCarousel.activeSlide === index}
                    aria-label={String(item?.highlight ?? item?.text ?? index)}
                    className={heroTitleCarousel.activeSlide === index ? 'is-active' : ''}
                    onClick={() => heroTitleCarousel.goToSlide(index)}
                  />
                ))}
              </div>
            ) : null}
            <p className="lead">{t('landing.heroLead')}</p>

            <div className="hero-actions" id="check">
              <Link to="/signin?next=/scan" className="btn btn-primary">{t('landing.runCheck')}</Link>
              <a href="#report" className="btn btn-ghost">{t('landing.viewSample')}</a>
            </div>

            <p className="hero-free-credit">
              <span className="hero-free-credit-spark" aria-hidden="true" />
              {t('landing.heroFreeCreditsPre')}
              <Link to="/signin?next=/scan" className="hero-free-credit-link">{t('landing.heroFreeCreditsLink')}</Link>
              {t('landing.heroFreeCreditsPost')}
            </p>

            <div className="trust-note">
              <span className="mini-shield" aria-hidden="true" />
              <span>{t('landing.trustNote')}</span>
            </div>

            <div className="trust-note detector-update-note">
              <span className="hero-free-credit-spark" aria-hidden="true" />
              <span>{t('landing.detectorUpdateNote')}</span>
            </div>
          </div>

          <HeroReviewPanel steps={heroReviewSteps} />
        </div>
      </section>

      <section className="trust-bar" aria-label={t('landing.audienceDetails')}>
        <div className="section-inner trust-bar-inner">
          <span>{t('landing.builtFor')}</span>
          <strong>{t('landing.students')}</strong>
          <strong>{t('landing.gradWriters')}</strong>
          <strong>{t('landing.eslWriters')}</strong>
          <strong>{t('landing.independentWriters')}</strong>
          <span>{t('landing.tokenRate')}</span>
          <span>{t('landing.pdfReport')}</span>
          <span>{t('landing.citationGrounding')}</span>
          <span>{t('landing.contentAwareRewrite')}</span>
        </div>
      </section>

      <ContentRiskCarousel
        anchorCards={anchorCards}
        anchorWorkflow={anchorWorkflow}
        useCases={useCases}
      />

      <ReportStrategyCarousel
        contentStrategies={contentStrategies}
        publicPath={publicPath}
      />

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

      <section id="compare" className="landing-section">
        <div className="section-inner">
          <p className="eyebrow">{t('featuresPage.tableLabel')}</p>
          <h2>{t('featuresPage.title')}</h2>
          <p className="section-lead" style={{ marginBottom: '1.25rem', color: 'var(--muted)' }}>{t('featuresPage.lead')}</p>
          <div className="feat-table-wrap">
            <table className="feat-table">
              <thead>
                <tr>
                  <th />
                  {competitors.map((name, i) => (
                    <th key={name} className={i === 0 ? 'feat-th-dp' : undefined}>{name}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {comparisonRows.map((row) => (
                  <tr key={row.label}>
                    <td>{row.label}</td>
                    {row.values.map((val, i) => (
                      <td key={i} className={i === 0 ? 'feat-td-dp' : undefined}>
                        {val === 'yes'
                          ? <span className="feat-yes" aria-label="yes">✓</span>
                          : val === 'no'
                          ? <span className="feat-no" aria-label="no">✗</span>
                          : <span className="feat-partial">{val}</span>}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p style={{ textAlign: 'center', marginTop: '1.25rem', fontSize: '0.875rem' }}>
            <Link to={publicPath('/features')}>{t('nav.features')} →</Link>
            {' · '}
            <Link to={publicPath('/turnitin-vs-ai-detectors')}>{t('footer.turnitinVsDetectors')} →</Link>
          </p>
        </div>
      </section>

      <section id="cta" className={`landing-cta${isCtaVisible ? '' : ' is-offscreen'}`} ref={ctaSectionRef}>
        <CodeTexture id="landingCta" />
        <div className="section-inner">
          <p className="brand-pill">{t('landing.ctaPill')}</p>
          <h2>{t('landing.ctaTitle')}</h2>
          <p>{t('landing.ctaBody')}</p>
          <Link to="/signin?next=/scan" className="btn btn-ghost">{t('landing.ctaButton')}</Link>
          <p className="cta-free-credit">{t('landing.ctaFreeCredits')}</p>
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
            <Link to={publicPath('/technology')}>{t('footer.howItWorks')}</Link>
            <a href="#report">{t('footer.sampleReport')}</a>
            <Link to={publicPath('/content-checker')}>{t('footer.essayChecker')}</Link>
            <Link to={publicPath('/academic-integrity-ai')}>{t('footer.academicIntegrity')}</Link>
            <Link to={publicPath('/turnitin-ai-score')}>{t('footer.turnitinScore')}</Link>
            <Link to={publicPath('/turnitin-vs-ai-detectors')}>{t('footer.turnitinVsDetectors')}</Link>
            <Link to={publicPath('/turnitin-alternatives')}>{t('footer.turnitinAlternatives')}</Link>
            <Link to={publicPath('/ai-declaration')}>{t('footer.aiDeclaration')}</Link>
            <Link to={publicPath('/reduce-ai-detection')}>{t('footer.reduceDetection')}</Link>
            <Link to={publicPath('/pricing')}>{t('footer.pricing')}</Link>
            <Link to={publicPath('/faq')}>{t('footer.faq')}</Link>
            <Link to={publicPath('/privacy')}>{t('footer.privacy')}</Link>
            <Link to={publicPath('/security')}>{t('footer.security')}</Link>
            <a href="https://www.reddit.com/r/DraftProofApp/" target="_blank" rel="noopener noreferrer">{t('footer.community')}</a>
            <a href={`mailto:${t('footer.supportEmail')}`}>{t('footer.supportEmail')}</a>
          </nav>
        </div>
      </footer>
    </main>
  );
}

function HeroReviewPanel({ steps }) {
  const { t } = useTranslation();
  const reviewSteps = Array.isArray(steps) ? steps : [];
  const carousel = useLandingCarousel(reviewSteps.length, 3800);
  const activeStep = reviewSteps[carousel.activeSlide] || reviewSteps[0];

  if (!activeStep) return null;

  return (
    <aside
      ref={carousel.sectionRef}
      className="review-panel"
      aria-label={t('landing.quickSummary')}
      onMouseEnter={carousel.pause}
      onMouseLeave={carousel.resume}
      onFocusCapture={carousel.pause}
      onTouchStart={carousel.pause}
    >
      <div className="review-panel-top">
        <p className="card-kicker">{t('landing.quickSummary')}</p>
        <span className="live-dot">{t('landing.livePreview')}</span>
      </div>

      <div className="hero-review-step-row">
        <span>{activeStep.stepLabel}</span>
        <div className="hero-review-steps" role="tablist" aria-label={t('landing.heroReviewStepsLabel')}>
          {reviewSteps.map((step, index) => (
            <button
              type="button"
              key={step.id}
              role="tab"
              aria-selected={carousel.activeSlide === index}
              aria-label={step.title}
              className={carousel.activeSlide === index ? 'is-active' : ''}
              onClick={() => carousel.goToSlide(index)}
            >
              {step.shortLabel}
            </button>
          ))}
        </div>
      </div>

      <div className="hero-review-stage" key={activeStep.id}>
        <h2>{activeStep.title}</h2>
        <p>{activeStep.body}</p>
        <HeroReviewVisual step={activeStep} />

        <div className="primary-fix">
          <div>
            <span>{activeStep.primaryLabel}</span>
            <strong>{activeStep.primaryValue}</strong>
          </div>
          <em>{activeStep.badge}</em>
        </div>
      </div>
    </aside>
  );
}

function HeroReviewVisual({ step }) {
  if (step.visual === 'findings') {
    const findings = Array.isArray(step.findings) ? step.findings : [];
    const summary = Array.isArray(step.summary) ? step.summary : [];

    return (
      <div className="hero-review-visual hero-findings-visual">
        <div className="hero-finding-stack">
          {findings.map((finding) => (
            <div className={`hero-finding-item ${toneClass(finding.tone)}`} key={finding.label}>
              <span>{finding.label}</span>
              <p>{finding.body}</p>
              <em>{finding.badge}</em>
            </div>
          ))}
        </div>
        <div className="hero-finding-summary">
          {summary.map((item) => (
            <span className={toneClass(item.tone)} key={item.label}>
              {item.label}
            </span>
          ))}
        </div>
      </div>
    );
  }

  if (step.visual === 'diff') {
    const diffRows = Array.isArray(step.diffRows) ? step.diffRows : [];
    const fixSteps = Array.isArray(step.fixSteps) ? step.fixSteps : [];

    return (
      <div className="hero-review-visual hero-diff-visual">
        <div className="hero-diff-card">
          {diffRows.map((row) => (
            <div className={`hero-diff-row ${toneClass(row.tone, ['remove', 'add', 'neutral'])}`} key={row.label}>
              <span>{row.label}</span>
              <p>{row.text}</p>
            </div>
          ))}
        </div>
        <div className="hero-fix-chips">
          {fixSteps.map((fix) => (
            <span key={fix}>{fix}</span>
          ))}
        </div>
      </div>
    );
  }

  const scanLines = Array.isArray(step.scanLines) ? step.scanLines : [];
  const scanStatus = Array.isArray(step.scanStatus) ? step.scanStatus : [];

  return (
    <div className="hero-review-visual hero-scan-visual">
      <div className="hero-scan-document">
        {scanLines.map((line) => (
          <div className={`hero-scan-line ${toneClass(line.tone)}`} key={line.label}>
            <span>{line.label}</span>
            <p>{line.text}</p>
          </div>
        ))}
      </div>
      <div className="hero-scan-status">
        {scanStatus.map((item) => (
          <div className={toneClass(item.tone)} key={item.label}>
            <span>{item.label}</span>
            <strong>{item.value}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

function toneClass(tone, allowed = ['positive', 'warning', 'neutral']) {
  return allowed.includes(tone) ? `is-${tone}` : 'is-neutral';
}

// Use-case slide artwork (asset paths, not translatable copy): id -> illustration.
const USE_CASE_IMAGES = {
  'score-vs-coaching': '/landing/use-case-score.svg',
  'ai-allowed': '/landing/use-case-allowed.svg',
  'ai-banned': '/landing/use-case-banned.svg',
};

function ContentRiskCarousel({ anchorCards, anchorWorkflow, useCases }) {
  const { t } = useTranslation();
  const slides = useMemo(() => ([
    {
      id: 'content-anchors',
      eyebrow: t('landing.anchorEyebrow'),
      title: t('landing.anchorTitle'),
      imageSrc: '/landing/content-anchors.svg',
      body: [t('landing.anchorBody1'), t('landing.anchorBody2')],
      renderPanel: () => (
        <div className="anchor-panel" aria-label={t('landing.anchorCardsLabel')}>
          <div className="anchor-card-grid">
            {anchorCards.map((card) => (
              <article className="anchor-card" key={card.title}>
                <span>{card.label}</span>
                <h3>{card.title}</h3>
                <p>{card.body}</p>
              </article>
            ))}
          </div>
          <ol className="anchor-workflow" aria-label={t('landing.anchorWorkflowLabel')}>
            {anchorWorkflow.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ol>
        </div>
      ),
    },
    {
      id: 'critical-thinking',
      eyebrow: t('landing.criticalEyebrow'),
      title: t('landing.criticalTitle'),
      body: [t('landing.criticalBody1'), t('landing.criticalBody2')],
      renderPanel: () => (
        <SignalPanel
          label={t('landing.criticalSignalsLabel')}
          signals={t('landing.criticalSignals', { returnObjects: true })}
          guardrails={[
            t('landing.criticalGuardrail1'),
            t('landing.criticalGuardrail2'),
          ]}
          punch={t('landing.criticalPunch')}
        />
      ),
    },
    ...(Array.isArray(useCases) ? useCases : [])
      .filter((uc) => uc.id === 'score-vs-coaching')
      .map((uc) => ({
        id: `use-case-${uc.id}`,
        eyebrow: uc.eyebrow,
        title: uc.title,
        imageSrc: USE_CASE_IMAGES[uc.id],
        body: Array.isArray(uc.body) ? uc.body : [uc.body],
        renderPanel: () => (
          <SignalPanel
            label={uc.eyebrow}
            signals={Array.isArray(uc.points) ? uc.points : []}
            guardrails={Array.isArray(uc.guardrails) ? uc.guardrails : []}
            punch={uc.punch || uc.tag}
          />
        ),
      })),
    {
      id: 'policy-aware',
      eyebrow: t('landing.policyAwareEyebrow'),
      title: t('landing.policyAwareTitle'),
      body: [t('landing.policyAwareBody')],
      renderPanel: () => (
        <PolicyAwarePanel
          useCases={(Array.isArray(useCases) ? useCases : []).filter(
            (uc) => uc.id === 'ai-allowed' || uc.id === 'ai-banned',
          )}
        />
      ),
    },
  ]), [anchorCards, anchorWorkflow, useCases, t]);
  const carousel = useLandingCarousel(slides.length);
  const { trackRef, height: activeSlideHeight } = useActiveSlideHeight(
    carousel.activeSlide,
    '.content-carousel-slide.is-active',
  );

  return (
    <section
      ref={carousel.sectionRef}
      className="landing-section content-carousel-section"
      aria-label={t('landing.contentCarouselLabel')}
      onMouseEnter={carousel.pause}
      onMouseLeave={carousel.resume}
      onFocusCapture={carousel.pause}
      onTouchStart={carousel.pause}
    >
      <div
        ref={trackRef}
        className="content-carousel-track"
        style={{
          '--active-slide': carousel.activeSlide,
          ...(activeSlideHeight ? { height: `${activeSlideHeight}px` } : {}),
        }}
      >
        {slides.map((slide, index) => (
          <article
            className={`section-inner content-carousel-slide${carousel.activeSlide === index ? ' is-active' : ''}`}
            key={slide.id}
            aria-hidden={carousel.activeSlide !== index}
            inert={carousel.activeSlide !== index ? '' : undefined}
          >
            <div className="content-carousel-copy">
              <p className="eyebrow">{slide.eyebrow}</p>
              {slide.imageSrc && (
                <img
                  className="content-carousel-visual"
                  src={slide.imageSrc}
                  alt=""
                  loading="lazy"
                  decoding="async"
                />
              )}
              <h2>{slide.title}</h2>
              {slide.body.map((paragraph) => (
                <p key={paragraph}>{paragraph}</p>
              ))}
              {slide.sourceUrl && (
                <p className="sample-reference-note">
                  <a href={slide.sourceUrl} target="_blank" rel="noopener noreferrer">
                    {slide.sourceLabel}
                  </a>
                </p>
              )}
            </div>
            {slide.renderPanel()}
          </article>
        ))}
      </div>

      <LandingCarouselControls
        activeSlide={carousel.activeSlide}
        dotsLabel={t('landing.contentCarouselTabsLabel')}
        nextLabel={t('landing.contentCarouselNext')}
        onNext={carousel.goToNextSlide}
        onSelect={carousel.goToSlide}
        slides={slides}
      />
    </section>
  );
}

function ReportStrategyCarousel({ contentStrategies, publicPath }) {
  const { t } = useTranslation();
  const slides = useMemo(() => ([
    {
      id: 'sample-report',
      eyebrow: t('landing.sampleEyebrow'),
      renderSlide: () => (
        <article className="report-carousel-panel sample-report-layout">
          <div className="sample-report-copy">
            <p className="eyebrow">{t('landing.sampleEyebrow')}</p>
            <h2>{t('landing.sampleTitle')}</h2>
            <p>{t('landing.sampleBody')}</p>
            <div className="sample-report-points">
              <span>{t('landing.samplePoint1')}</span>
              <span>{t('landing.samplePoint2')}</span>
              <span>{t('landing.samplePoint3')}</span>
            </div>

            <div className="sample-report-rewrite">
              <p className="sample-report-rewrite-caption">{t('landing.rewriteTeaserBody')}</p>
              <RewriteBeforeAfter
                before={t('landing.rewriteTeaserBefore')}
                after={t('landing.rewriteTeaserAfter')}
                marker={t('landing.rewriteTeaserMarker')}
                beforeLabel={t('landing.rewriteTeaserBeforeLabel')}
                afterLabel={t('landing.rewriteTeaserAfterLabel')}
              />
              <Link to={publicPath('/rewrite')} className="sample-report-rewrite-link">{t('landing.rewriteTeaserLink')}</Link>
            </div>

            <Link to={publicPath('/content-checker')} className="btn btn-primary">{t('landing.runOwnScan')}</Link>
          </div>

          <SampleReportPreview />
        </article>
      ),
    },
    {
      id: 'content-aware-help',
      eyebrow: t('landing.strategyEyebrow'),
      renderSlide: () => (
        <article className="report-carousel-panel strategy-carousel-panel">
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
        </article>
      ),
    },
  ]), [contentStrategies, publicPath, t]);
  const carousel = useLandingCarousel(slides.length, 7200);
  const { trackRef, height: activeSlideHeight } = useActiveSlideHeight(
    carousel.activeSlide,
    '.report-strategy-slide.is-active',
  );

  return (
    <section
      ref={carousel.sectionRef}
      id="report"
      className="landing-section report-strategy-carousel-section"
      aria-label={t('landing.reportStrategyCarouselLabel')}
      onMouseEnter={carousel.pause}
      onMouseLeave={carousel.resume}
      onFocusCapture={carousel.pause}
      onTouchStart={carousel.pause}
    >
      <div className="section-inner report-strategy-carousel-shell">
        <div
          ref={trackRef}
          className="report-strategy-track"
          style={{
            '--active-slide': carousel.activeSlide,
            ...(activeSlideHeight ? { height: `${activeSlideHeight}px` } : {}),
          }}
        >
          {slides.map((slide, index) => (
            <div
              className={`report-strategy-slide${carousel.activeSlide === index ? ' is-active' : ''}`}
              key={slide.id}
              aria-hidden={carousel.activeSlide !== index}
              inert={carousel.activeSlide !== index ? '' : undefined}
            >
              {slide.renderSlide()}
            </div>
          ))}
        </div>
      </div>

      <LandingCarouselControls
        activeSlide={carousel.activeSlide}
        className="report-carousel-controls"
        dotsLabel={t('landing.reportStrategyCarouselTabsLabel')}
        nextLabel={t('landing.reportStrategyCarouselNext')}
        onNext={carousel.goToNextSlide}
        onSelect={carousel.goToSlide}
        slides={slides}
      />
    </section>
  );
}

// Shared viewport-visibility gate: off-screen sections shouldn't keep
// auto-advancing carousels, decoding video, or animating — a user scrolling
// past loses motion they never saw, and it burns CPU/GPU/battery for nothing.
function useIsVisible(ref, threshold = 0.3) {
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    const node = ref.current;
    if (!node || typeof IntersectionObserver === 'undefined') return undefined;

    const observer = new IntersectionObserver(
      ([entry]) => setIsVisible(entry.isIntersecting),
      { threshold },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [ref, threshold]);

  return isVisible;
}

function useLandingCarousel(slideCount, intervalMs = 6500) {
  const [activeSlide, setActiveSlide] = useState(0);
  const [isPaused, setIsPaused] = useState(false);
  const sectionRef = useRef(null);
  const isVisible = useIsVisible(sectionRef, 0.3);

  useEffect(() => {
    if (isPaused || !isVisible || slideCount < 2) return undefined;
    // CSS handles prefers-reduced-motion (disables transitions) — don't stop rotation here

    const timer = window.setTimeout(() => {
      setActiveSlide((current) => (current + 1) % slideCount);
    }, intervalMs);

    return () => window.clearTimeout(timer);
  }, [activeSlide, intervalMs, isPaused, isVisible, slideCount]);

  return {
    activeSlide,
    sectionRef,
    goToSlide: (index) => {
      setActiveSlide(index);
      setIsPaused(true);
    },
    goToNextSlide: () => {
      setActiveSlide((current) => (current + 1) % slideCount);
      setIsPaused(true);
    },
    // Touch has no hover equivalent — treat a first tap in the carousel like
    // a mouseenter so mobile readers get the same "stop moving under me" pause.
    pause: () => setIsPaused(true),
    resume: () => setIsPaused(false),
  };
}

// The carousel tracks below lay every slide out in a single flex row (so the
// horizontal translateX animation works), which means the row's height
// defaults to the TALLEST slide — a short active slide leaves a dead gap
// where the tallest sibling's height used to be. Measure the active slide's
// own intrinsic height and pin the track to that instead, animating between
// heights so shorter/taller slides no longer leave (or need) empty space.
function useActiveSlideHeight(activeSlide, activeSelector) {
  const trackRef = useRef(null);
  const [height, setHeight] = useState(null);

  useLayoutEffect(() => {
    const activeEl = trackRef.current?.querySelector(activeSelector);
    if (activeEl) setHeight(activeEl.getBoundingClientRect().height);
  }, [activeSlide, activeSelector]);

  useEffect(() => {
    const activeEl = trackRef.current?.querySelector(activeSelector);
    if (!activeEl || typeof ResizeObserver === 'undefined') return undefined;

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) setHeight(entry.contentRect.height);
    });
    observer.observe(activeEl);
    return () => observer.disconnect();
  }, [activeSlide, activeSelector]);

  return { trackRef, height };
}

function LandingCarouselControls({ activeSlide, className = '', dotsLabel, nextLabel, onNext, onSelect, slides }) {
  const activeContentSlide = slides[activeSlide] || slides[0];

  return (
    <div className={`content-carousel-controls section-inner ${className}`.trim()}>
      <div className="content-carousel-dots" role="tablist" aria-label={dotsLabel}>
        {slides.map((slide, index) => (
          <button
            type="button"
            key={slide.id}
            role="tab"
            aria-selected={activeSlide === index}
            aria-label={slide.eyebrow}
            className={activeSlide === index ? 'is-active' : ''}
            onClick={() => onSelect(index)}
          />
        ))}
      </div>
      <button
        type="button"
        className="content-carousel-next"
        aria-label={nextLabel}
        onClick={onNext}
      >
        <span aria-hidden="true">&gt;</span>
      </button>
      <span className="content-carousel-status">{activeContentSlide.eyebrow}</span>
    </div>
  );
}

function SignalPanel({ label, signals, guardrails, punch }) {
  return (
    <div className="human-written-panel">
      <ul className="signal-list" aria-label={label}>
        {signals.map((signal) => (
          <li key={signal}>{signal}</li>
        ))}
      </ul>
      <div className="human-written-guardrails">
        {guardrails.map((guardrail) => (
          <span key={guardrail}>{guardrail}</span>
        ))}
        <strong>{punch}</strong>
      </div>
    </div>
  );
}

// Compact side-by-side "allowed vs banned" comparison — merges what used to
// be two separate carousel slides into one, so the policy-aware claim (the
// one comparison-table row every competitor loses) reads as a single beat.
function PolicyAwarePanel({ useCases }) {
  return (
    <div className="policy-aware-panel">
      {useCases.map((uc) => (
        <article className="policy-aware-column" key={uc.id}>
          <span className="policy-aware-tag">{uc.tag}</span>
          <h3>{uc.title}</h3>
          <ul>
            {(Array.isArray(uc.points) ? uc.points : []).slice(0, 3).map((point) => (
              <li key={point}>{point}</li>
            ))}
          </ul>
          <strong>{uc.punch}</strong>
        </article>
      ))}
    </div>
  );
}

function SampleReportPreview() {
  const { t } = useTranslation();
  const [activeSection, setActiveSection] = useState('authorshipBreakdown');
  const [isAutoPaused, setIsAutoPaused] = useState(false);
  const [isHoverPaused, setIsHoverPaused] = useState(false);
  const sectionRef = useRef(null);
  const isVisible = useIsVisible(sectionRef, 0.3);
  const sampleActionItems = t('landing.sampleActionItems', { returnObjects: true });
  const sampleFlaggedSentences = t('landing.sampleFlaggedSentences', { returnObjects: true });
  const sampleCriticalQuestions = t('landing.sampleCriticalQuestions', { returnObjects: true });
  const previewTabs = t('landing.reportPreviewTabs', { returnObjects: true });
  const previewTabIds = useMemo(() => previewTabs.map((tab) => tab.id), [previewTabs]);
  const currentTab = previewTabs.find((tab) => tab.id === activeSection) || previewTabs[0];
  const isPreviewPaused = isAutoPaused || isHoverPaused || !isVisible;

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
      ref={sectionRef}
      className={`sample-report-preview${isPreviewPaused ? ' is-paused' : ''}`}
      aria-label={t('landing.reportPreviewLabel')}
      onMouseEnter={() => setIsHoverPaused(true)}
      onMouseLeave={() => setIsHoverPaused(false)}
      onFocusCapture={() => setIsAutoPaused(true)}
      onTouchStart={() => setIsAutoPaused(true)}
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
        {activeSection === 'authorshipBreakdown' && (
          <MergedAuthorshipRisk
            t={t}
            sections={['header', 'composition']}
            breakdown={SAMPLE_AUTHORSHIP_BREAKDOWN}
            sr={SAMPLE_SUBMISSION_RISK}
            authoritativeTier="green"
            tierAuthority={SAMPLE_TIER_AUTHORITY}
          />
        )}
        {activeSection === 'aiSignal' && (
          <MergedAuthorshipRisk
            t={t}
            sections={['verdict', 'riskAxes', 'disclaimer']}
            breakdown={SAMPLE_AUTHORSHIP_BREAKDOWN}
            sr={SAMPLE_SUBMISSION_RISK}
            authoritativeTier="green"
            tierAuthority={SAMPLE_TIER_AUTHORITY}
          />
        )}

        {activeSection === 'actionPlan' && (
          <div className="fix-first-card">
            <div className="fix-first-head">
              <span>{t('report.whatToFixFirst.kicker')}</span>
              <h2>{t('report.whatToFixFirst.title')}</h2>
            </div>
            <div className="fix-first-list">
              {sampleActionItems.map((item, index) => (
                <div className="fix-first-item" key={item.title}>
                  <span className="fix-first-index">{index + 1}</span>
                  <span className="fix-first-copy">
                    <strong>{item.title}</strong>
                    <em>{item.body}</em>
                  </span>
                  {item.label && <span className="fix-first-label">{item.label}</span>}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* allow-hardcode: className values below are existing CSS hooks (.issue-*, .deberta-evidence-*
            defined in styles/site-master/07-report-submitted.css) reused verbatim from the real
            SignalHighlights issue-card markup — structural styling classes, not a content/scoring list. */}
        {activeSection === 'findings' && (
          <div className="issue-card is-open">
            <div className="issue-card-head">
              <span className="issue-card-main">
                <span className="issue-card-chips">
                  <em className="issue-card-num">{t('landing.findingsSamplePosition')}</em>
                  <em className="issue-chip issue-chip-tier is-high">{t('report.severities.high')}</em>
                  <em className="issue-chip">{t('landing.findingsSampleType')}</em>
                  <em className="issue-chip">{t('landing.findingsSampleCount')}</em>
                </span>
              </span>
            </div>
            <div className="issue-card-body">
              <p className="issue-card-summary">{t('landing.findingsSampleDescription')}</p>
              <div className="issue-action issue-action-thinking">
                <span className="issue-action-label">{t('report.submitted.criticalThinking')}</span>
                <p>
                  <strong>{t('report.criticalThinking.dimensions.evidence_grounding.label')}</strong>
                  {' — '}
                  {t('report.criticalThinking.dimensions.evidence_grounding.action')}
                </p>
              </div>
              <div className="issue-action issue-action-evidence">
                <span className="issue-action-label">{t('report.submitted.flaggedSentences')}</span>
                <ul className="deberta-evidence-list">
                  {sampleFlaggedSentences.map((sentence) => (
                    <li key={sentence.text}>
                      <span className="deberta-evidence-score">{sentence.score}%</span>
                      <span className="deberta-evidence-text">{sentence.text}</span>
                      <span className="deberta-evidence-suggestion">{sentence.suggestion}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}

        {activeSection === 'criticalThinking' && (
          <div className="sample-section-card">
            <div className="sample-section-card-head">
              <span>{t('report.criticalThinking.title')}</span>
              <h3>{t('report.criticalThinking.questionsTitle')}</h3>
              <p>{t('report.criticalThinking.questionsIntro')}</p>
            </div>
            <ol className="critical-thinking-questions">
              {sampleCriticalQuestions.map((item) => (
                <li className="critical-thinking-question" key={item.question}>
                  <span className="critical-thinking-q-quote">{`“${item.quote}”`}</span>
                  <span className="critical-thinking-q-text">{item.question}</span>
                </li>
              ))}
            </ol>
          </div>
        )}
      </div>
    </article>
  );
}
