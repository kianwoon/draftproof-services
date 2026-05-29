// LandingSections — the calm, paper-surface marketing sections that sit between
// the navy hero and navy CTA. All copy verbatim from production i18n (en).

function TrustBar() {
  return (
    <section className="trust-bar" aria-label="DraftProof audiences and review details">
      <div className="section-inner trust-bar-inner">
        <span>Built for</span>
        <strong>College students</strong>
        <strong>University students</strong>
        <strong>Graduate students</strong>
        <strong>ESL writers</strong>
        <span>1 token per 1,000 words</span>
        <span>PDF report</span>
        <span>Citation + similarity review</span>
        <span>Turnitin-aware revision guidance</span>
      </div>
    </section>
  );
}

function HumanWrittenSection() {
  const signals = [
    'AI-like predictability', 'Generic academic phrasing', 'Over-smooth rewriting',
    'Weak evidence and grounding', 'Low sentence variation', 'Missing author reasoning',
  ];
  return (
    <section className="landing-section human-written-section">
      <div className="section-inner human-written-layout">
        <div>
          <p className="eyebrow">AI-like signals</p>
          <h2>Human-written work can still look AI-generated.</h2>
          <p>AI writing reports do not know who wrote every sentence. They analyse writing
            patterns: predictability, sentence rhythm, generic phrasing, over-smooth revision,
            weak grounding, and other signals that may resemble AI-generated text.</p>
          <p>DraftProof helps students understand those signals before submission, so they can
            revise responsibly with stronger evidence, clearer reasoning, and better source grounding.</p>
        </div>

        <div className="human-written-panel">
          <ul className="signal-list" aria-label="Writing signals DraftProof can explain">
            {signals.map((s) => <li key={s}>{s}</li>)}
          </ul>
          <div className="human-written-guardrails">
            <span>We do not promise to bypass detectors.</span>
            <span>We do not guarantee any Turnitin result.</span>
            <strong>Human-written does not always mean low-risk. DraftProof explains why.</strong>
          </div>
        </div>
      </div>
    </section>
  );
}

function WhySection() {
  const whyCards = [
    { title: 'We learn from synthetic information', body: 'Search engines, chatbots, and writing assistants summarise knowledge before we reach the original source. Writing can become detached from its evidence.', note: 'DraftProof bridges that gap' },
    { title: 'Traditional media is no longer the only source', body: 'Information now moves through AI newsrooms, generated summaries, and reported material. Polished does not mean proven.', note: 'Check the source, check the claim' },
    { title: 'AI detection alone is not enough', body: 'A score is not feedback. DraftProof asks better questions: is the claim supported, and what needs fixing?', note: 'Actionable, not just a verdict' },
  ];
  return (
    <section id="product" className="landing-section">
      <div className="section-inner">
        <p className="eyebrow">Why Students Use DraftProof</p>
        <h2>Turnitin changed how students think about originality, citations, and AI-style writing.</h2>
        <div className="why-card-grid">
          {whyCards.map((card, i) => (
            <article className="why-card" key={card.title}>
              <span>{String(i + 1).padStart(2, '0')}</span>
              <h3>{card.title}</h3>
              <p>{card.body}</p>
              <small>{card.note}</small>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}

function EngineSection() {
  const checks = [
    { title: 'Citation gaps', body: "Identifies claims that need a source but don't have one." },
    { title: 'Source integrity', body: 'Checks whether cited sources actually support the claim made.' },
    { title: 'Generic phrasing', body: 'Flags writing that sounds AI-generic or unsupported by evidence.' },
    { title: 'Authorship signals', body: 'Surfaces review-only patterns that deserve a human look before submission.' },
  ];
  return (
    <section id="engine" className="landing-section checks-section">
      <CodeTexture id="landingChecks" />
      <div className="section-inner">
        <p className="eyebrow">How It Works</p>
        <h2>Four checks. One clear report.</h2>
        <p className="section-lead">DraftProof analyses your essay across four dimensions before you submit.</p>
        <div className="check-line">
          {checks.map((check, i) => (
            <article className="check-step" key={check.title}>
              <span>{i + 1}</span>
              <h3>{check.title}</h3>
              <p>{check.body}</p>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}

function BeliefsSection() {
  const beliefs = [
    'Every AI-like sentence is not misconduct.',
    'Every similarity match is not plagiarism.',
    'Students should not be judged by black-box scores.',
    'Rewriting everything does not make writing more honest.',
  ];
  return (
    <section id="trust" className="landing-section beliefs-section">
      <div className="section-inner">
        <p className="eyebrow">What DraftProof Believes</p>
        <h2>Writing tools should be fair, transparent, and useful.</h2>
        <div className="belief-row-grid">
          {beliefs.map((b) => (
            <div className="belief-row" key={b}><span aria-hidden="true">×</span>{b}</div>
          ))}
          <div className="belief-row belief-row-positive">
            <span aria-hidden="true">✓</span>
            We believe users deserve clear, evidence-based feedback that helps them improve their work.
          </div>
        </div>
      </div>
    </section>
  );
}

function LandingCTA({ onNav }) {
  return (
    <section id="cta" className="landing-cta">
      <CodeTexture id="landingCta" />
      <div className="section-inner">
        <p className="brand-pill">Before Turnitin, review what your instructor may question.</p>
        <h2>DraftProof helps you improve the essay before submission.</h2>
        <p>Before a paper, report, or essay is submitted, check that your work is original,
          properly cited, clearly written, and responsibly supported.</p>
        <a href="#" className="btn btn-ghost" onClick={onNav}>Review my essay</a>
        <small>1 token per 1,000 words · PDF report included · No bypass claims</small>
      </div>
    </section>
  );
}

function LandingFooter() {
  const links = ['Product', 'How it works', 'Sample report', 'Essay checker', 'Pricing', 'FAQ', 'Privacy', 'Security'];
  return (
    <footer className="landing-footer">
      <div className="section-inner landing-footer-inner">
        <div>
          <a href="#hero" className="footer-wordmark">DraftProof</a>
          <p>DraftProof provides writing integrity signals and review guidance. It does not
            determine misconduct, plagiarism, or AI authorship.</p>
        </div>
        <nav aria-label="Product">
          {links.map((l) => <a key={l} href="#">{l}</a>)}
          <a href="mailto:support@draftproof.app">support@draftproof.app</a>
        </nav>
      </div>
    </footer>
  );
}
