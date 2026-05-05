import { Link } from 'react-router-dom';

const checks = [
  {
    title: 'Citation gaps',
    body: "Identifies claims that need a source but don't have one.",
  },
  {
    title: 'Source integrity',
    body: 'Checks whether cited sources actually support the claim made.',
  },
  {
    title: 'Generic phrasing',
    body: 'Flags writing that sounds AI-generic or unsupported by evidence.',
  },
  {
    title: 'Authorship signals',
    body: 'Surfaces review-only patterns that deserve a human look before submission.',
  },
];

const whyCards = [
  {
    title: 'We learn from synthetic information',
    body: 'Search engines, chatbots, and writing assistants summarise knowledge before we reach the original source. Writing can become detached from its evidence.',
    note: 'DraftProof bridges that gap',
  },
  {
    title: 'Traditional media is no longer the only source',
    body: 'Information now moves through AI newsrooms, generated summaries, and reported material. Polished does not mean proven.',
    note: 'Check the source, check the claim',
  },
  {
    title: 'AI detection alone is not enough',
    body: 'A score is not feedback. DraftProof asks better questions: is the claim supported, and what needs fixing?',
    note: 'Actionable, not just a verdict',
  },
];

const beliefs = [
  'Every AI-like sentence is not misconduct.',
  'Every similarity match is not plagiarism.',
  'Students should not be judged by black-box scores.',
  'Rewriting everything does not make writing more honest.',
];

export default function Landing() {
  return (
    <main className="landing-page">
      <section id="hero" className="landing-hero">
        <HeroCodeField />
        <div className="section-inner landing-hero-grid">
          <div className="hero-copy">
            <p className="brand-pill">Writing integrity for education & research</p>
            <h1>
              Before you submit, prove your work is <span>grounded.</span>
            </h1>
            <p className="lead">
              DraftProof reviews your writing for citation gaps, source integrity,
              generic phrasing, and authorship signals so you can fix issues before submission.
            </p>

            <div className="hero-actions" id="check">
              <Link to="/scan" className="btn btn-ghost">Run a pre-submission check</Link>
              <a href="#report" className="btn btn-ghost">View sample report</a>
            </div>

            <div className="trust-note">
              <span className="mini-shield" aria-hidden="true" />
              <span>Not an AI detector · Not a plagiarism verdict · A writing integrity review you can act on.</span>
            </div>
          </div>

          <aside className="review-panel" id="report" aria-label="DraftProof quick summary">
            <div className="review-panel-top">
              <p className="card-kicker">DraftProof Quick Summary</p>
              <span className="live-dot">Live preview</span>
            </div>
            <h2>Pre-submission review</h2>
            <p>Running integrity check on your draft...</p>

            <div className="review-grid">
              <Metric label="Review tier" value="Medium" tone="warning" width="50%" />
              <Metric label="Grounding" value="Strong" tone="positive" width="82%" />
              <Metric label="Citation gaps" value="2 found" tone="warning" width="35%" />
              <Metric label="Source integrity" value="Verified" tone="positive" width="92%" />
            </div>

            <div className="primary-fix">
              <div>
                <span>Primary fix</span>
                <strong>1 citation</strong>
              </div>
              <em>Actionable</em>
            </div>
          </aside>
        </div>
      </section>

      <section className="trust-bar" aria-label="DraftProof audiences and review details">
        <div className="section-inner trust-bar-inner">
          <span>Built for</span>
          <strong>Students</strong>
          <strong>Academic researchers</strong>
          <strong>Educators</strong>
          <strong>Policy writers</strong>
          <span>1 token per 1,000 words</span>
          <span>PDF report</span>
          <span>Citation + source grounding</span>
        </div>
      </section>

      <section id="product" className="landing-section">
        <div className="section-inner">
          <p className="eyebrow">Why DraftProof Exists</p>
          <h2>The way people read, write, and cite has fundamentally changed.</h2>
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

      <section id="engine" className="landing-section checks-section">
        <div className="section-inner">
          <p className="eyebrow">How It Works</p>
          <h2>Four checks. One clear report.</h2>
          <p className="section-lead">DraftProof analyses your writing across four dimensions before you submit.</p>
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
          <p className="eyebrow">What DraftProof Believes</p>
          <h2>Writing tools should be fair, transparent, and useful.</h2>
          <div className="belief-row-grid">
            {beliefs.map((belief) => (
              <div className="belief-row" key={belief}>
                <span aria-hidden="true">×</span>
                {belief}
              </div>
            ))}
            <div className="belief-row belief-row-positive">
              <span aria-hidden="true">✓</span>
              We believe users deserve clear, evidence-based feedback that helps them improve their work.
            </div>
          </div>
        </div>
      </section>

      <section id="cta" className="landing-cta">
        <div className="section-inner">
          <p className="brand-pill">The world now produces more information than people can easily verify.</p>
          <h2>DraftProof is that review layer.</h2>
          <p>
            Before a paper, report, or essay is submitted, check that the work
            is properly grounded, clearly written, and responsibly supported.
          </p>
          <Link to="/scan" className="btn btn-ghost">Run a pre-submission review</Link>
          <small>1 token per 1,000 words · PDF report included · No AI verdict</small>
        </div>
      </section>

      <footer className="landing-footer">
        <div className="section-inner landing-footer-inner">
          <div>
            <Link to="/" className="footer-wordmark">DraftProof</Link>
            <p>
              DraftProof provides writing integrity signals and review guidance.
              It does not determine misconduct, plagiarism, or AI authorship.
            </p>
          </div>
          <nav aria-label="Footer">
            <a href="#product">Product</a>
            <a href="#engine">How it works</a>
            <a href="#report">Sample report</a>
            <Link to="/pricing">Pricing</Link>
            <Link to="/privacy">Privacy</Link>
            <Link to="/security">Security</Link>
          </nav>
        </div>
      </footer>
    </main>
  );
}

function HeroCodeField() {
  return (
    <svg
      className="hero-code-field"
      viewBox="0 0 1440 620"
      aria-hidden="true"
      focusable="false"
      preserveAspectRatio="none"
    >
      <defs>
        <linearGradient id="codeFade" x1="0" x2="1" y1="0" y2="0">
          <stop offset="0%" stopColor="#0D1B2A" stopOpacity="0" />
          <stop offset="12%" stopColor="#0D1B2A" stopOpacity="0.95" />
          <stop offset="78%" stopColor="#0D1B2A" stopOpacity="0.78" />
          <stop offset="100%" stopColor="#0D1B2A" stopOpacity="0" />
        </linearGradient>
        <mask id="codeMask">
          <rect width="1440" height="620" fill="url(#codeFade)" />
        </mask>
      </defs>

      <g mask="url(#codeMask)" className="hero-code-layer">
        <g className="code-row code-row-a">
          <CodeLine y="78" text="0101  source.check()  1100  citation.match  0010  grounded=true  1011" />
          <CodeLine y="174" text="claim.verify  1001  source:linked  0110  citation.add  1010  review.only" />
          <CodeLine y="270" text="0011  evidence.fit  1110  no_verdict  0101  grounded=true  1001" />
          <CodeLine y="366" text="source.check()  0111  claim.supported  1010  citation.gap  0001" />
          <CodeLine y="462" text="1010  draft.review  0011  source.grounding  1101  fix.before.submit" />
        </g>
        <g className="code-row code-row-b">
          <CodeLine y="126" text="grounded=true  0110  citation.match  1011  claim.verify  0010" />
          <CodeLine y="222" text="1100  source.integrity  0001  review.signal  1010  evidence.fit" />
          <CodeLine y="318" text="citation.gap  0101  source.check()  1110  no_ai_verdict  1001" />
          <CodeLine y="414" text="0010  claim.supported  1011  draftproof.scan  0110  grounded" />
          <CodeLine y="510" text="review.only  1001  fix.citation  0101  source.linked  1100" />
        </g>
      </g>
    </svg>
  );
}

function CodeLine({ y, text }) {
  return (
    <text x="0" y={y}>
      <tspan>{text}</tspan>
      <tspan dx="64">{text}</tspan>
    </text>
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
