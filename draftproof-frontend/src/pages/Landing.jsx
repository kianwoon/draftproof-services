import { Link } from 'react-router-dom';
import CodeTexture from '../components/CodeTexture';

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

const contentStrategies = [
  {
    type: 'Essay',
    strategy: 'Make the argument clearer',
    detail: 'Keep your main point and make the reasoning easier to follow.',
  },
  {
    type: 'Research with citations',
    strategy: 'Improve wording without moving sources',
    detail: 'Clean up weak sentences while keeping citations and source claims in place.',
  },
  {
    type: 'Technical writing',
    strategy: 'Keep exact meaning intact',
    detail: 'Protect terms, steps, definitions, and requirements from being over-polished.',
  },
  {
    type: 'Legal, policy, or medical text',
    strategy: 'Make careful edits only',
    detail: 'Change only what needs attention where wording mistakes can create risk.',
  },
  {
    type: 'Lists and tables',
    strategy: 'Keep the structure easy to follow',
    detail: 'Improve wording without breaking rows, bullets, comparisons, or findings.',
  },
  {
    type: 'Short answers',
    strategy: 'Add context or ask for more',
    detail: 'Tell you when the text is too short to judge instead of guessing.',
  },
  {
    type: 'Personal reflection',
    strategy: 'Keep your own voice',
    detail: 'Preserve your experience and viewpoint instead of making it sound generic.',
  },
  {
    type: 'Marketing copy',
    strategy: 'Match the audience and format',
    detail: 'Improve the message without forcing it into an academic writing style.',
  },
];

const sampleReportStats = [
  { label: 'Risk tier', value: 'Low Risk', tone: 'positive' },
  { label: 'Total findings', value: '50' },
  { label: 'Authorship rating', value: 'Good' },
  { label: 'Raw AI-style signal', value: '29.46%', tone: 'positive' },
  { label: 'Writing score', value: '32.84%', tone: 'accent' },
];

const sampleSignalGroups = [
  {
    title: 'AI Authorship Signals',
    description: 'Pattern, predictability, and model-like texture signals.',
    tone: 'ai',
    signals: [
      { label: 'Raw Top-k Predictability', value: 67 },
      { label: 'Patchwork variance', value: 48 },
      { label: 'AI likelihood', value: 29 },
    ],
  },
  {
    title: 'Human / Authenticity Signals',
    description: 'Grounding and anchor signals that reduce authorship certainty.',
    tone: 'human',
    signals: [
      { label: 'Human anchor', value: 84 },
      { label: 'Human anchor discount', value: 38 },
    ],
  },
  {
    title: 'Quality & Calibration Signals',
    description: 'Evidence quality, calibration confidence, and reporting caution.',
    tone: 'quality',
    signals: [
      { label: 'Calibration confidence', value: 57 },
      { label: 'Grounding risk', value: 50 },
      { label: 'Reporting suppression', value: 43 },
    ],
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
        <CodeTexture id="landingHero" className="hero-code-field" />
        <div className="section-inner landing-hero-grid">
          <div className="hero-copy">
            <p className="brand-pill">Writing integrity for education & research</p>
            <h1>
              Before you submit, prove your work is <span>grounded.</span>
            </h1>
            <p className="lead">
              DraftProof reviews your writing for citation gaps, source integrity,
              generic phrasing, and authorship signals, then applies the right strategy for
              the content type before you submit.
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

          <aside className="review-panel" aria-label="DraftProof quick summary">
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
          <span>Content-aware rewrite strategy</span>
        </div>
      </section>

      <section id="report" className="landing-section sample-report-section">
        <div className="section-inner sample-report-layout">
          <div className="sample-report-copy">
            <p className="eyebrow">Sample Report</p>
            <h2>A report that separates risk, authorship, and evidence quality.</h2>
            <p>
              DraftProof shows the difference between an AI-style signal, human
              grounding, and calibration confidence so reviewers can see why the
              result is not a simple verdict.
            </p>
            <div className="sample-report-points">
              <span>Risk summary before details</span>
              <span>Grouped signal families</span>
              <span>Actionable review context</span>
            </div>
            <Link to="/scan" className="btn btn-primary">Run your own scan</Link>
          </div>

          <SampleReportPreview />
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

      <section id="strategies" className="landing-section strategy-section">
        <div className="section-inner">
          <div className="strategy-heading">
            <div>
              <p className="eyebrow">Content-Aware Help</p>
              <h2>Different writing needs different help.</h2>
            </div>
            <p>
              An essay, a research paragraph, a policy note, and a personal reflection
              should not be rewritten the same way. DraftProof first understands what
              you wrote, then suggests the safest way to improve it.
            </p>
          </div>

          <div className="strategy-grid" aria-label="DraftProof content-aware rewrite strategies">
            {contentStrategies.map((item) => (
              <article className="strategy-card" key={item.type}>
                <span>{item.type}</span>
                <h3>{item.strategy}</h3>
                <p>{item.detail}</p>
              </article>
            ))}
          </div>

          <div className="strategy-proof">
            <strong>DraftProof does not blindly humanize text.</strong>
            <span>
              It helps you fix the right thing for the kind of writing you actually have.
            </span>
          </div>
        </div>
      </section>

      <section id="engine" className="landing-section checks-section">
        <CodeTexture id="landingChecks" />
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
        <CodeTexture id="landingCta" />
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

function SampleReportPreview() {
  return (
    <article className="sample-report-preview" aria-label="DraftProof sample analysis report preview">
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
          <span>Transformation Pattern</span>
          <h3>Human / uncertain pattern</h3>
          <div className="sample-report-tags">
            <em>Low Confidence</em>
            <em>Not a Verdict</em>
          </div>
        </div>
        <div className="sample-authorship-badge">
          <span>Authorship Rating</span>
          <strong>GOOD</strong>
          <small>11% calibrated risk</small>
        </div>
      </div>

      <div className="sample-contribution">
        <span>Estimated Contribution</span>
        <p>Human anchoring dominates, with limited AI transformation signal.</p>
        <div className="sample-contribution-bars">
          <SampleSignalBar label="Human Contribution" value={100} tone="human" />
          <SampleSignalBar label="AI Transformation" value={0} tone="ai" />
        </div>
      </div>

      <div className="sample-signal-profile">
        <div className="sample-signal-title">Signal Profile</div>
        {sampleSignalGroups.map((group) => (
          <div className={`sample-signal-group is-${group.tone}`} key={group.title}>
            <div className="sample-signal-group-head">
              <div>
                <h4>{group.title}</h4>
                <p>{group.description}</p>
              </div>
              <span>{group.signals.length} signals</span>
            </div>
            {group.signals.map((signal) => (
              <SampleSignalBar
                key={signal.label}
                label={signal.label}
                value={signal.value}
                tone={group.tone}
              />
            ))}
          </div>
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
