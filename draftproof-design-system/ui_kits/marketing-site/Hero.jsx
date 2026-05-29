// Hero — the navy "engine room" landing hero with the live code-texture motif
// and the floating "Quick Summary" review panel on the right. Copy is verbatim
// from production i18n (en).
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

function Hero({ onNav }) {
  return (
    <section id="hero" className="landing-hero">
      <CodeTexture id="landingHero" className="hero-code-field" />
      <div className="section-inner landing-hero-grid">
        <div className="hero-copy">
          <p className="brand-pill">Turnitin-safe essay review for students</p>
          <h1>Submit your essay with stronger evidence and <span className="hero-hl">lower Turnitin risk.</span></h1>
          <p className="lead">
            DraftProof helps students prepare for Turnitin-style review by finding citation
            gaps, weak source grounding, similarity risk, and AI-like writing patterns, then
            turning those signals into a revision plan.
          </p>

          <div className="hero-actions" id="check">
            <a href="#" className="btn btn-ghost" onClick={onNav}>Review my essay</a>
            <a href="#report" className="btn btn-ghost" onClick={onNav}>View sample report</a>
          </div>

          <div className="trust-note">
            <span className="mini-shield" aria-hidden="true" />
            <span>Not a Turnitin bypass · Not a misconduct verdict · A pre-submission essay review you can act on.</span>
          </div>
        </div>

        <aside className="review-panel" aria-label="DraftProof Quick Summary">
          <div className="review-panel-top">
            <p className="card-kicker">DraftProof Quick Summary</p>
            <span className="live-dot">Live preview</span>
          </div>
          <h2>Essay pre-submission review</h2>
          <p>Checking essay risks before submission...</p>

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
  );
}
