// DashboardScreen — the signed-in workspace. Navy hero with avatar + balance,
// a primary "start a scan" card beside two secondary cards, and the two-column
// scan/rewrite workflow explainer. Markup + copy mirror production Dashboard.jsx.
function DashboardScreen({ go, balance, email }) {
  const initials = (email || '?').charAt(0).toUpperCase();
  const scanSteps = [
    { title: 'Paste your draft', body: 'Add the text of your paper, report, or essay. Scans of 500 words or fewer are free.' },
    { title: 'Run the review', body: 'DraftProof checks citation gaps, source grounding, phrasing, and authorship signals.' },
    { title: 'Read the report', body: 'See a tier, a signal profile, and a prioritized list of what to fix first.' },
  ];
  const rewriteSteps = [
    { title: 'Start from a scan', body: 'A completed scan is required — guided revision uses the report to choose safe edits.' },
    { title: 'Review the changes', body: 'Improvements are highlighted; meaning and structure are preserved, not blindly humanized.' },
    { title: 'Keep the record', body: 'Download a PDF of the revised result for your pre-submission trail.' },
  ];

  return (
    <main className="dash-shell">
      <div className="container">
        <section className="dash-hero">
          <CodeTexture id="dashboardHero" />
          <div className="dash-welcome">
            <div className="dash-avatar">{initials}</div>
            <div>
              <p className="eyebrow">Your workspace</p>
              <h1>Welcome back</h1>
              <p className="dash-email">{email}</p>
            </div>
          </div>

          <div className="dash-hero-panel" aria-label="Account summary">
            <div>
              <span>Token balance</span>
              <strong>{balance} tokens</strong>
            </div>
            <a href="#" className="btn btn-ghost btn-small" onClick={(e) => e.preventDefault()}>Buy tokens</a>
          </div>
        </section>

        <section className="dash-grid" aria-label="Actions">
          <a href="#" className="dash-primary-card"
            onClick={(e) => { e.preventDefault(); go('scan'); }}>
            <div>
              <span className="brand-pill">Pre-submission review</span>
              <h2>Start a pre-submission scan</h2>
              <p>Paste your draft and review citation gaps, source grounding, generic phrasing,
                and authorship signals before you submit.</p>
              <p className="dash-free-note">Scans of 500 words or fewer are free.</p>
            </div>
            <div className="dash-card-footer">
              <span>1 token per 1,000 words</span>
              <strong>Start scan</strong>
            </div>
          </a>

          <div className="dash-side-stack">
            <a href="#" className="dash-small-card"
              onClick={(e) => { e.preventDefault(); go('report'); }}>
              <span className="dash-action-icon" aria-hidden="true">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                  <path d="M7 3.8h7.2L18 7.6v12.6H7V3.8Z" />
                  <path d="M14 3.8v4h4M9.5 11h5M9.5 14h5M9.5 17h3" />
                </svg>
              </span>
              <div>
                <h3>View reports</h3>
                <p>Open a completed review and its PDF.</p>
              </div>
            </a>

            <a href="#" className="dash-small-card" onClick={(e) => e.preventDefault()}>
              <span className="dash-action-icon" aria-hidden="true">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                  <path d="M12 6v6l3.5 2" />
                  <path d="M20 12a8 8 0 1 1-2.35-5.65" />
                  <path d="M20 4v5h-5" />
                </svg>
              </span>
              <div>
                <h3>Purchase history</h3>
                <p>Track token purchases and usage.</p>
              </div>
            </a>
          </div>
        </section>

        <section className="dash-section">
          <div className="dash-section-heading">
            <p className="eyebrow">How it works</p>
            <h2>Two workflows, one review trail</h2>
          </div>

          <div className="dash-workflow-grid">
            <div className="dash-workflow-panel">
              <div className="dash-workflow-panel-heading">
                <span className="brand-pill">Scan</span>
                <h3>Review the draft</h3>
              </div>
              <ol className="dash-steps">
                {scanSteps.map((s, i) => (
                  <li className="dash-step" key={s.title}>
                    <span className="step-num">{i + 1}</span>
                    <strong>{s.title}</strong>
                    <p>{s.body}</p>
                  </li>
                ))}
              </ol>
              <a href="#" className="dash-workflow-link"
                onClick={(e) => { e.preventDefault(); go('scan'); }}>Start scan</a>
            </div>

            <div className="dash-workflow-panel">
              <div className="dash-workflow-panel-heading">
                <span className="brand-pill">Guided revision</span>
                <h3>Improve responsibly</h3>
              </div>
              <ol className="dash-steps">
                {rewriteSteps.map((s, i) => (
                  <li className="dash-step" key={s.title}>
                    <span className="step-num">{i + 1}</span>
                    <strong>{s.title}</strong>
                    <p>{s.body}</p>
                  </li>
                ))}
              </ol>
              <a href="#" className="dash-workflow-link"
                onClick={(e) => { e.preventDefault(); go('report'); }}>Open reports</a>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
