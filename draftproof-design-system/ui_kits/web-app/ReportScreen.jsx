// ReportScreen — a completed review. Four headline metrics, a findings table
// (claim → evidence issue → severity → suggested fix), an action breakdown, and
// a right rail with the signal stats, the primary fix, and a pre-submission
// checklist. Uses the production .report-* / .action-* / .check-list classes and
// the product's real data model (tiers, citation repair, review-only signals).
function ReportScreen({ go }) {
  const findings = [
    { claim: 'Automated writing assistance has become ubiquitous in higher education.', issue: 'Strong claim with no cited source.', sev: 'high', fix: 'Add a citation or soften to a hedged statement.' },
    { claim: 'Synthetic summaries can detach a claim from its evidence.', issue: 'Source supports the topic, not this specific point.', sev: 'medium', fix: 'Cite a source that backs the detachment claim directly.' },
    { claim: 'Polished prose does not guarantee a grounded argument.', issue: 'Generic phrasing; reads as boilerplate.', sev: 'low', fix: 'Replace with a specific example from your own analysis.' },
  ];
  const sevLabel = { high: 'High', medium: 'Medium', low: 'Low' };

  return (
    <main className="app-page report-page">
      <div className="container">
        <section className="app-hero app-hero-dark">
          <CodeTexture id="reportHero" />
          <div>
            <p className="eyebrow">Review complete</p>
            <h1>Pre-submission report</h1>
            <p>A clear plan: what may be questioned, why the score moved, and what to fix first —
              signals, not a verdict.</p>
          </div>
          <div className="app-hero-stat">
            <span>Document</span>
            <strong>4,982 words</strong>
            <small>PDF emailed to your account</small>
          </div>
        </section>

        <section className="report-layout">
          <div className="report-main">
            <div className="report-metrics">
              <div>
                <span>Review tier</span>
                <strong className="tier-medium">Medium</strong>
              </div>
              <div>
                <span>AI-style signal</span>
                <strong className="tier-low">Low · 18%</strong>
              </div>
              <div>
                <span>Citation gaps</span>
                <strong className="tier-medium">2 found</strong>
              </div>
              <div>
                <span>Source grounding</span>
                <strong className="tier-low">Strong</strong>
              </div>
            </div>

            <div className="report-body">
              <table aria-label="Findings">
                <thead>
                  <tr>
                    <th style={{ width: '34%' }}>Claim</th>
                    <th style={{ width: '26%' }}>Evidence issue</th>
                    <th style={{ width: '12%' }}>Severity</th>
                    <th style={{ width: '28%' }}>Suggested fix</th>
                  </tr>
                </thead>
                <tbody>
                  {findings.map((f, i) => (
                    <tr key={i}>
                      <td>{f.claim}</td>
                      <td>{f.issue}</td>
                      <td><span className={`severity ${f.sev}`}>{sevLabel[f.sev]}</span></td>
                      <td>{f.fix}</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              <div className="action-grid" style={{ marginTop: '1.4rem', border: '1px solid var(--line)', borderRadius: 'var(--radius-sm)', overflow: 'hidden' }}>
                <div className="action-row">
                  <span className="report-action-label">Citation repair</span>
                  <strong className="tier-medium">1</strong>
                </div>
                <div className="action-row">
                  <span className="report-action-label">Rewrite priority</span>
                  <strong>None</strong>
                </div>
                <div className="action-row">
                  <span className="report-action-label">Review-only signals</span>
                  <strong>32</strong>
                </div>
                <div className="action-row">
                  <span className="report-action-label">No action</span>
                  <strong className="tier-low">Clean</strong>
                </div>
              </div>
            </div>
          </div>

          <aside className="report-aside">
            <p className="eyebrow">Signal profile</p>
            <div className="mini-stats">
              <div><span>Predictability</span><strong>30.0%</strong></div>
              <div><span>Human anchor</span><strong className="tier-low">82%</strong></div>
              <div><span>Calibrated risk</span><strong>15%</strong></div>
              <div><span>Confidence</span><strong>61%</strong></div>
            </div>

            <div className="primary-action">
              <strong>Primary fix:</strong> add one citation to the strongest unsupported claim
              before submission. This clears the only high-severity finding.
            </div>

            <ul className="check-list">
              <li><span>✓</span>Claims checked for source support</li>
              <li><span>✓</span>Similarity terms reviewed in context</li>
              <li><span>✓</span>Authorship signals labeled review-only</li>
              <li><span>✓</span>PDF review trail generated</li>
            </ul>

            <button className="btn btn-primary" style={{ width: '100%', justifyContent: 'center', marginTop: '1.2rem' }}
              onClick={() => go('scan')}>Start a guided revision</button>
            <button className="btn btn-secondary btn-small" style={{ width: '100%', justifyContent: 'center', marginTop: '.6rem' }}
              onClick={() => go('dashboard')}>Back to dashboard</button>
          </aside>
        </section>
      </div>
    </main>
  );
}
