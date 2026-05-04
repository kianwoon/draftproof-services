import { Link } from 'react-router-dom';

const icons = {
  citation: (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M6.5 4.5h8.2c1.7 0 2.8 1 2.8 2.6v12.4H8a2.5 2.5 0 0 1-2.5-2.5V5.5c0-.6.4-1 1-1Z" />
      <path d="M8.2 7.8h6.5M8.2 11h5.1M8.2 14.2h6.5" />
    </svg>
  ),
  source: (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M10.8 17.2a6.4 6.4 0 1 1 4.5-1.9l3.4 3.4" />
      <path d="M8.3 10.8l2.1 2.1 4.1-4.7" />
    </svg>
  ),
  phrasing: (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M5.4 18.6 6 15l8.7-8.7a2 2 0 0 1 2.8 0l.2.2a2 2 0 0 1 0 2.8L9 18l-3.6.6Z" />
      <path d="m13.4 7.6 3 3M5.8 21h12.4" />
    </svg>
  ),
  authorship: (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 3.2 19 6v5.3c0 4.6-2.8 7.6-7 9.5-4.2-1.9-7-4.9-7-9.5V6l7-2.8Z" />
      <path d="m8.7 11.8 2.1 2.1 4.6-5" />
    </svg>
  ),
  paste: (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M8 5.5H6.8A2.3 2.3 0 0 0 4.5 7.8v10.4a2.3 2.3 0 0 0 2.3 2.3h10.4a2.3 2.3 0 0 0 2.3-2.3V7.8a2.3 2.3 0 0 0-2.3-2.3H16" />
      <path d="M8.5 6.8h7v-2h-7v2ZM8 11h8M8 14.3h6" />
    </svg>
  ),
  scan: (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M5 7V5h2M17 5h2v2M19 17v2h-2M7 19H5v-2M7 12h10M9.5 9h5M9.5 15h5" />
    </svg>
  ),
  report: (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M7 3.8h7.2L18 7.6v12.6H7V3.8Z" />
      <path d="M14 3.8v4h4M9.5 11h5M9.5 14h5M9.5 17h3" />
    </svg>
  ),
  revise: (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M5 17.8h4l9.1-9.1a2 2 0 0 0 0-2.8 2 2 0 0 0-2.8 0L6.2 15 5 17.8Z" />
      <path d="m14.2 7 2.8 2.8M5 21h14" />
    </svg>
  ),
};

export default function Landing() {
  return (
    <main className="snap-shell">
      {/* HERO */}
      <section id="hero" className="snap-section hero-section">
        <div className="section-inner hero-grid">
          <div className="hero-copy">
            <p className="eyebrow">Writing integrity reviews for education and research</p>
            <h1>Before you submit, prove your work is grounded.</h1>
            <p className="lead">
              DraftProof reviews your writing for citation gaps, source integrity,
              generic phrasing, and review-only authorship signals so you can
              improve the paper before submission.
            </p>

            <div className="hero-actions" id="check">
              <Link to="/scan" className="btn btn-primary">Run a pre-submission check</Link>
              <a href="#report" className="btn btn-secondary">View sample report</a>
            </div>

            <div className="value-strip" aria-label="DraftProof review details">
              <span>1 token per 1,000 words</span>
              <span>PDF report</span>
              <span>Citation + source grounding</span>
            </div>

            <div className="trust-note">
              <span className="mini-shield" aria-hidden="true">✓</span>
              <span>Not an AI detector. Not a plagiarism verdict. A writing integrity review you can act on.</span>
            </div>
          </div>

          <aside className="summary-card" aria-label="DraftProof quick summary">
            <div className="card-top">
              <div>
                <p className="card-kicker">DraftProof Quick Summary</p>
                <h2>Pre-submission review</h2>
              </div>
              <span className="pill">Live preview</span>
            </div>

            <div className="metric-grid">
              <div className="metric">
                <span>Review Tier</span>
                <strong className="tier-medium">Medium</strong>
                <div className="bar bar-warm"><i></i><i></i><i></i><i className="muted"></i><i className="muted"></i></div>
              </div>
              <div className="metric">
                <span>Grounding</span>
                <strong className="tier-low">Strong</strong>
                <div className="bar bar-green"><i></i><i></i><i className="muted"></i><i className="muted"></i><i className="muted"></i></div>
              </div>
              <div className="metric callout-metric">
                <span>Primary Fix</span>
                <strong>1 citation</strong>
                <em>Actionable</em>
              </div>
            </div>
          </aside>
        </div>
      </section>

      {/* PRODUCT */}
      <section id="product" className="snap-section">
        <div className="section-inner">
          <div className="split-heading">
            <div>
              <p className="eyebrow">Product</p>
              <h2>What DraftProof reviews</h2>
            </div>
            <p className="lead">
              Every scan checks your document across four integrity dimensions —
              giving you specific, actionable feedback instead of a single score.
            </p>
          </div>

          <div className="feature-grid">
            <div className="feature-card">
              <div className="feature-icon">{icons.citation}</div>
              <h3>Citation gaps</h3>
              <p>Finds claims that reference external facts but lack proper citations.</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">{icons.source}</div>
              <h3>Source integrity</h3>
              <p>Checks whether cited sources actually support the claims they are attached to.</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">{icons.phrasing}</div>
              <h3>Generic phrasing</h3>
              <p>Flags passages that rely on boilerplate or vague language instead of original analysis.</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">{icons.authorship}</div>
              <h3>Authorship signals</h3>
              <p>Surfaces passages that deserve human review without treating scores as verdicts.</p>
            </div>
          </div>
        </div>
      </section>

      {/* TRUST */}
      <section id="trust" className="snap-section trust-section">
        <div className="section-inner">
          <div className="split-heading">
            <div>
              <p className="eyebrow">Trust posture</p>
              <h2>Built for guidance, not accusation</h2>
            </div>
            <p className="lead">
              DraftProof is designed as a pre-submission review layer. It helps
              users strengthen evidence, wording, and source support while
              leaving final judgment with the writer, teacher, or institution.
            </p>
          </div>

          <div className="trust-grid">
            <div className="trust-card">
              <strong>Private by design</strong>
              <p>Review text is handled as user-submitted draft material, not public content.</p>
            </div>
            <div className="trust-card">
              <strong>No misconduct verdicts</strong>
              <p>Reports highlight review signals and suggested actions, not accusations.</p>
            </div>
            <div className="trust-card">
              <strong>Evidence-first feedback</strong>
              <p>Priority goes to claims, citations, source fit, and fixable writing issues.</p>
            </div>
          </div>
        </div>
      </section>

      {/* HOW IT WORKS */}
      <section id="engine" className="snap-section">
        <div className="section-inner">
          <div style={{textAlign: 'center', marginBottom: '3rem'}}>
            <p className="eyebrow">How it works</p>
            <h2>Four steps to a grounded paper</h2>
          </div>

          <div className="process-line">
            <div>
              <div className="step-icon">{icons.paste}</div>
              <h3>Paste</h3>
              <p>Paste your draft text for a pre-submission review.</p>
            </div>
            <div>
              <div className="step-icon">{icons.scan}</div>
              <h3>Scan</h3>
              <p>DraftProof reviews every section for fixable integrity signals.</p>
            </div>
            <div>
              <div className="step-icon">{icons.report}</div>
              <h3>Report</h3>
              <p>Get a structured report with severity ratings and locations.</p>
            </div>
            <div>
              <div className="step-icon">{icons.revise}</div>
              <h3>Rewrite</h3>
              <p>Revise flagged passages for clarity, evidence, and academic tone.</p>
            </div>
          </div>
        </div>
      </section>

      {/* SAMPLE REPORT */}
      <section id="report" className="snap-section">
        <div className="section-inner">
          <div style={{textAlign: 'center', marginBottom: '2.5rem'}}>
            <p className="eyebrow">Sample report</p>
            <h2>See what a review looks like</h2>
          </div>

          <div className="report-layout">
            <div className="report-main">
              <div className="report-metrics">
                <div><span>Review Tier</span><strong className="tier-medium">Medium</strong></div>
                <div><span>Grounding</span><strong className="tier-low">Strong</strong></div>
                <div><span>Citation repairs</span><strong>1</strong></div>
                <div><span>Review-only</span><strong>32</strong></div>
              </div>

              <div className="report-body">
                <table>
                  <thead>
                    <tr>
                      <th>Claim</th>
                      <th>Evidence issue</th>
                      <th>Severity</th>
                      <th>Suggested fix</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>Remote learning improved retention by 40%</td>
                      <td>Cited source does not support the exact statistic</td>
                      <td><strong className="tier-high">High</strong></td>
                      <td>Add a supporting citation or qualify the claim</td>
                    </tr>
                    <tr>
                      <td>This proves technology is always beneficial</td>
                      <td>Overgeneralized conclusion</td>
                      <td><strong className="tier-medium">Medium</strong></td>
                      <td>Narrow the statement and connect it to evidence</td>
                    </tr>
                    <tr>
                      <td>In conclusion, it is important to note...</td>
                      <td>Generic phrasing signal</td>
                      <td><strong className="tier-low">Low</strong></td>
                      <td>Rewrite only if it weakens your own voice</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <aside className="report-aside">
              <div className="mini-stats">
                <div><span>Citation repair</span><strong>1</strong></div>
                <div><span>Review-only</span><strong>32</strong></div>
              </div>

              <div className="primary-action">
                <strong>Primary action:</strong> verify the unsupported statistic before submission.
              </div>

              <ul className="check-list">
                <li><span>✓</span> Evidence issue located <strong>Page 3</strong></li>
                <li><span>✓</span> Suggested fix provided <strong>Ready</strong></li>
                <li><span>✓</span> Verdict language avoided <strong>Fair</strong></li>
              </ul>
            </aside>
          </div>
        </div>
      </section>

      {/* CTA + Footer */}
      <section id="cta" className="snap-section cta-footer">
        <div className="cta-inner">
          <h2>Use DraftProof to find what actually needs fixing before submission.</h2>
          <div className="hero-actions" style={{justifyContent: 'center'}}>
            <Link to="/scan" className="btn btn-light">Start pre-submission review</Link>
            <a href="#engine" className="btn btn-ghost">See how DraftProof works</a>
          </div>
        </div>

        <footer className="site-footer">
          <Link to="/" className="brand footer-brand">
            <span className="brand-mark" aria-hidden="true">
              <svg viewBox="0 0 32 32" role="img">
                <path d="M16 3 27 7v8c0 7.2-4.6 11.8-11 14C9.6 26.8 5 22.2 5 15V7l11-4Z" />
                <path d="m10.8 15.9 3.4 3.3 7.4-8" />
              </svg>
            </span>
            <span>DraftProof</span>
          </Link>

          <div className="footer-links">
            <a href="#product">Product</a>
            <a href="#engine">How it works</a>
            <a href="#report">Sample report</a>
            <a href="#pricing">Pricing</a>
            <Link to="/privacy">Privacy</Link>
            <Link to="/security">Security</Link>
          </div>

          <p>
            DraftProof provides writing integrity signals and review guidance.
            It does not determine misconduct, plagiarism, or AI authorship.
          </p>
        </footer>
      </section>
    </main>
  );
}
