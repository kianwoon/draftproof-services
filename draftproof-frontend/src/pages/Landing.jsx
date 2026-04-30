import { Link } from 'react-router-dom';

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
              generic phrasing, and authorship-risk signals — so you can improve
              the paper before it becomes a problem.
            </p>

            <div className="hero-actions" id="check">
              <Link to="/scan" className="btn btn-primary">Run a pre-submission check</Link>
              <a href="#report" className="btn btn-secondary">View sample report</a>
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
                <span>Overall Tier</span>
                <strong className="tier-medium">Medium</strong>
                <div className="bar bar-warm"><i></i><i></i><i></i><i className="muted"></i><i className="muted"></i></div>
              </div>
              <div className="metric">
                <span>AI Likelihood</span>
                <strong className="tier-low">Low</strong>
                <div className="bar bar-green"><i></i><i></i><i className="muted"></i><i className="muted"></i><i className="muted"></i></div>
              </div>
              <div className="donut" style={{'--value': 16.8}}>
                <strong>16.8%</strong>
                <span>estimated</span>
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
              <div className="feature-icon">📚</div>
              <h3>Citation gaps</h3>
              <p>Finds claims that reference external facts but lack proper citations.</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">🔍</div>
              <h3>Source integrity</h3>
              <p>Checks whether cited sources actually support the claims they are attached to.</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">✍️</div>
              <h3>Generic phrasing</h3>
              <p>Flags passages that rely on boilerplate or vague language instead of original analysis.</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">🛡️</div>
              <h3>Authorship signals</h3>
              <p>Identifies patterns associated with AI-generated or ghostwritten content.</p>
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
              <div className="step-icon">📄</div>
              <h3>Paste</h3>
              <p>Paste your draft text for a pre-submission review.</p>
            </div>
            <div>
              <div className="step-icon">🔬</div>
              <h3>Scan</h3>
              <p>AI reviews every section for integrity signals.</p>
            </div>
            <div>
              <div className="step-icon">📊</div>
              <h3>Report</h3>
              <p>Get a structured report with severity ratings and locations.</p>
            </div>
            <div>
              <div className="step-icon">✏️</div>
              <h3>Rewrite</h3>
              <p>Apply suggested fixes or rewrite flagged passages yourself.</p>
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
                <div><span>Overall Tier</span><strong className="tier-medium">Medium</strong></div>
                <div><span>AI Likelihood</span><strong className="tier-low">16.8%</strong></div>
                <div><span>Predictability</span><strong>30.0%</strong></div>
                <div><span>Grounding</span><strong className="tier-low">Strong</strong></div>
              </div>

              <div className="report-body">
                <table>
                  <thead>
                    <tr>
                      <th>Issue</th>
                      <th>Severity</th>
                      <th>Location</th>
                      <th>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>Uncited statistical claim</td>
                      <td><strong className="tier-high">High</strong></td>
                      <td>Page 3, ¶2</td>
                      <td>Add citation</td>
                    </tr>
                    <tr>
                      <td>Generic transition phrase</td>
                      <td><strong className="tier-medium">Medium</strong></td>
                      <td>Page 5, ¶1</td>
                      <td>Review-only</td>
                    </tr>
                    <tr>
                      <td>Predictable sentence structure</td>
                      <td><strong className="tier-low">Low</strong></td>
                      <td>Page 7, ¶3</td>
                      <td>No action needed</td>
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
                <strong>Primary action:</strong> Add citation for one uncited claim.
              </div>

              <ul className="check-list">
                <li><span>✓</span> Citation repair <strong>1</strong></li>
                <li><span>✓</span> Review-only signals <strong>32</strong></li>
                <li><span>✓</span> No action needed <strong>OK</strong></li>
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
            <a href="#privacy">Privacy</a>
            <a href="#security">Security</a>
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
