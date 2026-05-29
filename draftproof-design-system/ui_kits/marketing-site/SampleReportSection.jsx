// SampleReportSection — the "See exactly what to fix" section with the
// interactive 3-tab report preview (AI Signal / Score Profile / Action Plan).
// Tabs auto-rotate every 4.5s, pause on hover/click. Verbatim copy + structure
// from the production SampleReportPreview component.
const { useState, useEffect, useMemo } = React;

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

const PREVIEW_TABS = [
  { id: 'aiSignal', label: 'AI Signal', summary: 'Authorship pattern' },
  { id: 'scoreProfile', label: 'Score Profile', summary: 'Why it moved' },
  { id: 'actionPlan', label: 'Action Plan', summary: 'What to fix' },
];

function SampleReportPreview() {
  const [activeSection, setActiveSection] = useState('aiSignal');
  const [isAutoPaused, setIsAutoPaused] = useState(false);
  const [isHoverPaused, setIsHoverPaused] = useState(false);
  const tabIds = useMemo(() => PREVIEW_TABS.map((t) => t.id), []);
  const isPaused = isAutoPaused || isHoverPaused;

  useEffect(() => {
    if (isPaused) return undefined;
    if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) return undefined;
    const timer = window.setTimeout(() => {
      setActiveSection((cur) => {
        const i = tabIds.indexOf(cur);
        return tabIds[(i + 1) % tabIds.length];
      });
    }, 4500);
    return () => window.clearTimeout(timer);
  }, [activeSection, isPaused, tabIds]);

  const select = (id) => { setActiveSection(id); setIsAutoPaused(true); };

  const scoreSignals = [
    { label: 'AI-style risk', value: 'Low', detail: 'Calibrated after human anchors', tone: 'warning' },
    { label: 'Source grounding', value: 'Review', detail: 'Claims checked for support', tone: 'quality' },
    { label: 'Fix priority', value: 'Clear', detail: 'Highest-impact issues first', tone: 'positive' },
  ];
  const actionItems = [
    { title: 'Add citation support', body: 'Two claims need clearer source backing before submission.', tone: 'warning' },
    { title: 'Strengthen source grounding', body: 'One paragraph should explain how the cited source supports the point.', tone: 'quality' },
    { title: 'Revise generic phrasing', body: 'Replace broad AI-style wording with specific reasoning and evidence.', tone: 'positive' },
  ];
  const notes = ['No single transformation pattern dominates', 'Human anchor reduced AI certainty', 'PDF report included'];

  return (
    <article
      className={`sample-report-preview${isPaused ? ' is-paused' : ''}`}
      aria-label="DraftProof sample essay review report preview"
      onMouseEnter={() => setIsHoverPaused(true)}
      onMouseLeave={() => setIsHoverPaused(false)}
    >
      <div className="sample-preview-tabs" role="tablist" aria-label="Sample report sections">
        {PREVIEW_TABS.map((tab) => (
          <button type="button" key={tab.id} role="tab"
            aria-selected={activeSection === tab.id}
            className={activeSection === tab.id ? 'is-active' : ''}
            onClick={() => select(tab.id)}>
            <span>{tab.label}</span>
            <em>{tab.summary}</em>
          </button>
        ))}
      </div>

      <div className="sample-preview-panel" role="tabpanel">
        {activeSection === 'aiSignal' && (
          <React.Fragment>
            <div className="sample-report-pattern">
              <div className="sample-report-pattern-main">
                <div className="sample-transformation-icon" aria-hidden="true">
                  <svg width="30" height="30" viewBox="0 0 30 30" fill="none">
                    <path d="M6 8.5h12.5M6 15h18M6 21.5h10" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round"/>
                    <path d="M21 7l3 3-3 3M18 18l-3 3 3 3" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </div>
                <div>
                  <span>Transformation Pattern</span>
                  <h3>Human / uncertain pattern</h3>
                  <div className="sample-report-tags">
                    <em>Low Confidence</em>
                    <em>Not a Verdict</em>
                  </div>
                </div>
              </div>
              <div className="sample-authorship-badge">
                <span>AI Signal</span>
                <strong>Low AI Signal</strong>
                <small>41% calibrated top-k · below 20% reference</small>
              </div>
            </div>

            <div className="sample-report-chart">
              <div className="sample-original-scan">
                <div className="sample-original-head">
                  <div>
                    <span>Original Scan</span>
                    <strong>Human / uncertain pattern</strong>
                  </div>
                  <em>18%</em>
                </div>
                <div className="sample-contribution">
                  <span>Estimated Contribution</span>
                  <p>Human anchoring dominates, with limited AI transformation signal.</p>
                  <div className="sample-report-tags">
                    <em>Calibrated AI risk 15%</em>
                    <em>Human anchor discount 38%</em>
                    <em>Calibration confidence 61%</em>
                    <em>Reporting suppression 39%</em>
                  </div>
                  <div className="sample-contribution-bars">
                    <SampleSignalBar label="Human Contribution" value={100} tone="human" />
                    <SampleSignalBar label="AI Transformation" value={0} tone="ai" />
                  </div>
                </div>
              </div>
              <div className="sample-report-notes">
                {notes.map((n) => <span key={n}>{n}</span>)}
              </div>
              <p className="sample-reference-note">Turnitin reference: AI scores below 20% may appear as *% instead of an exact percentage because low-range results are less reliable. DraftProof scores are review signals, not verdicts.</p>
            </div>
          </React.Fragment>
        )}

        {activeSection === 'scoreProfile' && (
          <div className="sample-section-card">
            <div className="sample-section-card-head">
              <span>Score Profile</span>
              <h3>Why the score moved</h3>
              <p>DraftProof groups scanner signals so you can see whether the score came from AI-like texture, weak grounding, or stronger human anchors.</p>
            </div>
            <div className="sample-score-profile-grid">
              {scoreSignals.map((s) => (
                <div className={`sample-score-signal is-${s.tone}`} key={s.label}>
                  <span>{s.label}</span>
                  <strong>{s.value}</strong>
                  <em>{s.detail}</em>
                </div>
              ))}
            </div>
            <div className="sample-profile-bars">
              <SampleSignalBar label="AI-style signal" value={18} tone="ai" />
              <SampleSignalBar label="Source grounding" value={64} tone="quality" />
              <SampleSignalBar label="Human anchor" value={82} tone="human" />
            </div>
          </div>
        )}

        {activeSection === 'actionPlan' && (
          <div className="sample-section-card">
            <div className="sample-section-card-head">
              <span>Action Plan</span>
              <h3>Fix the highest-impact issues first</h3>
              <p>The report turns scan signals into a practical review order before you revise or download the PDF.</p>
            </div>
            <div className="sample-action-list">
              {actionItems.map((item, i) => (
                <article className={`sample-action-item is-${item.tone}`} key={item.title}>
                  <span>{String(i + 1).padStart(2, '0')}</span>
                  <div>
                    <strong>{item.title}</strong>
                    <p>{item.body}</p>
                  </div>
                </article>
              ))}
            </div>
          </div>
        )}
      </div>
    </article>
  );
}

function SampleReportSection({ onNav }) {
  const valueCards = [
    { title: 'Explain the score', body: 'See the signal profile behind AI-style writing, grounding risk, similarity, and authorship uncertainty.' },
    { title: 'Prioritize the fix', body: 'Focus first on citation gaps, weak source support, and high-impact writing issues.' },
    { title: 'Review responsibly', body: 'Use rewrite guidance to improve clarity and evidence without blindly humanizing everything.' },
    { title: 'Keep a record', body: 'Download a PDF report showing what was checked before submission.' },
  ];
  return (
    <section id="report" className="landing-section sample-report-section">
      <div className="section-inner sample-report-layout">
        <div className="sample-report-copy">
          <p className="eyebrow">Sample Report</p>
          <h2>See exactly what to fix before you submit.</h2>
          <p>DraftProof turns a scan into a clear review plan: what may be questioned, why the
            score moved, which signals matter most, and how to revise responsibly without losing
            your own meaning.</p>
          <div className="sample-report-points">
            <span>Explain the score instead of guessing from one percentage</span>
            <span>Prioritize citation, source, similarity, and AI-style risks</span>
            <span>Keep a PDF review trail before submission</span>
          </div>
          <div className="sample-report-value-grid" aria-label="What DraftProof report gives you">
            {valueCards.map((c) => (
              <article key={c.title}>
                <strong>{c.title}</strong>
                <p>{c.body}</p>
              </article>
            ))}
          </div>
          <a href="#" className="btn btn-primary" onClick={onNav}>Run your own scan</a>
        </div>
        <SampleReportPreview />
      </div>
    </section>
  );
}
