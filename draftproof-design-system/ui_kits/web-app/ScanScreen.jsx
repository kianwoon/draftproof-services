// ScanScreen — paste-a-draft workspace. Navy app hero with the live balance,
// a textarea with live word count + token math (free ≤500 words, 1 token per
// started 1,000 after that), and a simulated progress run that hands off to the
// report. Right rail lists the four review dimensions. Mirrors production Scan.jsx.
const { useState: useScanState, useRef: useScanRef } = React;

const SAMPLE_ESSAY = `The shift toward synthetic information has changed how students encounter knowledge. Many learners now meet a topic through summaries, chatbots, and aggregated explanations before they ever reach a primary source. This is efficient, but it can quietly detach a claim from the evidence that should support it.

Recent commentary suggests that automated writing assistance has become ubiquitous in higher education. Polished prose, however, does not guarantee that an argument is grounded. A paragraph can read fluently while resting on assertions that no cited source actually backs.

Pre-submission review addresses this gap directly. Rather than returning a single score, it separates the work that genuinely needs attention — missing citations, weak source support, over-uniform phrasing — from the writing that should simply be left alone.`;

function countWords(t) {
  const s = t.trim();
  return s ? s.split(/\s+/).length : 0;
}
function tokensFor(words) {
  return words <= 500 ? 0 : Math.ceil(words / 1000);
}

function ScanScreen({ go, balance }) {
  const [text, setText] = useScanState(SAMPLE_ESSAY);
  const [busy, setBusy] = useScanState(false);
  const [pct, setPct] = useScanState(0);
  const [msg, setMsg] = useScanState(null);
  const timer = useScanRef(null);
  const words = countWords(text);
  const tokens = tokensFor(words);

  const stages = [
    [8, 'Queuing scan'],
    [28, 'Scanning citations'],
    [52, 'Checking source grounding'],
    [74, 'Reviewing phrasing signals'],
    [92, 'Compiling authorship signals'],
    [100, 'Scan complete'],
  ];

  const run = (e) => {
    e.preventDefault();
    if (busy || !text.trim()) return;
    setBusy(true); setPct(0);
    let i = 0;
    const tick = () => {
      const [p, m] = stages[i];
      setPct(p); setMsg(m); i += 1;
      if (i < stages.length) {
        timer.current = setTimeout(tick, 620);
      } else {
        timer.current = setTimeout(() => go('report'), 700);
      }
    };
    tick();
  };

  return (
    <main className="app-page scan-shell">
      <div className="container">
        <section className="app-hero app-hero-dark">
          <CodeTexture id="scanHero" />
          <div>
            <p className="eyebrow">Pre-submission review</p>
            <h1>Scan your draft for fixable integrity signals.</h1>
            <p>Paste your text to review citation gaps, source grounding, generic phrasing, and
              authorship signals before submission.</p>
          </div>
          <div className="app-hero-stat">
            <span>Available balance</span>
            <strong>{balance} tokens</strong>
            <small>Free through 500 words</small>
          </div>
        </section>

        <section className="scan-workspace">
          <form className="scan-form" onSubmit={run}>
            <label className="scan-label" htmlFor="scan-text">
              Document text
              <span>Paste plain text from your paper, report, or essay.</span>
            </label>
            <p className="scan-pricing-note">Scans with 500 words or fewer are free. Token billing starts at 501 words.</p>
            <textarea id="scan-text" className="scan-textarea"
              placeholder="Paste your document text here..."
              value={text} onChange={(e) => setText(e.target.value)} rows={16} />
            <div className="scan-meta-row">
              <span>{words} {words === 1 ? 'word' : 'words'}</span>
              {words > 0 && tokens === 0 && <strong>Free scan</strong>}
              {tokens > 0 && <strong>{tokens} {tokens === 1 ? 'token' : 'tokens'} required</strong>}
            </div>

            <button type="submit" className="btn btn-primary" disabled={busy}>
              {busy ? (msg || 'Scanning...') : 'Start scan'}
            </button>
            <p className="scan-delivery-note">When the scan completes, we email a PDF copy of your report to your account email.</p>

            {busy && (
              <div className="scan-progress" role="status" aria-live="polite">
                <div className="scan-progress-meta">
                  <span>{msg || 'Scanning...'}</span>
                  <span>{pct}%</span>
                </div>
                <div className="scan-progress-track" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow={pct}>
                  <div className="scan-progress-fill" style={{ width: `${pct}%` }} />
                </div>
              </div>
            )}
          </form>

          <aside className="scan-side-panel" aria-label="What DraftProof checks">
            <p className="eyebrow">Review scope</p>
            <h2>What gets checked</h2>
            <ul>
              <li><span>1</span>Citation gaps and unsupported claims</li>
              <li><span>2</span>Source fit against the claim being made</li>
              <li><span>3</span>Generic or boilerplate phrasing</li>
              <li><span>4</span>Review-only authorship signals</li>
            </ul>
          </aside>
        </section>
      </div>
    </main>
  );
}
