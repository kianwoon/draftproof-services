// Stylometric-consistency "Writing-style outliers" panel (advisory,
// informational-only). Renders result.consistency_display — the server-side
// composer's single source of truth (poc/report/consistency_panel.py's
// compose_consistency_display). Mirrors the PDF panel byte-for-byte:
// KEEP-IN-SYNC with poc/report/render_panels.py's render_consistency_panel and
// poc/report/report.py's caller (both consume/produce the SAME dict shape).
//
// Phase 1 is informational only (ConsistencyDetector.overall_risk is
// unconditionally 0.0 — see poc/detect/consistency.py's module docstring):
// this component NEVER touches the tier or the AI-likelihood score. Returns
// null when `display` is absent/malformed (flag off / no paragraph flagged /
// older report), so the page renders exactly as before — byte-identical.
//
// allow-hardcode: the student-facing copy strings below are PRESENTATION
// text (mirroring render_claim_graph_panel's copy in render_panels.py, which
// is also plain English rather than i18n-keyed, since the PDF has no i18n
// table) — not a matching/scoring word-list.
export default function ConsistencyRisk({ display }) {
  if (!display || typeof display !== 'object' || display.present !== true) return null;

  const rows = Array.isArray(display.rows) ? display.rows : [];
  if (rows.length === 0) return null;

  const flagged = (display.summary && display.summary.flagged_paragraphs) ?? rows.length;

  return (
    <div className="consistency-risk-panel">
      <p className="consistency-risk-title">
        Writing-style outliers{' '}
        <span className="consistency-risk-pill">Informational · advisory</span>
      </p>
      <p className="consistency-risk-sub">
        Paragraphs whose sentence structure and word choice read differently than
        the rest of the document. This is <strong>not</strong> standalone
        evidence of AI generation or outsourcing — a different voice can also
        come from a quoted passage, a section written on a different day, or a
        legitimate co-author. It does <strong>not</strong> change the
        AI-likelihood score.
      </p>
      <p className="consistency-risk-sub">
        <strong>{flagged}</strong> paragraph{flagged === 1 ? '' : 's'} flagged for review.
      </p>
      <div className="consistency-risk-rows">
        {rows.map((row, i) => {
          if (!row || typeof row !== 'object') return null;
          const score = row.outlier_score;
          const scoreLabel = typeof score === 'number' ? ` (score ${score})` : '';
          return (
            <div className="consistency-risk-row" key={row.paragraph_id || i}>
              <div className="consistency-risk-row-head">
                <span className="consistency-risk-badge">{row.paragraph_id}</span>
                <span className="consistency-risk-score">{scoreLabel}</span>
              </div>
              <p className="consistency-risk-excerpt">{row.excerpt}</p>
              <p className="consistency-risk-features">
                Deviates in: {row.features_label}
              </p>
            </div>
          );
        })}
      </div>
      <p className="consistency-risk-footer">
        Advisory only — review these paragraphs, do not assume they are
        AI-generated or need to be rewritten.
      </p>
    </div>
  );
}
