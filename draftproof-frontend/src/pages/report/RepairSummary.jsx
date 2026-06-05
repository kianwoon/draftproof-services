export default function RepairSummary({
  summary,
  canEdit,
  canRewrite,
  rewriteInProgress,
  rewriteLoading,
  onEditDraft,
  onRewrite,
  pdfUrl,
  downloadLabel,
  rewriteLabel,
  editLabel,
  ariaLabel,
  kicker,
  mainRiskLabel,
}) {
  if (!summary) return null;

  return (
    <section className="repair-summary-card" aria-label={ariaLabel}>
      <div className="repair-summary-main">
        <span className="repair-summary-kicker">{kicker}</span>
        <h2>{summary.status}</h2>
        <p>{summary.nextAction}</p>
        <div className="repair-summary-note">{summary.confidenceNote}</div>
      </div>
      <div className="repair-summary-side">
        <div className="repair-summary-risk">
          <span>{mainRiskLabel}</span>
          <strong>{summary.mainRisk}</strong>
        </div>
        <div className="repair-summary-actions">
          {canEdit && (
            <button type="button" className="btn btn-secondary" onClick={onEditDraft}>
              {editLabel}
            </button>
          )}
          {canRewrite && (
            <button type="button" className="rewrite-btn" onClick={onRewrite} disabled={rewriteLoading}>
              {rewriteLabel}
            </button>
          )}
          {pdfUrl && (
            <a href={pdfUrl} target="_blank" rel="noopener noreferrer" className="download-pdf-btn">
              {downloadLabel}
            </a>
          )}
          {rewriteInProgress && !canRewrite && (
            <button type="button" className="rewrite-btn" onClick={onRewrite} disabled={rewriteLoading}>
              {rewriteLabel}
            </button>
          )}
        </div>
      </div>
    </section>
  );
}
