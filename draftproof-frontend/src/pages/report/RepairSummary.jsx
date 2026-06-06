export default function RepairSummary({
  summary,
  ariaLabel,
  kicker,
  mainRiskLabel,
  actionLabel,
  onAction,
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
          {actionLabel && onAction && (
            <button type="button" className="repair-summary-action" onClick={onAction}>
              {actionLabel}
            </button>
          )}
        </div>
      </div>
    </section>
  );
}
