import { Link } from 'react-router-dom';
import {
  formatDate,
  formatMetricPercent,
} from './reportHelpers';
import EditPencilIcon from './EditPencilIcon';

export default function ReportHero({
  t,
  locale,
  report,
  tier,
  issuesCount,
  writingScore,
  canStartRewrite,
  rewriteLoading,
  rewriteCanceling,
  rewriteInProgress,
  rewriteTokenEstimate,
  currentRewrite,
  onRewrite,
  onCancelRewrite,
  repairSummary,
  repairMainRiskLabel,
  repairActionLabel,
  repairActionHint,
  onRepairAction,
  isRewrittenView = false,
}) {
  const showActions = report.report_pdf_url || canStartRewrite || rewriteLoading || rewriteInProgress;

  return (
    <>
      <Link to="/reports" className="report-back">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
          <path d="M10 3L5 8l5 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
        {t('report.back')}
      </Link>

      <div className="report-hero">
        <div className="report-hero-title-row">
          <div className="report-doc-icon" aria-hidden="true">
            <svg width="42" height="42" viewBox="0 0 42 42" fill="none">
              <rect x="8" y="9" width="26" height="24" rx="5" stroke="currentColor" strokeWidth="3"/>
              <path d="M13 25l6-6 5 5 6-8" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
          <div className="report-hero-info">
            <div className="report-eyebrow">{isRewrittenView ? t('report.eyebrowRewritten') : t('report.eyebrow')}</div>
            <h1>{t('report.documentTitle')}</h1>
            {isRewrittenView && (
              <p className="report-hero-rewrite-note">{t('report.rewrittenHeroNote')}</p>
            )}
            {report.created_at && (
              <p className="report-meta">
                <svg width="17" height="17" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                  <path d="M4.5 1.8v2M11.5 1.8v2M2.5 6h11M3.5 3.5h9A1.5 1.5 0 0114 5v7.5A1.5 1.5 0 0112.5 14h-9A1.5 1.5 0 012 12.5V5a1.5 1.5 0 011.5-1.5z" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"/>
                </svg>
                {formatDate(report.created_at, locale)}
                {report.word_count > 0 && (
                  <span className="report-meta-separator"> · {t('report.words', { count: report.word_count })}</span>
                )}
              </p>
            )}
          </div>
          <div className="report-hero-stats" aria-label={t('report.overview')}>
            <div className="report-hero-stat">
              <span>{t('report.summary.riskTier')}</span>
              <strong style={{ color: tier.color }}>{t(`report.tiers.${report.tier}`, { defaultValue: tier.label })}</strong>
            </div>
            <div className="report-hero-stat">
              <span>{t('report.summary.writingScore')}</span>
              <strong>{writingScore != null ? formatMetricPercent(writingScore, 0) : '-'}</strong>
            </div>
            <div className="report-hero-stat">
              <span>{t('reports.findings')}</span>
              <strong>{issuesCount}</strong>
            </div>
          </div>
        </div>

        {showActions && (
          <div className="report-hero-actions">
            {report.report_pdf_url && (
              <div className="report-download-group">
                <a href={report.report_pdf_url} target="_blank" rel="noopener noreferrer" className="download-pdf-btn">
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                    <path d="M3 10v2.5A1.5 1.5 0 004.5 14h7a1.5 1.5 0 001.5-1.5V10M8 2v8M5 7l3 3 3-3" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  {t('report.downloadPdf')}
                </a>
                <span>{t('report.retentionNotice')}</span>
              </div>
            )}
            {(canStartRewrite || rewriteLoading || rewriteInProgress) && (
              <div className="rewrite-action-group">
                <button
                  type="button"
                  className="rewrite-btn"
                  onClick={onRewrite}
                  disabled={rewriteLoading || rewriteCanceling}
                >
                  {rewriteLoading ? t('report.rewrite.starting') : rewriteInProgress ? t('report.rewrite.resume') : t('report.rewrite.rewriteAiSections')}
                </button>
                {rewriteTokenEstimate && (
                  <strong className="rewrite-token-estimate">{rewriteTokenEstimate}</strong>
                )}
                <span>{t('report.rewrite.emailPdfNotice')}</span>
              </div>
            )}
            {rewriteInProgress && currentRewrite?.id && (
              <button
                type="button"
                className="rewrite-btn rewrite-cancel-btn"
                onClick={onCancelRewrite}
                disabled={rewriteCanceling}
              >
                {rewriteCanceling ? t('report.rewrite.canceling') : t('report.rewrite.cancelRewrite')}
              </button>
            )}
          </div>
        )}

        {repairSummary && (
          <div className="report-hero-repair" aria-label={t('report.repairSummary.ariaLabel')}>
            <div className="report-hero-repair-plan">
              <span className="report-hero-repair-kicker">{t('report.repairSummary.kicker')}</span>
              <strong>{repairSummary.status}</strong>
              <p>{repairSummary.nextAction}</p>
              <span className="report-hero-repair-note">{repairSummary.confidenceNote}</span>
            </div>
            <div className="report-hero-repair-risk">
              <span>{repairMainRiskLabel}</span>
              <strong>{repairSummary.mainRisk}</strong>
              {repairActionHint && <p className="report-hero-repair-action-hint">{repairActionHint}</p>}
              {repairActionLabel && onRepairAction && (
                <button type="button" className="repair-summary-action" onClick={onRepairAction}>
                  <EditPencilIcon />
                  {repairActionLabel}
                </button>
              )}
            </div>
          </div>
        )}
      </div>
    </>
  );
}
