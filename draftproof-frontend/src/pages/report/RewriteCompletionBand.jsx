import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import {
  formatMetricPercent,
  calibratedReportAiScore,
  formatSignedDelta,
  formatPlainScore,
} from './reportHelpers';

export default function RewriteCompletionBand({
  hasRewriteResult,
  rewriteOutcome,
  rewriteBandTitle,
  rewriteBandDetail,
  rewriteResultSummary,
  currentRewrite,
}) {
  const { t } = useTranslation();
  return hasRewriteResult ? (
    <div className={`report-rewrite-summary-bar${rewriteOutcome === 'suggestion_only' ? ' is-preserved' : ''}${rewriteOutcome === 'topk_blocked' ? ' is-blocked' : ''}${rewriteOutcome === 'ai_mitigated' ? ' is-mitigated' : ''}`}>
      <div className="rewrite-summary-icon" aria-hidden="true">
        <span>
          <svg width="42" height="42" viewBox="0 0 42 42" fill="none">
            <circle cx="21" cy="21" r="15" fill="currentColor"/>
            {rewriteOutcome === 'suggestion_only' || rewriteOutcome === 'topk_blocked' ? (
              <path d="M15 15l12 12M27 15L15 27" stroke="#fff" strokeWidth="3" strokeLinecap="round"/>
            ) : (
              <path d="M14 21.5l4.5 4.5L28.5 16" stroke="#fff" strokeWidth="3.2" strokeLinecap="round" strokeLinejoin="round"/>
            )}
          </svg>
        </span>
      </div>
      <div className="rewrite-summary-main">
        <span className="rewrite-summary-kicker">{t('report.rewrite.completion')}</span>
        <strong>{rewriteBandTitle}</strong>
        <em>{rewriteBandDetail}</em>
      </div>
      <div className="rewrite-summary-stat">
        <span>{formatMetricPercent(calibratedReportAiScore(rewriteResultSummary?.original_ai_authorship ?? rewriteResultSummary?.original_risk), 0)}</span>
        <small>{t('report.rewrite.aiBefore')}</small>
      </div>
      <div className="rewrite-summary-stat">
        <span>{formatMetricPercent(calibratedReportAiScore(rewriteResultSummary?.rewritten_ai_authorship ?? rewriteResultSummary?.rewrite_risk), 0)}</span>
        <small>{t('report.rewrite.aiAfter')}</small>
      </div>
      <div className="rewrite-summary-stat">
        <span>{formatPlainScore(rewriteResultSummary?.human_shift_score, 1)}</span>
        <small>{t('report.rewrite.humanShift')}</small>
      </div>
      <div className="rewrite-summary-stat">
        <span>{formatSignedDelta(rewriteResultSummary?.original_human_contribution, rewriteResultSummary?.rewritten_human_contribution)}</span>
        <small>{t('report.rewrite.humanContribution')}</small>
      </div>
      <div className="rewrite-summary-stat">
        <span>{formatSignedDelta(rewriteResultSummary?.original_ai_transformation, rewriteResultSummary?.rewritten_ai_transformation)}</span>
        <small>{t('report.rewrite.aiTransformation')}</small>
      </div>
      <div className="rewrite-summary-stat">
        <span>{formatSignedDelta(rewriteResultSummary?.original_grounding_quality_risk, rewriteResultSummary?.rewritten_grounding_quality_risk)}</span>
        <small>{t('report.rewrite.groundingRisk')}</small>
      </div>
      <Link
        to={`/rewrite/${currentRewrite.id}`}
        className="rewrite-results-link"
      >
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
          <path d="M5 2.5h5.2L13 5.3v10.2H5V2.5z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round"/>
          <path d="M10 2.5v3h3M6.8 8.3h4M6.8 11h4" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/>
        </svg>
        {t('report.rewrite.viewResult')}
      </Link>
    </div>
  ) : null;
}
