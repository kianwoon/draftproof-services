import { SUBMISSION_RISK_AXES } from './reportHelpers';

// Leads the report with the additive 3-layer Submission-risk view: an overall
// level + plain-language reason, then a compact per-axis chip row (label ·
// level, color-coded, no per-chip body text — owner redesign 2026-07-04: the
// former 5 tall boxes made the hero too heavy). The AI-likelihood % stays
// visible in the single merged note line below the chips — it is load-bearing
// (the detector's absolute estimate the V7 panel's subtitle points to as
// "Text-pattern risk above"). Renders nothing when `sr` is absent (older
// reports / abstained diagnosis).
export default function SubmissionRiskBand({ t, sr }) {
  if (!sr || !sr.overall || !sr.overall.level) return null;
  const { overall, axes = {} } = sr;
  const level = overall.level;
  const textPattern = axes.text_pattern || {};
  const hasScore = typeof textPattern.display_score === 'number';

  return (
    <div className={`submission-risk-band is-${level}`} aria-label={t('report.submissionRisk.ariaLabel')}>
      <div className="submission-risk-headline">
        <span className="submission-risk-kicker">{t('report.submissionRisk.kicker')}</span>
        <strong className={`submission-risk-level is-${level}`}>
          {t(`report.submissionRisk.levels.${level}`)}
        </strong>
        <p className="submission-risk-reason submission-risk-ownership-lead">
          {t(`report.submissionRisk.ownershipLead.${level}`, { defaultValue: '' })}
        </p>
      </div>

      <div className="submission-risk-chips">
        {SUBMISSION_RISK_AXES.map((key) => {
          const lvl = (axes[key] || {}).level || 'unknown';
          return (
            <span className={`submission-risk-chip is-${lvl}`} key={key}>
              {t(`report.submissionRisk.axes.${key}`)}
              {' · '}
              <strong>{t(`report.submissionRisk.levels.${lvl}`)}</strong>
            </span>
          );
        })}
      </div>

      {/* One merged muted line: former per-chip AI-likelihood explainer + the
          band-wide note. Falls back to the note alone when no score exists. */}
      <p className="submission-risk-note">
        {hasScore
          ? t(
              sr._fused && sr._flagLine != null
                ? 'report.submissionRisk.compactNoteFusedAnchored'
                : sr._fused
                  ? 'report.submissionRisk.compactNoteFused'
                  : 'report.submissionRisk.compactNote',
              { score: Math.round(textPattern.display_score), flagLine: sr._flagLine },
            )
          : t('report.submissionRisk.note')}
      </p>
    </div>
  );
}
