import { SUBMISSION_RISK_AXES } from './reportHelpers';

// Leads the report with the additive 3-layer Submission-risk view: an overall
// level + plain-language reason, then the per-axis grid. The AI-likelihood % is
// demoted to the "text_pattern" axis (kept visible as the external trigger), and
// policy/declaration stays "self-declare" since the text can't reveal it.
// Renders nothing when `sr` is absent (older reports / abstained diagnosis).
export default function SubmissionRiskBand({ t, sr }) {
  if (!sr || !sr.overall || !sr.overall.level) return null;
  const { overall, axes = {} } = sr;
  const level = overall.level;

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

      <div className="submission-risk-axes">
        {SUBMISSION_RISK_AXES.map((key) => {
          const ax = axes[key] || {};
          const lvl = ax.level || 'unknown';
          return (
            <div className={`submission-risk-axis is-${lvl}`} key={key}>
              <span>{t(`report.submissionRisk.axes.${key}`)}</span>
              <strong>{t(`report.submissionRisk.levels.${lvl}`)}</strong>
              {key === 'text_pattern' && typeof ax.display_score === 'number' && (
                <em>{t('report.submissionRisk.externalTrigger', { score: Math.round(ax.display_score) })}</em>
              )}
              {key === 'policy_declaration' && (
                <em>{t('report.submissionRisk.selfDeclare')}</em>
              )}
            </div>
          );
        })}
      </div>

      <p className="submission-risk-note">{t('report.submissionRisk.note')}</p>
    </div>
  );
}
