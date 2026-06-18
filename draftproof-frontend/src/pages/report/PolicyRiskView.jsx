import { useState } from 'react';

// Two policy-interpreted scores from one signal engine. Leads with a Low/Moderate/
// High/Severe level (the score number is secondary). Each row has a "confirm
// yourself" checkbox for the text-underivable factor (declaration / process); checking
// it subtracts the server-computed confirm_delta — no scoring formula on the client.
const ROWS = [
  { key: 'ai_allowed', labelKey: 'allowedLabel', confirmKey: 'confirmAllowed' },
  { key: 'ai_restricted', labelKey: 'restrictedLabel', confirmKey: 'confirmRestricted' },
];

export default function PolicyRiskView({ t, pr }) {
  const [confirmed, setConfirmed] = useState({ ai_allowed: false, ai_restricted: false });
  if (!pr || !pr.ai_allowed || !pr.ai_allowed.level || pr.ai_allowed.level === 'unknown') return null;

  return (
    <div className="policy-risk" aria-label={t('report.policyRisk.heading')}>
      <div className="policy-risk-head">
        <span className="policy-risk-kicker">{t('report.policyRisk.heading')}</span>
        <p className="policy-risk-sub">{t('report.policyRisk.subheading')}</p>
      </div>

      <div className="policy-risk-rows">
      {ROWS.map((row) => {
        const p = pr[row.key] || {};
        const isConfirmed = confirmed[row.key];
        const level = isConfirmed ? (p.confirm_level || p.level) : p.level;
        const score = typeof p.score === 'number'
          ? Math.max(0, Math.round(p.score - (isConfirmed ? (p.confirm_delta || 0) : 0)))
          : null;
        return (
          <div className={`policy-risk-row is-${level}`} key={row.key}>
            <div className="policy-risk-row-head">
              <span className="policy-risk-mode">{t(`report.policyRisk.${row.labelKey}`)}</span>
              <strong className={`policy-risk-level is-${level}`}>
                {score != null && <span className="policy-risk-score">{score}</span>}
                <span className="policy-risk-band-label">
                  {t(`report.policyRisk.levels.${level}`, { defaultValue: level })}
                </span>
              </strong>
            </div>
            {p.main_issue && (
              <>
                <p className="policy-risk-issue">
                  <span>{t('report.policyRisk.mainIssuePrefix')}</span>{' '}
                  {t(`report.policyRisk.issues.${p.main_issue}`, { defaultValue: '' })}
                </p>
                <p className="policy-risk-fix">
                  <span>{t('report.policyRisk.bestFixPrefix')}</span>{' '}
                  {t(`report.policyRisk.fixes.${p.main_issue}`, { defaultValue: '' })}
                </p>
              </>
            )}
            {(p.confirm_delta || 0) > 0 && (
              <label className="policy-risk-confirm">
                <input
                  type="checkbox"
                  checked={isConfirmed}
                  onChange={(e) => setConfirmed((c) => ({ ...c, [row.key]: e.target.checked }))}
                />
                {t(`report.policyRisk.${row.confirmKey}`)}
              </label>
            )}
          </div>
        );
      })}
      </div>

      <p className="policy-risk-disclaimer">{t('report.policyRisk.disclaimer')}</p>
    </div>
  );
}
