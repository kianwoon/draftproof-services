import { useState } from 'react';

// Two policy-interpreted scores from one signal engine. Leads with a Low/Moderate/
// High/Severe level (the score number is secondary). Each row has a "confirm
// yourself" checkbox for the text-underivable factor (declaration / process); checking
// it subtracts the server-computed confirm_delta — no scoring formula on the client.
const ROWS = [
  { key: 'ai_allowed', labelKey: 'allowedLabel', confirmKey: 'confirmAllowed' },
  { key: 'ai_restricted', labelKey: 'restrictedLabel', confirmKey: 'confirmRestricted' },
];

// Phase 2 (docs/plans/policy_risk_external_review_response.md): pr.headline
// ("allowed"/"restricted"/"both", computed server-side by
// poc/detect/policy_risk.py's _select_headline) picks which row to show when the
// assignment's real ai_policy is known. Absent/"both" (older reports, or ai_policy
// not offered/unknown) renders both rows exactly as before Phase 2 -- pure
// selection, never a client-side scoring decision.
const HEADLINE_ROW_KEY = { allowed: 'ai_allowed', restricted: 'ai_restricted' };

export default function PolicyRiskView({ t, pr }) {
  const [confirmed, setConfirmed] = useState({ ai_allowed: false, ai_restricted: false });
  if (!pr || !pr.ai_allowed || !pr.ai_allowed.level || pr.ai_allowed.level === 'unknown') return null;

  const headline = pr.headline || 'both';
  const rows = headline === 'both' ? ROWS : ROWS.filter((row) => row.key === HEADLINE_ROW_KEY[headline]);

  return (
    <div className="policy-risk" aria-label={t('report.policyRisk.heading')}>
      <div className="policy-risk-head">
        <span className="policy-risk-kicker">{t('report.policyRisk.heading')}</span>
        <p className="policy-risk-sub">
          {t(headline === 'both' ? 'report.policyRisk.subheading' : 'report.policyRisk.subheadingKnown')}
        </p>
      </div>

      {headline === 'restricted' && pr.headline_policy === 'editing_only' && (
        <p className="policy-risk-editing-note">{t('report.policyRisk.editingOnlyNote')}</p>
      )}

      <div className="policy-risk-rows">
      {rows.map((row) => {
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
            {/* Ordering-floor disclosure (poc/detect/policy_risk.py:_apply_ordering_floor):
                when the strict lens organically scored LOWER than the permissive one, the
                displayed score is matched upward. Silent flooring reads as a bug (two
                identical scores); this line names the rule without showing a second
                number that could be misread as "the real score". Hidden once EITHER
                confirm checkbox is ticked: the client-side confirm_delta subtraction
                can legitimately push one card below the other, which would make this
                "never reads lower" claim literally false on screen. */}
            {p.floored_to_ai_allowed === true
              && !confirmed.ai_allowed && !confirmed.ai_restricted && (
              <p className="policy-risk-floor-note">{t('report.policyRisk.flooredNote')}</p>
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
