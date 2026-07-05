// draftproof-frontend/src/pages/report/MergedAuthorshipRisk.jsx
// Unified V7 card: one AI-likelihood headline + two labeled lenses (composition bars
// vs risk axes) + one DraftProof scale. Replaces AuthorshipClarityBreakdown.jsx +
// SubmissionRiskBand.jsx. Additive/null-safe: renders nothing when both `breakdown`
// and `sr` are absent; degrades gracefully when only one is present.
import { SUBMISSION_RISK_AXES } from './reportHelpers';

const CATEGORY_ORDER = [
  'student_owned',
  'ai_assisted_polished',
  'ai_paraphrased',
  'ai_generated_like',
];

// Bar-fill color per category (spec Global Constraints — must match PDF pdf.py).
// allow-hardcode: presentation colors (bar fills), not a scoring/matching list.
const CATEGORY_COLOR = {
  student_owned: '#1D9E75',
  ai_assisted_polished: '#888780',
  ai_paraphrased: '#D85A30',
  ai_generated_like: '#D85A30',
};

const KNOWN_DEEP_SCAN_BANDS = ['insufficient', 'amber', 'orange', 'red'];
const KNOWN_AUTHORITATIVE_TIERS = ['green', 'amber', 'orange', 'red'];
// Single source of truth: band label + chip color both derive from the tier the
// backend assigned. Do NOT recompute a band from the fused score against frontend
// cutoffs (duplicates backend threshold logic — a no-hardcode violation).
const TIER_TO_BAND = { green: 'low', amber: 'moderate', orange: 'high', red: 'critical' };

function CategoryBar({ t, category, raw, band }) {
  const hasRaw = typeof raw === 'number' && Number.isFinite(raw);
  const widthPct = hasRaw ? Math.max(0, Math.min(100, raw * 100)) : 0;
  const bandLabel = band
    ? t(`report.authorshipBreakdown.bands.${band}`)
    : t('report.authorshipBreakdown.bands.None');
  return (
    <div className="merged-comp-row">
      <div className="merged-comp-row-head">
        <span className="merged-comp-label">
          {t(`report.authorshipBreakdown.categories.${category}`)}
        </span>
        <span className="merged-comp-band">
          {hasRaw ? `${bandLabel} · ${Math.round(widthPct)}%` : bandLabel}
        </span>
      </div>
      <div className="merged-comp-track">
        <div
          className="merged-comp-fill"
          style={{ width: `${widthPct}%`, background: CATEGORY_COLOR[category] }}
        />
      </div>
    </div>
  );
}

// The ONE headline. Fused score (verdict) when tier_authority present, else the
// deep-scan-only estimate. The ownership lead + flag-line note ride alongside.
function VerdictBand({ t, breakdown, sr, authoritativeTier, tierAuthority }) {
  const hasAuthoritativeTier = KNOWN_AUTHORITATIVE_TIERS.includes(authoritativeTier);
  const level = sr && sr.overall ? sr.overall.level : null;

  let valueEl = null;
  let chipEl = null;
  let evidenceEl = null;

  if (tierAuthority && typeof tierAuthority.fused_score === 'number') {
    const band = hasAuthoritativeTier ? TIER_TO_BAND[authoritativeTier] : null;
    valueEl = <strong className="merged-verdict-value">{Math.round(tierAuthority.fused_score)}%</strong>;
    if (band) {
      chipEl = (
        <span className={`merged-verdict-chip is-${authoritativeTier}`}>
          {t(`report.authorshipBreakdown.fusedHeadline.bands.${band}`)}
        </span>
      );
    }
    if (typeof tierAuthority.composite_score === 'number') {
      evidenceEl = (
        <p className="merged-verdict-evidence">
          {t('report.authorshipBreakdown.fusedHeadline.evidence', {
            composite: Math.round(tierAuthority.composite_score),
            deepScan: Math.round((tierAuthority.proportion || 0) * 100),
          })}
        </p>
      );
    }
  } else if (breakdown && breakdown.deep_scan && KNOWN_DEEP_SCAN_BANDS.includes(breakdown.deep_scan.band)) {
    const ds = breakdown.deep_scan;
    const insufficient = ds.band === 'insufficient';
    const hasProp = typeof ds.proportion === 'number' && Number.isFinite(ds.proportion);
    if (insufficient || !hasProp) {
      chipEl = <span className="merged-verdict-chip is-insufficient">{t('report.authorshipBreakdown.deepScan.insufficientChip')}</span>;
    } else {
      valueEl = <strong className="merged-verdict-value">{Math.round(ds.proportion * 100)}%</strong>;
      const chipTier = hasAuthoritativeTier ? authoritativeTier : ds.band;
      chipEl = (
        <span className={`merged-verdict-chip is-${chipTier}`}>
          {hasAuthoritativeTier
            ? t('report.authorshipBreakdown.deepScan.bandDefersToTier', {
                band: t(`report.authorshipBreakdown.deepScan.bands.${ds.band}`),
                tier: t(`report.tiers.${authoritativeTier}`),
              })
            : t(`report.authorshipBreakdown.deepScan.bands.${ds.band}`)}
        </span>
      );
    }
    evidenceEl = <p className="merged-verdict-evidence">{t('report.authorshipBreakdown.deepScan.notTurnitin')}</p>;
  }

  const ownershipLead = level
    ? t(`report.submissionRisk.ownershipLead.${level}`, { defaultValue: '' })
    : '';
  const textPattern = (sr && sr.axes && sr.axes.text_pattern) || {};
  const hasScore = typeof textPattern.display_score === 'number';
  const flagNote = hasScore
    ? t(
        sr._fused && sr._flagLine != null
          ? 'report.submissionRisk.compactNoteFusedAnchored'
          : sr._fused
            ? 'report.submissionRisk.compactNoteFused'
            : 'report.submissionRisk.compactNote',
        { score: Math.round(textPattern.display_score), flagLine: sr._flagLine },
      )
    : (sr ? t('report.submissionRisk.note') : '');

  return (
    <div className={`merged-verdict${level ? ` is-${level}` : ''}`}>
      <div className="merged-verdict-headline">
        <span className="merged-verdict-kicker">{t('report.authorshipBreakdown.fusedHeadline.label')}</span>
        {valueEl}
        {chipEl}
      </div>
      {ownershipLead && <p className="merged-verdict-lead">{ownershipLead}</p>}
      {evidenceEl}
      {flagNote && <p className="merged-verdict-note">{flagNote}</p>}
    </div>
  );
}

export default function MergedAuthorshipRisk({ t, breakdown, sr, authoritativeTier, tierAuthority }) {
  const hasBreakdown = !!breakdown;
  const hasSr = !!(sr && sr.overall && sr.overall.level);
  if (!hasBreakdown && !hasSr) return null;

  const rawShares = (breakdown && breakdown.document_breakdown_raw) || {};
  const bandShares = (breakdown && breakdown.document_breakdown_bands) || {};
  const axes = (sr && sr.axes) || {};
  const textPattern = axes.text_pattern || {};
  const hasScore = typeof textPattern.display_score === 'number';

  return (
    <section className="merged-authorship-risk" aria-label={t('report.merged.title')}>
      <div className="merged-head">
        <h3>
          {t('report.merged.title')}
          <span className="merged-beta-chip">{t('report.authorshipBreakdown.betaChip')}</span>
        </h3>
        <p className="merged-subtitle">{t('report.merged.subtitle')}</p>
      </div>

      <VerdictBand
        t={t}
        breakdown={breakdown}
        sr={hasSr ? sr : null}
        authoritativeTier={authoritativeTier}
        tierAuthority={tierAuthority}
      />

      <div className="merged-lenses">
        {hasBreakdown && (
          <div className="merged-lens">
            <p className="merged-lens-head">
              {t('report.merged.compositionLens')}{' '}
              <span className="merged-lens-note">· {t('report.merged.compositionLensNote')}</span>
            </p>
            <div className="merged-comp-bars">
              {CATEGORY_ORDER.map((category) => (
                <CategoryBar key={category} t={t} category={category} raw={rawShares[category]} band={bandShares[category]} />
              ))}
            </div>
          </div>
        )}

        {hasSr && (
          <div className="merged-lens">
            <p className="merged-lens-head">
              {t('report.merged.riskLens')}{' '}
              <span className="merged-lens-note">· {t('report.merged.riskLensNote')}</span>
            </p>
            <div className="merged-risk-axes">
              {SUBMISSION_RISK_AXES.map((key) => {
                const lvl = (axes[key] || {}).level || 'unknown';
                return (
                  <div className={`merged-axis is-${lvl}`} key={key}>
                    <span>{t(`report.submissionRisk.axes.${key}`)}</span>
                    <strong>{t(`report.submissionRisk.levels.${lvl}`)}</strong>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {hasSr && hasScore && (
        <details className="merged-scale" open>
          <summary>{t('report.submissionRisk.scale.toggle')}</summary>
          <div className="merged-scale-content">
            <table className="merged-scale-table">
              <thead>
                <tr>
                  <th>{t('report.submissionRisk.scale.headers.score')}</th>
                  <th>{t('report.submissionRisk.scale.headers.reads')}</th>
                  <th>{t('report.submissionRisk.scale.headers.measured')}</th>
                </tr>
              </thead>
              <tbody>
                <tr><td>0–32</td><td>{t('report.submissionRisk.scale.rows.low.reads')}</td><td>{t('report.submissionRisk.scale.rows.low.measured')}</td></tr>
                <tr><td>32–48</td><td>{t('report.submissionRisk.scale.rows.medium.reads')}</td><td>{t('report.submissionRisk.scale.rows.medium.measured')}</td></tr>
                <tr><td>48–65</td><td>{t('report.submissionRisk.scale.rows.high.reads')}</td><td>{t('report.submissionRisk.scale.rows.high.measured')}</td></tr>
                <tr><td>65+</td><td>{t('report.submissionRisk.scale.rows.critical.reads')}</td><td>{t('report.submissionRisk.scale.rows.critical.measured')}</td></tr>
              </tbody>
            </table>
            <p className="merged-scale-footnote">{t('report.submissionRisk.scale.notTurnitinComparable')}</p>
          </div>
        </details>
      )}

      <p className="merged-disclaimer">
        {(breakdown && breakdown.disclaimer) || t('report.authorshipBreakdown.disclaimer')}
      </p>
      <p className="merged-feedback">
        {t('report.authorshipBreakdown.feedbackPrompt')}{' '}
        <button
          type="button"
          className="merged-feedback-link"
          onClick={() => window.dispatchEvent(new Event('draftproof:open-feedback'))}
        >
          {t('report.authorshipBreakdown.feedbackAction')}
        </button>
      </p>
    </section>
  );
}
