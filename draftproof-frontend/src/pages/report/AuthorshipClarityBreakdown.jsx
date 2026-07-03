// draftproof-frontend/src/pages/report/AuthorshipClarityBreakdown.jsx
// Additive V7 "Authorship Clarity Breakdown" panel: a 4-category read on how much of the
// document reads student-owned vs AI-assisted-polished vs AI-paraphrased vs AI-generated-like.
// Renders nothing if the badge has no authorship_breakdown (flag off / older report) — matches
// the additive/null-safe house pattern used by DebertaSignal.jsx / AuthenticityDashboard.jsx.
//
// display_mode is always "bands" today (Phase 1) — document_breakdown_raw is used ONLY to size
// bar-fill widths for a visual sense of proportion; the exact percentage is NEVER rendered as
// text (false precision until a validation dataset exists — see project decision).
//
// NOTE: styling is not yet written — this component uses the `authorship-breakdown-*` class
// naming convention (sibling to `deberta-*` / `authn-*`) but the CSS itself is a TODO for the
// follow-up task that wires this into Report.jsx.

const CATEGORY_ORDER = [
  'student_owned',
  'ai_assisted_polished',
  'ai_paraphrased',
  'ai_generated_like',
];

function CategoryBar({ t, category, raw, band }) {
  const widthPct = typeof raw === 'number' && Number.isFinite(raw)
    ? Math.max(0, Math.min(100, raw * 100))
    : 0;
  return (
    <div className="authorship-breakdown-row">
      <span className="authorship-breakdown-row-label">
        {t(`report.authorshipBreakdown.categories.${category}`)}
      </span>
      <div className="authorship-breakdown-bar-track">
        <div
          className="authorship-breakdown-bar-fill"
          style={{ width: `${widthPct}%` }}
        />
      </div>
      <span className="authorship-breakdown-row-band">
        {band ? t(`report.authorshipBreakdown.bands.${band}`) : t('report.authorshipBreakdown.bands.None')}
      </span>
    </div>
  );
}

// Uncertainty flags we know how to explain to users. Unknown flags are silently
// skipped — never render raw flag strings.
const KNOWN_UNCERTAINTY_FLAGS = [
  'deep_scan_uncalibrated',
  'deep_scan_below_reliability_floor',
  'paraphrase_without_original_draft',
];

export default function AuthorshipClarityBreakdown({ t, breakdown }) {
  if (!breakdown) return null;

  const rawShares = breakdown.document_breakdown_raw || {};
  const bandShares = breakdown.document_breakdown_bands || {};
  const showCaveat = breakdown.confidence === 'low' || breakdown.degraded_display === true;
  const uncertaintyFlags = Array.isArray(breakdown.uncertainty_flags)
    ? KNOWN_UNCERTAINTY_FLAGS.filter((flag) => breakdown.uncertainty_flags.includes(flag))
    : [];

  return (
    <section
      className="authorship-breakdown"
      aria-label={t('report.authorshipBreakdown.ariaLabel')}
    >
      <div className="authorship-breakdown-head">
        <h3>
          {t('report.authorshipBreakdown.title')}
          <span className="authorship-breakdown-beta-chip">
            {t('report.authorshipBreakdown.betaChip')}
          </span>
        </h3>
      </div>

      <div className="authorship-breakdown-bars">
        {CATEGORY_ORDER.map((category) => (
          <CategoryBar
            key={category}
            t={t}
            category={category}
            raw={rawShares[category]}
            band={bandShares[category]}
          />
        ))}
      </div>

      {showCaveat && (
        <p className="authorship-breakdown-caveat">
          {t('report.authorshipBreakdown.lowConfidenceCaveat')}
        </p>
      )}

      {uncertaintyFlags.map((flag) => (
        <p key={flag} className="authorship-breakdown-caveat">
          {t(`report.authorshipBreakdown.uncertaintyFlags.${flag}`)}
        </p>
      ))}

      <p className="authorship-breakdown-disclaimer">
        {t('report.authorshipBreakdown.disclaimer')}
      </p>

      <p className="authorship-breakdown-feedback">
        {t('report.authorshipBreakdown.feedbackPrompt')}{' '}
        <button
          type="button"
          className="authorship-breakdown-feedback-link"
          onClick={() => window.dispatchEvent(new Event('draftproof:open-feedback'))}
        >
          {t('report.authorshipBreakdown.feedbackAction')}
        </button>
      </p>
    </section>
  );
}
