// draftproof-frontend/src/pages/report/AuthorshipClarityBreakdown.jsx
// Additive V7 "Authorship Clarity Breakdown" panel: a 4-category read on how much of the
// document reads student-owned vs AI-assisted-polished vs AI-paraphrased vs AI-generated-like.
// Renders nothing if the badge has no authorship_breakdown (flag off / older report) — matches
// the additive/null-safe house pattern used by DebertaSignal.jsx / AuthenticityDashboard.jsx.
//
// Percentages ALWAYS display next to the band ("Some · 37%") — owner decision 2026-07-04.
// On near-uniform documents the digits sit within formula noise; the flatness-gated
// "mixed signals" caveat carries that warning rather than hiding the number.
// Bar-fill widths always use document_breakdown_raw for visual proportion.

const CATEGORY_ORDER = [
  'student_owned',
  'ai_assisted_polished',
  'ai_paraphrased',
  'ai_generated_like',
];

function CategoryBar({ t, category, raw, band, showPercent }) {
  const hasRaw = typeof raw === 'number' && Number.isFinite(raw);
  const widthPct = hasRaw ? Math.max(0, Math.min(100, raw * 100)) : 0;
  const bandLabel = band
    ? t(`report.authorshipBreakdown.bands.${band}`)
    : t('report.authorshipBreakdown.bands.None');
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
        {showPercent && hasRaw ? `${bandLabel} · ${Math.round(widthPct)}%` : bandLabel}
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
  // Percentages ALWAYS display (owner decision 2026-07-04, superseding the
  // earlier confidence-gated approach: a number that appears only sometimes
  // reads as ambiguous/confusing). On near-uniform documents the printed
  // digits are within formula noise — the "mixed signals" caveat below
  // (flatness-gated, confidence === 'low') carries that warning instead of
  // hiding the number. degraded_display is deliberately not consulted here:
  // it is structurally always true in Phase 1A (three signals unbuilt) and
  // that situation is disclosed via the uncertainty-flag caveat lines.
  const showCaveat = breakdown.confidence === 'low';
  const showPercent = true;
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
            showPercent={showPercent}
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
