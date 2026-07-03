// draftproof-frontend/src/pages/report/AuthorshipClarityBreakdown.jsx
// Additive V7 "Authorship Clarity Breakdown" panel: a 4-category read on how much of the
// document reads student-owned vs AI-assisted-polished vs AI-paraphrased vs AI-generated-like.
// Renders nothing if the badge has no authorship_breakdown (flag off / older report) — matches
// the additive/null-safe house pattern used by DebertaSignal.jsx / AuthenticityDashboard.jsx.
//
// Percentage display is CONFIDENCE-GATED (owner decision 2026-07-04): when the flatness
// guard fires (confidence === 'low' or degraded_display — i.e. a near-uniform distribution
// where a printed digit would imply meaningless precision), only bands render. When one
// category clearly dominates, the rounded share renders next to the band ("Strong · 54%").
// Bar-fill widths always use document_breakdown_raw for visual proportion either way.

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
  // "Mixed signals" caveat + percent suppression are gated on FLATNESS ONLY
  // (confidence === 'low', i.e. no category decisively wins). Do NOT gate on
  // degraded_display: in Phase 1A three signals are always unbuilt, so every
  // document is structurally degraded and that flag is ALWAYS true — gating on
  // it made the percent unreachable on every report (verified live 2026-07-04,
  // report 95d3de1f: degraded_paragraph_count == paragraph_count on all docs).
  // The unbuilt-signal situation is already disclosed via the uncertainty-flag
  // caveat lines below.
  const showCaveat = breakdown.confidence === 'low';
  const showPercent = !showCaveat;
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
