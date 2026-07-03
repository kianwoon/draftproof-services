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

// Deep-scan bands we have i18n labels + chip colors for. Unknown bands are
// silently skipped — never render raw band strings. "green" is deliberately
// absent: the backend never emits it for deep_scan.
const KNOWN_DEEP_SCAN_BANDS = ['insufficient', 'amber', 'orange', 'red'];

// Tiers the authoritative badge can carry (poc/report/builder.py:
// "tier": authoritative_tier or layer3.tier.value — lowercase green/amber/orange/red).
const KNOWN_AUTHORITATIVE_TIERS = ['green', 'amber', 'orange', 'red'];

// Headline row between the subtitle and the bars: the deep-scan detector's
// absolute AI estimate for this document. For band "insufficient" the reading
// is below the reliability floor, so the number is weak evidence — show a
// muted "insufficient evidence" chip instead of a percentage-as-verdict.
//
// Chip COLOR authority (owner decision 2026-07-04): a raw deep-scan band (e.g.
// "amber") is only one ingredient feeding the fused ai_likelihood_score at 60%
// weight (poc/detect_v7/weights.json tier_authority) — it must never carry its
// own independent alarm color that can contradict the authoritative tier shown
// in the hero (e.g. an amber chip next to a green "Low Risk" verdict). The chip
// color always follows `authoritativeTier` when supplied; the raw band label
// still prints in the chip text so the underlying signal isn't hidden. Older
// reports / tier-authority-off carry no authoritativeTier, so we fall back to
// the pre-existing band-driven color rather than breaking.
function DeepScanHeadline({ t, deepScan, authoritativeTier }) {
  if (!deepScan || !KNOWN_DEEP_SCAN_BANDS.includes(deepScan.band)) return null;
  const hasProportion =
    typeof deepScan.proportion === 'number' && Number.isFinite(deepScan.proportion);
  const insufficient = deepScan.band === 'insufficient';
  const hasAuthoritativeTier = KNOWN_AUTHORITATIVE_TIERS.includes(authoritativeTier);
  // "insufficient" stays muted regardless of authoritativeTier — it's a
  // data-quality state (too few sentences crossed the reliability floor), not
  // a risk color, so it must never borrow the authoritative tier's alarm hue.
  const chipColorTier = insufficient ? null : (hasAuthoritativeTier ? authoritativeTier : deepScan.band);
  const chipClassName = insufficient
    ? 'authorship-breakdown-deepscan-chip is-insufficient'
    : `authorship-breakdown-deepscan-chip is-${chipColorTier}`;
  const bandLabel = t(`report.authorshipBreakdown.deepScan.bands.${deepScan.band}`);
  return (
    <div className="authorship-breakdown-deepscan">
      <span className="authorship-breakdown-deepscan-label">
        {t('report.authorshipBreakdown.deepScan.label')}
      </span>
      {insufficient || !hasProportion ? (
        <span className="authorship-breakdown-deepscan-chip is-insufficient">
          {t('report.authorshipBreakdown.deepScan.insufficientChip')}
        </span>
      ) : (
        <>
          <strong className="authorship-breakdown-deepscan-value">
            {Math.round(deepScan.proportion * 100)}%
          </strong>
          <span className={chipClassName}>
            {hasAuthoritativeTier
              ? t('report.authorshipBreakdown.deepScan.bandDefersToTier', {
                  band: bandLabel,
                  tier: t(`report.tiers.${authoritativeTier}`),
                })
              : bandLabel}
          </span>
        </>
      )}
      <span className="authorship-breakdown-deepscan-note">
        {t('report.authorshipBreakdown.deepScan.notTurnitin')}
      </span>
    </div>
  );
}

// The band label + chip color BOTH derive from `authoritativeTier` — the tier
// the backend actually assigned to this document (green/amber/orange/red, from
// poc/report/builder.py). We deliberately do NOT recompute a band from the fused
// score against frontend cutoffs: that duplicates the backend's threshold logic
// (a no-hardcode violation) AND risks the label disagreeing with the color if the
// two drift. One source of truth = always consistent with the hero tier.
const TIER_TO_BAND = { green: 'low', amber: 'moderate', orange: 'high', red: 'critical' };

// Headline row for when tier_authority (the fused AI-likelihood driving the
// authoritative tier) is available: the FUSED score is the verdict, so it
// leads the panel — the deep-scan/composite ingredients demote to a muted
// evidence line beneath it (owner decision 2026-07-04).
function FusedHeadline({ t, tierAuthority, authoritativeTier }) {
  const fusedScore = tierAuthority.fused_score;
  const hasAuthoritativeTier = KNOWN_AUTHORITATIVE_TIERS.includes(authoritativeTier);
  const band = hasAuthoritativeTier ? TIER_TO_BAND[authoritativeTier] : null;
  const composite = tierAuthority.composite_score;
  const deepScanPct = Math.round((tierAuthority.proportion || 0) * 100);

  return (
    <div className="authorship-breakdown-fused">
      <div className="authorship-breakdown-fused-headline">
        <span className="authorship-breakdown-fused-label">
          {t('report.authorshipBreakdown.fusedHeadline.label')}
        </span>
        <strong className="authorship-breakdown-fused-value">
          {Math.round(fusedScore)}%
        </strong>
        {band && (
          <span className={`authorship-breakdown-fused-band-chip is-${authoritativeTier}`}>
            {t(`report.authorshipBreakdown.fusedHeadline.bands.${band}`)}
          </span>
        )}
      </div>
      <p className="authorship-breakdown-fused-evidence">
        {t('report.authorshipBreakdown.fusedHeadline.evidence', {
          composite: Math.round(composite),
          deepScan: deepScanPct,
        })}
      </p>
    </div>
  );
}

export default function AuthorshipClarityBreakdown({ t, breakdown, authoritativeTier, tierAuthority }) {
  if (!breakdown) return null;

  const rawShares = breakdown.document_breakdown_raw || {};
  const bandShares = breakdown.document_breakdown_bands || {};
  // Percentages ALWAYS display (owner decision 2026-07-04, superseding the
  // earlier confidence-gated approach: a number that appears only sometimes
  // reads as ambiguous/confusing).
  const showPercent = true;

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
        {/* Semantic bridge to the hero above: these shares are a COMPOSITION
            (always sum to 100%), not an AI-probability — without this line,
            "Student-owned · 37%" reads as contradicting the hero's
            "Ownership risk: Low / you can defend this as your own work". */}
        <p className="authorship-breakdown-subtitle">
          {t('report.authorshipBreakdown.subtitle')}
        </p>
      </div>

      {tierAuthority && typeof tierAuthority.fused_score === 'number' ? (
        <FusedHeadline t={t} tierAuthority={tierAuthority} authoritativeTier={authoritativeTier} />
      ) : (
        <DeepScanHeadline t={t} deepScan={breakdown.deep_scan} authoritativeTier={authoritativeTier} />
      )}

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
