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

// V8 three-way display fallback (poc/detect_v7/pipeline_bridge.py::_compose_display_fallback,
// weights.json display_fallback.mode == "three_way"): ai_paraphrased + ai_generated_like are
// measurably indistinguishable from single-document evidence, so the DISPLAY merges them into
// one "ai_transformed" share. Used only when breakdown.display_taxonomy === 'three_way' &&
// breakdown.display_shares is present; the four-way CATEGORY_ORDER above remains the fallback
// path for legacy payloads (mode 'four_way' / older reports with no display_* fields).
const THREE_WAY_CATEGORY_ORDER = [
  'student_owned',
  'ai_assisted_polished',
  'ai_transformed',
];

// Bar-fill color per category (spec Global Constraints — must match PDF pdf.py).
// allow-hardcode: presentation colors (bar fills), not a scoring/matching list.
const CATEGORY_COLOR = {
  student_owned: '#1D9E75',
  ai_assisted_polished: '#888780',
  ai_paraphrased: '#D85A30',
  ai_generated_like: '#D85A30',
  // ai_transformed merges ai_paraphrased + ai_generated_like — same fill color, both
  // legacy categories already shared it, so no new color decision is needed.
  ai_transformed: '#D85A30',
};

const KNOWN_DEEP_SCAN_BANDS = ['insufficient', 'amber', 'orange', 'red'];
const KNOWN_AUTHORITATIVE_TIERS = ['green', 'amber', 'orange', 'red'];
// Single source of truth: band label + chip color both derive from the tier the
// backend assigned. Do NOT recompute a band from the fused score against frontend
// cutoffs (duplicates backend threshold logic — a no-hardcode violation).
// Exported: the rewrite-comparison column chips (Report.jsx) rate both columns on
// this same fused-tier scale so "original vs rewritten" is one vocabulary, not two.
export const TIER_TO_BAND = { green: 'low', amber: 'moderate', orange: 'high', red: 'critical' };

function CategoryBar({ t, category, raw, band }) {
  const hasRaw = typeof raw === 'number' && Number.isFinite(raw);
  const widthPct = hasRaw ? Math.max(0, Math.min(100, raw * 100)) : 0;
  // `band` is only supplied for the four-way categories (backend-computed,
  // document_breakdown_bands). The V8 three-way merged "ai_transformed" share has
  // no corresponding band field (the backend deliberately emits only display_shares,
  // not thresholded bands, for the merge) — recomputing a band client-side would
  // duplicate the backend's threshold logic (a no-hardcode violation, same reasoning
  // as TIER_TO_BAND above), so that row shows the percentage alone.
  const bandLabel = band ? t(`report.authorshipBreakdown.bands.${band}`) : null;
  return (
    <div className="merged-comp-row">
      <div className="merged-comp-row-head">
        <span className="merged-comp-label">
          {t(`report.authorshipBreakdown.categories.${category}`)}
        </span>
        <span className="merged-comp-band">
          {hasRaw
            ? (bandLabel ? `${bandLabel} · ${Math.round(widthPct)}%` : `${Math.round(widthPct)}%`)
            : (bandLabel || t('report.authorshipBreakdown.bands.None'))}
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

// Headline-confidence caution (badge.headline_confidence, poc/report/headline_confidence.py):
// a clean-looking headline (green/0%) contradicted by other evidence on the same report
// (second-opinion tile, raw pattern tier, thin sample, below-floor deep scan). Composes the
// reason clauses via i18n. Annotate-not-suppress: qualifies the number, never hides it.
// KEEP IN SYNC with render.py _HEADLINE_CONFIDENCE_CLAUSES (PDF wording).
function headlineConfidenceNote(t, headlineConfidence) {
  const reasons = (headlineConfidence && headlineConfidence.reasons) || [];
  if (!reasons.length) return '';
  const detail = headlineConfidence.detail || {};
  const clauses = reasons
    .map((reason) => t(`report.aiLikelihood.headlineConfidence.clauses.${reason}`, {
      pct: detail.second_opinion_pct,
      rawTier: detail.raw_tier,
      defaultValue: '',
    }))
    .filter(Boolean);
  if (!clauses.length) return '';
  return t('report.aiLikelihood.headlineConfidence.frame', { clauses: clauses.join('; ') });
}

// One "How this score is built" mini-bar: label + thin track (fill width = pct) +
// value, with an optional vertical flag-line marker (used on the Fused bar). `kind`
// is the tier code (green/amber/orange/red) or 'muted' — reuses the verdict palette.
function ScoreBar({ label, pct, kind, marker }) {
  const w = Math.max(0, Math.min(100, pct));
  const m = typeof marker === 'number' && Number.isFinite(marker)
    ? Math.max(0, Math.min(100, marker))
    : null;
  return (
    <div className="merged-scorebar-row">
      <span className="merged-scorebar-label">{label}</span>
      <div className="merged-scorebar-track">
        <div className={`merged-scorebar-fill is-${kind || 'muted'}`} style={{ width: `${w}%` }} />
        {m != null && <span className="merged-scorebar-marker" style={{ left: `${m}%` }} aria-hidden="true" />}
      </div>
      <span className="merged-scorebar-val">{Math.round(pct)}%</span>
    </div>
  );
}

// The ONE headline. Fused score (verdict) when tier_authority present, else the
// deep-scan-only estimate. VISUAL layout (owner redesign 2026-07-16): score hero →
// "How this score is built" mini-bars → action callout → two always-visible caveat
// lines. Same data as before, ~1/3 the words, nothing collapsed. The meta-framing
// line + the "detail is in the breakdown below" nav hint are intentionally DROPPED.
function VerdictBand({ t, breakdown, sr, authoritativeTier, tierAuthority, headlineConfidence }) {
  const hasAuthoritativeTier = KNOWN_AUTHORITATIVE_TIERS.includes(authoritativeTier);
  const level = sr && sr.overall ? sr.overall.level : null;

  let valueEl = null;
  let chipEl = null;
  let miniBars = null;
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
    // Mini-bars replace the old "Behind this score: composite …, deep-scan …" prose.
    // They only render when the composite number exists; otherwise the fused value in
    // the hero above stands alone (graceful fallback — no empty bars).
    if (typeof tierAuthority.composite_score === 'number') {
      const flagLine = typeof sr?._flagLine === 'number' ? sr._flagLine : null;
      const fusedKind = hasAuthoritativeTier ? authoritativeTier : null;
      miniBars = (
        <div className="merged-scorebuild">
          <p className="merged-scorebuild-head">{t('report.merged.scoreBuild.head')}</p>
          <ScoreBar
            label={t('report.merged.scoreBuild.composite')}
            pct={Math.round(tierAuthority.composite_score)}
            kind="muted"
          />
          <ScoreBar
            label={t('report.merged.scoreBuild.deepScan')}
            pct={Math.round((tierAuthority.proportion || 0) * 100)}
            kind="muted"
          />
          <ScoreBar
            label={t('report.merged.scoreBuild.fused')}
            pct={Math.round(tierAuthority.fused_score)}
            kind={fusedKind}
            marker={flagLine}
          />
          {flagLine != null && (
            <p className="merged-scorebuild-cap">
              {t('report.merged.scoreBuild.flagLineCaption', { flagLine: Math.round(flagLine) })}
            </p>
          )}
        </div>
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
    // Deep-scan-only path has no composite/fused to chart, so it keeps a single
    // one-line caption instead of the mini-bars.
    evidenceEl = <p className="merged-verdict-evidence">{t('report.authorshipBreakdown.deepScan.notTurnitin')}</p>;
  }

  const cautionNote = headlineConfidenceNote(t, headlineConfidence);
  const ownershipLead = level
    ? t(`report.submissionRisk.ownershipLead.${level}`, { defaultValue: '' })
    : '';

  return (
    <div className={`merged-verdict${level ? ` is-${level}` : ''}`}>
      <div className="merged-verdict-headline">
        <span className="merged-verdict-kicker">{t('report.authorshipBreakdown.fusedHeadline.label')}</span>
        {valueEl}
        {chipEl}
      </div>
      {miniBars}
      {evidenceEl}
      {cautionNote && <p className="merged-comp-caveat merged-verdict-caution">{cautionNote}</p>}
      {/* Action callout: the ownership lead in a highlighted (warning-tint) box. */}
      {ownershipLead && (
        <div className={`merged-verdict-action${level ? ` is-${level}` : ''}`}>{ownershipLead}</div>
      )}
      {/* Two always-visible caveat lines (condensed from the old long paragraph). No
          collapse/expander — they stay visible text. */}
      <p className="merged-verdict-caveat">
        <span className="merged-verdict-caveat-ico" aria-hidden="true">ⓘ</span>
        <span>{t('report.merged.caveats.headsUp')}</span>
      </p>
      <p className="merged-verdict-caveat">
        <span className="merged-verdict-caveat-ico" aria-hidden="true">✎</span>
        <span>{t('report.merged.caveats.declaration')}</span>
      </p>
    </div>
  );
}

// ── Phase-0 Authorship Evidence Level panel (advisory, display-only) ──────
// Re-presents EXISTING signals as Evidence Level N/5 + per-lens bands + banded
// confidence + typed coverage + limitations. Additive/null-safe: renders null
// when the badge carries no authorship_evidence_levels (flag off / older report).
// Band/level CODES map to i18n keys (report.evidenceLevels.*) — KEEP IN SYNC
// with poc/report/render_panels.py render_authorship_evidence_levels.
const AEL_LENSES = ['ai_pattern', 'grounding', 'citation', 'reasoning'];
// Consolidated-header swatch: each dimension carries the SAME colour as its
// per-sentence underline in SignalHighlights (red = reads as AI, amber = weak
// grounding, purple = reasoning jump). The swatch <span> reuses the shared
// `.submitted-issue-swatch is-{color}` CSS (07-report-submitted.css), whose
// hexes ARE the SignalHighlights ISSUE_COLOR_HEX values (#dc2626/#f59e0b/#7c3aed)
// — one source of truth, no new colour. `citation` (Source traceability) is a
// DOCUMENT-level dimension with no sentence underline, so it gets NO swatch
// (honest: there is nothing to point at in the text).
const AEL_LENS_SWATCH = { ai_pattern: 'red', grounding: 'amber', reasoning: 'purple' };
// Map each lens's raw band/level code to a display band token + level class.
const AEL_TIER_BAND = { green: 'low', clean: 'low', low: 'low', amber: 'moderate', orange: 'high', red: 'high', high: 'high' };
const AEL_GROUNDING_BAND = { likely_human: 'strong', some_texture: 'moderate', moderate: 'moderate', strong: 'elevated', very_strong: 'elevated' };
const AEL_CRITICAL_BAND = { strong_control: 'strong', acceptable_control: 'moderate', weak_control: 'developing', high_dependency: 'limited', very_high_dependency: 'limited' };
const AEL_CITATION_LEVEL = { low: 'strong', medium: 'moderate', high: 'elevated' };
// Display band token -> chip level class (reuse merged-axis is-* colouring).
const AEL_BAND_CLASS = { low: 'low', strong: 'low', moderate: 'medium', developing: 'medium', elevated: 'high', high: 'high', limited: 'high' };

function aelLensBand(lensId, lens) {
  if (!lens || lens.available === false) return null;
  if (lensId === 'ai_pattern') return AEL_TIER_BAND[String(lens.band || '').toLowerCase()] || null;
  if (lensId === 'grounding') return AEL_GROUNDING_BAND[String(lens.band || '').toLowerCase()] || null;
  if (lensId === 'citation') return AEL_CITATION_LEVEL[String(lens.level || '').toLowerCase()] || null;
  return AEL_CRITICAL_BAND[String(lens.band || '').toLowerCase()] || null; // reasoning
}

// Anchor headline/fix are emitted by the server composer with an i18n `code` +
// interpolation `params` (rendered via t() for en/zh) AND a ready English
// string (server fallback). The `example` is verbatim user text — NEVER
// translated. See poc/report/authorship_evidence_levels.py (_attach_anchors).
function aelAnchorText(t, code, params, fallback, keyspace) {
  if (code) {
    const key = `report.evidenceLevels.${keyspace}.${code}`;
    const s = t(key, params || {});
    if (s && s !== key) return s;
  }
  return fallback || '';
}

export function EvidenceLevelPanel({ t, ael }) {
  if (!ael || typeof ael !== 'object' || !ael.lenses) return null;
  const level = ael.level;
  const maxLevel = ael.max_level_assessable != null ? ael.max_level_assessable : 2;
  return (
    <div className="merged-lens merged-ael">
      <p className="merged-lens-head">
        {t('report.evidenceLevels.head')}{' '}
        <span className="merged-beta-chip">{t('report.evidenceLevels.betaChip')}</span>
      </p>
      <p className="merged-verdict-lead">
        {t('report.evidenceLevels.levelLine', { level: String(level), max: String(maxLevel) })}
      </p>
      <div className="merged-risk-axes merged-ael-axes">
        {AEL_LENSES.map((lensId) => {
          const lens = ael.lenses[lensId];
          const band = aelLensBand(lensId, lens);
          const cls = band ? (AEL_BAND_CLASS[band] || 'medium') : 'unknown';
          const conf = lens && lens.assessment_confidence;
          const anchor = lens && lens.anchor;
          const headline = anchor
            ? aelAnchorText(t, anchor.headline_code, anchor.params, anchor.headline, 'anchors')
            : null;
          const fix = anchor && anchor.fix
            ? aelAnchorText(t, anchor.fix_code, null, anchor.fix, 'anchorFix')
            : null;
          const swatch = AEL_LENS_SWATCH[lensId] || null;
          return (
            <div className="merged-ael-lens" key={lensId}>
              <div className={`merged-axis is-${cls}`}>
                <span className="merged-ael-lens-name">
                  {swatch ? (
                    <span className={`submitted-issue-swatch is-${swatch}`} aria-hidden="true" />
                  ) : null}
                  {t(`report.evidenceLevels.lenses.${lensId}`)}
                </span>
                <strong>
                  {band ? t(`report.evidenceLevels.bands.${band}`) : t('report.evidenceLevels.bands.notAssessed')}
                  {conf ? ` · ${t(`report.evidenceLevels.confidence.${conf}`)}` : ''}
                </strong>
              </div>
              {anchor && (headline || anchor.example || fix) ? (
                <div className="merged-ael-anchor">
                  {headline ? <p className="merged-ael-anchor-head">{headline}</p> : null}
                  {anchor.example ? (
                    <p className="merged-ael-anchor-eg">{`“${anchor.example}”`}</p>
                  ) : null}
                  {fix ? <p className="merged-ael-anchor-fix">{fix}</p> : null}
                </div>
              ) : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// Claim-graph "Source grounding" row status -> chip level class (reuse the
// merged-axis is-* colouring). Verified = low/green (good), contradicted =
// high/red (the ONLY red), paywalled + unresolved = neutral ("not held against
// you"). Presentation-only; the composer (poc/report/claim_graph_panel.py) is
// the single source of truth for the data itself.
const CG_STATUS_CLASS = { verified: 'low', contradicted: 'high', paywalled: 'unknown', unresolved: 'unknown' };

// Claim-graph "Source grounding" evidence panel. Advisory + display-only: renders
// authorship_evidence.claim_graph_display (the server-side composer's single
// source of truth). Returns null when absent (claim graph off / older report),
// so the report is byte-identical when the flag is off. NEVER affects the score.
function ClaimGraphPanel({ t, cg }) {
  if (!cg || typeof cg !== 'object' || cg.present !== true) return null;
  const summary = cg.summary || {};
  const checks = Array.isArray(cg.source_checks) ? cg.source_checks : [];
  const questions = Array.isArray(cg.questions) ? cg.questions : [];
  const channels = Array.isArray(cg.coverage_channels) ? cg.coverage_channels : [];
  const limitations = Array.isArray(cg.limitations) ? cg.limitations : [];
  return (
    <div className="merged-lens merged-claim-graph">
      <p className="merged-lens-head">
        {t('report.claimGraph.title')}{' '}
        <span className="merged-beta-chip">{t('report.claimGraph.advisoryPill')}</span>
      </p>
      <p className="merged-verdict-lead">{t('report.claimGraph.subtitle')}</p>
      <div className="merged-risk-axes merged-cg-summary">
        <div className="merged-axis is-unknown">
          <span>{t('report.claimGraph.specificClaims')}</span>
          <strong>{String(summary.specific_claims != null ? summary.specific_claims : 0)}</strong>
        </div>
        <div className="merged-axis is-low">
          <span>{t('report.claimGraph.verifiedToSource')}</span>
          <strong>{String(summary.verified_to_source != null ? summary.verified_to_source : 0)}</strong>
        </div>
        <div className="merged-axis is-medium">
          <span>{t('report.claimGraph.needGrounding')}</span>
          <strong>{String(summary.need_grounding != null ? summary.need_grounding : 0)}</strong>
        </div>
      </div>
      {checks.length > 0 && (
        <>
          <p className="merged-lens-note"><strong>{t('report.claimGraph.citedSources')}</strong></p>
          <div className="merged-cg-checks">
            {checks.map((c, i) => {
              const status = String((c && c.status) || '').toLowerCase();
              const cls = CG_STATUS_CLASS[status] || 'unknown';
              const statusKey = {
                verified: 'statusVerified', contradicted: 'statusContradicted',
                paywalled: 'statusPaywalled', unresolved: 'statusUnresolved',
              }[status] || 'statusUnresolved';
              return (
                <div className={`merged-cg-check is-${cls}`} key={i}>
                  <span className="merged-cg-check-status">{t(`report.claimGraph.${statusKey}`)}</span>
                  <span className="merged-cg-check-body">
                    {(c && c.claim_excerpt) || ''}
                    {c && c.locator ? <em className="merged-cg-check-locator">{c.locator}</em> : null}
                  </span>
                </div>
              );
            })}
          </div>
        </>
      )}
      {questions.length > 0 && (
        <>
          <p className="merged-lens-note"><strong>{t('report.claimGraph.questionsHeader')}</strong></p>
          <ul className="merged-cg-questions">
            {questions.map((q, i) => <li key={i}>{q}</li>)}
          </ul>
        </>
      )}
      {(channels.length > 0 || limitations.length > 0) && (
        <p className="merged-lens-note">
          {channels.length > 0 ? `${t('report.claimGraph.sourcesVia')}: ${channels.join(', ')}` : ''}
          {channels.length > 0 && limitations.length > 0 ? ' · ' : ''}
          {limitations.length > 0 ? `${t('report.claimGraph.limitationsLabel')}: ${limitations.join('; ')}` : ''}
        </p>
      )}
      <p className="merged-lens-note merged-cg-footer">{t('report.claimGraph.notHeldAgainstYou')} {t('report.claimGraph.footer')}</p>
    </div>
  );
}

// `sections` lets a caller render only a slice of this real card (e.g. the landing
// page's marketing preview splits the composition lens and the verdict+risk lens
// across two tabs) instead of re-implementing the markup as a separate mockup.
// Omitting `sections` (every Report.jsx call site) renders every section, unchanged.
// NOTE: 'evidenceLevels' is intentionally ABSENT — the Authorship-evidence-level
// panel (EvidenceLevelPanel, exported above) was lifted OUT of this card and now
// renders as the consolidated header of the "Read full document" section
// (SignalHighlights.jsx), directly above the underlined document it describes.
// This card no longer double-renders it.
const ALL_SECTIONS = ['header', 'verdict', 'composition', 'riskAxes', 'deepScanParagraphs', 'claimGraph', 'scale', 'disclaimer'];

export default function MergedAuthorshipRisk({ t, breakdown, sr, authoritativeTier, tierAuthority, headlineConfidence, claimGraphDisplay, sections }) {
  const hasBreakdown = !!breakdown;
  const hasSr = !!(sr && sr.overall && sr.overall.level);
  if (!hasBreakdown && !hasSr) return null;

  const activeSections = sections || ALL_SECTIONS;
  const show = (name) => activeSections.includes(name);

  const rawShares = (breakdown && breakdown.document_breakdown_raw) || {};
  const bandShares = (breakdown && breakdown.document_breakdown_bands) || {};
  // V8 three-way display fallback: when the backend emits display_taxonomy ==
  // 'three_way' (weights.json display_fallback.mode), render the merged
  // student_owned / ai_assisted_polished / ai_transformed shares instead of the
  // four-way breakdown. Absent display_shares (legacy payloads / mode 'four_way')
  // falls back to the byte-identical four-way rendering below.
  const isThreeWay = !!(breakdown && breakdown.display_taxonomy === 'three_way' && breakdown.display_shares);
  const compositionShares = isThreeWay ? breakdown.display_shares : rawShares;
  const compositionOrder = isThreeWay ? THREE_WAY_CATEGORY_ORDER : CATEGORY_ORDER;
  const axes = (sr && sr.axes) || {};
  const textPattern = axes.text_pattern || {};
  const hasScore = typeof textPattern.display_score === 'number';

  return (
    <section className="merged-authorship-risk" aria-label={t('report.merged.title')}>
      {show('header') && (
        <div className="merged-head">
          <h3>
            {t('report.merged.title')}
            <span className="merged-beta-chip">{t('report.authorshipBreakdown.betaChip')}</span>
          </h3>
          <p className="merged-subtitle">{t('report.merged.subtitle')}</p>
        </div>
      )}

      {show('verdict') && (
        <VerdictBand
          t={t}
          breakdown={breakdown}
          sr={hasSr ? sr : null}
          authoritativeTier={authoritativeTier}
          tierAuthority={tierAuthority}
          headlineConfidence={headlineConfidence}
        />
      )}

      {(show('composition') || show('riskAxes')) && (
        <div className="merged-lenses">
          {show('composition') && hasBreakdown && (
            <div className="merged-lens">
              <p className="merged-lens-head">
                {t('report.merged.compositionLens')}{' '}
                {/* PR #17's "sums to 100%" caveat, EXTENDED (not stacked) to also carry the
                    owner-mandated interpretation framing vs the verdict's AI-risk authority. */}
                <span className="merged-lens-note">· {t('report.merged.compositionLensNote')}</span>
              </p>
              {breakdown.presentation === 'mixed_signals' && (
                <p className="merged-comp-caveat">{t('report.merged.compositionMixedSignalsCaveat')}</p>
              )}
              {isThreeWay && (
                <p className="merged-comp-caveat merged-comp-caveat--info">
                  {t('report.merged.threeWayExplainer')}
                </p>
              )}
              <div className="merged-comp-bars">
                {compositionOrder.map((category) => (
                  <CategoryBar
                    key={category}
                    t={t}
                    category={category}
                    raw={compositionShares[category]}
                    band={isThreeWay ? undefined : bandShares[category]}
                  />
                ))}
              </div>
            </div>
          )}

          {show('riskAxes') && hasSr && (
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
      )}

      {show('deepScanParagraphs')
        && hasBreakdown
        && breakdown.deep_scan
        && Array.isArray(breakdown.deep_scan.paragraphs)
        && breakdown.deep_scan.paragraphs.length > 0 && (
        <div className="merged-lens merged-dsp">
          <p className="merged-lens-head">
            {t('report.authorshipBreakdown.deepScan.paragraphs.head')}{' '}
            <span className="merged-lens-note">· {t('report.authorshipBreakdown.deepScan.paragraphs.note')}</span>
          </p>
          <div className="merged-dsp-rows">
            {breakdown.deep_scan.paragraphs.map((p) => {
              const insufficient = p.band === 'insufficient';
              const pct = Math.round((p.proportion || 0) * 100);
              const floor = breakdown.deep_scan.reliability_floor;
              const floorPct = typeof floor === 'number' ? Math.round(floor * 100) : null;
              const hasDetail = typeof p.flagged_count === 'number' && floorPct !== null;
              return (
                <div className="merged-dsp-row" key={p.index}>
                  <div className="merged-dsp-main">
                    <span className="merged-dsp-label">
                      {t('report.authorshipBreakdown.deepScan.paragraphs.row', { index: p.index + 1 })}
                    </span>
                    {/* Same pattern as the verdict line: plain bold value +
                        band-label chip — never a raw number inside a chip. */}
                    {!insufficient && <strong className="merged-dsp-value">{pct}%</strong>}
                    <span className={`merged-verdict-chip is-${p.band}`}>
                      {insufficient
                        ? t('report.authorshipBreakdown.deepScan.insufficientChip')
                        : t(`report.authorshipBreakdown.deepScan.bands.${p.band}`)}
                    </span>
                  </div>
                  <span className="merged-lens-note merged-dsp-detail">
                    {hasDetail
                      ? `${t('report.authorshipBreakdown.deepScan.paragraphs.flaggedDetail', {
                          flagged: p.flagged_count,
                          count: p.sentence_count,
                        })} · ${
                          insufficient
                            ? t('report.authorshipBreakdown.deepScan.paragraphs.belowFloor', { pct, floor: floorPct })
                            : t('report.authorshipBreakdown.deepScan.paragraphs.atOrAboveFloor', { floor: floorPct })
                        }`
                      : t('report.authorshipBreakdown.deepScan.paragraphs.sentences', { count: p.sentence_count })}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {show('claimGraph') && claimGraphDisplay && (
        <ClaimGraphPanel t={t} cg={claimGraphDisplay} />
      )}

      {show('scale') && hasSr && hasScore && (
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

      {show('disclaimer') && (
        <>
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
        </>
      )}
    </section>
  );
}
