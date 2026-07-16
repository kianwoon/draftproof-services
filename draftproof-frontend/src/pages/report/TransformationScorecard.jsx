import MergedAuthorshipRisk from './MergedAuthorshipRisk';
import PolicyRiskView from './PolicyRiskView';
import {
  transformationLabel,
  confidenceLabel,
  aiLikelihoodBands,
  submissionRisk,
  policyRisk,
  groundingDiagnosis,
  GROUNDING_DIAGNOSIS_LEAD_ENABLED,
  GROUNDING_DIAGNOSIS_BUCKETS,
  EXTERNAL_ESTIMATE_DISPLAY_ENABLED,
} from './reportHelpers';

export default function TransformationScorecard({
  t,
  report,
  hasRewriteSignalComparison,
  rewrittenBadge,
  rewrittenAiScore,
  originalComparisonBadge,
  aiScore,
  transformation,
  transformationSignals,
  transformationSummary,
  transformationOriginalScore,
  originalColumnRatingBadge,
  rewrittenTransformation,
  rewrittenTransformationSummary,
  transformationRewrittenScore,
  rewrittenColumnRatingBadge,
  rewriteResultSummary,
}) {
  const renderAiLikelihoodHeadline = (variantBadge) => {
    const bands = aiLikelihoodBands(variantBadge);
    if (!bands.draftproof) return null;
    const dp = bands.draftproof;
    const ext = bands.external;
    const ratingLabel = variantBadge?.authorship_rating_label;
    const diag = groundingDiagnosis(variantBadge);
    const driver = GROUNDING_DIAGNOSIS_LEAD_ENABLED && diag?.primary_driver ? diag : null;
    const driverBuckets = driver?.buckets || {};
    // Driver bar first (emphasized), remaining dimensions by gap score descending.
    const orderedDriverKeys = driver
      ? [
          driver.primary_driver,
          ...GROUNDING_DIAGNOSIS_BUCKETS
            .filter((k) => k !== driver.primary_driver && driverBuckets[k])
            .sort((a, b) => (driverBuckets[b]?.score || 0) - (driverBuckets[a]?.score || 0)),
        ].filter((k) => driverBuckets[k])
      : [];
    // Scoped detector-signal verdict (NOT an overall verdict — the Submission-risk
    // band leads that). Derived from badge tier + external band + grounding driver.
    const tierKey = String(dp.tier || '').toLowerCase();
    const signalPhrase = t(`report.aiLikelihood.verdictSignal.${tierKey}`, { defaultValue: t('report.aiLikelihood.title') });
    const flagPhrase = (EXTERNAL_ESTIMATE_DISPLAY_ENABLED && ext)
      ? t(`report.aiLikelihood.verdictFlag.${ext.band}`, { defaultValue: '' }) : '';
    const driverLabelText = driver ? t(`report.groundingDiagnosis.drivers.${driver.primary_driver}.label`) : '';
    const verdictLine = `${signalPhrase}${flagPhrase ? ` — ${flagPhrase}` : ''}.`
      + (driverLabelText ? ` ${t('report.aiLikelihood.verdictFix', { driver: driverLabelText })}` : '');
    return (
      <div className="ai-likelihood-block">
        <div className="ai-likelihood-caption">{t('report.aiLikelihood.title')}</div>
        <div className="ai-likelihood-verdict" style={{ fontWeight: 600, fontSize: '15px', margin: '2px 0 14px', lineHeight: 1.45 }}>{verdictLine}</div>
        {driver && (
          <div className="ai-likelihood-driver" style={{ marginBottom: '12px' }}>
            <div className="ai-likelihood-driver-label" style={{ fontWeight: 500 }}>
              {t('report.aiLikelihood.mainFixLabel')}: {t(`report.groundingDiagnosis.drivers.${driver.primary_driver}.label`)}
            </div>
            <div className="ai-likelihood-driver-action" style={{ color: 'var(--color-text-secondary, #64748b)' }}>
              {t(`report.groundingDiagnosis.drivers.${driver.primary_driver}.action`)}
              {driver.caveat ? ` (${t('report.groundingDiagnosis.tentative')})` : ''}
            </div>
            <div className="ai-likelihood-bars-heading" style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: 'var(--color-text-secondary, #64748b)', textTransform: 'uppercase', letterSpacing: '.04em', margin: '12px 0 6px' }}>
              <span>{t('report.groundingDiagnosis.bucketsHeading')}</span>
              <span>{t('report.groundingDiagnosis.lowerIsBetter')}</span>
            </div>
            <div className="ai-likelihood-bars" style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {orderedDriverKeys.map((k) => {
                const score = Math.round(driverBuckets[k].score || 0);
                const isDriver = k === driver.primary_driver;
                return (
                  <div key={k} className="ai-likelihood-bar-row">
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', marginBottom: '2px' }}>
                      <span style={{ fontWeight: isDriver ? 500 : 400 }}>{t(`report.groundingDiagnosis.buckets.${k}`)}</span>
                      <span style={{ color: isDriver ? '#854F0B' : 'var(--color-text-secondary, #64748b)', fontWeight: isDriver ? 500 : 400 }}>{score}</span>
                    </div>
                    <div style={{ height: '10px', background: 'var(--color-background-secondary, #eef0f2)', borderRadius: '5px', overflow: 'hidden' }}>
                      <div style={{ width: `${score}%`, height: '100%', background: isDriver ? '#EF9F27' : '#B4B2A9', borderRadius: '5px' }} />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
        {policyRisk(variantBadge) ? (
          <PolicyRiskView t={t} pr={policyRisk(variantBadge)} />
        ) : (
          <div className="ai-likelihood-pair">
            <div className="ai-likelihood-metric">
              <div className="ai-likelihood-caption">{t('report.aiLikelihood.draftproof')}</div>
              <div className="ai-likelihood-score" style={{ color: dp.color }}>{dp.score}%</div>
              <div className="ai-likelihood-band">{dp.tier}</div>
              <div className="ai-likelihood-actionable">{t('report.aiLikelihood.draftproofNote')}</div>
            </div>
            {EXTERNAL_ESTIMATE_DISPLAY_ENABLED && (
              <div className="ai-likelihood-metric">
                <div className="ai-likelihood-caption">{t('report.aiLikelihood.external')}</div>
                {ext ? (
                  <>
                    <div className="ai-likelihood-score" style={{ color: ext.color }}>~{ext.score}%</div>
                    <div className="ai-likelihood-band">{t(`report.aiLikelihood.externalBand.${ext.band}`, { defaultValue: '' })}</div>
                  </>
                ) : (
                  <div className="ai-likelihood-unavailable">{t('report.aiLikelihood.externalUnavailable')}</div>
                )}
              </div>
            )}
          </div>
        )}
        {!policyRisk(variantBadge) && (
          <>
            {ratingLabel && (
              <div className="ai-likelihood-meta">{ratingLabel}</div>
            )}
            <div className="ai-likelihood-why">{EXTERNAL_ESTIMATE_DISPLAY_ENABLED ? t('report.aiLikelihood.whyDiffer') : t('report.aiLikelihood.externalDemoted')}</div>
          </>
        )}
      </div>
    );
  };

  // The scan section's card is the read users learn the document through: fused verdict
  // (fused %, composite + deep-scan evidence), "How the writing reads" composition, and
  // "Where the risk sits" axes. The comparison columns previously spoke a different
  // vocabulary (transformation contributor bars / grounding-driver numbers that appear
  // nowhere in the scan card), so the original column looked unrelated to the scan the
  // user just read. Render the IDENTICAL card slices per column: original = the scan's
  // own card, rewritten = the same card recomputed on the rewritten text. The coaching
  // block (main-fix driver + policy cards) stays below as secondary detail. Falls back
  // to nothing for legacy rewrites whose stored badges predate the V7 compaction
  // keep-list (component renders only the sections whose data is present).
  const renderFusedVerdictBand = (variantBadge) => {
    if (!variantBadge?.tier_authority && !variantBadge?.authorship_breakdown) return null;
    return (
      <MergedAuthorshipRisk
        t={t}
        breakdown={variantBadge.authorship_breakdown || null}
        sr={submissionRisk(variantBadge)}
        authoritativeTier={variantBadge.tier || report.tier}
        tierAuthority={variantBadge.tier_authority || null}
        headlineConfidence={variantBadge.headline_confidence || null}
        sections={['verdict', 'composition', 'riskAxes']}
      />
    );
  };

  const renderTransformationDetails = (variant, pattern, summary, _variantAiScore, ratingBadge = null) => {
    const variantBadge = variant === 'rewritten'
      ? { ...rewrittenBadge, ai_likelihood_score: rewrittenBadge.ai_likelihood_score ?? rewrittenAiScore }
      : { ...originalComparisonBadge, ai_likelihood_score: originalComparisonBadge.ai_likelihood_score ?? aiScore };
    const fusedCard = hasRewriteSignalComparison ? renderFusedVerdictBand(variantBadge) : null;
    return (
      <div className={`transformation-detail ${variant === 'rewritten' ? 'is-rewritten' : 'is-original'}`}>
        {/* Column identity FIRST — "Original/Rewritten Scan" is the label a reader orients
            by; below the content it read as a stray footer. In single-scan mode the label
            just repeats the card header h2, so it stays hidden there. */}
        {(hasRewriteSignalComparison || ratingBadge?.label) && (
          <div className="transformation-detail-head">
            <div>
              <span>{variant === 'rewritten' ? t('report.transformation.rewrittenScan') : t('report.transformation.originalScan')}</span>
              <strong>{transformationLabel(pattern, t) || (variant === 'rewritten' ? t('report.transformation.rewrittenPatternFallback') : t('report.transformation.originalPatternFallback'))}</strong>
              {ratingBadge?.label && (
                <div
                  className="transformation-column-rating"
                  style={{
                    '--rating-color': ratingBadge.tone?.color || '#334155',
                    '--rating-bg': ratingBadge.tone?.bg || '#f8fafc',
                  }}
                  title={ratingBadge.fullLabel || ratingBadge.label}
                >
                  <span>{ratingBadge.caption}</span>
                  <b>{ratingBadge.label}</b>
                </div>
              )}
            </div>
          </div>
        )}
        {/* Comparison column = the scan card, nothing else. The AI-writing-signal
            narrative and the policy cards both repeat the card's story in other
            vocabularies (live feedback 2026-07-07); columns without V7 data (legacy
            rewrites) keep the previous full layout. */}
        {fusedCard || renderAiLikelihoodHeadline(variantBadge)}
      </div>
    );
  };

  // The "Human / uncertain pattern" classification already conveys the uncertainty in its
  // label, so a "Low confidence" pill alongside it is redundant and reads as contradictory
  // (reassuring "Human /" vs "Low confidence"). Show the confidence pill only for the
  // specific AI patterns, where it adds real signal.
  const showTransformationConfidence = Boolean(transformation?.confidence) && transformation?.code !== 'human_uncertain';
  return transformation && transformationSignals.length > 0 ? (
    <section className="transformation-scorecard" aria-label={t('report.transformation.scorecard')}>
      <div className="transformation-header">
        <div className="transformation-summary">
          <div className="transformation-icon" aria-hidden="true">
            <svg width="30" height="30" viewBox="0 0 30 30" fill="none">
              <path d="M6 8.5h12.5M6 15h18M6 21.5h10" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round"/>
              <path d="M21 7l3 3-3 3M18 18l-3 3 3 3" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
          <div>
            <span className="transformation-kicker">{t('report.transformation.kicker')}</span>
            <h2>{hasRewriteSignalComparison ? t('report.transformation.originalVsRewritten') : transformationLabel(transformation, t) || t('report.transformation.patternAnalysis')}</h2>
            {!hasRewriteSignalComparison && t(`report.transformation.subtitles.${transformation.code}`, { defaultValue: '' }) && (
              <p className="transformation-subtitle" style={{ margin: '4px 0 0', color: 'var(--color-text-secondary, #64748b)', fontSize: '14px', maxWidth: '52ch' }}>
                {t(`report.transformation.subtitles.${transformation.code}`)}
              </p>
            )}
          </div>
        </div>
        {(showTransformationConfidence || hasRewriteSignalComparison) && (
          <div className="transformation-meta-row">
            {showTransformationConfidence && (
              <span className="transformation-pill" title={t('report.transformation.confidenceTooltip')}>
                {t(`report.transformation.confidenceBadge.${transformation.confidence}`, { defaultValue: t('report.transformation.confidence', { value: confidenceLabel(transformation.confidence, t) }) })}
              </span>
            )}
            {hasRewriteSignalComparison && (
              <span className="transformation-pill">{t('report.transformation.rewriteComparison')}</span>
            )}
          </div>
        )}
      </div>
      <div className="transformation-chart">
        {hasRewriteSignalComparison ? (
          <div className="transformation-comparison-grid">
            {renderTransformationDetails('original', transformation, transformationSummary, transformationOriginalScore, originalColumnRatingBadge)}
            {renderTransformationDetails('rewritten', rewrittenTransformation, rewrittenTransformationSummary, transformationRewrittenScore, rewrittenColumnRatingBadge)}
          </div>
        ) : (
          renderTransformationDetails('original', transformation, transformationSummary, transformationOriginalScore)
        )}
      </div>
      {rewriteResultSummary?.verdict_label ? (
        // Gap-resolution verdict reframe: headline the content-gap outcome, keep the
        // AI-likelihood after-score visible with the make-it-yours note (annotate, don't
        // suppress). Only renders when production.py emitted the field (reframe on).
        <div className="rewrite-verdict-reframe">
          <p className="rewrite-verdict-headline">
            {t(`report.rewriteVerdict.labels.${rewriteResultSummary.verdict_label}`, {
              defaultValue: t('report.rewriteVerdict.labels.draft_for_review'),
            })}
          </p>
          <p className="rewrite-verdict-sub">
            {t('report.rewriteVerdict.sub', {
              findings: rewriteResultSummary.gap_resolution?.findings_resolved ?? 0,
              anchors: rewriteResultSummary.gap_resolution?.anchors_added ?? 0,
            })}
          </p>
          <p className="rewrite-verdict-note">
            {rewriteResultSummary.ai_likelihood_note || t('report.rewriteVerdict.note')}
          </p>
        </div>
      ) : null}
    </section>
  ) : null;
}
