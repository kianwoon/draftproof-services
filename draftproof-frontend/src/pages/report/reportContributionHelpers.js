function metricValue(value) {
  if (value == null || Number.isNaN(Number(value))) return null;
  const number = Number(value);
  return Math.abs(number) <= 1 ? number * 100 : number;
}

function clampPercent(value) {
  const percent = metricValue(value);
  if (percent == null) return null;
  return Math.max(0, Math.min(100, percent));
}

function getScanIntelligence(scan) {
  return scan?.scan_intelligence || scan?.results_json?.scan_intelligence || {};
}

function contributionPair(humanValue, aiValue) {
  const human = clampPercent(humanValue);
  const ai = clampPercent(aiValue);
  if (human == null && ai == null) return null;
  const normalizedHuman = clampPercent(human ?? 100 - ai);
  return {
    humanContribution: Math.round(normalizedHuman),
    aiTransformation: Math.round(100 - normalizedHuman),
  };
}

function getScanContributionSummary(scan) {
  if (!scan) return null;
  const intelligence = getScanIntelligence(scan);
  const contribution = intelligence.transformation?.contribution || {};
  const layers = scan.integrity_layers?.layers || intelligence.integrity_layers?.layers || {};
  const humanLayerScore = layers.human_contribution_signal?.score;
  const aiLayerScore = layers.ai_transformation_risk?.score;
  const pair = contributionPair(
    contribution.human_contribution_ratio ?? contribution.human_contribution ?? contribution.human_ratio ?? humanLayerScore,
    contribution.ai_transformation_ratio ?? contribution.ai_transformation ?? contribution.transformation_ratio ?? aiLayerScore
  );

  if (!pair) return null;
  return {
    ...pair,
    adjustedAiRisk: Math.round(clampPercent(contribution.calibrated_ai_risk ?? contribution.adjusted_ai_risk) ?? 0),
    rawAdjustedAiRisk: Math.round(clampPercent(contribution.adjusted_ai_risk ?? contribution.calibrated_ai_risk) ?? 0),
    humanAnchorDiscount: Math.round(clampPercent(contribution.human_anchor_discount) ?? 0),
    calibrationConfidence: Math.round(clampPercent(contribution.calibration_confidence) ?? 0),
    reportingSuppression: Math.round(clampPercent(contribution.reporting_suppression) ?? 0),
    summary: contribution.summary || '',
  };
}

function buildRewriteContributionOverride(rewriteResultSummary, variant = 'rewritten') {
  if (!rewriteResultSummary) return null;
  return contributionPair(
    variant === 'original'
      ? rewriteResultSummary.original_human_contribution
      : rewriteResultSummary.rewritten_human_contribution,
    variant === 'original'
      ? rewriteResultSummary.original_ai_transformation
      : rewriteResultSummary.rewritten_ai_transformation
  );
}

export {
  buildRewriteContributionOverride,
  getScanContributionSummary,
};
