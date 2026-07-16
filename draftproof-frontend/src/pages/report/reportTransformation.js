import { TIER_CONFIG, SIGNAL_COLORS, signalLabel, clampPercent } from './reportHelpers';

export const TRANSFORMATION_SIGNAL_LABELS = {
  topk_pattern: 'Raw Top-k predictability',
  topk_pattern_raw: 'Raw Top-k predictability',
  topk_calibrated_risk: 'Calibrated Top-k risk',
  ai_likelihood: 'AI likelihood',
  adjusted_ai_risk: 'Adjusted AI risk',
  calibrated_ai_risk: 'Calibrated AI risk',
  human_anchor_score: 'Human anchor',
  human_anchor_discount: 'Human anchor discount',
  rewrite_smoothness: 'Rewrite smoothness',
  semantic_uniformity_risk: 'Semantic uniformity',
  discourse_regularity_risk: 'Discourse regularity',
  source_similarity: 'Source similarity',
  surface_similarity: 'Surface similarity',
  paraphrase_transformation_risk: 'Paraphrase transformation',
  outline_to_text_expansion: 'Expansion pattern',
  section_style_variance: 'Patchwork variance',
  grounding_quality_risk: 'Grounding risk',
  citation_grounding_risk: 'Citation grounding',
  signal_agreement_score: 'Signal agreement',
  calibration_confidence: 'Calibration confidence',
  reporting_suppression: 'Reporting suppression',
};

export const TRANSFORMATION_SIGNAL_DESCRIPTIONS = {
  topk_pattern: 'How often the writing chooses very expected words. A very high score means the wording follows common predictable paths. Lower is better.',
  topk_pattern_raw: 'How often the writing chooses very expected words. A very high score means the wording follows common predictable paths. Lower is better.',
  topk_calibrated_risk: 'How predictable the wording looks after DraftProof adjusts the raw score. Very high scores can make writing look machine-smoothed. Lower is better.',
  ai_likelihood: 'Overall AI-style writing pattern detected in the text. This is a signal, not proof of authorship. Lower is better.',
  adjusted_ai_risk: 'AI-style risk after DraftProof accounts for human details and other context. Lower is better.',
  calibrated_ai_risk: 'The final authorship-risk score after DraftProof applies cautious checks. Lower is better.',
  human_anchor_score: 'Concrete personal, local, or experience-based details that make the writing feel more human-authored. Higher is better.',
  human_anchor_discount: 'How much the human details reduce AI concern. Higher means the writing has stronger human evidence.',
  rewrite_smoothness: 'How polished and even the writing flow appears. Higher scores may suggest the text was heavily smoothed. Lower is better.',
  semantic_uniformity_risk: 'How evenly the ideas move from paragraph to paragraph. Very even pacing can look less natural. Lower is better.',
  discourse_regularity_risk: 'How predictable the paragraph flow feels. Very regular flow can make the writing look templated. Lower is better.',
  source_similarity: 'How close the meaning is to known source material. Higher scores may suggest heavy reliance on a source.',
  surface_similarity: 'How close the wording is to known source material. Higher scores may suggest copied or lightly changed text.',
  paraphrase_transformation_risk: 'Whether source meaning appears to be kept while the wording is heavily changed. Lower is better.',
  outline_to_text_expansion: 'Whether short ideas appear expanded into fuller writing in a structured way. Lower is better.',
  section_style_variance: 'Whether sections look stitched together or written in different passes. Lower is better.',
  grounding_quality_risk: 'How well claims are grounded in concrete, lived, specific detail (the grounding the rewrite actually improves). Lower is better. Separate from the citation row below, which only your real sources can fix.',
  citation_grounding_risk: 'Whether claims are backed by real citations or sources. The rewrite cannot add real citations for you, so this stays high until you add your own — it is separate from the concrete, lived-detail grounding the rewrite improves. An integrity signal, not proof of AI.',
  signal_agreement_score: 'How much the different DraftProof checks point in the same direction. Higher means the result is more consistent.',
  calibration_confidence: 'How confident DraftProof is that the available evidence is strong enough to report. Higher means more confidence.',
  reporting_suppression: 'DraftProof is being cautious because some evidence is uncertain or not strong enough. Higher means more caution was applied.',
};

const TRANSFORMATION_SIGNAL_ORDER = [
  'topk_calibrated_risk',
  'topk_pattern_raw',
  'topk_pattern',
  'ai_likelihood',
  'adjusted_ai_risk',
  'calibrated_ai_risk',
  'human_anchor_score',
  'human_anchor_discount',
  'rewrite_smoothness',
  'semantic_uniformity_risk',
  'discourse_regularity_risk',
  'source_similarity',
  'surface_similarity',
  'paraphrase_transformation_risk',
  'outline_to_text_expansion',
  'section_style_variance',
  'grounding_quality_risk',
  'citation_grounding_risk',
  'signal_agreement_score',
  'calibration_confidence',
  'reporting_suppression',
];

const TRANSFORMATION_SIGNAL_GROUPS = [
  {
    id: 'ai_authorship',
    label: 'AI Authorship Signals',
    description: 'Pattern, predictability, and model-like texture signals.',
    keys: [
      'topk_calibrated_risk',
      'topk_pattern_raw',
      'topk_pattern',
      'ai_likelihood',
      'rewrite_smoothness',
      'semantic_uniformity_risk',
      'section_style_variance',
      'outline_to_text_expansion',
      'discourse_regularity_risk',
    ],
  },
  {
    id: 'human_authenticity',
    label: 'Human / Authenticity Signals',
    description: 'Grounding and anchor signals that reduce authorship certainty.',
    keys: [
      'human_anchor_score',
      'human_anchor_discount',
    ],
  },
  {
    id: 'quality_calibration',
    label: 'Quality & Calibration Signals',
    description: 'Evidence quality, calibration confidence, and reporting caution.',
    keys: [
      'grounding_quality_risk',
      'citation_grounding_risk',
      'calibration_confidence',
      'reporting_suppression',
      'signal_agreement_score',
      'adjusted_ai_risk',
      'calibrated_ai_risk',
      'source_similarity',
      'surface_similarity',
      'paraphrase_transformation_risk',
    ],
  },
];

const TRANSFORMATION_SIGNAL_GROUP_BY_KEY = TRANSFORMATION_SIGNAL_GROUPS.reduce((acc, group) => {
  group.keys.forEach((key) => {
    acc[key] = group.id;
  });
  return acc;
}, {});

const TRANSFORMATION_SIGNAL_IMPROVEMENT_DIRECTION = {
  topk_pattern: 'lower',
  topk_pattern_raw: 'lower',
  topk_calibrated_risk: 'lower',
  ai_likelihood: 'lower',
  adjusted_ai_risk: 'lower',
  calibrated_ai_risk: 'lower',
  grounding_risk: 'lower',
  grounding_quality_risk: 'lower',
  citation_grounding_risk: 'lower',
  rewrite_smoothness: 'lower',
  semantic_uniformity_risk: 'lower',
  discourse_regularity_risk: 'lower',
  source_similarity: 'lower',
  surface_similarity: 'lower',
  paraphrase_transformation_risk: 'lower',
  outline_to_text_expansion: 'lower',
  section_style_variance: 'lower',
  human_anchor_score: 'higher',
  human_anchor_discount: 'higher',
};

export function buildTransformationSignals(features = {}, suppliedSignals = []) {
  const suppliedByKey = new Map(
    (Array.isArray(suppliedSignals) ? suppliedSignals : [])
      .filter((signal) => signal?.key)
      .map((signal) => [signal.key, signal])
  );

  return TRANSFORMATION_SIGNAL_ORDER
    .map((key) => {
      const supplied = suppliedByKey.get(key);
      const value = clampPercent(supplied?.score ?? features[key]);
      if (value == null) return null;
      return {
        key,
        label: supplied?.label || TRANSFORMATION_SIGNAL_LABELS[key] || key.replaceAll('_', ' '),
        description: TRANSFORMATION_SIGNAL_DESCRIPTIONS[key] || supplied?.description || 'Scanner signal used to interpret the transformation pattern.',
        family: supplied?.family,
        higherScoreMeans: supplied?.higher_score_means,
        color: SIGNAL_COLORS[key],
        value,
      };
    })
    .filter(Boolean)
    .sort((a, b) => b.value - a.value);
}

export function transformationSignalFeatureMap(signals = []) {
  return (Array.isArray(signals) ? signals : []).reduce((acc, signal) => {
    if (!signal?.key) return acc;
    const value = clampPercent(signal.score ?? signal.value);
    if (value != null) acc[signal.key] = value;
    return acc;
  }, {});
}

export function sortTransformationSignalsForComparison(signals = []) {
  return [...signals].sort((a, b) => {
    const scoreDelta = Number(b.value || 0) - Number(a.value || 0);
    if (Math.abs(scoreDelta) >= 0.01) return scoreDelta;

    const labelCompare = String(a.label || '').localeCompare(String(b.label || ''), undefined, {
      sensitivity: 'base',
    });
    if (labelCompare !== 0) return labelCompare;
    return String(a.key || '').localeCompare(String(b.key || ''), undefined, {
      sensitivity: 'base',
    });
  });
}

export function buildPairedTransformationSignals(originalSignals = [], rewrittenSignals = []) {
  const originalByKey = new Map(originalSignals.filter((signal) => signal?.key).map((signal) => [signal.key, signal]));
  const rewrittenByKey = new Map(rewrittenSignals.filter((signal) => signal?.key).map((signal) => [signal.key, signal]));
  const originalOrder = sortTransformationSignalsForComparison(originalSignals).map((signal) => signal.key);
  const rewrittenOnlyOrder = sortTransformationSignalsForComparison(rewrittenSignals)
    .map((signal) => signal.key)
    .filter((key) => !originalByKey.has(key));

  return [...originalOrder, ...rewrittenOnlyOrder].map((key) => {
    const original = originalByKey.get(key);
    const rewritten = rewrittenByKey.get(key);
    const reference = original || rewritten || {};
    return {
      key,
      label: reference.label || TRANSFORMATION_SIGNAL_LABELS[key] || key.replaceAll('_', ' '),
      description: reference.description || TRANSFORMATION_SIGNAL_DESCRIPTIONS[key] || 'Scanner signal used to interpret the transformation pattern.',
      color: reference.color || SIGNAL_COLORS[key],
      original,
      rewritten,
    };
  });
}

export function groupTransformationSignals(signals = []) {
  const groups = TRANSFORMATION_SIGNAL_GROUPS.map((group) => ({
    ...group,
    signals: [],
  }));
  const other = { id: 'other', label: 'Other Signals', keys: [], signals: [] };
  const byId = new Map(groups.map((group) => [group.id, group]));

  (Array.isArray(signals) ? signals : []).forEach((signal) => {
    const groupId = TRANSFORMATION_SIGNAL_GROUP_BY_KEY[signal?.key] || 'other';
    (byId.get(groupId) || other).signals.push(signal);
  });

  return [...groups, other].filter((group) => group.signals.length > 0);
}

export function getTransformationSignalImprovement(signal, baselineSignal) {
  if (!signal || !baselineSignal) return null;
  const direction = TRANSFORMATION_SIGNAL_IMPROVEMENT_DIRECTION[signal.key];
  if (!direction) return null;

  const baselineValue = Number(baselineSignal.value);
  const rewrittenValue = Number(signal.value);
  if (!Number.isFinite(baselineValue) || !Number.isFinite(rewrittenValue)) return null;

  const delta = rewrittenValue - baselineValue;
  const changedEnough = Math.abs(delta) >= 0.5;
  if (!changedEnough) return null;
  if (direction === 'lower' && delta < 0) return { from: baselineValue, to: rewrittenValue, delta };
  if (direction === 'higher' && delta > 0) return { from: baselineValue, to: rewrittenValue, delta };
  return null;
}

// Which way is "good" for a signal: 'lower' (value high = concerning) or
// 'higher' (value low = concerning). Returns null when the signal has no known
// polarity, so callers can fall back to neutral treatment.
export function transformationSignalDirection(key) {
  return TRANSFORMATION_SIGNAL_IMPROVEMENT_DIRECTION[key] || null;
}

// Resolve a signal value to a severity tier USING the signal's polarity, so the
// bar's colour encodes "how concerning" rather than signal identity. Bar length
// already carries magnitude; this frees hue to carry good/bad. Reuses the
// existing TIER_CONFIG palette — no new colour list. Unknown-polarity signals
// return a neutral slate so they read as informational, not alarming.
export function transformationSignalSeverity(key, value) {
  const direction = TRANSFORMATION_SIGNAL_IMPROVEMENT_DIRECTION[key];
  const numeric = Number(value);
  if (!direction || !Number.isFinite(numeric)) {
    return { tier: 'neutral', color: '#94a3b8', direction: direction || null };
  }
  // "badness" = how far the value sits in the undesirable direction (0..100).
  const badness = direction === 'higher' ? 100 - numeric : numeric;
  const tier = badness >= 67 ? 'high' : badness >= 34 ? 'moderate' : 'low';
  return { tier, color: TIER_CONFIG[tier].color, direction };
}

export function buildTransformationSummary(features = {}, signals = [], contributionOverride = null, t = null) {
  const tr = t || ((key, options = {}) => options.defaultValue || key);
  const humanAnchor = clampPercent(features.human_anchor_score) ?? 0;
  const groundingQuality = 100 - (clampPercent(features.citation_grounding_risk) ?? 0);
  const semanticOriginality = 100 - Math.max(
    clampPercent(features.source_similarity) ?? 0,
    clampPercent(features.surface_similarity) ?? 0
  );

  const aiLikelihood = clampPercent(features.calibrated_ai_risk) ?? clampPercent(features.adjusted_ai_risk) ?? clampPercent(features.ai_likelihood) ?? 0;
  const rewriteSmoothness = clampPercent(features.rewrite_smoothness) ?? 0;
  const expansionPattern = clampPercent(features.outline_to_text_expansion) ?? 0;
  const patchworkVariance = clampPercent(features.section_style_variance) ?? 0;
  const groundingRisk = clampPercent(features.citation_grounding_risk) ?? 0;
  const sourceSimilarity = clampPercent(features.source_similarity) ?? 0;

  const humanRaw = (
    humanAnchor * 0.55 +
    groundingQuality * 0.25 +
    semanticOriginality * 0.20
  );
  const aiRaw = (
    aiLikelihood * 0.35 +
    rewriteSmoothness * 0.20 +
    expansionPattern * 0.15 +
    groundingRisk * 0.15 +
    patchworkVariance * 0.10 +
    sourceSimilarity * 0.05
  );
  const total = Math.max(humanRaw + aiRaw, 1);
  const overrideHuman = clampPercent(contributionOverride?.humanContribution);
  const overrideAi = clampPercent(contributionOverride?.aiTransformation);
  const humanContribution = Math.round(
    overrideHuman ?? (overrideAi != null ? 100 - overrideAi : (humanRaw / total) * 100)
  );
  const aiTransformation = Math.round(
    overrideAi ?? (overrideHuman != null ? 100 - overrideHuman : 100 - humanContribution)
  );
  const topSignals = signals
    .filter((signal) => signal.key !== 'human_anchor_score')
    .slice(0, 2)
    .map((signal) => signalLabel(signal.key, signal.label, tr).toLowerCase());

  let summary = tr('report.transformation.summaryMixed');
  if (aiTransformation >= 70) {
    summary = tr('report.transformation.summaryAiDominates');
  } else if (aiTransformation >= 55) {
    summary = tr('report.transformation.summaryAiStronger');
  } else if (humanContribution >= 65) {
    summary = tr('report.transformation.summaryHumanStronger');
  }
  if (topSignals.length) {
    summary += ` ${tr('report.transformation.mainDrivers', { drivers: topSignals.join(' and ') })}`;
  }

  return {
    humanContribution,
    aiTransformation,
    adjustedAiRisk: Math.round(aiLikelihood),
    rawAdjustedAiRisk: Math.round(clampPercent(features.adjusted_ai_risk) ?? aiLikelihood),
    humanAnchorDiscount: Math.round(clampPercent(features.human_anchor_discount) ?? 0),
    calibrationConfidence: Math.round(clampPercent(features.calibration_confidence) ?? 0),
    reportingSuppression: Math.round(clampPercent(features.reporting_suppression) ?? 0),
    summary,
  };
}

