import { buildApiEventUrl } from '../../api/draftproofApi';
import {
  buildRewriteContributionOverride,
  getScanContributionSummary,
} from './reportContributionHelpers';

const TIER_CONFIG = {
  low:      { label: 'Low Risk',      color: '#22c55e', bg: '#f0fdf4', icon: 'M12 15.5l-3-3 1.4-1.4L12 12.6l4.6-4.6L18 9.5z' },
  moderate: { label: 'Moderate Risk',  color: '#f59e0b', bg: '#fffbeb', icon: 'M12 9v4M12 15h.01' },
  high:     { label: 'High Risk',      color: '#ef4444', bg: '#fef2f2', icon: 'M12 9v4M12 15h.01M4.93 4.93l14.14 14.14' },
  green:    { label: 'Low Risk',       color: '#22c55e', bg: '#f0fdf4', icon: 'M12 15.5l-3-3 1.4-1.4L12 12.6l4.6-4.6L18 9.5z' },
  amber:    { label: 'Moderate Risk',  color: '#f59e0b', bg: '#fffbeb', icon: 'M12 9v4M12 15h.01' },
  orange:   { label: 'High Risk',      color: '#f97316', bg: '#fff7ed', icon: 'M12 9v4M12 15h.01M4.93 4.93l14.14 14.14' },
  red:      { label: 'Critical Risk',  color: '#ef4444', bg: '#fef2f2', icon: 'M12 9v4M12 15h.01M4.93 4.93l14.14 14.14' },
};

const SEVERITY_CONFIG = {
  critical: { color: '#dc2626', bg: '#fef2f2', label: 'CRITICAL' },
  high:     { color: '#ef4444', bg: '#fef2f2', label: 'HIGH' },
  medium:   { color: '#f59e0b', bg: '#fffbeb', label: 'MEDIUM' },
  low:      { color: '#22c55e', bg: '#f0fdf4', label: 'LOW' },
  info:     { color: '#3b82f6', bg: '#eff6ff', label: 'INFO' },
};

const SIGNAL_COLORS = {
  topk_pattern: '#d97706',
  topk_pattern_raw: '#d97706',
  topk_calibrated_risk: '#ea580c',
  ai_likelihood: '#d97706',
  adjusted_ai_risk: '#dc2626',
  calibrated_ai_risk: '#b91c1c',
  grounding_risk: '#dc2626',
  grounding_quality_risk: '#dc2626',
  citation_grounding_risk: '#dc2626',
  human_anchor_score: '#15803d',
  human_anchor_discount: '#16a34a',
  rewrite_smoothness: '#7c3aed',
  semantic_uniformity_risk: '#9333ea',
  discourse_regularity_risk: '#0891b2',
  outline_to_text_expansion: '#7c3aed',
  source_similarity: '#0f766e',
  surface_similarity: '#0f766e',
  paraphrase_transformation_risk: '#0e7490',
  section_style_variance: '#2563eb',
  predictability: '#d97706',
  writing_quality: '#7c3aed',
  genericity: '#7c3aed',
};

function signalClassName(key) {
  return String(key || 'scan_signal').replace(/[^a-zA-Z0-9_-]/g, '-');
}

function formatDate(iso, locale = 'en-SG') {
  if (!iso) return '';
  return new Date(iso).toLocaleDateString(locale, { day: 'numeric', month: 'long', year: 'numeric', hour: '2-digit', minute: '2-digit' });
}

function humanizeKey(key) {
  return String(key || '').replaceAll('_', ' ');
}

function titleizeKey(key) {
  return humanizeKey(key).replace(/\b\w/g, (char) => char.toUpperCase());
}

function signalLabel(key, fallback, t) {
  if (!key) return fallback || '';
  return t(`report.signals.labels.${key}`, { defaultValue: fallback || titleizeKey(key) });
}

function signalDescription(key, fallback, t) {
  if (!key) return fallback || t('report.signals.fallbackDescription');
  return t(`report.signals.descriptions.${key}`, {
    defaultValue: fallback || t('report.signals.fallbackDescription'),
  });
}

function translatedSignal(signal, t) {
  if (!signal) return signal;
  return {
    ...signal,
    label: signalLabel(signal.key, signal.label, t),
    description: signalDescription(signal.key, signal.description, t),
    pairedLabel: signalLabel(signal.key, signal.pairedLabel || signal.label, t),
    pairedDescription: signalDescription(signal.key, signal.pairedDescription || signal.description, t),
  };
}

function translatedGroup(group, t) {
  return {
    ...group,
    label: t(`report.signalGroups.${group.id}.label`, { defaultValue: group.label }),
    description: t(`report.signalGroups.${group.id}.description`, { defaultValue: group.description || '' }),
    signals: group.signals,
  };
}

function transformationLabel(pattern, t) {
  if (!pattern) return '';
  return t(`report.transformation.labels.${pattern.code}`, { defaultValue: pattern.label || t('report.transformation.patternAnalysis') });
}

function confidenceLabel(confidence, t) {
  const key = String(confidence || '').toLowerCase();
  return t(`report.transformation.confidenceLevels.${key}`, { defaultValue: confidence });
}

function evidenceLabel(evidence, t) {
  const normalized = String(evidence || '').toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
  return t(`report.transformation.evidence.${normalized}`, { defaultValue: evidence });
}

function translateAuthorshipRating(rating = {}, t) {
  const normalized = String(rating.code || rating.short_label || rating.label || '').toLowerCase();
  const fallbackCode = normalized.includes('low signal')
    ? 'low_signal'
    : normalized === 'likely ai'
      ? 'likely_ai_fallback'
      : null;
  const code = rating.code || fallbackCode;
  if (!code) return rating;
  return {
    ...rating,
    label: t(`report.authorship.${code}.label`, { defaultValue: rating.label }),
    short_label: t(`report.authorship.${code}.short`, { defaultValue: rating.short_label || rating.label }),
  };
}

function pct(value) {
  if (value == null || Number.isNaN(Number(value))) return null;
  return `${(Number(value) * 100).toFixed(0)}%`;
}

function formatMetricPercent(value, digits = 0) {
  if (value == null || Number.isNaN(Number(value))) return '—';
  const number = Number(value);
  const percent = Math.abs(number) <= 1 ? number * 100 : number;
  return `${percent.toFixed(digits)}%`;
}

const TURNITIN_AI_REFERENCE_THRESHOLD = 20;
const AI_SIGNAL_STAMP_LEVELS = [
  {
    min: 60,
    labelKey: 'report.aiSignalStamp.high',
    color: '#b91c1c',
    bg: '#fef2f2',
  },
  {
    min: 40,
    labelKey: 'report.aiSignalStamp.likely',
    color: '#c2410c',
    bg: '#fff7ed',
  },
  {
    min: TURNITIN_AI_REFERENCE_THRESHOLD,
    labelKey: 'report.aiSignalStamp.review',
    color: '#b45309',
    bg: '#fff7ed',
  },
  {
    min: 0,
    labelKey: 'report.aiSignalStamp.low',
    color: '#15803d',
    bg: '#f0fdf4',
  },
];

// Returns the DraftProof AI-likelihood percent (badge ai_likelihood_score) directly — the
// single canonical "AI score" shown across the report page, scan list, PDF, and email.
// (Previously halved by a 0.5 "display multiplier"; removed so every surface agrees.)
function calibratedReportAiScore(value) {
  return metricValue(value);
}

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

function metricCount(value) {
  if (value == null || Number.isNaN(Number(value))) return null;
  return Math.max(0, Number(value));
}

const TRANSFORMATION_SIGNAL_LABELS = {
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

const TRANSFORMATION_SIGNAL_DESCRIPTIONS = {
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

function buildTransformationSignals(features = {}, suppliedSignals = []) {
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

function transformationSignalFeatureMap(signals = []) {
  return (Array.isArray(signals) ? signals : []).reduce((acc, signal) => {
    if (!signal?.key) return acc;
    const value = clampPercent(signal.score ?? signal.value);
    if (value != null) acc[signal.key] = value;
    return acc;
  }, {});
}

function sortTransformationSignalsForComparison(signals = []) {
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

function buildPairedTransformationSignals(originalSignals = [], rewrittenSignals = []) {
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

function groupTransformationSignals(signals = []) {
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

function getTransformationSignalImprovement(signal, baselineSignal) {
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
function transformationSignalDirection(key) {
  return TRANSFORMATION_SIGNAL_IMPROVEMENT_DIRECTION[key] || null;
}

// Resolve a signal value to a severity tier USING the signal's polarity, so the
// bar's colour encodes "how concerning" rather than signal identity. Bar length
// already carries magnitude; this frees hue to carry good/bad. Reuses the
// existing TIER_CONFIG palette — no new colour list. Unknown-polarity signals
// return a neutral slate so they read as informational, not alarming.
function transformationSignalSeverity(key, value) {
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

function buildTransformationSummary(features = {}, signals = [], contributionOverride = null, t = null) {
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

function deriveAuthorshipRatingFallback(score, tierValue, writingScore, aiComponents = {}, writingComponents = {}) {
  if (score == null || Number.isNaN(Number(score))) return null;
  const percent = metricValue(score);
  const writingPercent = metricValue(writingScore);
  const tier = String(tierValue || '').toLowerCase();
  const highComponentAlignment = (
    percent >= 58 &&
    (
      (metricValue(aiComponents.topk_pattern) || 0) >= 80 ||
      (metricValue(aiComponents.qualifying_text_ai_density) || 0) >= 70
    ) &&
    (metricValue(aiComponents.generic_assertion_risk) || 0) >= 80 &&
    (
      (metricValue(writingComponents.unsupported_claim_risk) || 0) >= 80 ||
      (metricValue(writingComponents.source_grounding_risk) || 0) >= 70 ||
      (metricValue(writingComponents.broad_claim_risk) || 0) >= 70
    )
  );
  const highDensityAlignment = (
    percent >= 45 &&
    (metricValue(aiComponents.qualifying_text_ai_density) || 0) >= 70 &&
    (metricValue(aiComponents.topk_pattern) || 0) >= 60 &&
    (metricValue(aiComponents.generic_assertion_risk) || 0) >= 80 &&
    (
      (metricValue(writingComponents.unsupported_claim_risk) || 0) >= 75 ||
      (metricValue(writingComponents.source_grounding_risk) || 0) >= 65 ||
      (metricValue(writingComponents.broad_claim_risk) || 0) >= 65
    )
  );
  const likelyComponentAlignment = (
    percent >= 48 &&
    (
      (metricValue(aiComponents.topk_pattern) || 0) >= 70 ||
      (metricValue(aiComponents.generic_assertion_risk) || 0) >= 70 ||
      (
        writingPercent != null &&
        writingPercent >= 55 &&
        (
          (metricValue(writingComponents.unsupported_claim_risk) || 0) >= 70 ||
          (metricValue(writingComponents.source_grounding_risk) || 0) >= 70 ||
          (metricValue(writingComponents.broad_claim_risk) || 0) >= 65
        )
      )
    )
  );

  if (percent >= 65 || tier === 'red' || (percent >= 60 && writingPercent != null && writingPercent >= 65) || highComponentAlignment || highDensityAlignment) {
    return { label: 'AI-Generated / AI-Paraphrased Signals', short_label: 'AI Signals' };
  }
  if (likelyComponentAlignment) {
    return { label: 'Likely AI', short_label: 'Likely AI' };
  }
  if (percent >= 32 || tier === 'amber' || tier === 'moderate') {
    return { label: 'Possible AI-Assisted', short_label: 'Possible AI' };
  }
  if (percent >= 20) {
    return { label: 'Unlikely AI', short_label: 'Unlikely AI' };
  }
  return { label: 'Low AI Signal', short_label: 'Low Signal' };
}

const CALIBRATED_AUTHORSHIP_LEVELS = [
  { min: 60, level: 4, label: 'Strong AI-Style Signal', short_label: 'Strong AI Signal', code: 'ai_generated_signals' },
  { min: 45, level: 3, label: 'Likely AI-Assisted', short_label: 'Likely AI-Assisted', code: 'likely_ai' },
  { min: 32, level: 2, label: 'Possible AI-Assisted', short_label: 'Possible AI-Assisted', code: 'possible_ai_assisted' },
  { min: 20, level: 1, label: 'Unlikely AI-Assisted', short_label: 'Unlikely AI-Assisted', code: 'unlikely_ai' },
  { min: 0, level: 0, label: 'Good', short_label: 'Good', code: 'low_ai_signal' },
];

function ratingForCalibratedPercent(percent) {
  const rating = CALIBRATED_AUTHORSHIP_LEVELS.find((item) => percent >= item.min) || CALIBRATED_AUTHORSHIP_LEVELS.at(-1);
  return { ...rating };
}

function getAuthorshipSampleLimit(sampleContext = {}, topkCalibrationEligible = null) {
  const wordCount = metricCount(sampleContext?.word_count);
  const sentenceCount = metricCount(sampleContext?.sentence_count);
  const hasSampleContext = wordCount != null || sentenceCount != null || topkCalibrationEligible === false;
  if (!hasSampleContext) return null;

  const veryShort = (
    topkCalibrationEligible === false ||
    (wordCount != null && wordCount < 30) ||
    (sentenceCount != null && sentenceCount < 3)
  );
  const limited = (
    veryShort ||
    (wordCount != null && wordCount < 150) ||
    (sentenceCount != null && sentenceCount < 6)
  );

  return {
    wordCount,
    sentenceCount,
    veryShort,
    limited,
  };
}

function strongestSupportingAiShapeSignal(signals = {}) {
  const candidates = [
    ['ai_likelihood', 'AI likelihood', signals.ai_likelihood],
    ['semantic_uniformity_risk', 'Semantic uniformity', signals.semantic_uniformity_risk],
    ['section_style_variance', 'Patchwork variance', signals.section_style_variance],
    ['rewrite_smoothness', 'Rewrite smoothness', signals.rewrite_smoothness],
    ['outline_to_text_expansion', 'Expansion pattern', signals.outline_to_text_expansion],
    ['discourse_regularity_risk', 'Discourse regularity', signals.discourse_regularity_risk],
  ]
    .map(([key, label, value]) => ({ key, label, score: clampPercent(value) }))
    .filter((signal) => signal.score != null);

  return candidates.find((signal) => signal.score >= 50) || null;
}

function deriveCalibratedAuthorshipRating(
  score,
  topkPatternScore = null,
  topkCalibratedRisk = null,
  supportingSignals = {},
  sampleContext = {},
  topkCalibrationEligible = null
) {
  const calibratedPercent = clampPercent(score);
  const topkPercent = clampPercent(topkPatternScore);
  const topkRiskPercent = clampPercent(topkCalibratedRisk);
  const supportingSignal = strongestSupportingAiShapeSignal(supportingSignals);
  const sampleLimit = getAuthorshipSampleLimit(sampleContext, topkCalibrationEligible);
  const aiLikelihoodPercent = clampPercent(supportingSignals?.ai_likelihood);
  const humanAnchorPercent = clampPercent(supportingSignals?.human_anchor_score);
  const semanticUniformityPercent = clampPercent(supportingSignals?.semantic_uniformity_risk);
  if (sampleLimit?.veryShort) {
    return {
      label: 'Too Short to Assess',
      short_label: 'Too Short',
      code: 'insufficient_sample',
      level: -1,
      sample_limited: true,
      sample_context: sampleLimit,
      score: calibratedPercent,
      topk_score: topkPercent,
      topk_calibrated_risk: topkRiskPercent,
    };
  }
  let rating = calibratedPercent == null ? null : ratingForCalibratedPercent(calibratedPercent);

  const turnitinZeroLikeHumanProfile = (
    calibratedPercent != null &&
    calibratedPercent <= 14 &&
    humanAnchorPercent != null &&
    humanAnchorPercent >= 75 &&
    (aiLikelihoodPercent == null || aiLikelihoodPercent <= 35) &&
    (topkRiskPercent == null || topkRiskPercent <= 55) &&
    (semanticUniformityPercent == null || semanticUniformityPercent <= 35)
  );
  if (turnitinZeroLikeHumanProfile) {
    rating = {
      ...CALIBRATED_AUTHORSHIP_LEVELS.find((item) => item.code === 'low_ai_signal'),
      turnitin_zero_like_human_profile: true,
    };
  }

  const applyTopkFloor = (floor, extra = {}) => {
    if (turnitinZeroLikeHumanProfile) return;
    if (!rating || rating.level < floor.level) {
      rating = {
        ...floor,
        topk_escalated: true,
        topk_score: topkPercent,
        topk_calibrated_risk: topkRiskPercent,
        supporting_signal: supportingSignal,
        ...extra,
      };
    }
  };

  const strongTopkWholeProfile = (
    topkPercent != null &&
    topkRiskPercent != null &&
    topkPercent >= 90 &&
    topkRiskPercent >= 90 &&
    supportingSignal &&
    calibratedPercent != null &&
    calibratedPercent >= 35 &&
    aiLikelihoodPercent != null &&
    aiLikelihoodPercent >= 55 &&
    (humanAnchorPercent == null || humanAnchorPercent <= 50)
  );
  if (strongTopkWholeProfile) {
    applyTopkFloor(CALIBRATED_AUTHORSHIP_LEVELS.find((item) => item.code === 'ai_generated_signals'), {
      topk_strong_signal: true,
    });
  }

  if (topkRiskPercent != null) {
    if (topkRiskPercent >= 80) {
      applyTopkFloor(CALIBRATED_AUTHORSHIP_LEVELS.find((item) => item.code === 'likely_ai'));
    } else if (topkRiskPercent >= 70) {
      applyTopkFloor(CALIBRATED_AUTHORSHIP_LEVELS.find((item) => item.code === 'possible_ai_assisted'));
    }
  }

  const moderateAiTexture = (
    calibratedPercent != null &&
    aiLikelihoodPercent != null &&
    topkPercent != null &&
    topkRiskPercent != null &&
    calibratedPercent >= 15 &&
    aiLikelihoodPercent >= 32 &&
    aiLikelihoodPercent < 45 &&
    topkPercent >= 70 &&
    topkRiskPercent >= 40
  );
  if (moderateAiTexture) {
    applyTopkFloor(CALIBRATED_AUTHORSHIP_LEVELS.find((item) => item.code === 'possible_ai_assisted'), {
      moderate_ai_texture: true,
    });
  }

  if (!rating) return null;
  if (sampleLimit?.limited && rating.level > 2) {
    rating = {
      ...CALIBRATED_AUTHORSHIP_LEVELS.find((item) => item.code === 'possible_ai_assisted'),
      sample_limited: true,
      sample_context: sampleLimit,
      original_rating: rating,
    };
  }
  return {
    ...rating,
    score: calibratedPercent,
    topk_score: rating.topk_score ?? topkPercent,
    topk_calibrated_risk: rating.topk_calibrated_risk ?? topkRiskPercent,
    supporting_signal: rating.supporting_signal ?? supportingSignal,
  };
}

function formatAuthorshipSealDetail({
  rating = {},
  topkPatternScore = null,
  topkCalibratedRisk = null,
  calibratedAuthorshipRisk = null,
  fallbackScore = null,
}, t) {
  if (rating.code === 'insufficient_sample') {
    const words = rating.sample_context?.wordCount;
    const sentences = rating.sample_context?.sentenceCount;
    if (words != null || sentences != null) {
      return words != null
        ? t('report.seal.wordsNotEnough', { count: Math.round(words) })
        : t('report.seal.sentencesNotEnough', { count: Math.round(sentences) });
    }
    return t('report.seal.notEnough');
  }
  if (rating.sample_limited) {
    const words = rating.sample_context?.wordCount;
    const sentences = rating.sample_context?.sentenceCount;
    if (words != null || sentences != null) {
      return words != null
        ? t('report.seal.wordsSampleLimited', { count: Math.round(words) })
        : t('report.seal.sentencesSampleLimited', { count: Math.round(sentences) });
    }
    return t('report.seal.sampleLimited');
  }
  if (rating.topk_strong_signal && (topkCalibratedRisk != null || topkPatternScore != null) && rating.supporting_signal) {
    const topkLabel = topkCalibratedRisk != null
      ? t('report.seal.calibratedTopk', { value: formatMetricPercent(topkCalibratedRisk, 0) })
      : t('report.seal.topk', { value: formatMetricPercent(topkPatternScore, 0) });
    return `${topkLabel} · ${formatMetricPercent(rating.supporting_signal.score, 0)} ${signalLabel(rating.supporting_signal.key, rating.supporting_signal.label, t).toLowerCase()}`;
  }
  if (rating.topk_escalated && (topkCalibratedRisk != null || topkPatternScore != null)) {
    const topkLabel = t('report.seal.calibratedTopk', { value: formatMetricPercent(topkCalibratedRisk ?? topkPatternScore, 0) });
    if (rating.supporting_signal) {
      return `${topkLabel} · ${formatMetricPercent(rating.supporting_signal.score, 0)} ${signalLabel(rating.supporting_signal.key, rating.supporting_signal.label, t).toLowerCase()}`;
    }
    return topkLabel;
  }
  if (calibratedAuthorshipRisk != null) {
    return t('report.seal.calibratedRisk', { value: formatMetricPercent(calibratedAuthorshipRisk, 0) });
  }
  return t('report.seal.rawSignal', { value: formatMetricPercent(fallbackScore, 0) });
}

function formatAiReferenceSuffix(score, t) {
  const value = metricValue(score);
  if (value == null) return null;
  return value < TURNITIN_AI_REFERENCE_THRESHOLD
    ? t('report.seal.belowReference')
    : t('report.seal.thresholdExceeded');
}

function formatAuthorshipSealDetailWithReference(detail, referenceScore, t) {
  const suffix = formatAiReferenceSuffix(referenceScore, t);
  if (!suffix) return detail;
  return detail ? `${detail} · ${suffix}` : suffix;
}

function getAiSignalStamp(score, t) {
  const value = metricValue(score);
  const level = AI_SIGNAL_STAMP_LEVELS.find((item) => value != null && value >= item.min) || {
    labelKey: 'report.aiSignalStamp.review',
    color: '#334155',
    bg: '#f8fafc',
  };
  return {
    label: t(level.labelKey),
    score: value,
    tone: {
      color: level.color,
      bg: level.bg,
    },
  };
}

function getAuthorshipTone(rating = {}) {
  const code = String(rating.code || rating.short_label || rating.label || '').toLowerCase();
  if (code.includes('insufficient') || code.includes('too short')) {
    return { color: '#475569', bg: '#f8fafc' };
  }
  if (code.includes('low_signal') || code.includes('low signal')) {
    return { color: '#15803d', bg: '#f0fdf4' };
  }
  if (code.includes('unlikely')) {
    return { color: '#0f766e', bg: '#f0fdfa' };
  }
  if (code.includes('possible')) {
    return { color: '#b45309', bg: '#fff7ed' };
  }
  if (code.includes('likely')) {
    return { color: '#c2410c', bg: '#fff7ed' };
  }
  if (code.includes('generated') || code.includes('signals')) {
    return { color: '#b91c1c', bg: '#fef2f2' };
  }
  return { color: '#334155', bg: '#f8fafc' };
}

function formatSignedDelta(original, next) {
  if (original == null || next == null) return '—';
  const delta = Number(next) - Number(original);
  if (Number.isNaN(delta)) return '—';
  const rounded = Number.isInteger(delta) ? String(delta) : delta.toFixed(1);
  if (delta > 0) return `+${rounded}`;
  return rounded;
}

function formatPlainScore(value, digits = 1) {
  if (value == null) return '—';
  const numeric = Number(value);
  if (Number.isNaN(numeric)) return '—';
  return Number.isInteger(numeric) ? String(numeric) : numeric.toFixed(digits);
}

function countRewriteFindings(findings) {
  if (!findings || typeof findings !== 'object') return null;
  return ['critical', 'high', 'medium', 'low', 'info'].reduce((total, tier) => {
    const rows = findings[tier];
    return total + (Array.isArray(rows) ? rows.length : 0);
  }, 0);
}

function getRewritePayloadSummary(rewriteReport) {
  return rewriteReport?.summary || rewriteReport?.rewrite_summary || rewriteReport || {};
}

function getOriginalDetectScan(rewriteReport) {
  const summary = getRewritePayloadSummary(rewriteReport);
  return (
    summary.detect_scan_original_saved ||
    summary.detect_scan_original ||
    rewriteReport?.detect_scan_original_saved ||
    rewriteReport?.detect_scan_original ||
    null
  );
}

function isRewriteOriginalPreserved(rewriteReport) {
  const summary = getRewritePayloadSummary(rewriteReport);
  return Boolean(
    rewriteReport?.status === 'original_preserved' ||
    rewriteReport?.pipeline_status === 'original_preserved' ||
    summary.status === 'original_preserved' ||
    summary.pipeline_status === 'original_preserved' ||
    summary.outcome === 'original_preserved' ||
    summary.no_text_change
  );
}

function isRewriteSummaryOriginalPreserved(summary) {
  if (!summary) return false;
  return Boolean(
    summary.status === 'original_preserved' ||
    summary.pipeline_status === 'original_preserved' ||
    summary.outcome === 'original_preserved' ||
    summary.no_text_change === true ||
    summary.kpi_finalization_status === 'original_preserved'
  );
}

function isRewriteStrictSafeFinalization(summary) {
  if (!summary) return false;
  return Boolean(
    summary.strict_safe_band_achieved === true ||
    summary.kpi_finalization_status === 'strict_safe_auto_finalized' ||
    summary.kpi_finalization_status === 'strict_safe_author_review_required'
  );
}

function requiresRewriteAuthorReview(summary) {
  if (!summary || isRewriteSummaryOriginalPreserved(summary)) return false;
  const authorProxyContext = summary.author_proxy_context || {};
  const strictSafeFinalization = isRewriteStrictSafeFinalization(summary);
  return Boolean(
    summary.best_candidate_author_review_required === true ||
    summary.public_candidate_warning === 'author_proxy_candidate_requires_review' ||
    summary.outcome === 'rewrite_candidate_generated_needs_author_review' ||
    summary.status === 'rewrite_candidate_generated_needs_author_review' ||
    summary.kpi_finalization_status === 'strict_safe_author_review_required' ||
    (strictSafeFinalization && authorProxyContext.review_required === true)
  );
}

function requiresRewriteExternalReview(summary) {
  if (!summary || isRewriteSummaryOriginalPreserved(summary)) return false;
  return false;
}

function hasRewriteComparisonData(rewriteReport) {
  if (!rewriteReport) return false;
  const summary = getRewritePayloadSummary(rewriteReport);
  const rewrittenScan = getRewrittenDetectScan(rewriteReport);
  const originalScan = getOriginalDetectScan(rewriteReport);
  const detectScores = summary.detect_scores || {};
  return Boolean(
    hasObjectData(originalScan) ||
    hasObjectData(rewrittenScan) ||
    detectScores.rewritten_ai != null ||
    detectScores.rewritten_ai_authorship != null ||
    detectScores.original_ai != null ||
    detectScores.original_ai_authorship != null ||
    summary.final_risk != null
  );
}

function getRewrittenDetectScan(rewriteReport) {
  const summary = getRewritePayloadSummary(rewriteReport);
  if (isRewriteOriginalPreserved(rewriteReport)) {
    return summary.detect_scan_rewritten || getOriginalDetectScan(rewriteReport);
  }
  return summary.detect_scan_rewritten || rewriteReport?.detect_scan_rewritten || null;
}

function hasObjectData(value) {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value) && Object.keys(value).length > 0);
}

function mergeScanSummary(baseScan, savedScan) {
  if (!savedScan) return baseScan || null;
  if (!baseScan) return savedScan;
  const baseBadge = baseScan.ai_risk_badge || {};
  const savedBadge = savedScan.ai_risk_badge || {};
  return {
    ...baseScan,
    ...savedScan,
    ai_risk_badge: {
      ...baseBadge,
      ...savedBadge,
      transformation_classification: savedBadge.transformation_classification || baseBadge.transformation_classification,
    },
    scan_intelligence: hasObjectData(savedScan.scan_intelligence) ? savedScan.scan_intelligence : baseScan.scan_intelligence,
    integrity_layers: hasObjectData(savedScan.integrity_layers) ? savedScan.integrity_layers : baseScan.integrity_layers,
  };
}

function getScanIntelligence(scan) {
  return scan?.scan_intelligence || scan?.results_json?.scan_intelligence || {};
}

function getScanAiComponents(scan) {
  return scan?.ai_risk_badge?.ai_components || scan?.results_json?.ai_risk_badge?.ai_components || {};
}

function getScanDocumentContext(scan) {
  const intelligence = getScanIntelligence(scan);
  return scan?.document_context || scan?.results_json?.document_context || intelligence.document || {};
}

function getScanTransformationSignals(scan) {
  const coreSignals = getScanIntelligence(scan).transformation?.core_signals || [];
  const aiComponents = getScanAiComponents(scan);
  const appended = [...coreSignals];
  [
    ['topk_calibrated_risk', clampPercent(aiComponents.topk_calibrated_risk)],
    ['topk_pattern_raw', clampPercent(aiComponents.topk_pattern_raw ?? aiComponents.topk_pattern)],
  ].forEach(([key, score]) => {
    if (score == null || appended.some((signal) => signal?.key === key)) return;
    appended.push({
      key,
      label: TRANSFORMATION_SIGNAL_LABELS[key],
      description: TRANSFORMATION_SIGNAL_DESCRIPTIONS[key],
      family: 'ai_authorship_risk',
      higher_score_means: key === 'topk_calibrated_risk' ? 'higher calibrated token-route risk' : 'more predictable raw token routes',
      score,
      metric_source: 'ai_components',
    });
  });
  return appended;
}

function mergeTransformationSummary(baseSummary, authoritativeSummary) {
  if (!authoritativeSummary) return baseSummary;
  if (!baseSummary) return authoritativeSummary;
  const merged = { ...baseSummary };
  Object.entries(authoritativeSummary).forEach(([key, value]) => {
    if (value !== null && value !== undefined && value !== '') {
      merged[key] = value;
    }
  });
  return merged;
}

function buildRewriteResultSummary(rewriteReport) {
  const summary = getRewritePayloadSummary(rewriteReport);
  const detectScores = summary.detect_scores || {};
  const originalScan = summary.detect_scan_original_saved || summary.detect_scan_original || {};
  const rewrittenScan = getRewrittenDetectScan(rewriteReport) || {};
  const originalBadge = originalScan.ai_risk_badge || {};
  const rewrittenBadge = rewrittenScan.ai_risk_badge || {};
  const originalContribution = getScanContributionSummary(originalScan);
  const rewrittenContribution = getScanContributionSummary(rewrittenScan);
  const originalComponents = originalBadge.ai_components || {};
  const rewrittenComponents = rewrittenBadge.ai_components || {};
  const originalFindings = countRewriteFindings(originalScan.findings) ?? detectScores.original_findings;
  const rewrittenFindings = countRewriteFindings(rewrittenScan.findings) ?? detectScores.rewritten_findings;
  const changedSentences = (rewriteReport?.sentence_comparison || []).filter(
    (row) => String(row.orig_sentence ?? row.original ?? '').trim() !== String(row.new_sentence ?? row.rewritten ?? '').trim()
  ).length;
  const originalHumanContribution = detectScores.original_human_contribution ?? originalContribution?.humanContribution;
  const rewrittenHumanContribution = detectScores.rewritten_human_contribution ?? rewrittenContribution?.humanContribution;

  return {
    outcome: summary.outcome || '',
    status: rewriteReport?.status || summary.status || '',
    public_candidate_warning: summary.public_candidate_warning || '',
    strict_goal_status: summary.strict_goal_status || '',
    best_candidate_external_review_required: summary.best_candidate_external_review_required === true,
    best_candidate_author_review_required: summary.best_candidate_author_review_required === true,
    strict_safe_band_achieved: summary.strict_safe_band_achieved === true,
    kpi_finalization_status: summary.kpi_finalization_status || '',
    no_text_change: summary.no_text_change === true || rewriteReport?.no_text_change === true,
    author_proxy_context: summary.author_proxy_context || rewriteReport?.author_proxy_context || null,
    author_review_cards: summary.author_review_cards || rewriteReport?.author_review_cards || [],
    engine_mode: summary.rewrite_engine_mode || '',
    gate: summary.authenticity_mitigation?.selected_gate || summary.authenticity_mitigation?.best_attempt?.gate || null,
    ai_mitigation_selected: Boolean(summary.authenticity_mitigation?.selected || summary.ai_mitigation_search?.selected),
    original_ai_authorship: detectScores.original_ai_authorship ?? originalBadge.ai_likelihood_score ?? originalScan.ai_score,
    rewritten_ai_authorship: detectScores.rewritten_ai_authorship ?? rewrittenBadge.ai_likelihood_score ?? rewrittenScan.ai_score,
    original_human_contribution: originalHumanContribution,
    rewritten_human_contribution: rewrittenHumanContribution,
    original_ai_transformation: detectScores.original_ai_transformation ?? originalContribution?.aiTransformation,
    rewritten_ai_transformation: detectScores.rewritten_ai_transformation ?? rewrittenContribution?.aiTransformation,
    original_grounding_quality_risk: detectScores.original_grounding_quality_risk ?? originalComponents.source_grounding_risk ?? originalComponents.unsupported_claim_risk,
    rewritten_grounding_quality_risk: detectScores.rewritten_grounding_quality_risk ?? rewrittenComponents.source_grounding_risk ?? rewrittenComponents.unsupported_claim_risk,
    human_shift_score: detectScores.human_shift_score
      ?? (originalHumanContribution != null && rewrittenHumanContribution != null ? rewrittenHumanContribution - originalHumanContribution : null)
      ?? summary.authenticity_mitigation?.selected_human_shift_score
      ?? summary.ai_mitigation_search?.selected_human_shift_score,
    human_shift_components: detectScores.human_shift_components ?? summary.authenticity_mitigation?.selected_human_shift_components ?? summary.ai_mitigation_search?.selected_human_shift_components,
    original_risk: originalBadge.ai_likelihood_score ?? detectScores.original_ai ?? summary.original_risk,
    rewrite_risk: rewrittenBadge.ai_likelihood_score ?? detectScores.rewritten_ai ?? summary.final_risk,
    original_findings: originalFindings,
    rewritten_findings: rewrittenFindings,
    changed_sentences: changedSentences,
  };
}

function findingDescription(issue) {
  if (issue.title === 'low_specificity' && issue.evidence?.metrics) {
    const m = issue.evidence.metrics;
    const risk = pct(issue.evidence.adjusted_specificity_concern ?? m.specificity_risk);
    const specificity = pct(m.specificity_score);
    const parts = [
      risk ? `Specificity concern: ${risk}` : null,
      specificity ? `specificity score: ${specificity}` : null,
      m.named_entities != null ? `named entities: ${m.named_entities}` : null,
      m.numbers != null ? `numbers: ${m.numbers}` : null,
      m.dates != null ? `dates: ${m.dates}` : null,
      m.domain_term_count != null ? `domain terms: ${m.domain_term_count}` : null,
    ].filter(Boolean);
    return parts.join(', ');
  }

  return issue.description;
}

function normalizeSignal(signal = {}, issue = {}) {
  const key = signal.key || issue.signal_category || issue.category || 'scan_signal';
  const severity = issue.severity ? SEVERITY_CONFIG[issue.severity] : null;
  return {
    finding_id: signal.finding_id || issue.id,
    key,
    label: signal.label || TRANSFORMATION_SIGNAL_LABELS[key] || key.replaceAll('_', ' '),
    description: signal.description || findingDescription(issue) || issue.recommendation || 'Scanner signal attached to this section.',
    score: signal.score ?? clampPercent(issue.score) ?? clampPercent(issue.top10_ratio) ?? null,
    color: SIGNAL_COLORS[key] || signal.color || severity?.color || '#475569',
    title: signal.title || issue.title || '',
    tier: signal.tier || issue.severity || '',
    actionability: signal.actionability || issue.actionability || '',
    recommendation: signal.recommendation || issue.recommendation || '',
  };
}

function signalPriority(signal) {
  const tierRank = { critical: 5, high: 4, medium: 3, low: 2, info: 1 };
  return [
    tierRank[signal.tier] || 0,
    clampPercent(signal.score) ?? 0,
    signal.count || 0,
  ];
}

function compareSignals(a, b) {
  const left = signalPriority(a);
  const right = signalPriority(b);
  for (let i = 0; i < left.length; i += 1) {
    if (right[i] !== left[i]) return right[i] - left[i];
  }
  return String(a.key).localeCompare(String(b.key));
}

function uniqueCompact(values) {
  return Array.from(new Set(values.map((value) => String(value || '').trim()).filter(Boolean)));
}

function mergeParagraphSignals(segments) {
  const grouped = new Map();
  segments.forEach((segment) => {
    segment.signals.forEach((signal) => {
      const current = grouped.get(signal.key) || {
        ...signal,
        count: 0,
        finding_ids: [],
        sentence_ids: [],
        descriptions: [],
        recommendations: [],
      };
      current.count += 1;
      current.score = Math.max(current.score || 0, signal.score || 0);
      current.finding_ids.push(signal.finding_id);
      current.sentence_ids.push(segment.sentence_id);
      current.descriptions.push(signal.description);
      current.recommendations.push(signal.recommendation);
      grouped.set(signal.key, current);
    });
  });

  return Array.from(grouped.values())
    .map((signal) => {
      const descriptions = uniqueCompact(signal.descriptions);
      const recommendations = uniqueCompact(signal.recommendations);
      return {
        ...signal,
        finding_ids: uniqueCompact(signal.finding_ids),
        sentence_ids: uniqueCompact(signal.sentence_ids),
        descriptions,
        recommendations,
        description: descriptions[0] || signal.description,
        recommendation: recommendations.join(' '),
      };
    })
    .sort(compareSignals);
}

function summarizeSentenceIds(segments) {
  const ids = uniqueCompact(segments.map((segment) => segment.sentence_id));
  if (ids.length <= 2) return ids.join(', ');
  return `${ids[0]}-${ids[ids.length - 1]}`;
}

function paragraphExplanationsById(results) {
  const rows = results?.paragraph_explanations?.paragraphs;
  if (!Array.isArray(rows)) return new Map();
  return new Map(
    rows
      .filter((row) => row && row.paragraph_id)
      .map((row) => [String(row.paragraph_id), row])
  );
}

function buildSubmittedContentModel(report) {
  const results = report?.results_json || {};
  const intel = results.scan_intelligence || {};
  const rawSegments = intel.document?.segments || results.highlight_segments || [];
  const explanationByParagraph = paragraphExplanationsById(results);
  const issueById = new Map((report?.issues || []).map((issue) => [String(issue.id), issue]));
  const issuesBySentence = new Map();

  (report?.issues || []).forEach((issue) => {
    if (!issue.location) return;
    const list = issuesBySentence.get(issue.location) || [];
    list.push(issue);
    issuesBySentence.set(issue.location, list);
  });

  let segments = [];
  if (Array.isArray(rawSegments) && rawSegments.length > 0) {
    segments = rawSegments
      .map((segment, index) => {
        const sentenceId = segment.sentence_id || segment.segment_id || `s${index + 1}`;
        const directSignals = Array.isArray(segment.signals) ? segment.signals : [];
        const fallbackSignals = issuesBySentence.get(sentenceId) || [];
        const signals = directSignals.length
          ? directSignals.map((signal) => normalizeSignal(signal, issueById.get(String(signal.finding_id)) || {}))
          : fallbackSignals.map((issue) => normalizeSignal({}, issue));
        const primarySignal = signals[0] || null;
        return {
          id: segment.segment_id || sentenceId,
          sentence_id: sentenceId,
          paragraph_id: segment.paragraph_id || 'p001',
          start_char: segment.start_char ?? index,
          text: segment.text || segment.sentence || '',
          signals,
          primarySignal,
        };
      })
      .filter((segment) => segment.text);
  } else if (results.sentence_map && typeof results.sentence_map === 'object') {
    segments = Object.entries(results.sentence_map)
      .map(([sentenceId, sentence], index) => {
        const fallbackSignals = issuesBySentence.get(sentenceId) || [];
        const signals = fallbackSignals.map((issue) => normalizeSignal({}, issue));
        return {
          id: sentenceId,
          sentence_id: sentenceId,
          paragraph_id: sentence.paragraph_id || 'p001',
          start_char: sentence.start_char ?? index,
          text: sentence.text || '',
          signals,
          primarySignal: signals[0] || null,
        };
      })
      .filter((segment) => segment.text)
      .sort((a, b) => a.start_char - b.start_char);
  }

  const paragraphs = [];
  const paragraphMap = new Map();
  segments.forEach((segment) => {
    const paragraphId = segment.paragraph_id || 'p001';
    if (!paragraphMap.has(paragraphId)) {
      const paragraph = { id: paragraphId, segments: [] };
      paragraphMap.set(paragraphId, paragraph);
      paragraphs.push(paragraph);
    }
    paragraphMap.get(paragraphId).segments.push(segment);
  });

  paragraphs.forEach((paragraph) => {
    paragraph.segments.sort((a, b) => a.start_char - b.start_char);
    const text = paragraph.segments.map((segment) => segment.text).join(' ').trim();
    const signals = mergeParagraphSignals(paragraph.segments);
    paragraph.text = text;
    paragraph.signals = signals;
    paragraph.primarySignal = signals[0] || null;
    paragraph.sentence_id = summarizeSentenceIds(paragraph.segments);
    paragraph.sentence_ids = uniqueCompact(paragraph.segments.map((segment) => segment.sentence_id));
    paragraph.signalCount = paragraph.segments.reduce((count, segment) => count + segment.signals.length, 0);
    paragraph.explanation = explanationByParagraph.get(String(paragraph.id)) || null;
  });

  const signalMap = new Map();
  paragraphs.forEach((paragraph) => {
    paragraph.signals.forEach((signal) => {
      const current = signalMap.get(signal.key);
      signalMap.set(signal.key, {
        ...signal,
        count: (current?.count || 0) + 1,
        score: Math.max(current?.score || 0, signal.score || 0),
      });
    });
  });
  const legend = Array.from(signalMap.values()).sort((a, b) => b.count - a.count || (b.score || 0) - (a.score || 0));

  return {
    paragraphs,
    segments,
    legend,
    highlightedCount: paragraphs.filter((paragraph) => paragraph.signals.length > 0).length,
  };
}

// Tier-weighted severity for the per-paragraph density bar. critical worst -> low least; info trivial.
const PARAGRAPH_SEVERITY_TIER_WEIGHT = { critical: 4, high: 3, medium: 2, low: 1, info: 0.5 };
const PARAGRAPH_SEVERITY_TIER_RANK = { critical: 4, high: 3, medium: 2, low: 1, info: 0 };

// Build the per-paragraph severity heatmap bar from the submitted-content paragraph model.
// Severity = tier-weighted finding density per paragraph (weight / length), normalised across the doc.
// Width is proportional to each paragraph's share of the document length. Pure derivation from existing
// scan data (findings tier via signal.tier) -- no new backend computation, works on existing reports.
function buildParagraphSeverityBar(paragraphs) {
  if (!Array.isArray(paragraphs) || paragraphs.length === 0) return null;
  const rows = paragraphs.map((paragraph, index) => {
    const signals = (paragraph.segments || []).flatMap((segment) => segment.signals || []);
    let weight = 0;
    let topTier = '';
    let topRank = -1;
    signals.forEach((signal) => {
      const tier = String(signal.tier || '').toLowerCase();
      weight += PARAGRAPH_SEVERITY_TIER_WEIGHT[tier] ?? 1;
      const rank = PARAGRAPH_SEVERITY_TIER_RANK[tier] ?? 0;
      if (rank > topRank) {
        topRank = rank;
        topTier = tier;
      }
    });
    const length = Math.max(1, (paragraph.text || '').length);
    return {
      id: paragraph.id,
      index: index + 1,
      length,
      findingCount: signals.length,
      topTier,
      density: weight / length,
    };
  });
  const totalLength = rows.reduce((sum, row) => sum + row.length, 0) || 1;
  const maxDensity = rows.reduce((max, row) => Math.max(max, row.density), 0);
  return rows.map((row) => ({
    id: row.id,
    index: row.index,
    findingCount: row.findingCount,
    topTier: row.topTier,
    widthPct: (row.length / totalLength) * 100,
    // 0..1 relative heatmap: the most finding-dense paragraph reaches full intensity.
    intensity: maxDensity > 0 ? row.density / maxDensity : 0,
  }));
}

function formatRewriteStatus(status, t) {
  if (status === 'pending') return t('report.rewrite.queued');
  if (status === 'processing') return t('report.rewrite.processing');
  if (status === 'retrying') return t('report.rewrite.retrying');
  if (status === 'completed') return t('report.rewrite.complete');
  if (status === 'canceled') return t('report.rewrite.canceled');
  if (status === 'failed') return t('report.rewrite.failed');
  return t('report.rewrite.processing');
}

function isRewriteActive(status) {
  return ['pending', 'processing', 'retrying'].includes(status);
}

function normalizeRewriteProgressMessage(message, status, t) {
  if (!message) return formatRewriteStatus(status, t);
  const normalized = String(message).trim().toLowerCase();
  if (
    normalized.includes('rewriting your document') ||
    normalized.includes('this may take 1-3 minutes')
  ) {
    return t('report.rewrite.processing');
  }
  return message;
}

function normalizeRewriteJob(job, t) {
  if (!job) return job;
  return {
    ...job,
    progress_message: normalizeRewriteProgressMessage(job.progress_message, job.status, t),
  };
}

function formatElapsed(seconds) {
  const total = Math.max(0, Number(seconds) || 0);
  if (total < 60) return `${total} sec`;
  const minutes = Math.floor(total / 60);
  const remainingSeconds = total % 60;
  return `${minutes} min ${String(remainingSeconds).padStart(2, '0')} sec`;
}

function getRewriteProgressDetail({ status, progress, elapsedSeconds, sseUnavailable, t }) {
  if (status === 'pending') return t('report.rewrite.waiting');
  if (status === 'retrying') return t('report.rewrite.retryingDetail');
  if (sseUnavailable) return t('report.rewrite.checking');
  if (progress >= 35 && progress <= 45 && elapsedSeconds >= 30) {
    return t('report.rewrite.longestStep');
  }
  const messages = t('report.rewrite.progressMessages', { returnObjects: true });
  const index = Math.floor(Math.max(0, elapsedSeconds) / 8) % messages.length;
  return messages[index];
}
const REVIEW_ONLY_REWRITE_PATTERNS = [
  'no rewriteable ai sections',
  'no auto-fixable findings',
  'no rephrasable findings',
  'review-only',
  'review only',
];

function isReviewOnlyRewriteMessage(message) {
  if (!message) return false;
  const normalized = String(message).toLowerCase();
  return REVIEW_ONLY_REWRITE_PATTERNS.some((pattern) => normalized.includes(pattern));
}

function buildRewriteEventsUrl(rewriteId) {
  return buildApiEventUrl(`/rewrites/${rewriteId}/events`);
}

const DRAFTPROOF_TIER_COLORS = { GREEN: '#16a34a', AMBER: '#d97706', ORANGE: '#ea580c', RED: '#dc2626' };
const EXTERNAL_BAND_COLORS = { low: '#16a34a', elevated: '#d97706', high: '#dc2626' };

// The badge carries DraftProof's grouped external-detector proxy. It is surfaced as an ESTIMATE,
// not a Turnitin/vendor result; legacy estimates may appear under alternates for auditability.
// Mirror of render.EXTERNAL_ESTIMATE_DISPLAY_ENABLED — keep both in sync.
const EXTERNAL_ESTIMATE_DISPLAY_ENABLED = true;

// Mirror of report.render._ai_likelihood_bands (same band table; see spec). Returns the
// external band KEY; the label is resolved via i18n (report.aiLikelihood.externalBand.<band>).
function aiLikelihoodBands(badge) {
  const b = badge || {};
  const score = b.ai_likelihood_score;
  const tier = String(b.tier || '').toUpperCase();
  const draftproof = typeof score === 'number'
    ? { score: Math.round(score), tier: tier || 'AMBER', color: DRAFTPROOF_TIER_COLORS[tier] || '#d97706' }
    : null;
  const ext = b.external_detector_estimate || {};
  const band = String(ext.band || '').toLowerCase();
  const external = typeof ext.score === 'number'
    ? { score: Math.round(ext.score), band, color: EXTERNAL_BAND_COLORS[band] || '#475569', note: ext.note || '' }
    : null;
  return { draftproof, external };
}

const REWRITE_VERDICT_TONES = {
  high: { color: '#b91c1c', bg: '#fef2f2' },
  elevated: { color: '#c2410c', bg: '#fff7ed' },
  low: { color: '#15803d', bg: '#f0fdf4' },
};

// Rewrite seal verdict tracks the DETECTOR reality (external/Turnitin band), never the rosy
// internal calibrated risk -- users read a reassuring verdict as "Turnitin-safe", which no
// rewrite can honestly promise (the fluency estimate has a floor no rewrite removes). A
// reassuring ("low") verdict is earned only when detectors genuinely stop flagging.
function rewriteDetectorVerdict(band, t) {
  // INTERIM: the external band over-flags real Turnitin, so we don't assert a detector verdict.
  if (!EXTERNAL_ESTIMATE_DISPLAY_ENABLED) {
    return { label: t('report.transformation.verdict.grounded'), tone: { color: '#0f766e', bg: '#ecfdf5' } };
  }
  const key = String(band || '').toLowerCase();
  if (key === 'high') return { label: t('report.transformation.verdict.stillFlagged'), tone: REWRITE_VERDICT_TONES.high };
  if (key === 'elevated') return { label: t('report.transformation.verdict.riskRemains'), tone: REWRITE_VERDICT_TONES.elevated };
  if (key === 'low') return { label: t('report.transformation.verdict.riskLow'), tone: REWRITE_VERDICT_TONES.low };
  return { label: t('report.transformation.verdict.reviewed'), tone: { color: '#334155', bg: '#f8fafc' } };
}

function firstNonEmpty(...values) {
  return values.find((value) => typeof value === 'string' && value.trim())?.trim() || '';
}

function buildRepairSummary({
  report,
  submittedContent,
  authorshipEvidence,
  transformationSummary,
  status,
  pattern,
  t,
}) {
  const highlightedCount = Number(submittedContent?.highlightedCount || 0);
  const primaryParagraph = (submittedContent?.paragraphs || []).find((paragraph) => paragraph.primarySignal);
  const primarySignal = primaryParagraph?.primarySignal;
  const primarySignalLabel = primarySignal
    ? signalLabel(primarySignal.key, primarySignal.label, t)
    : '';
  const statusText = firstNonEmpty(
    status,
    pattern?.label,
    report?.ai_risk_badge?.authorship_rating_label,
    t('report.repairSummary.statusFallback')
  );
  const mainRisk = firstNonEmpty(
    primarySignalLabel,
    transformationSummary?.summary,
    t('report.repairSummary.mainRiskFallback')
  );
  const nextAction = highlightedCount > 0
    ? t('report.repairSummary.nextActionHighlighted', { count: highlightedCount })
    : authorshipEvidence?.thin_signals?.[0]?.action
      ? authorshipEvidence.thin_signals[0].action
      : t('report.repairSummary.nextActionFallback');

  return {
    status: statusText,
    mainRisk,
    nextAction,
    highlightedCount,
    confidenceNote: t('report.repairSummary.confidenceNote'),
  };
}

function buildFixFirstItems({ submittedContent, authorshipEvidence, t }) {
  const items = [];
  const seen = new Set();
  const addItem = (item) => {
    const title = firstNonEmpty(item.title);
    const body = firstNonEmpty(item.body);
    const key = `${title}|${body}|${item.paragraphId || ''}`;
    if (!title || seen.has(key)) return;
    seen.add(key);
    items.push({ ...item, title, body });
  };

  (submittedContent?.paragraphs || [])
    .filter((paragraph) => paragraph.primarySignal)
    .slice(0, 3)
    .forEach((paragraph) => {
      const guidance = paragraph.explanation || {};
      const signal = paragraph.primarySignal;
      addItem({
        paragraphId: paragraph.id,
        label: paragraph.sentence_id,
        title: firstNonEmpty(
          guidance.main_issue,
          signalLabel(signal.key, signal.label, t),
          t('report.whatToFixFirst.paragraphFallbackTitle')
        ),
        body: firstNonEmpty(
          guidance.recommendation,
          signal.recommendation,
          guidance.rewrite_hint,
          signalDescription(signal.key, signal.description, t)
        ),
      });
    });

  (authorshipEvidence?.thin_signals || []).slice(0, 2).forEach((signal) => {
    addItem({
      title: firstNonEmpty(signal.label, t('report.whatToFixFirst.authorshipFallbackTitle')),
      body: firstNonEmpty(signal.action, t('report.whatToFixFirst.authorshipFallbackBody')),
    });
  });

  (authorshipEvidence?.strengthen_examples || []).slice(0, 2).forEach((example, index) => {
    addItem({
      title: t('report.whatToFixFirst.exampleTitle', { count: index + 1 }),
      body: example,
    });
  });

  if (!items.length) {
    addItem({
      title: t('report.whatToFixFirst.reviewTitle'),
      body: t('report.whatToFixFirst.reviewBody'),
    });
  }

  return items.slice(0, 5);
}

export {
  TIER_CONFIG,
  SEVERITY_CONFIG,
  signalClassName,
  formatDate,
  signalLabel,
  signalDescription,
  translatedSignal,
  translatedGroup,
  transformationLabel,
  confidenceLabel,
  evidenceLabel,
  translateAuthorshipRating,
  formatMetricPercent,
  calibratedReportAiScore,
  clampPercent,
  buildTransformationSignals,
  transformationSignalFeatureMap,
  sortTransformationSignalsForComparison,
  buildPairedTransformationSignals,
  groupTransformationSignals,
  getTransformationSignalImprovement,
  transformationSignalDirection,
  transformationSignalSeverity,
  buildTransformationSummary,
  deriveAuthorshipRatingFallback,
  deriveCalibratedAuthorshipRating,
  formatAuthorshipSealDetail,
  formatAuthorshipSealDetailWithReference,
  getAiSignalStamp,
  getAuthorshipTone,
  formatSignedDelta,
  formatPlainScore,
  getOriginalDetectScan,
  getRewrittenDetectScan,
  hasRewriteComparisonData,
  isRewriteStrictSafeFinalization,
  requiresRewriteAuthorReview,
  requiresRewriteExternalReview,
  mergeScanSummary,
  getScanDocumentContext,
  getScanTransformationSignals,
  getScanContributionSummary,
  mergeTransformationSummary,
  buildRewriteResultSummary,
  buildRewriteContributionOverride,
  buildSubmittedContentModel,
  buildParagraphSeverityBar,
  isRewriteActive,
  normalizeRewriteProgressMessage,
  normalizeRewriteJob,
  formatElapsed,
  getRewriteProgressDetail,
  isReviewOnlyRewriteMessage,
  buildRewriteEventsUrl,
  aiLikelihoodBands,
  rewriteDetectorVerdict,
  buildRepairSummary,
  buildFixFirstItems,
  EXTERNAL_ESTIMATE_DISPLAY_ENABLED,
};
