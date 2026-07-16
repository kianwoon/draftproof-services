import { signalLabel, formatMetricPercent, TURNITIN_AI_REFERENCE_THRESHOLD, AI_SIGNAL_STAMP_LEVELS, metricValue, clampPercent, metricCount } from './reportHelpers';

export function deriveAuthorshipRatingFallback(score, tierValue, writingScore, aiComponents = {}, writingComponents = {}) {
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

export function deriveCalibratedAuthorshipRating(
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

export function formatAuthorshipSealDetail({
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

export function formatAuthorshipSealDetailWithReference(detail, referenceScore, t) {
  const suffix = formatAiReferenceSuffix(referenceScore, t);
  if (!suffix) return detail;
  return detail ? `${detail} · ${suffix}` : suffix;
}

export function getAiSignalStamp(score, t) {
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

export function getAuthorshipTone(rating = {}) {
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

