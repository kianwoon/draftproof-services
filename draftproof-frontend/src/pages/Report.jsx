import { useCallback, useEffect, useRef, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getReport, createRewrite, cancelRewrite, getRewriteStatus, getRewriteReport, buildApiEventUrl } from '../api/draftproofApi';
import ErrorReload from '../components/ErrorReload';
import ConfirmDialog from '../components/ConfirmDialog';
import { useAuth } from '../context/AuthContext';

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
  ai_likelihood: '#9a3412',
  adjusted_ai_risk: '#dc2626',
  calibrated_ai_risk: '#b91c1c',
  grounding_risk: '#9a3412',
  citation_grounding_risk: '#9a3412',
  human_anchor_score: '#15803d',
  human_anchor_discount: '#16a34a',
  rewrite_smoothness: '#4338ca',
  semantic_uniformity_risk: '#7c3aed',
  discourse_regularity_risk: '#4f46e5',
  outline_to_text_expansion: '#4338ca',
  source_similarity: '#0369a1',
  surface_similarity: '#0369a1',
  paraphrase_transformation_risk: '#0e7490',
  section_style_variance: '#2563eb',
  predictability: '#9a3412',
  writing_quality: '#4338ca',
  genericity: '#4338ca',
};

function formatDate(iso) {
  if (!iso) return '';
  return new Date(iso).toLocaleDateString('en-SG', { day: 'numeric', month: 'long', year: 'numeric', hour: '2-digit', minute: '2-digit' });
}

function pct(value) {
  if (value == null || Number.isNaN(Number(value))) return null;
  return `${(Number(value) * 100).toFixed(0)}%`;
}

function formatMetricPercent(value, digits = 1) {
  if (value == null || Number.isNaN(Number(value))) return '—';
  const number = Number(value);
  const percent = Math.abs(number) <= 1 ? number * 100 : number;
  return `${percent.toFixed(digits)}%`;
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

const TRANSFORMATION_SIGNAL_LABELS = {
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
  citation_grounding_risk: 'Grounding risk',
  signal_agreement_score: 'Signal agreement',
  calibration_confidence: 'Calibration confidence',
  reporting_suppression: 'Reporting suppression',
};

const TRANSFORMATION_SIGNAL_DESCRIPTIONS = {
  ai_likelihood: 'Statistical AI-style signal based on predictability, token concentration, generic phrasing, and sentence regularity.',
  adjusted_ai_risk: 'AI likelihood after reducing certainty for strong human anchoring and calibration checks.',
  calibrated_ai_risk: 'Final calibrated authorship risk after thresholding, signal agreement, and reporting safeguards.',
  human_anchor_score: 'Strength of concrete lived experience, local context, operational memory, and specific human reasoning.',
  human_anchor_discount: 'How much strong human anchoring reduces AI certainty in the calibrated risk model.',
  rewrite_smoothness: 'Likelihood that a human draft was polished into cleaner, more even AI-assisted prose.',
  semantic_uniformity_risk: 'Embedding-based signal for overly even paragraph meaning or low semantic shape variation.',
  discourse_regularity_risk: 'Embedding-based signal for unusually regular paragraph progression and smooth discourse flow.',
  source_similarity: 'Meaning-level closeness to source material, useful for detecting paraphrased source content.',
  surface_similarity: 'Wording-level closeness to source material after direct text comparison.',
  paraphrase_transformation_risk: 'Risk that source meaning was retained while wording and sentence structure were heavily changed.',
  outline_to_text_expansion: 'Risk that short notes or an outline were expanded into longer prose with limited new evidence.',
  section_style_variance: 'Section-level style shifts that can suggest stitched writing from different chunks or passes.',
  citation_grounding_risk: 'Claims, citations, or academic statements that appear weakly supported by evidence.',
  signal_agreement_score: 'How strongly separate scanner layers agree with each other instead of firing in isolation.',
  calibration_confidence: 'Confidence that available signals are strong enough and sufficiently aligned for reporting.',
  reporting_suppression: 'Amount of risk held back because the evidence is uncertain, limited, or not institutionally defensible.',
};

const TRANSFORMATION_SIGNAL_ORDER = [
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
  'citation_grounding_risk',
  'signal_agreement_score',
  'calibration_confidence',
  'reporting_suppression',
];

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
        description: supplied?.description || TRANSFORMATION_SIGNAL_DESCRIPTIONS[key] || 'Scanner signal used to interpret the transformation pattern.',
        family: supplied?.family,
        higherScoreMeans: supplied?.higher_score_means,
        value,
      };
    })
    .filter(Boolean)
    .sort((a, b) => b.value - a.value);
}

function sortTransformationSignalsForComparison(signals = []) {
  return [...signals].sort((a, b) => {
    const labelCompare = String(a.label || '').localeCompare(String(b.label || ''), undefined, {
      sensitivity: 'base',
    });
    if (labelCompare !== 0) return labelCompare;
    return String(a.key || '').localeCompare(String(b.key || ''), undefined, {
      sensitivity: 'base',
    });
  });
}

function buildTransformationSummary(features = {}, signals = [], contributionOverride = null) {
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
    .map((signal) => signal.label.toLowerCase());

  let summary = 'Mixed authorship pattern: human anchoring and AI transformation signals are both visible.';
  if (aiTransformation >= 70) {
    summary = 'AI transformation dominates this scan pattern.';
  } else if (aiTransformation >= 55) {
    summary = 'AI transformation signals are stronger than the human anchor.';
  } else if (humanContribution >= 65) {
    summary = 'Human contribution remains the stronger signal.';
  }
  if (topSignals.length) {
    summary += ` Main drivers: ${topSignals.join(' and ')}.`;
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

function getRewrittenDetectScan(rewriteReport) {
  const summary = getRewritePayloadSummary(rewriteReport);
  return summary.detect_scan_rewritten || rewriteReport?.detect_scan_rewritten || null;
}

function buildRewriteResultSummary(rewriteReport) {
  const summary = getRewritePayloadSummary(rewriteReport);
  const detectScores = summary.detect_scores || {};
  const originalScan = summary.detect_scan_original_saved || summary.detect_scan_original || {};
  const rewrittenScan = getRewrittenDetectScan(rewriteReport) || {};
  const originalBadge = originalScan.ai_risk_badge || {};
  const rewrittenBadge = rewrittenScan.ai_risk_badge || {};
  const originalFindings = countRewriteFindings(originalScan.findings) ?? detectScores.original_findings;
  const rewrittenFindings = countRewriteFindings(rewrittenScan.findings) ?? detectScores.rewritten_findings;
  const changedSentences = (rewriteReport?.sentence_comparison || []).filter(
    (row) => String(row.orig_sentence || '').trim() !== String(row.new_sentence || '').trim()
  ).length;

  return {
    outcome: summary.outcome || '',
    engine_mode: summary.rewrite_engine_mode || '',
    gate: summary.authenticity_mitigation?.selected_gate || summary.authenticity_mitigation?.best_attempt?.gate || null,
    ai_mitigation_selected: Boolean(summary.authenticity_mitigation?.selected || summary.ai_mitigation_search?.selected),
    original_ai_authorship: detectScores.original_ai_authorship,
    rewritten_ai_authorship: detectScores.rewritten_ai_authorship,
    original_human_contribution: detectScores.original_human_contribution,
    rewritten_human_contribution: detectScores.rewritten_human_contribution,
    original_ai_transformation: detectScores.original_ai_transformation,
    rewritten_ai_transformation: detectScores.rewritten_ai_transformation,
    original_grounding_quality_risk: detectScores.original_grounding_quality_risk,
    rewritten_grounding_quality_risk: detectScores.rewritten_grounding_quality_risk,
    human_shift_score: detectScores.human_shift_score ?? summary.authenticity_mitigation?.selected_human_shift_score ?? summary.ai_mitigation_search?.selected_human_shift_score,
    human_shift_components: detectScores.human_shift_components ?? summary.authenticity_mitigation?.selected_human_shift_components ?? summary.ai_mitigation_search?.selected_human_shift_components,
    original_risk: originalBadge.ai_likelihood_score ?? detectScores.original_ai ?? summary.original_risk,
    rewrite_risk: rewrittenBadge.ai_likelihood_score ?? detectScores.rewritten_ai ?? summary.final_risk,
    original_findings: originalFindings,
    rewritten_findings: rewrittenFindings,
    changed_sentences: changedSentences,
  };
}

function buildRewriteContributionOverride(rewriteResultSummary) {
  if (!rewriteResultSummary) return null;
  const human = clampPercent(rewriteResultSummary.rewritten_human_contribution);
  const ai = clampPercent(rewriteResultSummary.rewritten_ai_transformation);
  if (human == null && ai == null) return null;
  return {
    humanContribution: human ?? 100 - ai,
    aiTransformation: ai ?? 100 - human,
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

function findingEvidenceSummary(issue) {
  if (issue.evidence?.summary) return issue.evidence.summary;
  if (typeof issue.evidence === 'string') return issue.evidence;
  return '';
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

function buildSubmittedContentModel(report) {
  const results = report?.results_json || {};
  const intel = results.scan_intelligence || {};
  const rawSegments = intel.document?.segments || results.highlight_segments || [];
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
  });

  const signalMap = new Map();
  segments.forEach((segment) => {
    segment.signals.forEach((signal) => {
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
    highlightedCount: segments.filter((segment) => segment.signals.length > 0).length,
  };
}

function formatRewriteStatus(status) {
  if (status === 'pending') return 'Queued';
  if (status === 'processing') return 'Rewriting AI sections';
  if (status === 'retrying') return 'Retrying rewrite';
  if (status === 'completed') return 'Rewrite complete';
  if (status === 'canceled') return 'Rewrite canceled';
  if (status === 'failed') return 'Rewrite failed';
  return 'Rewriting AI sections';
}

function isRewriteActive(status) {
  return ['pending', 'processing', 'retrying'].includes(status);
}

function normalizeRewriteProgressMessage(message, status) {
  if (!message) return formatRewriteStatus(status);
  const normalized = String(message).trim().toLowerCase();
  if (
    normalized.includes('rewriting your document') ||
    normalized.includes('this may take 1-3 minutes')
  ) {
    return 'Rewriting AI sections';
  }
  return message;
}

function normalizeRewriteJob(job) {
  if (!job) return job;
  return {
    ...job,
    progress_message: normalizeRewriteProgressMessage(job.progress_message, job.status),
  };
}

const REWRITE_PROGRESS_MESSAGES = [
  'Rewriting flagged passages while preserving the original meaning.',
  'Checking revised text against the source draft.',
  'Improving clarity, specificity, and academic tone.',
  'Preparing highlighted rewrite results for this report.',
];

function formatElapsed(seconds) {
  const total = Math.max(0, Number(seconds) || 0);
  if (total < 60) return `${total} sec`;
  const minutes = Math.floor(total / 60);
  const remainingSeconds = total % 60;
  return `${minutes} min ${String(remainingSeconds).padStart(2, '0')} sec`;
}

function getRewriteProgressDetail({ status, progress, elapsedSeconds, sseUnavailable }) {
  if (status === 'pending') return 'Waiting for the rewrite worker to start.';
  if (status === 'retrying') return 'The rewrite worker is retrying this step automatically.';
  if (sseUnavailable) return 'Still checking the rewrite status every few seconds.';
  if (progress >= 35 && progress <= 45 && elapsedSeconds >= 30) {
    return 'This is usually the longest step. Longer reports can stay here while DraftProof rewrites and verifies flagged sections.';
  }
  const index = Math.floor(Math.max(0, elapsedSeconds) / 8) % REWRITE_PROGRESS_MESSAGES.length;
  return REWRITE_PROGRESS_MESSAGES[index];
}

const REVIEW_ONLY_REWRITE_TITLE = 'No rewriteable AI sections';
const REVIEW_ONLY_REWRITE_MESSAGE = 'This report only has review-only signals. There is nothing DraftProof can rewrite automatically, so no tokens were deducted.';
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

export default function Report() {
  const { id } = useParams();
  const { refreshBalance } = useAuth();
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedIssue, setExpandedIssue] = useState(null);
  const [rewriteJob, setRewriteJob] = useState(null);
  const [rewriteLoading, setRewriteLoading] = useState(false);
  const [rewriteCanceling, setRewriteCanceling] = useState(false);
  const [showCancelRewriteDialog, setShowCancelRewriteDialog] = useState(false);
  const [rewriteError, setRewriteError] = useState(null);
  const [rewriteStartedHere, setRewriteStartedHere] = useState(false);
  const [rewriteSseUnavailable, setRewriteSseUnavailable] = useState(false);
  const [rewriteNotice, setRewriteNotice] = useState(null);
  const [rewriteResultSummary, setRewriteResultSummary] = useState(null);
  const [rewriteResultReport, setRewriteResultReport] = useState(null);
  const [rewriteElapsedSeconds, setRewriteElapsedSeconds] = useState(0);
  const [selectedSegmentId, setSelectedSegmentId] = useState(null);
  const rewritePollRef = useRef(null);
  const rewriteEventSourceRef = useRef(null);
  const rewriteTimerStartRef = useRef(null);

  const showReviewOnlyRewriteNotice = useCallback((message) => {
    setRewriteJob(null);
    setRewriteError(null);
    setRewriteLoading(false);
    setRewriteStartedHere(false);
    setRewriteNotice({
      title: REVIEW_ONLY_REWRITE_TITLE,
      message: isReviewOnlyRewriteMessage(message) && String(message).includes('token')
        ? message
        : REVIEW_ONLY_REWRITE_MESSAGE,
    });
  }, []);

  const syncRewriteJob = useCallback((job) => {
    const normalizedJob = normalizeRewriteJob(job);
    setRewriteJob(normalizedJob);
    if (normalizedJob?.status && !['failed', 'canceled'].includes(normalizedJob.status)) {
      setRewriteError(null);
    }
    if (normalizedJob?.status === 'completed') {
      setReport((prev) => prev ? { ...prev, rewrite: normalizedJob } : prev);
      setRewriteStartedHere(false);
    }
  }, []);

  const pollRewriteStatus = useCallback(async (rewriteId) => {
    try {
      const { data } = await getRewriteStatus(rewriteId);
      syncRewriteJob(data);
      if (data.status === 'failed') {
        const failedMessage = data.error || 'Rewrite failed';
        if (isReviewOnlyRewriteMessage(failedMessage)) {
          showReviewOnlyRewriteNotice(failedMessage);
        } else {
          setRewriteError(failedMessage);
        }
      }
      if (data.status === 'canceled') {
        setRewriteError(null);
      }
      return data;
    } catch (err) {
      setRewriteError(err.response?.data?.detail || 'Failed to check rewrite status');
      return null;
    }
  }, [showReviewOnlyRewriteNotice, syncRewriteJob]);

  const closeRewriteEventSource = useCallback(() => {
    if (rewriteEventSourceRef.current) {
      rewriteEventSourceRef.current.close();
      rewriteEventSourceRef.current = null;
    }
  }, []);

  const connectRewriteEvents = useCallback((rewriteId) => {
    closeRewriteEventSource();
    if (!window.EventSource) return false;

    const source = new EventSource(buildRewriteEventsUrl(rewriteId), { withCredentials: true });
    rewriteEventSourceRef.current = source;

    source.addEventListener('progress', (event) => {
      let data;
      try {
        data = JSON.parse(event.data);
      } catch {
        closeRewriteEventSource();
        setRewriteSseUnavailable(true);
        pollRewriteStatus(rewriteId);
        return;
      }

      syncRewriteJob(data);
      if (data.status === 'failed') {
        const failedMessage = data.error || 'Rewrite failed';
        if (isReviewOnlyRewriteMessage(failedMessage)) {
          showReviewOnlyRewriteNotice(failedMessage);
        } else {
          setRewriteError(failedMessage);
        }
        closeRewriteEventSource();
      }
      if (data.status === 'completed' || data.status === 'canceled') {
        closeRewriteEventSource();
      }
    });

    source.addEventListener('rewrite-error', () => {
      setRewriteError('Rewrite failed');
      closeRewriteEventSource();
    });

    source.addEventListener('error', () => {
      closeRewriteEventSource();
      setRewriteSseUnavailable(true);
      pollRewriteStatus(rewriteId);
    });

    return true;
  }, [closeRewriteEventSource, pollRewriteStatus, showReviewOnlyRewriteNotice, syncRewriteJob]);

  useEffect(() => {
    const ac = new AbortController();
    getReport(id, { signal: ac.signal })
      .then(({ data }) => {
        setReport(data);
        if (data.rewrite) {
          setRewriteSseUnavailable(false);
          setRewriteJob(normalizeRewriteJob(data.rewrite));
          if (data.rewrite.id && isRewriteActive(data.rewrite.status)) {
            connectRewriteEvents(data.rewrite.id);
          }
        }
      })
      .catch((err) => {
        if (err.name === 'AbortError' || err.code === 'ERR_CANCELED') return;
        setError(err.response?.data?.detail || 'Failed to load report');
      })
      .finally(() => setLoading(false));
    return () => {
      ac.abort();
      closeRewriteEventSource();
    };
  }, [id, closeRewriteEventSource, connectRewriteEvents]);

  useEffect(() => {
    if (rewritePollRef.current) {
      clearInterval(rewritePollRef.current);
      rewritePollRef.current = null;
    }

    if (!rewriteJob?.id || !isRewriteActive(rewriteJob.status)) {
      return undefined;
    }

    if (rewriteEventSourceRef.current) {
      return undefined;
    }

    if (rewriteSseUnavailable || !connectRewriteEvents(rewriteJob.id)) {
      rewritePollRef.current = setInterval(() => {
        pollRewriteStatus(rewriteJob.id);
      }, 5000);
    }

    return () => {
      if (rewritePollRef.current) {
        clearInterval(rewritePollRef.current);
        rewritePollRef.current = null;
      }
    };
  }, [rewriteJob?.id, rewriteJob?.status, rewriteSseUnavailable, pollRewriteStatus, connectRewriteEvents]);

  useEffect(() => {
    const completedRewrite = rewriteJob?.status === 'completed' ? rewriteJob : report?.rewrite;
    if (!completedRewrite?.id || completedRewrite.status !== 'completed') {
      setRewriteResultSummary(null);
      setRewriteResultReport(null);
      return undefined;
    }

    let cancelled = false;
    getRewriteReport(completedRewrite.id)
      .then(({ data }) => {
        if (cancelled) return;
        setRewriteResultReport(data);
        setRewriteResultSummary(buildRewriteResultSummary(data));
      })
      .catch(() => {
        if (!cancelled) {
          setRewriteResultReport(null);
          setRewriteResultSummary(null);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [rewriteJob, report?.rewrite]);

  const activeRewriteForTimer = rewriteJob || report?.rewrite;
  const rewriteTimerActive = rewriteLoading || isRewriteActive(activeRewriteForTimer?.status);

  useEffect(() => {
    if (!rewriteTimerActive) {
      rewriteTimerStartRef.current = null;
      setRewriteElapsedSeconds(0);
      return undefined;
    }

    if (!rewriteTimerStartRef.current) {
      rewriteTimerStartRef.current = Date.now();
      setRewriteElapsedSeconds(0);
    }

    const timer = setInterval(() => {
      setRewriteElapsedSeconds(Math.floor((Date.now() - rewriteTimerStartRef.current) / 1000));
    }, 1000);

    return () => clearInterval(timer);
  }, [rewriteTimerActive, activeRewriteForTimer?.id]);

  if (loading) return (
    <main className="dash-shell">
      <div className="container">
        <div className="report-loading">
          <div className="report-pulse" />
          <p>Analyzing your report...</p>
        </div>
      </div>
    </main>
  );

  if (error) return (
    <main className="dash-shell">
      <div className="container"><ErrorReload message={error} /></div>
    </main>
  );

  if (!report) return (
    <main className="dash-shell">
      <div className="container"><p>Report not found.</p></div>
    </main>
  );

  const tier = TIER_CONFIG[report.tier] || TIER_CONFIG.moderate;
  const badge = report.ai_risk_badge || {};
  const aiScore = report.ai_score ?? badge.ai_likelihood_score ?? null;
  const writingScore = report.writing_score ?? badge.writing_quality_score ?? null;
  const transformation = badge.transformation_classification || null;
  const transformationSignalMetadata = report.scan_intelligence?.transformation?.core_signals || [];
  const transformationSignals = buildTransformationSignals(transformation?.features, transformationSignalMetadata);
  const transformationSummary = transformation
    ? buildTransformationSummary(transformation.features, transformationSignals)
    : null;
  const rewrittenScan = getRewrittenDetectScan(rewriteResultReport) || {};
  const rewrittenBadge = rewrittenScan.ai_risk_badge || {};
  const rewrittenTransformation = rewrittenBadge.transformation_classification || null;
  const rewrittenTransformationSignalMetadata = rewrittenScan.scan_intelligence?.transformation?.core_signals || [];
  const rewrittenTransformationSignals = buildTransformationSignals(
    rewrittenTransformation?.features,
    rewrittenTransformationSignalMetadata
  );
  const rewrittenContributionOverride = buildRewriteContributionOverride(rewriteResultSummary);
  const rewrittenTransformationSummary = rewrittenTransformation
    ? buildTransformationSummary(rewrittenTransformation.features, rewrittenTransformationSignals, rewrittenContributionOverride)
    : rewrittenContributionOverride
      ? {
        humanContribution: Math.round(rewrittenContributionOverride.humanContribution),
        aiTransformation: Math.round(rewrittenContributionOverride.aiTransformation),
        adjustedAiRisk: Math.round(rewriteResultSummary?.rewrite_risk ?? 0),
        rawAdjustedAiRisk: Math.round(rewriteResultSummary?.rewrite_risk ?? 0),
        humanAnchorDiscount: 0,
        calibrationConfidence: 0,
        reportingSuppression: 0,
        summary: 'Rewritten contribution estimate from the completed rewrite scan.',
      }
    : null;
  const rewrittenAiScore = rewrittenScan.ai_score ?? rewrittenBadge.ai_likelihood_score ?? rewriteResultSummary?.rewrite_risk ?? null;
  const authorshipRating = badge.authorship_rating || deriveAuthorshipRatingFallback(
    aiScore,
    badge.tier || report.tier,
    writingScore,
    badge.ai_components,
    badge.writing_components
  ) || {};
  const authorshipRatingFullLabel = authorshipRating.label || badge.authorship_rating_label || null;
  const authorshipRatingLabel = authorshipRating.short_label || authorshipRatingFullLabel;
  const issueCounts = { critical: 0, high: 0, medium: 0, low: 0, info: 0 };
  report.issues.forEach((iss) => { if (issueCounts[iss.severity] !== undefined) issueCounts[iss.severity]++; });
  const submittedContent = buildSubmittedContentModel(report);
  const selectedSegment = (
    submittedContent.segments.find((segment) => segment.id === selectedSegmentId) ||
    submittedContent.segments.find((segment) => segment.signals.length > 0) ||
    null
  );

  const hasAIFindings = report.issues.some(i =>
    i.category === 'ai_generation' ||
    i.scanner === 'ai_generation' ||
    i.signal_category === 'authorship_risk' ||
    i.actionability === 'auto_rewrite_candidate'
  );
  const currentRewrite = rewriteJob || report.rewrite;
  const rewriteInProgress = isRewriteActive(currentRewrite?.status);
  const hasCompletedRewrite = currentRewrite?.status === 'completed';
  const hasRewriteResult = hasCompletedRewrite && Boolean(currentRewrite?.id);
  const hasRewriteSignalComparison = Boolean(
    hasRewriteResult &&
    (rewrittenTransformation || rewrittenTransformationSummary)
  );
  const canStartRewrite = hasAIFindings && !hasRewriteResult;
  const rewriteProgress = currentRewrite
    ? Math.max(0, Math.min(100, Number(currentRewrite.progress_percent) || (rewriteInProgress ? 5 : hasCompletedRewrite ? 100 : 0)))
    : 0;
  const rewriteProgressMessage = normalizeRewriteProgressMessage(
    currentRewrite?.progress_message,
    currentRewrite?.status
  );
  const rewriteProgressDetail = !rewriteError && rewriteInProgress
    ? getRewriteProgressDetail({
      status: currentRewrite?.status,
      progress: rewriteProgress,
      elapsedSeconds: rewriteElapsedSeconds,
      sseUnavailable: rewriteSseUnavailable,
    })
    : null;
  const rewriteElapsedLabel = rewriteElapsedSeconds > 0 ? formatElapsed(rewriteElapsedSeconds) : null;
  const showRewriteProgress = !hasRewriteResult && (
    rewriteStartedHere || rewriteInProgress || rewriteLoading || rewriteCanceling || rewriteError
  );

  const handleRewrite = async (event) => {
    event?.preventDefault();
    event?.stopPropagation();
    if (rewriteLoading || hasRewriteResult) return;
    setRewriteStartedHere(true);
    setRewriteLoading(true);
    setRewriteError(null);
    setRewriteSseUnavailable(false);
    setRewriteJob({
      id: null,
      scan_id: id,
      status: 'pending',
      progress_percent: 3,
      progress_message: 'Queuing rewrite',
    });
    try {
      const { data } = await createRewrite(id);
      syncRewriteJob(data);
      if (data.id) {
        if (!connectRewriteEvents(data.id)) {
          await pollRewriteStatus(data.id);
        }
      }
    } catch (err) {
      const msg = err.response?.data?.detail || 'Failed to start rewrite';
      if (err.response?.status === 402) {
        setRewriteJob(null);
        setRewriteError(msg);
      } else if (err.response?.status === 422 || isReviewOnlyRewriteMessage(msg)) {
        showReviewOnlyRewriteNotice(msg);
      } else {
        setRewriteJob((prev) => prev ? {
          ...prev,
          status: 'failed',
          progress_message: 'Rewrite failed',
        } : null);
        setRewriteError(msg);
      }
    } finally {
      setRewriteLoading(false);
    }
  };

  const handleCancelRewrite = async (event) => {
    event?.preventDefault();
    event?.stopPropagation();
    if (!currentRewrite?.id || !rewriteInProgress || rewriteCanceling) return;
    setShowCancelRewriteDialog(true);
  };

  const confirmCancelRewrite = async () => {
    if (!currentRewrite?.id || !rewriteInProgress || rewriteCanceling) {
      setShowCancelRewriteDialog(false);
      return;
    }
    setRewriteCanceling(true);
    setRewriteError(null);
    try {
      const { data } = await cancelRewrite(currentRewrite.id);
      closeRewriteEventSource();
      syncRewriteJob(data);
      setRewriteStartedHere(false);
      setShowCancelRewriteDialog(false);
      refreshBalance?.();
    } catch (err) {
      setRewriteError(err.response?.data?.detail || 'Failed to cancel rewrite');
    } finally {
      setRewriteCanceling(false);
    }
  };

  const reportSummaryBar = (
    <div className="report-summary-bar">
      <div className="report-stat report-risk-stat" style={{ background: tier.bg }}>
        <span className="report-risk-icon" style={{ color: tier.color }} aria-hidden="true">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <path d={tier.icon} />
            <circle cx="12" cy="12" r="10" />
          </svg>
        </span>
        <span className="report-risk-copy">
          <span className="report-risk-value" style={{ color: tier.color }}>{tier.label}</span>
          <span className="report-stat-label">Risk Tier</span>
        </span>
      </div>
      <div className="report-stat">
        <span className="report-stat-value">{report.issues.length}</span>
        <span className="report-stat-label">Total Findings</span>
      </div>
      {authorshipRatingLabel && (
        <div className="report-stat">
          <span className="report-stat-value" style={{ color: tier.color }} title={authorshipRatingFullLabel || authorshipRatingLabel}>
            {authorshipRatingLabel}
          </span>
          <span className="report-stat-label">Authorship Rating</span>
        </div>
      )}
      {aiScore != null && (
        <div className="report-stat">
          <span className="report-stat-value" style={{ color: tier.color }}>{formatMetricPercent(aiScore, 2)}</span>
          <span className="report-stat-label">AI Score</span>
        </div>
      )}
      {writingScore != null && (
        <div className="report-stat">
          <span className="report-stat-value" style={{ color: '#6366f1' }}>{formatMetricPercent(writingScore, 2)}</span>
          <span className="report-stat-label">Writing Score</span>
        </div>
      )}
      {Object.entries(issueCounts).filter(([, v]) => v > 0).map(([sev, count]) => {
        const sc = SEVERITY_CONFIG[sev];
        return (
          <div key={sev} className="report-stat">
            <span className="report-stat-value" style={{ color: sc.color }}>{count}</span>
            <span className="report-stat-label">{sc.label}</span>
          </div>
        );
      })}
    </div>
  );

  const renderTransformationDetails = (variant, pattern, summary, signals, variantAiScore) => {
    const comparisonSignals = sortTransformationSignalsForComparison(signals);

    return (
      <div className={`transformation-detail ${variant === 'rewritten' ? 'is-rewritten' : 'is-original'}`}>
        <div className="transformation-detail-head">
          <div>
            <span>{variant === 'rewritten' ? 'Rewritten Scan' : 'Original Scan'}</span>
            <strong>{pattern?.label || (variant === 'rewritten' ? 'Rewritten contribution pattern' : 'Pattern analysis')}</strong>
          </div>
          <em>{formatMetricPercent(variantAiScore, 1)}</em>
        </div>
        {summary && (
          <div className="transformation-ratio-summary">
            <div className="transformation-ratio-copy">
              <span>Estimated Contribution</span>
              <p>{summary.summary}</p>
              <div className="transformation-adjustment-row">
                <strong>Calibrated AI risk {summary.adjustedAiRisk}%</strong>
                <strong>Human anchor discount {summary.humanAnchorDiscount}%</strong>
                <strong>Calibration confidence {summary.calibrationConfidence}%</strong>
                <strong>Reporting suppression {summary.reportingSuppression}%</strong>
              </div>
            </div>
            <div className="transformation-ratio-bars" aria-label={`${variant === 'rewritten' ? 'Rewritten' : 'Original'} human contribution versus AI transformation estimate`}>
              <div className="transformation-ratio-row">
                <span>Human Contribution</span>
                <strong>{summary.humanContribution}%</strong>
                <div className="transformation-ratio-track">
                  <div className="transformation-ratio-fill is-human" style={{ width: `${summary.humanContribution}%` }} />
                </div>
              </div>
              <div className="transformation-ratio-row">
                <span>AI Transformation</span>
                <strong>{summary.aiTransformation}%</strong>
                <div className="transformation-ratio-track">
                  <div className="transformation-ratio-fill is-ai" style={{ width: `${summary.aiTransformation}%` }} />
                </div>
              </div>
            </div>
          </div>
        )}
        {comparisonSignals.length > 0 && (
          <>
            <div className="transformation-chart-head">
              <span>Core Signals</span>
            </div>
            <div className="transformation-bars">
              {comparisonSignals.map((signal) => (
                <div
                  key={`${variant}-${signal.key}`}
                  className="transformation-bar-row"
                  data-tooltip={signal.description}
                  tabIndex={0}
                  aria-label={`${variant === 'rewritten' ? 'Rewritten' : 'Original'} ${signal.label}: ${signal.value.toFixed(0)}%. ${signal.description}`}
                  title={signal.description}
                >
                  <div className="transformation-bar-label">
                    <span>{signal.label}</span>
                    <strong>{signal.value.toFixed(0)}%</strong>
                  </div>
                  <div className="transformation-bar-track" aria-hidden="true">
                    <div
                      className={`transformation-bar-fill transformation-bar-${signal.key}`}
                      style={{ width: `${signal.value}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    );
  };

  const transformationScorecard = transformation && transformationSignals.length > 0 ? (
    <section className="transformation-scorecard" aria-label="Transformation pattern scorecard">
      <div className="transformation-header">
        <div className="transformation-summary">
          <div className="transformation-icon" aria-hidden="true">
            <svg width="30" height="30" viewBox="0 0 30 30" fill="none">
              <path d="M6 8.5h12.5M6 15h18M6 21.5h10" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round"/>
              <path d="M21 7l3 3-3 3M18 18l-3 3 3 3" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
          <div>
            <span className="transformation-kicker">Transformation Pattern</span>
            <h2>{hasRewriteSignalComparison ? 'Original vs rewritten pattern' : transformation.label || 'Pattern analysis'}</h2>
            <div className="transformation-meta-row">
              {transformation.confidence && (
                <span className="transformation-pill">{transformation.confidence} confidence</span>
              )}
              {hasRewriteSignalComparison && (
                <span className="transformation-pill">rewrite comparison</span>
              )}
              <span className="transformation-pill">not a verdict</span>
            </div>
          </div>
        </div>
        <div className="transformation-ai-score">
          <span>AI Score</span>
          <strong>
            {hasRewriteSignalComparison
              ? `${formatMetricPercent(aiScore, 1)} -> ${formatMetricPercent(rewrittenAiScore, 1)}`
              : formatMetricPercent(aiScore, 1)}
          </strong>
        </div>
      </div>
      <div className="transformation-chart">
        {hasRewriteSignalComparison ? (
          <div className="transformation-comparison-grid">
            {renderTransformationDetails('original', transformation, transformationSummary, transformationSignals, aiScore)}
            {renderTransformationDetails('rewritten', rewrittenTransformation, rewrittenTransformationSummary, rewrittenTransformationSignals, rewrittenAiScore)}
          </div>
        ) : (
          renderTransformationDetails('original', transformation, transformationSummary, transformationSignals, aiScore)
        )}
        {Array.isArray(transformation.evidence) && transformation.evidence.length > 0 && (
          <div className="transformation-evidence">
            {transformation.evidence.slice(0, 3).map((item) => (
              <span key={item}>{item}</span>
            ))}
          </div>
        )}
      </div>
    </section>
  ) : null;

  const rewriteOutcome = rewriteResultSummary?.outcome || '';
  const rewriteOutcomeText = rewriteOutcome
    ? rewriteOutcome.replaceAll('_', ' ')
    : hasCompletedRewrite
      ? 'completed'
      : '';
  const rewriteBandTitle = rewriteOutcome === 'ai_mitigated'
    ? 'AI-Mitigation accepted'
    : rewriteOutcome === 'suggestion_only'
      ? 'Original preserved'
      : rewriteOutcomeText || 'Rewrite complete';
  const rewriteBandDetail = rewriteOutcome === 'ai_mitigated'
    ? 'A candidate passed the mitigation gate and was kept.'
    : rewriteOutcome === 'suggestion_only'
      ? 'No candidate passed the rewrite gate; review the guidance before trying again.'
      : rewriteResultSummary?.ai_mitigation_selected
        ? 'A mitigation candidate was selected after scanning.'
        : 'Rewrite finished and the result is ready to review.';
  const rewriteCompletionBand = hasRewriteResult ? (
    <div className={`report-rewrite-summary-bar${rewriteOutcome === 'suggestion_only' ? ' is-preserved' : ''}${rewriteOutcome === 'ai_mitigated' ? ' is-mitigated' : ''}`}>
      <div className="rewrite-summary-icon" aria-hidden="true">
        <span>
          <svg width="42" height="42" viewBox="0 0 42 42" fill="none">
            <circle cx="21" cy="21" r="15" fill="currentColor"/>
            {rewriteOutcome === 'suggestion_only' ? (
              <path d="M15 15l12 12M27 15L15 27" stroke="#fff" strokeWidth="3" strokeLinecap="round"/>
            ) : (
              <path d="M14 21.5l4.5 4.5L28.5 16" stroke="#fff" strokeWidth="3.2" strokeLinecap="round" strokeLinejoin="round"/>
            )}
          </svg>
        </span>
      </div>
      <div className="rewrite-summary-main">
        <span className="rewrite-summary-kicker">Rewrite completion</span>
        <strong>{rewriteBandTitle}</strong>
        <em>{rewriteBandDetail}</em>
      </div>
      <div className="rewrite-summary-stat">
        <span>{formatMetricPercent(rewriteResultSummary?.original_ai_authorship ?? rewriteResultSummary?.original_risk, 1)}</span>
        <small>Rewrite AI authorship before</small>
      </div>
      <div className="rewrite-summary-stat">
        <span>{formatMetricPercent(rewriteResultSummary?.rewritten_ai_authorship ?? rewriteResultSummary?.rewrite_risk, 1)}</span>
        <small>Rewrite AI authorship after</small>
      </div>
      <div className="rewrite-summary-stat">
        <span>{formatPlainScore(rewriteResultSummary?.human_shift_score, 1)}</span>
        <small>Human Shift Score</small>
      </div>
      <div className="rewrite-summary-stat">
        <span>{formatSignedDelta(rewriteResultSummary?.original_human_contribution, rewriteResultSummary?.rewritten_human_contribution)}</span>
        <small>Rewrite human contribution</small>
      </div>
      <div className="rewrite-summary-stat">
        <span>{formatSignedDelta(rewriteResultSummary?.original_ai_transformation, rewriteResultSummary?.rewritten_ai_transformation)}</span>
        <small>Rewrite AI transformation</small>
      </div>
      <div className="rewrite-summary-stat">
        <span>{formatSignedDelta(rewriteResultSummary?.original_grounding_quality_risk, rewriteResultSummary?.rewritten_grounding_quality_risk)}</span>
        <small>Rewrite grounding risk</small>
      </div>
      <Link
        to={`/rewrite/${currentRewrite.id}`}
        className="rewrite-results-link"
      >
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
          <path d="M5 2.5h5.2L13 5.3v10.2H5V2.5z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round"/>
          <path d="M10 2.5v3h3M6.8 8.3h4M6.8 11h4" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/>
        </svg>
        View Rewrite Result
      </Link>
    </div>
  ) : null;

  return (
    <main className="dash-shell">
      <RewriteNoticeDialog
        open={Boolean(rewriteNotice)}
        title={rewriteNotice?.title}
        message={rewriteNotice?.message}
        onClose={() => setRewriteNotice(null)}
      />
      <ConfirmDialog
        open={showCancelRewriteDialog}
        title="Cancel this rewrite?"
        message="DraftProof will stop tracking this rewrite, release the reserved tokens, and ignore any late worker result for this job."
        confirmLabel={rewriteCanceling ? 'Canceling...' : 'Cancel rewrite'}
        onConfirm={confirmCancelRewrite}
        onCancel={() => {
          if (!rewriteCanceling) setShowCancelRewriteDialog(false);
        }}
      />
      <div className="container">
        {/* Back link */}
        <Link to="/reports" className="report-back">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M10 3L5 8l5 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          Back to Reports
        </Link>

        {/* Report header */}
        <div className="report-hero">
          <div className="report-hero-title-row">
            <div className="report-doc-icon" aria-hidden="true">
              <svg width="42" height="42" viewBox="0 0 42 42" fill="none">
                <rect x="8" y="9" width="26" height="24" rx="5" stroke="currentColor" strokeWidth="3"/>
                <path d="M13 25l6-6 5 5 6-8" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </div>
            <div className="report-hero-info">
              <div className="report-eyebrow">Analysis Report</div>
              <h1>{report.document_name}</h1>
              {report.created_at && (
                <p className="report-meta">
                  <svg width="17" height="17" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                    <path d="M4.5 1.8v2M11.5 1.8v2M2.5 6h11M3.5 3.5h9A1.5 1.5 0 0114 5v7.5A1.5 1.5 0 0112.5 14h-9A1.5 1.5 0 012 12.5V5a1.5 1.5 0 011.5-1.5z" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"/>
                  </svg>
                  {formatDate(report.created_at)}
                </p>
              )}
            </div>
          </div>
          {(report.report_pdf_url || canStartRewrite || rewriteLoading || rewriteInProgress) && (
            <div className="report-hero-actions">
              {report.report_pdf_url && (
                <a href={report.report_pdf_url} target="_blank" rel="noopener noreferrer" className="download-pdf-btn">
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                    <path d="M3 10v2.5A1.5 1.5 0 004.5 14h7a1.5 1.5 0 001.5-1.5V10M8 2v8M5 7l3 3 3-3" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  Download PDF
                </a>
              )}
              {(canStartRewrite || rewriteLoading || rewriteInProgress) && (
                <button
                  type="button"
                  className="rewrite-btn"
                  onClick={handleRewrite}
                  disabled={rewriteLoading || rewriteCanceling}
                >
                  {rewriteLoading ? 'Starting rewrite...' : rewriteInProgress ? 'Resume Rewrite' : 'Rewrite AI Sections'}
                </button>
              )}
              {rewriteInProgress && currentRewrite?.id && (
                <button
                  type="button"
                  className="rewrite-btn rewrite-cancel-btn"
                  onClick={handleCancelRewrite}
                  disabled={rewriteCanceling}
                >
                  {rewriteCanceling ? 'Canceling...' : 'Cancel Rewrite'}
                </button>
              )}
            </div>
          )}
        </div>
        {showRewriteProgress && (
          <div className={`report-rewrite-progress${rewriteError ? ' has-error' : ''}${hasCompletedRewrite ? ' is-complete' : ''}`}>
            <div className="scan-progress" role="status" aria-live="polite">
              <div className="scan-progress-meta">
                <span>
                  {rewriteError || rewriteProgressMessage || 'Rewriting AI sections'}
                  {rewriteInProgress && <em> Keep this report open; results will appear when ready.</em>}
                </span>
                <span>{hasCompletedRewrite ? 'Done' : `${rewriteProgress}%`}</span>
              </div>
              <div
                className="scan-progress-track"
                role="progressbar"
                aria-valuemin="0"
                aria-valuemax="100"
                aria-valuenow={rewriteProgress}
              >
                <div
                  className="scan-progress-fill"
                  style={{ width: `${hasCompletedRewrite ? 100 : rewriteProgress}%` }}
                />
              </div>
              {rewriteProgressDetail && (
                <div className="rewrite-progress-detail">
                  <span className="rewrite-progress-pulse" aria-hidden="true" />
                  <span>{rewriteProgressDetail}</span>
                </div>
              )}
              {rewriteInProgress && (
                <div className="rewrite-progress-footnote">
                  {rewriteElapsedLabel && <span>Elapsed: {rewriteElapsedLabel}</span>}
                  <span>Keep this page open; results will appear automatically.</span>
                </div>
              )}
            </div>
          </div>
        )}

        {rewriteCompletionBand}

        {transformationScorecard ? (
          <section className="report-overview-card" aria-label="Report overview">
            {reportSummaryBar}
            {transformationScorecard}
          </section>
        ) : (
          reportSummaryBar
        )}

        {submittedContent.segments.length > 0 && (
          <section className="submitted-content-review" aria-label="Submitted content with scan signals">
            <div className="submitted-content-head">
              <div>
                <span className="submitted-content-kicker">Submitted Content</span>
                <h2>Signal highlights</h2>
              </div>
              <div className="submitted-content-count">
                <strong>{submittedContent.highlightedCount}</strong>
                <span>highlighted sections</span>
              </div>
            </div>
            {submittedContent.legend.length > 0 && (
              <div className="submitted-signal-legend" aria-label="Signal color legend">
                {submittedContent.legend.slice(0, 6).map((signal) => (
                  <span key={signal.key} className="submitted-signal-chip" style={{ '--signal-color': signal.color }}>
                    <i aria-hidden="true" />
                    {signal.label}
                    <strong>{signal.count}</strong>
                  </span>
                ))}
              </div>
            )}
            <div className="submitted-content-grid">
              <div className="submitted-document" aria-label="Submitted document text">
                {submittedContent.paragraphs.map((paragraph) => (
                  <p key={paragraph.id}>
                    {paragraph.segments.map((segment) => {
                      const signal = segment.primarySignal;
                      const isSelected = selectedSegment?.id === segment.id;
                      if (!signal) {
                        return <span key={segment.id}>{segment.text} </span>;
                      }
                      return (
                        <button
                          key={segment.id}
                          type="button"
                          className={`submitted-highlight${isSelected ? ' is-selected' : ''}`}
                          style={{ '--signal-color': signal.color }}
                          title={signal.description}
                          onMouseEnter={() => setSelectedSegmentId(segment.id)}
                          onFocus={() => setSelectedSegmentId(segment.id)}
                          onClick={() => {
                            setSelectedSegmentId(segment.id);
                            const linkedIndex = report.issues.findIndex((issue) => (
                              segment.signals.some((s) => String(s.finding_id) === String(issue.id))
                            ));
                            if (linkedIndex >= 0) setExpandedIssue(linkedIndex);
                          }}
                        >
                          {segment.text}
                        </button>
                      );
                    })}
                  </p>
                ))}
              </div>
              <aside className="submitted-signal-panel" aria-label="Selected signal detail">
                {selectedSegment?.primarySignal ? (
                  <>
                    <span className="submitted-panel-kicker">{selectedSegment.sentence_id}</span>
                    <h3>{selectedSegment.primarySignal.label}</h3>
                    <p>{selectedSegment.primarySignal.description}</p>
                    <div className="submitted-panel-meta">
                      {selectedSegment.primarySignal.score != null && (
                        <span>{Math.round(selectedSegment.primarySignal.score)}% signal strength</span>
                      )}
                      {selectedSegment.primarySignal.tier && (
                        <span>{selectedSegment.primarySignal.tier} priority</span>
                      )}
                      {selectedSegment.primarySignal.actionability && (
                        <span>{selectedSegment.primarySignal.actionability.replaceAll('_', ' ')}</span>
                      )}
                    </div>
                    {selectedSegment.signals.length > 1 && (
                      <div className="submitted-panel-stack">
                        <span>Also detected</span>
                        {selectedSegment.signals.slice(1, 4).map((signal) => (
                          <em key={`${selectedSegment.id}-${signal.key}-${signal.finding_id}`}>{signal.label}</em>
                        ))}
                      </div>
                    )}
                    {selectedSegment.primarySignal.recommendation && (
                      <div className="submitted-panel-note">
                        <span>Recommendation</span>
                        <p>{selectedSegment.primarySignal.recommendation}</p>
                      </div>
                    )}
                  </>
                ) : (
                  <>
                    <span className="submitted-panel-kicker">No highlighted signal</span>
                    <h3>Content map ready</h3>
                    <p>The submitted text is available for review. New scans include richer signal highlights for each affected sentence.</p>
                  </>
                )}
              </aside>
            </div>
          </section>
        )}

        {/* Findings list */}
        {report.issues.length > 0 ? (
          <div className="report-findings">
            <h2>Findings</h2>
            <div className="findings-list">
              {report.issues.map((issue, i) => {
                const sc = SEVERITY_CONFIG[issue.severity] || SEVERITY_CONFIG.info;
                const isExpanded = expandedIssue === i;
                const hasScores = issue.score != null || issue.top10_ratio != null;
                return (
                  <div
                    key={issue.id || i}
                    className={`finding-card${isExpanded ? ' expanded' : ''}`}
                    onClick={() => setExpandedIssue(isExpanded ? null : i)}
                    style={{ borderLeftColor: sc.color }}
                  >
                    <div className="finding-header">
                      <span className="finding-severity" style={{ color: sc.color, background: sc.bg }}>
                        {sc.label}
                      </span>
                      <span className="finding-number">#{i + 1}</span>
                      {issue.title && <span className="finding-title-tag">{issue.title.replace(/_/g, ' ')}</span>}
                      {issue.location && <span className="finding-location">{issue.location}</span>}
                      <svg
                        className="finding-chevron"
                        width="14" height="14" viewBox="0 0 14 14" fill="none"
                        style={{ transform: isExpanded ? 'rotate(180deg)' : 'rotate(0)', transition: 'transform .2s' }}
                      >
                        <path d="M3 5l4 4 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                      </svg>
                    </div>
                    <p className="finding-desc">{findingDescription(issue)}</p>
                    {isExpanded && (
                      <div className="finding-detail" onClick={(e) => e.stopPropagation()}>
                        {issue.scanner && (
                          <div className="finding-meta-row">
                            <span className="finding-meta-label">Scanner</span>
                            <span className="finding-meta-value">{issue.scanner}</span>
                          </div>
                        )}
                        {issue.category && (
                          <div className="finding-meta-row">
                            <span className="finding-meta-label">Category</span>
                            <span className="finding-meta-value">{issue.category}</span>
                          </div>
                        )}
                        {issue.signal_category && (
                          <div className="finding-meta-row">
                            <span className="finding-meta-label">Signal</span>
                            <span className="finding-meta-value">{issue.signal_category.replace(/_/g, ' ')}</span>
                          </div>
                        )}
                        {issue.actionability && (
                          <div className="finding-meta-row">
                            <span className="finding-meta-label">Action</span>
                            <span className={`finding-action-badge finding-action-${issue.actionability}`}>
                              {issue.actionability.replace(/_/g, ' ')}
                            </span>
                          </div>
                        )}
                        {hasScores && (
                          <div className="finding-scores">
                            {issue.score != null && (
                              <div className="finding-score-item">
                                <span className="finding-score-label">Risk Score</span>
                                <div className="finding-score-bar">
                                  <div className="finding-score-fill" style={{ width: `${Math.min(issue.score * 100, 100)}%`, background: sc.color }} />
                                </div>
                                <span className="finding-score-value">{(issue.score * 100).toFixed(0)}%</span>
                              </div>
                            )}
                            {issue.top10_ratio != null && (
                              <div className="finding-score-item">
                                <span className="finding-score-label">Common Predictability</span>
                                <div className="finding-score-bar">
                                  <div className="finding-score-fill" style={{ width: `${Math.min(issue.top10_ratio * 100, 100)}%`, background: '#8b5cf6' }} />
                                </div>
                                <span className="finding-score-value">{(issue.top10_ratio * 100).toFixed(0)}%</span>
                              </div>
                            )}
                          </div>
                        )}
                        {issue.evidence && (
                          <div className="finding-evidence">
                            <span className="finding-meta-label">Evidence</span>
                            {findingEvidenceSummary(issue) ? (
                              <p>{findingEvidenceSummary(issue)}</p>
                            ) : (
                              null
                            )}
                            {typeof issue.evidence === 'object' && issue.evidence.sentence && (
                              <blockquote className="finding-quote">&ldquo;{issue.evidence.sentence}&rdquo;</blockquote>
                            )}
                          </div>
                        )}
                        {issue.sentence_text && !(issue.evidence && typeof issue.evidence === 'object' && issue.evidence.sentence) && (
                          <div className="finding-evidence">
                            <span className="finding-meta-label">Sentence</span>
                            <blockquote className="finding-quote">&ldquo;{issue.sentence_text}&rdquo;</blockquote>
                          </div>
                        )}
                        {issue.recommendation && (
                          <div className="finding-recommendation">
                            <span className="finding-meta-label">Recommendation</span>
                            <p>{issue.recommendation}</p>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        ) : (
          <div className="report-clean">
            <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
              <circle cx="24" cy="24" r="20" stroke="#22c55e" strokeWidth="2"/>
              <path d="M16 24l5 5 11-11" stroke="#22c55e" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            <h3>No issues found</h3>
            <p>Your document looks clean. No findings were detected.</p>
          </div>
        )}

      </div>
    </main>
  );
}

function RewriteNoticeDialog({ open, title, message, onClose }) {
  const closeButtonRef = useRef(null);

  useEffect(() => {
    if (open) {
      closeButtonRef.current?.focus();
    }
  }, [open]);

  useEffect(() => {
    if (!open) return undefined;
    const onKeyDown = (event) => {
      if (event.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal-card"
        role="dialog"
        aria-modal="true"
        aria-labelledby="rewrite-notice-title"
        onClick={(event) => event.stopPropagation()}
      >
        <h3 id="rewrite-notice-title" className="modal-title">{title}</h3>
        <p className="modal-message">{message}</p>
        <div className="modal-actions">
          <button
            ref={closeButtonRef}
            type="button"
            className="btn btn-primary btn-small"
            onClick={onClose}
          >
            OK
          </button>
        </div>
      </div>
    </div>
  );
}

function buildRewriteEventsUrl(rewriteId) {
  return buildApiEventUrl(`/rewrites/${rewriteId}/events`);
}
