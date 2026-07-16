import { buildApiEventUrl } from '../../api/draftproofApi';
import {
  buildRewriteContributionOverride,
  getScanContributionSummary,
} from './reportContributionHelpers';

import {
  TRANSFORMATION_SIGNAL_LABELS,
  TRANSFORMATION_SIGNAL_DESCRIPTIONS,
  buildTransformationSignals,
  transformationSignalFeatureMap,
  sortTransformationSignalsForComparison,
  buildPairedTransformationSignals,
  groupTransformationSignals,
  getTransformationSignalImprovement,
  transformationSignalDirection,
  transformationSignalSeverity,
  buildTransformationSummary,
} from './reportTransformation';
import {
  deriveAuthorshipRatingFallback,
  deriveCalibratedAuthorshipRating,
  formatAuthorshipSealDetail,
  formatAuthorshipSealDetailWithReference,
  getAiSignalStamp,
  getAuthorshipTone,
} from './reportAuthorship';
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
  // DeBERTa learned-classifier heatmap (drives the Signal-highlights paragraph colors).
  // Backend carries the per-band color on each signal; this is a fallback only.
  ai_signal_deberta: '#dc2626',
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
    // Gap-resolution verdict reframe (additive, produced by production.py when
    // DRAFTPROOF_REWRITE_VERDICT_REFRAME is on). Null on legacy rewrites → the web
    // verdict falls back to the score-delta path (annotate-never-suppress).
    verdict_label: summary.verdict_label ?? rewriteReport?.verdict_label ?? null,
    gap_resolution: summary.gap_resolution ?? rewriteReport?.gap_resolution ?? null,
    ai_likelihood_note: summary.ai_likelihood_note ?? rewriteReport?.ai_likelihood_note ?? null,
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
      current.finding_ids.push(signal.finding_id);
      current.sentence_ids.push(segment.sentence_id);
      current.descriptions.push(signal.description);
      current.recommendations.push(signal.recommendation);
      // The dominant (highest-score) sentence of this key drives the tier/label/summary so the
      // merged signal is coherent — a paragraph with a 100% sentence must read "high", not "low".
      // The FIRST encountered signal seeded {tier,label,...} above; adopt the stronger one's.
      if ((signal.score || 0) > (current.score || 0)) {
        current.score = signal.score;
        current.tier = signal.tier;
        current.label = signal.label;
        current.color = signal.color;
        current.description = signal.description;
        current.reader_summary = signal.reader_summary;
        current.recommendation = signal.recommendation;
      }
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

function buildSubmittedContentModel(report) {
  const results = report?.results_json || {};
  const intel = results.scan_intelligence || {};
  // Per-sentence suggestions from the LLM explainer, keyed by sentence_id. When present they
  // replace the single paragraph-level recommendation with a targeted fix per flagged sentence.
  const suggestionBySentence = new Map();
  ((results.paragraph_explanations || {}).paragraphs || []).forEach((p) => {
    (p?.sentence_suggestions || []).forEach((s) => {
      if (s && s.sentence_id && s.suggestion) suggestionBySentence.set(String(s.sentence_id), s.suggestion);
    });
  });
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
        // DeBERTa-only contract: the Signal-highlights section shows ONLY the learned-
        // classifier signal on each segment (the backend emits ai_signal_deberta as the sole
        // signal). The legacy perplexity/genericity issues must NOT be injected here — doing so
        // made the legend mix "Learned-classifier AI signal" + "Predictability" + "Genericity"
        // chips (three methodologies) and contradict the Second-opinion tile. Those legacy
        // findings remain available elsewhere (issue-card guidance), not as highlight signals.
        const directSignals = Array.isArray(segment.signals) ? segment.signals : [];
        const signals = directSignals
          .map((signal) => normalizeSignal(signal, issueById.get(String(signal.finding_id)) || {}))
          .filter((signal) => signal.key === 'ai_signal_deberta');
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
    // Legacy reports predate segment-level DeBERTa signals. Leave these plain rather than
    // resurrecting perplexity issues as highlight colors (same DeBERTa-only contract).
    segments = Object.entries(results.sentence_map)
      .map(([sentenceId, sentence], index) => ({
        id: sentenceId,
        sentence_id: sentenceId,
        paragraph_id: sentence.paragraph_id || 'p001',
        start_char: sentence.start_char ?? index,
        text: sentence.text || '',
        signals: [],
        primarySignal: null,
      }))
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
    // DeBERTa-native per-sentence evidence: the flagged sentences in this paragraph with their
    // band/score/guidance. This is the SOLE source of the issue-card body — no perplexity text.
    const flaggedSentences = paragraph.segments
      .flatMap((segment) => (segment.signals || [])
        .filter((signal) => signal.key === 'ai_signal_deberta')
        .map((signal) => ({
          sentence_id: segment.sentence_id,
          text: segment.text || '',
          score: signal.score ?? 0,
          tier: signal.tier || '',
          recommendation: signal.recommendation || '',
          reader_summary: signal.reader_summary || '',
          suggestion: suggestionBySentence.get(String(segment.sentence_id)) || '',
        })))
      .sort((a, b) => (b.score || 0) - (a.score || 0));
    paragraph.text = text;
    paragraph.signals = signals;
    paragraph.primarySignal = signals[0] || null;
    paragraph.sentence_id = summarizeSentenceIds(paragraph.segments);
    paragraph.sentence_ids = uniqueCompact(paragraph.segments.map((segment) => segment.sentence_id));
    paragraph.signalCount = paragraph.segments.reduce((count, segment) => count + segment.signals.length, 0);
    paragraph.flaggedSentences = flaggedSentences;
    // True when at least one flagged sentence has a tailored LLM suggestion. The issue card uses
    // this to choose per-sentence suggestions (preferred) vs the single paragraph recommendation
    // below (fail-open fallback when the explainer produced none).
    paragraph.hasSentenceSuggestions = flaggedSentences.some((s) => s.suggestion);
    // Paragraph-dominant guidance from the strongest flagged sentence (DeBERTa-native). The
    // band-level text is templated, so we anchor it to the actual sentence the classifier flagged
    // — naming the words to change makes the advice concrete instead of generic boilerplate.
    const top = flaggedSentences[0];
    const topSnippet = top?.text ? `"${top.text.split(/\s+/).slice(0, 14).join(' ')}${top.text.split(/\s+/).length > 14 ? '…' : ''}"` : '';
    const flaggedCount = flaggedSentences.length;
    paragraph.readerSummary = top
      ? `${top.reader_summary} ${flaggedCount > 1 ? `${flaggedCount} sentences in this paragraph read this way; the strongest is ${topSnippet}.` : `The strongest is ${topSnippet}.`}`.trim()
      : '';
    paragraph.recommendation = top ? `${top.recommendation}${topSnippet ? ` Start with ${topSnippet}.` : ''}`.trim() : '';
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
    // Which detector produced these per-sentence ai_signal_deberta scores on this report:
    // "deep_scan" (V7 Modal detector, same one the panel headlines) or "fakespot" (fail-open
    // default). Backend: poc/report/report.py's document.signal_highlight_source. Lets the
    // legend label the map's actual source instead of always implying the fakespot second
    // opinion when deep scan is really driving it.
    signalSource: intel.document?.signal_highlight_source || 'fakespot',
    // Per-sentence issue-tag underline layer (poc/report/sentence_issue_tags.py):
    // { sentences: {sid: [{type,color,label_code,fix_code,...}]}, document_level: [...],
    //   legend: [...] } or absent. The SOLE source for the colored issue underlines in
    // the "Read full document" view. Null on older reports / clean docs -> the view
    // renders exactly as before (byte-identical fallback).
    sentenceIssueTags: results.sentence_issue_tags || null,
  };
}

// Tier-weighted severity for the per-paragraph density bar. critical worst -> low least; info trivial.
const PARAGRAPH_SEVERITY_TIER_WEIGHT = { critical: 4, high: 3, medium: 2, low: 1, info: 0.5 };
const PARAGRAPH_SEVERITY_TIER_RANK = { critical: 4, high: 3, medium: 2, low: 1, info: 0 };

// Build the per-paragraph severity heatmap bar from the submitted-content paragraph model.
// Severity = tier-weighted finding density per paragraph (weight / length), normalised across the doc.
// Width is proportional to each paragraph's share of the document length. Pure derivation from existing
// scan data (findings tier via signal.tier) -- no new backend computation, works on existing reports.
function buildParagraphSeverityBar(paragraphs, sentenceIssueTags = null) {
  if (!Array.isArray(paragraphs) || paragraphs.length === 0) return null;
  // Per-SENTENCE segments (not per-paragraph) so the bar shows graduated color across the whole
  // document — matching the "Read full document" heatmap. Each sentence = one colored segment,
  // width proportional to its text length. Colour reflects ALL findings, not just AI:
  //   red = reads as AI (high DeBERTa band / 'ai' issue tag),
  //   amber = a non-AI finding (grounding / reasoning / review candidate),
  //   green = clean. This is the same information the underlines carry, so the bar
  //   agrees with the document instead of being AI-only. Issue tags come from
  //   submittedContent.sentenceIssueTags (poc/report/sentence_issue_tags.py).
  const tagsBySid = new Map(Object.entries((sentenceIssueTags && sentenceIssueTags.sentences) || {}));
  const rows = [];
  paragraphs.forEach((paragraph) => {
    (paragraph.segments || []).forEach((segment) => {
      const deberta = (segment.signals || []).find((sg) => sg.key === 'ai_signal_deberta');
      // Only the VERDICT-GATED 'high' band drives the red severity color; a muted 'review'
      // candidate (green-doc saturation) must NOT paint the bar red, so its score is not counted
      // here (falls back to the gated topTier color). Single source: the band decides, not the raw
      // score, which stays 100 on muted sentences. See report.py::_gate_heatmap_bands.
      const debBand = deberta ? String(deberta.title || '').replace('deberta_', '') : '';
      const sid = segment.sentence_id || segment.id;
      const tags = tagsBySid.get(String(sid)) || [];
      const hasAiTag = tags.some((tg) => tg && tg.type === 'ai');
      const hasOtherTag = tags.some((tg) => tg && tg.type !== 'ai');
      rows.push({
        id: sid || rows.length,
        paragraphId: paragraph.id,
        length: Math.max(1, (segment.text || '').length),
        findingCount: (segment.signals || []).length,
        maxDebertaScore: debBand === 'high' ? (Number(deberta.score) || 0) : 0,
        // A muted 'review' candidate must render the SAME amber the full-document underline uses
        // (is-severity-review), NOT fall through to the 'low' tier green. Without this flag the bar
        // disagreed with the "Read full document" heatmap: amber sentence -> green bar segment.
        reviewBand: debBand === 'review',
        // Non-AI findings (grounding amber / reasoning purple underlines) → amber on the severity
        // bar so a grounding-only sentence reads as a finding, not clean green.
        hasAiTag,
        hasOtherTag,
        topTier: deberta ? (deberta.tier || '') : '',
      });
    });
  });
  const totalLength = rows.reduce((sum, row) => sum + row.length, 0) || 1;
  return rows.map((row, index) => ({
    id: row.id,
    paragraphId: row.paragraphId,
    index: index + 1,
    findingCount: row.findingCount,
    topTier: row.topTier,
    maxDebertaScore: row.maxDebertaScore,
    reviewBand: row.reviewBand,
    hasAiTag: row.hasAiTag,
    hasOtherTag: row.hasOtherTag,
    widthPct: (row.length / totalLength) * 100,
    intensity: 1,
  }));
}

// DeBERTa score -> color, matching the full-document per-sentence heatmap scale exactly:
// amber (80-89) < deep orange (90-98) < red (>=99). Below 80 = no color (clean).
const DEBERTA_SEVERITY_COLORS = [
  { min: 99, color: '#dc2626' },
  { min: 90, color: '#f97316' },
  { min: 80, color: '#f59e0b' },
];

function debertaSeverityColor(score) {
  for (const { min, color } of DEBERTA_SEVERITY_COLORS) {
    if (score >= min) return color;
  }
  return null;
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

// Mirror of render.GROUNDING_DIAGNOSIS_LEAD_ENABLED — keep both in sync. When ON, the report
// leads with the primary grounding-diagnosis DRIVER (what to fix) instead of the vague risk %.
// Driver label/action strings live in i18n report.groundingDiagnosis.drivers.* (en + zh), kept
// in sync with detect.grounding_diagnosis.DRIVER_LABELS.
const GROUNDING_DIAGNOSIS_LEAD_ENABLED = true;

const GROUNDING_DIAGNOSIS_BUCKETS = ['concrete_grounding', 'authorship_trace', 'llm_patterning', 'language_texture'];

// Accessor for the additive 4-bucket grounding diagnosis on the badge (or null).
function groundingDiagnosis(badge) {
  const diag = (badge || {}).grounding_diagnosis;
  return diag && typeof diag === 'object' ? diag : null;
}

// Critical Thinking Control diagnosis (additive, non-gating). Returns null when
// absent or when the score abstained (insufficient_data) so the section hides.
// Display flag mirrors the GROUNDING_DIAGNOSIS_LEAD_ENABLED convention: the
// section runs on every scan and could not be E2E-verified pre-deploy, so this
// toggle hides it without a code change if the live render is wrong.
const CRITICAL_THINKING_CONTROL_ENABLED = true;
const CRITICAL_THINKING_DIMENSIONS = [
  'specific_context', 'student_judgement', 'reasoning_trail', 'ai_dependency', 'evidence_grounding',
];

function criticalThinkingControl(badge) {
  const ctc = (badge || {}).critical_thinking_control;
  if (!ctc || typeof ctc !== 'object') return null;
  return typeof ctc.score === 'number' ? ctc : null;
}

// Additive Submission-risk view (3-layer: text-pattern / ownership / academic).
// Returns null when the diagnosis abstained (overall.level === 'unknown') so the
// report falls back to leading with the AI-likelihood headline. The axis ORDER is
// fixed for display; policy_declaration is always 'unknown — self-declare'.
const SUBMISSION_RISK_AXES = [
  'text_pattern', 'ownership', 'citation', 'defence_readiness', 'policy_declaration',
];

function submissionRisk(badge) {
  const sr = (badge || {}).submission_risk;
  if (!sr || typeof sr !== 'object') return null;
  const level = sr.overall && sr.overall.level;
  if (!level || level === 'unknown') return null;
  // When the V7 tier-authority override fired, the badge's ai_likelihood (and
  // therefore the % in the band's note) is the FUSED score, not the composite —
  // the note must label it correctly (mislabel observed live 2026-07-04).
  return {
    ...sr,
    _fused: Boolean((badge || {}).tier_authority),
    _flagLine: ((badge || {}).tier_authority || {}).flag_line || null,
  };
}

// Two policy-interpreted scores (AI-allowed vs AI-restricted). Returns null when the
// diagnosis abstained, so the card falls back to the raw detector two-number block.
function policyRisk(badge) {
  const pr = (badge || {}).policy_risk;
  if (!pr || typeof pr !== 'object') return null;
  const level = pr.ai_allowed && pr.ai_allowed.level;
  if (!level || level === 'unknown') return null;
  return pr;
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
  // Grounding-diagnosis driver: when ON, lead the MAIN RISK with the specific driver
  // (e.g. "grounding gap") instead of the vague generic risk label.
  const diag = groundingDiagnosis(report?.ai_risk_badge);
  const driverKey = GROUNDING_DIAGNOSIS_LEAD_ENABLED && diag?.primary_driver ? diag.primary_driver : null;
  const driverLabel = driverKey ? t(`report.groundingDiagnosis.drivers.${driverKey}.label`) : '';
  const driverAction = driverKey ? t(`report.groundingDiagnosis.drivers.${driverKey}.action`) : '';
  const statusText = firstNonEmpty(
    status,
    pattern?.label,
    report?.ai_risk_badge?.authorship_rating_label,
    t('report.repairSummary.statusFallback')
  );
  const mainRisk = firstNonEmpty(
    driverLabel,
    primarySignalLabel,
    transformationSummary?.summary,
    t('report.repairSummary.mainRiskFallback')
  );
  const nextAction = highlightedCount > 0
    ? t('report.repairSummary.nextActionHighlighted', { count: highlightedCount })
    : firstNonEmpty(
        driverAction,
        authorshipEvidence?.thin_signals?.[0]?.action,
        t('report.repairSummary.nextActionFallback')
      );

  return {
    status: statusText,
    mainRisk,
    nextAction,
    highlightedCount,
    confidenceNote: t('report.repairSummary.confidenceNote'),
  };
}

function buildFixFirstItems({ submittedContent, t }) {
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
      const signal = paragraph.primarySignal;
      // DeBERTa-native guidance only (readerSummary + recommendation). The perplexity-fed LLM
      // explanation (paragraph.explanation.main_issue/recommendation/rewrite_hint) is NOT read
      // here — it leaked "predictable, generic phrasing" advice from the abandoned methodology.
      addItem({
        paragraphId: paragraph.id,
        label: paragraph.sentence_id,
        title: firstNonEmpty(
          paragraph.readerSummary,
          signalLabel(signal.key, signal.label, t),
          t('report.whatToFixFirst.paragraphFallbackTitle')
        ),
        // body intentionally omitted: the per-sentence fix guidance lives in
        // Signal highlights below. This row is a prioritized, clickable locator
        // (the grounding instruction is stated once in the section intro), not a
        // restatement of paragraph.recommendation (which duplicated the highlights).
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
  SIGNAL_COLORS,
  TURNITIN_AI_REFERENCE_THRESHOLD,
  AI_SIGNAL_STAMP_LEVELS,
  metricValue,
  metricCount,
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
  debertaSeverityColor,
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
  groundingDiagnosis,
  GROUNDING_DIAGNOSIS_LEAD_ENABLED,
  GROUNDING_DIAGNOSIS_BUCKETS,
  criticalThinkingControl,
  CRITICAL_THINKING_DIMENSIONS,
  CRITICAL_THINKING_CONTROL_ENABLED,
  submissionRisk,
  SUBMISSION_RISK_AXES,
  policyRisk,
};
