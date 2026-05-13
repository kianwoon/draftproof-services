import { useCallback, useEffect, useRef, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { getReport, createRewrite, cancelRewrite, getRewriteStatus, getRewriteReport } from '../api/draftproofApi';
import ErrorReload from '../components/ErrorReload';
import ConfirmDialog from '../components/ConfirmDialog';
import { useAuth } from '../context/AuthContext';
import RewriteNoticeDialog from './report/RewriteNoticeDialog';
import {
  TIER_CONFIG,
  SEVERITY_CONFIG,
  signalClassName,
  formatDate,
  signalLabel,
  translatedSignal,
  translatedGroup,
  transformationLabel,
  confidenceLabel,
  evidenceLabel,
  translateAuthorshipRating,
  formatMetricPercent,
  clampPercent,
  buildTransformationSignals,
  transformationSignalFeatureMap,
  sortTransformationSignalsForComparison,
  buildPairedTransformationSignals,
  groupTransformationSignals,
  getTransformationSignalImprovement,
  buildTransformationSummary,
  deriveAuthorshipRatingFallback,
  deriveCalibratedAuthorshipRating,
  formatAuthorshipSealDetail,
  getAuthorshipTone,
  formatSignedDelta,
  formatPlainScore,
  getOriginalDetectScan,
  getRewrittenDetectScan,
  mergeScanSummary,
  getScanDocumentContext,
  getScanTransformationSignals,
  getScanContributionSummary,
  mergeTransformationSummary,
  buildRewriteResultSummary,
  buildRewriteContributionOverride,
  buildSubmittedContentModel,
  isRewriteActive,
  normalizeRewriteProgressMessage,
  normalizeRewriteJob,
  formatElapsed,
  getRewriteProgressDetail,
  isReviewOnlyRewriteMessage,
  buildRewriteEventsUrl,
} from './report/reportHelpers';

export default function Report() {
  const { id } = useParams();
  const { refreshBalance } = useAuth();
  const { t, i18n } = useTranslation();
  const locale = i18n.resolvedLanguage?.startsWith('zh') ? 'zh-CN' : 'en-SG';
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
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
      title: t('report.rewrite.noRewriteableTitle'),
      message: isReviewOnlyRewriteMessage(message) && String(message).includes('token')
        ? message
        : t('report.rewrite.noRewriteableMessage'),
    });
  }, [t]);

  const syncRewriteJob = useCallback((job) => {
    const normalizedJob = normalizeRewriteJob(job, t);
    setRewriteJob(normalizedJob);
    if (normalizedJob?.status && !['failed', 'canceled'].includes(normalizedJob.status)) {
      setRewriteError(null);
    }
    if (normalizedJob?.status === 'completed') {
      setReport((prev) => prev ? { ...prev, rewrite: normalizedJob } : prev);
      setRewriteStartedHere(false);
    }
  }, [t]);

  const pollRewriteStatus = useCallback(async (rewriteId) => {
    try {
      const { data } = await getRewriteStatus(rewriteId);
      syncRewriteJob(data);
      if (data.status === 'failed') {
        const failedMessage = data.error || t('report.rewrite.failed');
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
      setRewriteError(err.response?.data?.detail || t('report.rewrite.checkingFailed'));
      return null;
    }
  }, [showReviewOnlyRewriteNotice, syncRewriteJob, t]);

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
        const failedMessage = data.error || t('report.rewrite.failed');
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
      setRewriteError(t('report.rewrite.failed'));
      closeRewriteEventSource();
    });

    source.addEventListener('error', () => {
      closeRewriteEventSource();
      setRewriteSseUnavailable(true);
      pollRewriteStatus(rewriteId);
    });

    return true;
  }, [closeRewriteEventSource, pollRewriteStatus, showReviewOnlyRewriteNotice, syncRewriteJob, t]);

  useEffect(() => {
    const ac = new AbortController();
    getReport(id, { signal: ac.signal })
      .then(({ data }) => {
        setReport(data);
        if (data.rewrite) {
          setRewriteSseUnavailable(false);
          setRewriteJob(normalizeRewriteJob(data.rewrite, t));
          if (data.rewrite.id && isRewriteActive(data.rewrite.status)) {
            connectRewriteEvents(data.rewrite.id);
          }
        }
      })
      .catch((err) => {
        if (err.name === 'AbortError' || err.code === 'ERR_CANCELED') return;
        setError(err.response?.data?.detail || t('report.loadFailed'));
      })
      .finally(() => setLoading(false));
    return () => {
      ac.abort();
      closeRewriteEventSource();
    };
  }, [id, closeRewriteEventSource, connectRewriteEvents, t]);

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
          <p>{t('report.loading')}</p>
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
      <div className="container"><p>{t('report.notFound')}</p></div>
    </main>
  );

  const tier = TIER_CONFIG[report.tier] || TIER_CONFIG.moderate;
  const issues = Array.isArray(report.issues) ? report.issues : [];
  const badge = report.ai_risk_badge || {};
  const aiScore = report.ai_score ?? badge.ai_likelihood_score ?? null;
  const writingScore = report.writing_score ?? badge.writing_quality_score ?? null;
  const reportScanJson = report.results_json || {};
  const originalComparisonScan = mergeScanSummary(reportScanJson, getOriginalDetectScan(rewriteResultReport));
  const originalComparisonBadge = originalComparisonScan?.ai_risk_badge || badge;
  const originalDocumentContext = getScanDocumentContext(originalComparisonScan || reportScanJson);
  const originalComparisonAiScore = originalComparisonScan
    ? (originalComparisonScan.ai_score ?? originalComparisonBadge.ai_likelihood_score ?? rewriteResultSummary?.original_risk ?? aiScore)
    : aiScore;
  const transformation = originalComparisonBadge.transformation_classification || null;
  const transformationSignalMetadata = getScanTransformationSignals(originalComparisonScan);
  const transformationFeatureFallbacks = transformationSignalFeatureMap(transformationSignalMetadata);
  const authorshipFeatures = {
    ...transformationFeatureFallbacks,
    ...(transformation?.features || {}),
  };
  const transformationSignals = buildTransformationSignals(authorshipFeatures, transformationSignalMetadata);
  const originalScanContributionSummary = getScanContributionSummary(originalComparisonScan);
  const originalContributionOverride = buildRewriteContributionOverride(rewriteResultSummary, 'original') || originalScanContributionSummary;
  if (authorshipFeatures.calibrated_ai_risk == null && originalContributionOverride?.adjustedAiRisk != null) {
    authorshipFeatures.calibrated_ai_risk = originalContributionOverride.adjustedAiRisk;
  }
  const transformationSummary = transformation
    ? mergeTransformationSummary(
      buildTransformationSummary(authorshipFeatures, transformationSignals, originalContributionOverride, t),
      originalScanContributionSummary
    )
    : null;
  const rewrittenScan = getRewrittenDetectScan(rewriteResultReport) || {};
  const rewrittenBadge = rewrittenScan.ai_risk_badge || {};
  const rewrittenTransformation = rewrittenBadge.transformation_classification || null;
  const rewrittenTransformationSignalMetadata = getScanTransformationSignals(rewrittenScan);
  const rewrittenTransformationFeatureFallbacks = transformationSignalFeatureMap(rewrittenTransformationSignalMetadata);
  const rewrittenAuthorshipFeatures = {
    ...rewrittenTransformationFeatureFallbacks,
    ...(rewrittenTransformation?.features || {}),
  };
  const rewrittenTransformationSignals = buildTransformationSignals(
    rewrittenAuthorshipFeatures,
    rewrittenTransformationSignalMetadata
  );
  const rewrittenScanContributionSummary = getScanContributionSummary(rewrittenScan);
  const rewrittenContributionOverride = buildRewriteContributionOverride(rewriteResultSummary, 'rewritten') || rewrittenScanContributionSummary;
  if (rewrittenAuthorshipFeatures.calibrated_ai_risk == null && rewrittenContributionOverride?.adjustedAiRisk != null) {
    rewrittenAuthorshipFeatures.calibrated_ai_risk = rewrittenContributionOverride.adjustedAiRisk;
  }
  const rewrittenTransformationSummary = rewrittenTransformation
    ? mergeTransformationSummary(
      buildTransformationSummary(rewrittenAuthorshipFeatures, rewrittenTransformationSignals, rewrittenContributionOverride, t),
      rewrittenScanContributionSummary
    )
    : rewrittenContributionOverride
      ? {
        humanContribution: Math.round(rewrittenContributionOverride.humanContribution),
        aiTransformation: Math.round(rewrittenContributionOverride.aiTransformation),
        adjustedAiRisk: Math.round(rewriteResultSummary?.rewrite_risk ?? 0),
        rawAdjustedAiRisk: Math.round(rewriteResultSummary?.rewrite_risk ?? 0),
        humanAnchorDiscount: 0,
        calibrationConfidence: 0,
        reportingSuppression: 0,
        summary: t('report.transformation.rewrittenContributionEstimate'),
      }
    : null;
  const rewrittenAiScore = rewrittenScan.ai_score ?? rewrittenBadge.ai_likelihood_score ?? rewriteResultSummary?.rewrite_risk ?? null;
  const rewrittenWritingScore = rewrittenScan.writing_score ?? rewrittenBadge.writing_quality_score ?? null;
  const calibratedAuthorshipRisk = clampPercent(authorshipFeatures.calibrated_ai_risk);
  const topkPatternScore = clampPercent(originalComparisonBadge.ai_components?.topk_pattern_raw ?? originalComparisonBadge.ai_components?.topk_pattern);
  const topkCalibratedRisk = clampPercent(originalComparisonBadge.ai_components?.topk_calibrated_risk);
  const rewrittenCalibratedAuthorshipRisk = clampPercent(rewrittenAuthorshipFeatures.calibrated_ai_risk);
  const rewrittenTopkPatternScore = clampPercent(rewrittenBadge.ai_components?.topk_pattern_raw ?? rewrittenBadge.ai_components?.topk_pattern);
  const rewrittenTopkCalibratedRisk = clampPercent(rewrittenBadge.ai_components?.topk_calibrated_risk);
  const rewrittenDocumentContext = getScanDocumentContext(rewrittenScan);
  const rawAuthorshipSignal = aiScore;
  const storedAuthorshipRating = badge.authorship_rating || deriveAuthorshipRatingFallback(
    aiScore,
    badge.tier || report.tier,
    writingScore,
    badge.ai_components,
    badge.writing_components
  ) || {};
  const authorshipRating = translateAuthorshipRating(deriveCalibratedAuthorshipRating(
    calibratedAuthorshipRisk,
    topkPatternScore,
    topkCalibratedRisk,
    authorshipFeatures,
    originalDocumentContext,
    originalComparisonBadge.ai_components?.topk_calibration_eligible
  ) || storedAuthorshipRating, t);
  const authorshipTone = getAuthorshipTone(authorshipRating);
  const authorshipRatingFullLabel = authorshipRating.label || badge.authorship_rating_label || null;
  const authorshipRatingLabel = authorshipRating.short_label || authorshipRatingFullLabel;
  const authorshipSealDetail = formatAuthorshipSealDetail({
    rating: authorshipRating,
    topkPatternScore,
    topkCalibratedRisk,
    calibratedAuthorshipRisk,
    fallbackScore: originalComparisonAiScore,
  }, t);
  const rewrittenStoredAuthorshipRating = rewrittenBadge.authorship_rating || deriveAuthorshipRatingFallback(
    rewrittenAiScore,
    rewrittenBadge.tier || badge.tier || report.tier,
    rewrittenWritingScore,
    rewrittenBadge.ai_components,
    rewrittenBadge.writing_components
  ) || {};
  const rewrittenAuthorshipRating = translateAuthorshipRating(deriveCalibratedAuthorshipRating(
    rewrittenCalibratedAuthorshipRisk,
    rewrittenTopkPatternScore,
    rewrittenTopkCalibratedRisk,
    rewrittenAuthorshipFeatures,
    rewrittenDocumentContext,
    rewrittenBadge.ai_components?.topk_calibration_eligible
  ) || rewrittenStoredAuthorshipRating, t);
  const rewrittenAuthorshipSealDetail = formatAuthorshipSealDetail({
    rating: rewrittenAuthorshipRating,
    topkPatternScore: rewrittenTopkPatternScore,
    topkCalibratedRisk: rewrittenTopkCalibratedRisk,
    calibratedAuthorshipRisk: rewrittenCalibratedAuthorshipRisk,
    fallbackScore: rewrittenAiScore,
  }, t);
  const rewrittenAuthorshipTone = getAuthorshipTone(rewrittenAuthorshipRating);
  const rewrittenAuthorshipRatingFullLabel = rewrittenAuthorshipRating.label || rewrittenBadge.authorship_rating_label || null;
  const rewrittenAuthorshipRatingLabel = rewrittenAuthorshipRating.short_label || rewrittenAuthorshipRatingFullLabel;
  const originalColumnRatingBadge = {
    caption: t('report.transformation.originalRating'),
    label: authorshipRatingLabel,
    fullLabel: authorshipRatingFullLabel,
    tone: authorshipTone,
  };
  const rewrittenColumnRatingBadge = {
    caption: t('report.transformation.rewrittenRating'),
    label: rewrittenAuthorshipRatingLabel,
    fullLabel: rewrittenAuthorshipRatingFullLabel,
    tone: rewrittenAuthorshipTone,
  };
  const issueCounts = { critical: 0, high: 0, medium: 0, low: 0, info: 0 };
  issues.forEach((iss) => { if (issueCounts[iss.severity] !== undefined) issueCounts[iss.severity]++; });
  const normalizedReport = { ...report, issues };
  const submittedContent = buildSubmittedContentModel(normalizedReport);
  const selectedSegment = (
    submittedContent.segments.find((segment) => segment.id === selectedSegmentId) ||
    submittedContent.segments.find((segment) => segment.signals.length > 0) ||
    null
  );

  const hasAIFindings = issues.some(i =>
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
  const sealAuthorshipRating = hasRewriteSignalComparison && rewrittenAuthorshipRating
    ? rewrittenAuthorshipRating
    : authorshipRating;
  const sealAuthorshipTone = getAuthorshipTone(sealAuthorshipRating);
  const sealAuthorshipFullLabel = sealAuthorshipRating.label || (hasRewriteSignalComparison ? rewrittenBadge.authorship_rating_label : badge.authorship_rating_label) || null;
  const sealAuthorshipLabel = sealAuthorshipRating.short_label || sealAuthorshipFullLabel;
  const sealAuthorshipDetail = hasRewriteSignalComparison
    ? rewrittenAuthorshipSealDetail
    : authorshipSealDetail;
  const pairedTransformationSignals = hasRewriteSignalComparison
    ? buildPairedTransformationSignals(transformationSignals, rewrittenTransformationSignals)
    : null;
  const transformationOriginalScore = hasRewriteSignalComparison
    ? (rewriteResultSummary?.original_ai_authorship ?? rewriteResultSummary?.original_risk ?? originalComparisonAiScore)
    : originalComparisonAiScore;
  const transformationRewrittenScore = hasRewriteSignalComparison
    ? (rewriteResultSummary?.rewritten_ai_authorship ?? rewriteResultSummary?.rewrite_risk ?? rewrittenAiScore)
    : rewrittenAiScore;
  const canStartRewrite = hasAIFindings && !hasRewriteResult;
  const rewriteProgress = currentRewrite
    ? Math.max(0, Math.min(100, Number(currentRewrite.progress_percent) || (rewriteInProgress ? 5 : hasCompletedRewrite ? 100 : 0)))
    : 0;
  const rewriteProgressMessage = normalizeRewriteProgressMessage(
    currentRewrite?.progress_message,
    currentRewrite?.status,
    t
  );
  const rewriteProgressDetail = !rewriteError && rewriteInProgress
    ? getRewriteProgressDetail({
      status: currentRewrite?.status,
      progress: rewriteProgress,
      elapsedSeconds: rewriteElapsedSeconds,
      sseUnavailable: rewriteSseUnavailable,
      t,
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
      progress_message: t('report.rewrite.queuing'),
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
      const msg = err.response?.data?.detail || t('report.rewrite.startFailed');
      if (err.response?.status === 402) {
        setRewriteJob(null);
        setRewriteError(msg);
      } else if (err.response?.status === 422 || isReviewOnlyRewriteMessage(msg)) {
        showReviewOnlyRewriteNotice(msg);
      } else {
        setRewriteJob((prev) => prev ? {
          ...prev,
          status: 'failed',
          progress_message: t('report.rewrite.failed'),
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
      setRewriteError(err.response?.data?.detail || t('report.rewrite.cancelFailed'));
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
          <span className="report-risk-value" style={{ color: tier.color }}>{t(`report.tiers.${report.tier}`, { defaultValue: tier.label })}</span>
          <span className="report-stat-label">{t('report.summary.riskTier')}</span>
        </span>
      </div>
      <div className="report-stat">
        <span className="report-stat-value">{issues.length}</span>
        <span className="report-stat-label">{t('report.summary.totalFindings')}</span>
      </div>
      {authorshipRatingLabel && (
        <div className="report-stat">
          <span className="report-stat-value" style={{ color: authorshipTone.color }} title={authorshipRatingFullLabel || authorshipRatingLabel}>
            {authorshipRatingLabel}
          </span>
          <span className="report-stat-label">{t('report.summary.authorshipRating')}</span>
        </div>
      )}
      {rawAuthorshipSignal != null && (
        <div className="report-stat">
          <span className="report-stat-value" style={{ color: tier.color }}>{formatMetricPercent(rawAuthorshipSignal, 2)}</span>
          <span className="report-stat-label">{t('report.summary.rawAiSignal')}</span>
        </div>
      )}
      {writingScore != null && (
        <div className="report-stat">
          <span className="report-stat-value" style={{ color: '#6366f1' }}>{formatMetricPercent(writingScore, 2)}</span>
          <span className="report-stat-label">{t('report.summary.writingScore')}</span>
        </div>
      )}
      {Object.entries(issueCounts).filter(([, v]) => v > 0).map(([sev, count]) => {
        const sc = SEVERITY_CONFIG[sev];
        return (
          <div key={sev} className="report-stat">
            <span className="report-stat-value" style={{ color: sc.color }}>{count}</span>
            <span className="report-stat-label">{t(`report.severities.${sev}`, { defaultValue: sc.label })}</span>
          </div>
        );
      })}
    </div>
  );

  const renderTransformationDetails = (variant, pattern, summary, signals, variantAiScore, pairedSignals = null, ratingBadge = null) => {
    const comparisonSignals = pairedSignals
      ? pairedSignals.map((pair) => ({
        ...(pair[variant] || {
          key: pair.key,
          label: pair.label,
          description: pair.description,
          value: null,
          isMissing: true,
        }),
        pairedLabel: pair.label,
        pairedDescription: pair.description,
      }))
      : sortTransformationSignalsForComparison(signals);
    const localizedComparisonSignals = comparisonSignals.map((signal) => translatedSignal(signal, t));

    return (
      <div className={`transformation-detail ${variant === 'rewritten' ? 'is-rewritten' : 'is-original'}`}>
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
          <em>{formatMetricPercent(variantAiScore, 1)}</em>
        </div>
        {summary && (
          <div className="transformation-ratio-summary">
            <div className="transformation-ratio-copy">
              <span>{t('report.transformation.estimatedContribution')}</span>
              <p>{summary.summary}</p>
              <div className="transformation-adjustment-row">
                <strong>{t('report.transformation.calibratedAiRisk', { value: summary.adjustedAiRisk })}</strong>
                <strong>{t('report.transformation.humanAnchorDiscount', { value: summary.humanAnchorDiscount })}</strong>
                <strong>{t('report.transformation.calibrationConfidence', { value: summary.calibrationConfidence })}</strong>
                <strong>{t('report.transformation.reportingSuppression', { value: summary.reportingSuppression })}</strong>
              </div>
            </div>
            <div className="transformation-ratio-bars" aria-label={t('report.transformation.contributionEstimate', { variant: variant === 'rewritten' ? t('report.transformation.rewritten') : t('report.transformation.original') })}>
              <div className="transformation-ratio-row">
                <span>{t('report.transformation.humanContribution')}</span>
                <strong>{summary.humanContribution}%</strong>
                <div className="transformation-ratio-track">
                  <div className="transformation-ratio-fill is-human" style={{ width: `${summary.humanContribution}%` }} />
                </div>
              </div>
              <div className="transformation-ratio-row">
                <span>{t('report.transformation.aiTransformation')}</span>
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
              <span>{t('report.transformation.signalProfile')}</span>
            </div>
            <div className="transformation-bars">
              {groupTransformationSignals(localizedComparisonSignals).map((group) => translatedGroup(group, t)).map((group) => (
                <div className={`transformation-signal-group transformation-signal-group-${group.id}`} key={`${variant}-${group.id}`}>
                  <div className="transformation-signal-group-head">
                    <div>
                      <h4>{group.label}</h4>
                      {group.description && <p>{group.description}</p>}
                    </div>
                    <span>{t('report.transformation.signals', { count: group.signals.length })}</span>
                  </div>
                  {group.signals.map((signal) => {
                    const baselineSignal = variant === 'rewritten'
                      ? transformationSignals.find((item) => item.key === signal.key)
                      : null;
                    const improvement = signal.isMissing ? null : getTransformationSignalImprovement(signal, baselineSignal);
                    const improvementCopy = improvement
                      ? t('report.transformation.improvedFromTo', { from: improvement.from.toFixed(0), to: improvement.to.toFixed(0) })
                      : '';
                    const tooltip = [
                      signal.pairedDescription || signal.description,
                      signal.isMissing ? t('report.transformation.notPresent') : improvementCopy,
                    ].filter(Boolean).join(' ');
                    const valueLabel = signal.isMissing ? '—' : `${signal.value.toFixed(0)}%`;
                    const barWidth = signal.isMissing ? 0 : signal.value;

                    return (
                      <div
                        key={`${variant}-${signal.key}`}
                        className={`transformation-bar-row${improvement ? ' is-improved' : ''}${signal.isMissing ? ' is-missing' : ''}`}
                        data-tooltip={tooltip}
                        tabIndex={0}
                        aria-label={`${variant === 'rewritten' ? t('report.transformation.rewritten') : t('report.transformation.original')} ${signal.pairedLabel || signal.label}: ${valueLabel}. ${tooltip}`}
                        title={tooltip}
                      >
                        <div className="transformation-bar-label">
                          <span className="transformation-bar-name">{signal.pairedLabel || signal.label}</span>
                          <span
                            className={`transformation-improvement-slot${improvement ? ' is-visible' : ''}`}
                            aria-label={improvement ? t('report.transformation.improvedFromTo', { from: improvement.from.toFixed(0), to: improvement.to.toFixed(0) }) : undefined}
                            title={improvement ? t('report.transformation.improvedFromTo', { from: improvement.from.toFixed(0), to: improvement.to.toFixed(0) }) : undefined}
                          >
                            {improvement && (
                              <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
                                <path d="M3.2 6.1l1.8 1.8 3.8-4.2" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
                              </svg>
                            )}
                          </span>
                          <strong>{valueLabel}</strong>
                        </div>
                        <div className="transformation-bar-track" aria-hidden="true">
                          <div
                            className={`transformation-bar-fill transformation-bar-${signal.key}`}
                            style={{ width: `${barWidth}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    );
  };

  const transformationScorecard = transformation && transformationSignals.length > 0 ? (
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
            <div className="transformation-meta-row">
              {transformation.confidence && (
                <span className="transformation-pill">{t('report.transformation.confidence', { value: confidenceLabel(transformation.confidence, t) })}</span>
              )}
              {hasRewriteSignalComparison && (
                <span className="transformation-pill">{t('report.transformation.rewriteComparison')}</span>
              )}
              <span className="transformation-pill">{t('report.transformation.notVerdict')}</span>
            </div>
          </div>
        </div>
        <div
          className="transformation-authorship-seal"
          style={{
            '--rating-color': sealAuthorshipTone.color,
            '--rating-bg': sealAuthorshipTone.bg,
          }}
        >
          <span>{hasRewriteSignalComparison ? t('report.transformation.rewrittenOutcome') : t('report.summary.authorshipRating')}</span>
          <strong title={sealAuthorshipFullLabel || sealAuthorshipLabel || undefined}>
            {sealAuthorshipLabel || t('report.transformation.notRated')}
          </strong>
          <em>
            {sealAuthorshipDetail}
          </em>
        </div>
      </div>
      <div className="transformation-chart">
        {hasRewriteSignalComparison ? (
          <div className="transformation-comparison-grid">
            {renderTransformationDetails('original', transformation, transformationSummary, transformationSignals, transformationOriginalScore, pairedTransformationSignals, originalColumnRatingBadge)}
            {renderTransformationDetails('rewritten', rewrittenTransformation, rewrittenTransformationSummary, rewrittenTransformationSignals, transformationRewrittenScore, pairedTransformationSignals, rewrittenColumnRatingBadge)}
          </div>
        ) : (
          renderTransformationDetails('original', transformation, transformationSummary, transformationSignals, transformationOriginalScore)
        )}
        {Array.isArray(transformation.evidence) && transformation.evidence.length > 0 && (
          <div className="transformation-evidence">
            {transformation.evidence.slice(0, 3).map((item) => (
              <span key={item}>{evidenceLabel(item, t)}</span>
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
    ? t('report.rewrite.accepted')
    : rewriteOutcome === 'topk_blocked'
      ? t('report.rewrite.topkBlocked')
    : rewriteOutcome === 'suggestion_only'
      ? t('report.rewrite.originalPreserved')
      : rewriteOutcomeText || t('report.rewrite.completeTitle');
  const rewriteBandDetail = rewriteOutcome === 'ai_mitigated'
    ? t('report.rewrite.acceptedDetail')
    : rewriteOutcome === 'topk_blocked'
      ? t('report.rewrite.topkBlockedDetail')
    : rewriteOutcome === 'suggestion_only'
      ? t('report.rewrite.originalPreservedDetail')
      : rewriteResultSummary?.ai_mitigation_selected
        ? t('report.rewrite.selectedDetail')
        : t('report.rewrite.finishedDetail');
  const rewriteCompletionBand = hasRewriteResult ? (
    <div className={`report-rewrite-summary-bar${rewriteOutcome === 'suggestion_only' ? ' is-preserved' : ''}${rewriteOutcome === 'topk_blocked' ? ' is-blocked' : ''}${rewriteOutcome === 'ai_mitigated' ? ' is-mitigated' : ''}`}>
      <div className="rewrite-summary-icon" aria-hidden="true">
        <span>
          <svg width="42" height="42" viewBox="0 0 42 42" fill="none">
            <circle cx="21" cy="21" r="15" fill="currentColor"/>
            {rewriteOutcome === 'suggestion_only' || rewriteOutcome === 'topk_blocked' ? (
              <path d="M15 15l12 12M27 15L15 27" stroke="#fff" strokeWidth="3" strokeLinecap="round"/>
            ) : (
              <path d="M14 21.5l4.5 4.5L28.5 16" stroke="#fff" strokeWidth="3.2" strokeLinecap="round" strokeLinejoin="round"/>
            )}
          </svg>
        </span>
      </div>
      <div className="rewrite-summary-main">
        <span className="rewrite-summary-kicker">{t('report.rewrite.completion')}</span>
        <strong>{rewriteBandTitle}</strong>
        <em>{rewriteBandDetail}</em>
      </div>
      <div className="rewrite-summary-stat">
        <span>{formatMetricPercent(rewriteResultSummary?.original_ai_authorship ?? rewriteResultSummary?.original_risk, 1)}</span>
        <small>{t('report.rewrite.aiBefore')}</small>
      </div>
      <div className="rewrite-summary-stat">
        <span>{formatMetricPercent(rewriteResultSummary?.rewritten_ai_authorship ?? rewriteResultSummary?.rewrite_risk, 1)}</span>
        <small>{t('report.rewrite.aiAfter')}</small>
      </div>
      <div className="rewrite-summary-stat">
        <span>{formatPlainScore(rewriteResultSummary?.human_shift_score, 1)}</span>
        <small>{t('report.rewrite.humanShift')}</small>
      </div>
      <div className="rewrite-summary-stat">
        <span>{formatSignedDelta(rewriteResultSummary?.original_human_contribution, rewriteResultSummary?.rewritten_human_contribution)}</span>
        <small>{t('report.rewrite.humanContribution')}</small>
      </div>
      <div className="rewrite-summary-stat">
        <span>{formatSignedDelta(rewriteResultSummary?.original_ai_transformation, rewriteResultSummary?.rewritten_ai_transformation)}</span>
        <small>{t('report.rewrite.aiTransformation')}</small>
      </div>
      <div className="rewrite-summary-stat">
        <span>{formatSignedDelta(rewriteResultSummary?.original_grounding_quality_risk, rewriteResultSummary?.rewritten_grounding_quality_risk)}</span>
        <small>{t('report.rewrite.groundingRisk')}</small>
      </div>
      <Link
        to={`/rewrite/${currentRewrite.id}`}
        className="rewrite-results-link"
      >
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
          <path d="M5 2.5h5.2L13 5.3v10.2H5V2.5z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round"/>
          <path d="M10 2.5v3h3M6.8 8.3h4M6.8 11h4" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/>
        </svg>
        {t('report.rewrite.viewResult')}
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
        title={t('report.rewrite.cancelTitle')}
        message={t('report.rewrite.cancelMessage')}
        confirmLabel={rewriteCanceling ? t('report.rewrite.canceling') : t('report.rewrite.cancelRewrite')}
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
          {t('report.back')}
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
              <div className="report-eyebrow">{t('report.eyebrow')}</div>
              <h1>{report.document_name}</h1>
              {report.created_at && (
                <p className="report-meta">
                  <svg width="17" height="17" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                    <path d="M4.5 1.8v2M11.5 1.8v2M2.5 6h11M3.5 3.5h9A1.5 1.5 0 0114 5v7.5A1.5 1.5 0 0112.5 14h-9A1.5 1.5 0 012 12.5V5a1.5 1.5 0 011.5-1.5z" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"/>
                  </svg>
                  {formatDate(report.created_at, locale)}
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
                  {t('report.downloadPdf')}
                </a>
              )}
              {(canStartRewrite || rewriteLoading || rewriteInProgress) && (
                <button
                  type="button"
                  className="rewrite-btn"
                  onClick={handleRewrite}
                  disabled={rewriteLoading || rewriteCanceling}
                >
                  {rewriteLoading ? t('report.rewrite.starting') : rewriteInProgress ? t('report.rewrite.resume') : t('report.rewrite.rewriteAiSections')}
                </button>
              )}
              {rewriteInProgress && currentRewrite?.id && (
                <button
                  type="button"
                  className="rewrite-btn rewrite-cancel-btn"
                  onClick={handleCancelRewrite}
                  disabled={rewriteCanceling}
                >
                  {rewriteCanceling ? t('report.rewrite.canceling') : t('report.rewrite.cancelRewrite')}
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
                  {rewriteError || rewriteProgressMessage || t('report.rewrite.processing')}
                  {rewriteInProgress && <em> {t('report.rewrite.keepOpenInline')}</em>}
                </span>
                <span>{hasCompletedRewrite ? t('report.rewrite.done') : `${rewriteProgress}%`}</span>
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
                  {rewriteElapsedLabel && <span>{t('report.rewrite.elapsed', { elapsed: rewriteElapsedLabel })}</span>}
                  <span>{t('report.rewrite.keepOpen')}</span>
                </div>
              )}
            </div>
          </div>
        )}

        {rewriteCompletionBand}

        {transformationScorecard ? (
          <section className={`report-overview-card${hasRewriteSignalComparison ? ' is-rewrite-comparison' : ''}`} aria-label={t('report.overview')}>
            {hasRewriteSignalComparison ? (
              <>
                {transformationScorecard}
                <div className="report-baseline-summary" aria-label={t('report.originalScanSummary')}>
                  <span className="report-baseline-label">{t('report.originalScanBaseline')}</span>
                  {reportSummaryBar}
                </div>
              </>
            ) : (
              <>
                {reportSummaryBar}
                {transformationScorecard}
              </>
            )}
          </section>
        ) : (
          reportSummaryBar
        )}

        {submittedContent.segments.length > 0 && (
          <section className="submitted-content-review" aria-label={t('report.submitted.sectionLabel')}>
            <div className="submitted-content-head">
              <div>
                <span className="submitted-content-kicker">{t('report.submitted.kicker')}</span>
                <h2>{t('report.submitted.title')}</h2>
              </div>
              <div className="submitted-content-count">
                <strong>{submittedContent.highlightedCount}</strong>
                <span>{t('report.submitted.highlightedSections')}</span>
              </div>
            </div>
            {submittedContent.legend.length > 0 && (
              <div className="submitted-signal-legend" aria-label={t('report.submitted.legend')}>
                {submittedContent.legend.slice(0, 6).map((signal) => (
                  <span
                    key={signal.key}
                    className={`submitted-signal-chip signal-style-${signalClassName(signal.key)}`}
                    style={{ '--signal-color': signal.color }}
                  >
                    <i aria-hidden="true" />
                    {signalLabel(signal.key, signal.label, t)}
                    <strong>{signal.count}</strong>
                  </span>
                ))}
              </div>
            )}
            <div className="submitted-content-grid">
              <div className="submitted-document" aria-label={t('report.submitted.documentText')}>
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
                          className={`submitted-highlight signal-style-${signalClassName(signal.key)}${isSelected ? ' is-selected' : ''}`}
                          style={{ '--signal-color': signal.color }}
                          title={signalDescription(signal.key, signal.description, t)}
                          onMouseEnter={() => setSelectedSegmentId(segment.id)}
                          onFocus={() => setSelectedSegmentId(segment.id)}
                          onClick={() => {
                            setSelectedSegmentId(segment.id);
                          }}
                        >
                          {segment.text}
                        </button>
                      );
                    })}
                  </p>
                ))}
              </div>
              <aside className="submitted-signal-panel" aria-label={t('report.submitted.selectedSignal')}>
                {selectedSegment?.primarySignal ? (
                  <>
                    <span className="submitted-panel-kicker">{selectedSegment.sentence_id}</span>
                    <h3>{signalLabel(selectedSegment.primarySignal.key, selectedSegment.primarySignal.label, t)}</h3>
                    <p>{signalDescription(selectedSegment.primarySignal.key, selectedSegment.primarySignal.description, t)}</p>
                    <div className="submitted-panel-meta">
                      {selectedSegment.primarySignal.score != null && (
                        <span>{t('report.submitted.signalStrength', { value: Math.round(selectedSegment.primarySignal.score) })}</span>
                      )}
                      {selectedSegment.primarySignal.tier && (
                        <span>{t('report.submitted.priority', { value: t(`report.severities.${selectedSegment.primarySignal.tier}`, { defaultValue: selectedSegment.primarySignal.tier }) })}</span>
                      )}
                      {selectedSegment.primarySignal.actionability && (
                        <span>{selectedSegment.primarySignal.actionability.replaceAll('_', ' ')}</span>
                      )}
                    </div>
                    {selectedSegment.signals.length > 1 && (
                      <div className="submitted-panel-stack">
                        <span>{t('report.submitted.alsoDetected')}</span>
                        {selectedSegment.signals.slice(1, 4).map((signal) => (
                          <em key={`${selectedSegment.id}-${signal.key}-${signal.finding_id}`}>{signalLabel(signal.key, signal.label, t)}</em>
                        ))}
                      </div>
                    )}
                    {selectedSegment.primarySignal.recommendation && (
                      <div className="submitted-panel-note">
                        <span>{t('report.submitted.recommendation')}</span>
                        <p>{selectedSegment.primarySignal.recommendation}</p>
                      </div>
                    )}
                  </>
                ) : (
                  <>
                    <span className="submitted-panel-kicker">{t('report.submitted.noSignal')}</span>
                    <h3>{t('report.submitted.mapReady')}</h3>
                    <p>{t('report.submitted.mapReadyBody')}</p>
                  </>
                )}
              </aside>
            </div>
          </section>
        )}

      </div>
    </main>
  );
}
