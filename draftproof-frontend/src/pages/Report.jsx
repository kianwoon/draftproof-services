// max-file-lines-guard: allow — Core scan-report page. Reduced 2442 → ~1900 (dead-code
// removal + safe component extractions into pages/report/*: TransformationScorecard,
// RewriteCompletionBand, SubmittedSignalGauge; helpers split into reportTransformation/
// reportAuthorship). The remaining <1500 reduction is the return-JSX / editor-modal
// surgery, deliberately deferred: it needs live app-render verification, not just the
// build gate. Owner-approved exception (2026-07-16). Re-evaluate if this grows further.
import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate, useParams, useSearchParams, Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { getReport, createRewrite, cancelRewrite, getRewriteStatus, getRewriteReport, getRewriteDownload, getScanStatus, startScanWithText, translateText, isAuthExpiryError } from '../api/draftproofApi';
import ErrorReload from '../components/ErrorReload';
import ConfirmDialog from '../components/ConfirmDialog';
import { useAuth } from '../context/AuthContext';
import { deleteReportDraft, getReportDraft, saveReportDraft } from '../utils/reportDraftStorage';
import { countWords, paidScanTokens } from '../utils/scanBilling';
import {
  requestBrowserNotificationPermission,
  showBrowserNotification,
} from '../utils/browserNotifications';
import RewriteNoticeDialog from './report/RewriteNoticeDialog';
import SignalHighlights from './report/SignalHighlights';
import FixFirstChecklist from './report/FixFirstChecklist';
import CriticalThinkingControl from './report/CriticalThinkingControl';
import DefenceCheck from '../components/DefenceCheck';
import ReportHero from './report/ReportHero';
import TransformationScorecard from './report/TransformationScorecard';
import MergedAuthorshipRisk, { TIER_TO_BAND } from './report/MergedAuthorshipRisk';
import ConsistencyRisk from './report/ConsistencyRisk';
import PolicyRiskView from './report/PolicyRiskView';
import RewriteCompletionBand from './report/RewriteCompletionBand';
import SubmittedSignalGauge from './report/SubmittedSignalGauge';
import useTextareaCaretOverlay from './report/useTextareaCaretOverlay';
import { buildTrackedDiff, trackedDiffToPlainText, trackedDiffToHtml } from './report/trackedDiff';
import {
  RESCAN_POLL_INTERVAL,
  RESCAN_MAX_POLLS,
  SUBMITTED_EDITOR_TRANSITION_MS,
  REWRITE_REPORT_RETRY_LIMIT,
  REWRITE_REPORT_RETRY_DELAY_MS,
  sleep,
  addScoreProfileFeature,
  groundingQualityComposite,
  submittedContentToText,
  findTextRange,
  changedTextRange,
  adjustHighlightedRange,
  adjustHighlightedRanges,
  buildOriginalSegmentRanges,
  highlightedEditorParts,
} from './report/reportTextUtils';
import {
  TIER_CONFIG,
  SEVERITY_CONFIG,
  formatDate,
  signalLabel,
  signalDescription,
  translatedSignal,
  translatedGroup,
  transformationLabel,
  confidenceLabel,
  translateAuthorshipRating,
  formatMetricPercent,
  calibratedReportAiScore,
  clampPercent,
  buildTransformationSignals,
  buildPairedTransformationSignals,
  groupTransformationSignals,
  getTransformationSignalImprovement,
  transformationSignalDirection,
  transformationSignalSeverity,
  transformationSignalFeatureMap,
  buildTransformationSummary,
  deriveAuthorshipRatingFallback,
  deriveCalibratedAuthorshipRating,
  getAuthorshipTone,
  formatSignedDelta,
  formatPlainScore,
  getOriginalDetectScan,
  getRewrittenDetectScan,
  hasRewriteComparisonData,
  mergeScanSummary,
  getScanDocumentContext,
  getScanTransformationSignals,
  getScanContributionSummary,
  mergeTransformationSummary,
  buildRewriteResultSummary,
  buildRewriteContributionOverride,
  buildSubmittedContentModel,
  buildParagraphSeverityBar,
  requiresRewriteAuthorReview,
  requiresRewriteExternalReview,
  isRewriteActive,
  normalizeRewriteProgressMessage,
  normalizeRewriteJob,
  formatElapsed,
  getRewriteProgressDetail,
  isReviewOnlyRewriteMessage,
  buildRewriteEventsUrl,
  aiLikelihoodBands,
  submissionRisk,
  policyRisk,
  rewriteDetectorVerdict,
  buildRepairSummary,
  groundingDiagnosis,
  GROUNDING_DIAGNOSIS_LEAD_ENABLED,
  GROUNDING_DIAGNOSIS_BUCKETS,
  buildFixFirstItems,
  EXTERNAL_ESTIMATE_DISPLAY_ENABLED,
} from './report/reportHelpers';

export default function Report() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const { refreshBalance, balance, logout } = useAuth();
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
  const [selectedParagraphId, setSelectedParagraphId] = useState(null);
  const [lockedParagraphId, setLockedParagraphId] = useState(null);
  const [submittedEditorOpen, setSubmittedEditorOpen] = useState(false);
  const [submittedEditorClosing, setSubmittedEditorClosing] = useState(false);
  const [submittedDraftText, setSubmittedDraftText] = useState('');
  const [submittedDraftLoaded, setSubmittedDraftLoaded] = useState(false);
  const [submittedDraftStatus, setSubmittedDraftStatus] = useState('idle');
  const [submittedDraftUpdatedAt, setSubmittedDraftUpdatedAt] = useState(null);
  const [submittedRescanBusy, setSubmittedRescanBusy] = useState(false);
  const [submittedRescanStatus, setSubmittedRescanStatus] = useState(null);
  const [submittedRescanError, setSubmittedRescanError] = useState(null);
  const [submittedRescanNeedsTokens, setSubmittedRescanNeedsTokens] = useState(false);
  const [submittedTranslateBusy, setSubmittedTranslateBusy] = useState(false);
  const [submittedTranslateError, setSubmittedTranslateError] = useState(null);
  const [submittedPreTranslateText, setSubmittedPreTranslateText] = useState(null);
  const [submittedHighlightRanges, setSubmittedHighlightRanges] = useState({});
  const [submittedTrackedCopyStatus, setSubmittedTrackedCopyStatus] = useState('idle');
  const rewritePollRef = useRef(null);
  const rewriteEventSourceRef = useRef(null);
  const rewriteTimerStartRef = useRef(null);
  const watchedRewriteIdsRef = useRef(new Set());
  const notifiedRewriteIdsRef = useRef(new Set());
  const submittedEditorRef = useRef(null);
  const autoOpenedEditorRef = useRef(false);
  const submittedHighlightRef = useRef(null);
  const submittedDocumentRef = useRef(null);
  const submittedEditorCloseTimerRef = useRef(null);
  const submittedTrackedCopyTimerRef = useRef(null);
  const {
    caret: submittedCaret,
    hideCaret: hideSubmittedCaret,
    scheduleCaretUpdate: scheduleSubmittedCaretUpdate,
  } = useTextareaCaretOverlay(submittedEditorRef);

  const clearSubmittedEditorCloseTimer = useCallback(() => {
    if (submittedEditorCloseTimerRef.current) {
      window.clearTimeout(submittedEditorCloseTimerRef.current);
      submittedEditorCloseTimerRef.current = null;
    }
  }, []);

  const clearSubmittedTrackedCopyTimer = useCallback(() => {
    if (submittedTrackedCopyTimerRef.current) {
      window.clearTimeout(submittedTrackedCopyTimerRef.current);
      submittedTrackedCopyTimerRef.current = null;
    }
  }, []);

  const closeSubmittedEditor = useCallback((immediate = false) => {
    clearSubmittedEditorCloseTimer();
    const reduceMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    if (immediate || reduceMotion) {
      setSubmittedEditorOpen(false);
      setSubmittedEditorClosing(false);
      return;
    }
    setSubmittedEditorClosing(true);
    submittedEditorCloseTimerRef.current = window.setTimeout(() => {
      setSubmittedEditorOpen(false);
      setSubmittedEditorClosing(false);
      submittedEditorCloseTimerRef.current = null;
    }, SUBMITTED_EDITOR_TRANSITION_MS);
  }, [clearSubmittedEditorCloseTimer]);

  const openSubmittedEditor = useCallback(() => {
    clearSubmittedEditorCloseTimer();
    setSubmittedEditorOpen(true);
    setSubmittedEditorClosing(false);
  }, [clearSubmittedEditorCloseTimer]);

  const notifyRewriteCompleted = useCallback((job) => {
    if (!job?.id || notifiedRewriteIdsRef.current.has(job.id)) return;
    if (!watchedRewriteIdsRef.current.has(job.id)) return;

    const shown = showBrowserNotification({
      title: t('report.rewrite.notificationTitle'),
      body: t('report.rewrite.notificationBody'),
      tag: `draftproof-rewrite-${job.id}`,
      url: `/rewrite/${job.id}`,
    });

    if (shown) {
      notifiedRewriteIdsRef.current.add(job.id);
    }
  }, [t]);

  const showReviewOnlyRewriteNotice = useCallback((message) => {
    setRewriteJob(null);
    setRewriteError(null);
    setRewriteLoading(false);
    setRewriteStartedHere(false);
    setRewriteNotice({
      title: t('report.rewrite.noRewriteableTitle'),
      message: isReviewOnlyRewriteMessage(message) && String(message).includes('token')
        ? String(message).replace(/\bTokens\b/g, 'Credits').replace(/\btokens\b/g, 'credits').replace(/\btoken\b/g, 'credit')
        : t('report.rewrite.noRewriteableMessage'),
    });
  }, [t]);

  const syncRewriteJob = useCallback((job) => {
    const normalizedJob = normalizeRewriteJob(job, t);
    if (normalizedJob?.id && isRewriteActive(normalizedJob.status)) {
      watchedRewriteIdsRef.current.add(normalizedJob.id);
    }
    setRewriteJob(normalizedJob);
    if (normalizedJob?.status && !['failed', 'canceled'].includes(normalizedJob.status)) {
      setRewriteError(null);
    }
    if (normalizedJob?.status === 'completed') {
      setReport((prev) => prev ? { ...prev, rewrite: normalizedJob } : prev);
      setRewriteStartedHere(false);
      notifyRewriteCompleted(normalizedJob);
    }
  }, [notifyRewriteCompleted, t]);

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
            watchedRewriteIdsRef.current.add(data.rewrite.id);
            connectRewriteEvents(data.rewrite.id);
          }
        }
      })
      .catch((err) => {
        if (err.name === 'AbortError' || err.code === 'ERR_CANCELED') return;
        // Session expired → the global 401 interceptor is already redirecting to
        // /signin; skip the ErrorReload countdown so the redirect is immediate.
        if (isAuthExpiryError(err)) return;
        setError(err.response?.data?.detail || t('report.loadFailed'));
      })
      .finally(() => setLoading(false));
    return () => {
      ac.abort();
      closeRewriteEventSource();
    };
  }, [id, closeRewriteEventSource, connectRewriteEvents, t]);

  useEffect(() => {
    closeSubmittedEditor(true);
    setSubmittedDraftText('');
    setSubmittedDraftLoaded(false);
    setSubmittedDraftStatus('idle');
    setSubmittedDraftUpdatedAt(null);
    setSubmittedRescanBusy(false);
    setSubmittedRescanStatus(null);
    setSubmittedRescanError(null);
    setSubmittedRescanNeedsTokens(false);
    setSubmittedTranslateBusy(false);
    setSubmittedTranslateError(null);
    setSubmittedPreTranslateText(null);
    setSubmittedHighlightRanges({});
    setLockedParagraphId(null);
  }, [id, closeSubmittedEditor]);

  useEffect(() => () => {
    clearSubmittedEditorCloseTimer();
    clearSubmittedTrackedCopyTimer();
  }, [clearSubmittedEditorCloseTimer, clearSubmittedTrackedCopyTimer]);

  // Which text the "Manual Rewrite / Correction" editor works on:
  //   • before any rewrite          → the original submission
  //   • after a completed rewrite    → the REWRITTEN draft (rewriteResultReport.final_text)
  // final_text loads asynchronously (see the loadRewriteReport effect), so editingRewriteDraft
  // only flips true once it is present — that keeps the editor from ever seeding blank and lets
  // us gate the open buttons until the correct baseline is ready. Drafts are namespaced per mode
  // so an original-mode draft and a rewrite-mode draft never clobber each other in IndexedDB.
  const completedRewriteJob = rewriteJob?.status === 'completed' ? rewriteJob : report?.rewrite;
  const rewriteFinalText = (completedRewriteJob?.id && completedRewriteJob.status === 'completed')
    ? (rewriteResultReport?.final_text || '')
    : '';
  const editingRewriteDraft = Boolean(rewriteFinalText);
  const submittedDraftStorageKey = editingRewriteDraft
    ? `${id}:rewrite:${completedRewriteJob.id}`
    : id;

  useEffect(() => {
    if (!report) return undefined;

    let cancelled = false;
    const reportIssues = Array.isArray(report.issues) ? report.issues : [];
    const model = buildSubmittedContentModel({ ...report, issues: reportIssues });
    const originalText = submittedContentToText(model);
    const baselineText = editingRewriteDraft ? rewriteFinalText : originalText;

    setSubmittedDraftLoaded(false);
    setSubmittedDraftText(baselineText);
    setSubmittedDraftStatus('idle');
    setSubmittedDraftUpdatedAt(null);

    getReportDraft(submittedDraftStorageKey)
      .then((draft) => {
        if (cancelled) return;
        if (draft?.text) {
          setSubmittedDraftText(draft.text);
          setSubmittedDraftStatus('saved');
          setSubmittedDraftUpdatedAt(draft.updatedAt || null);
        }
        setSubmittedDraftLoaded(true);
      })
      .catch(() => {
        if (!cancelled) {
          setSubmittedDraftLoaded(true);
          setSubmittedDraftStatus('error');
        }
      });

    return () => {
      cancelled = true;
    };
  }, [id, report?.id, editingRewriteDraft, rewriteFinalText, submittedDraftStorageKey]);

  useEffect(() => {
    if (!report || !submittedDraftLoaded) return undefined;

    const reportIssues = Array.isArray(report.issues) ? report.issues : [];
    const model = buildSubmittedContentModel({ ...report, issues: reportIssues });
    const originalText = submittedContentToText(model);
    const baselineText = editingRewriteDraft ? rewriteFinalText : originalText;
    if (submittedDraftText === baselineText) {
      setSubmittedDraftStatus('idle');
      setSubmittedDraftUpdatedAt(null);
      deleteReportDraft(submittedDraftStorageKey).catch(() => {});
      return undefined;
    }

    setSubmittedDraftStatus('saving');
    const timer = window.setTimeout(() => {
      saveReportDraft(submittedDraftStorageKey, submittedDraftText)
        .then((draft) => {
          setSubmittedDraftStatus('saved');
          setSubmittedDraftUpdatedAt(draft?.updatedAt || null);
        })
        .catch(() => {
          setSubmittedDraftStatus('error');
        });
    }, 650);

    return () => window.clearTimeout(timer);
  }, [id, report?.id, submittedDraftLoaded, submittedDraftText, editingRewriteDraft, rewriteFinalText, submittedDraftStorageKey]);

  useEffect(() => {
    if (!submittedEditorOpen) return undefined;

    const handleKeyDown = (event) => {
      if (event.key === 'Escape' && !submittedRescanBusy) {
        closeSubmittedEditor();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [closeSubmittedEditor, submittedEditorOpen, submittedRescanBusy]);

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
    let retryTimer = null;

    const loadRewriteReport = (attempt = 0) => {
      getRewriteReport(completedRewrite.id)
        .then(({ data }) => {
          if (cancelled) return;
          setRewriteResultReport(data);
          setRewriteResultSummary(buildRewriteResultSummary(data));
          if (!hasRewriteComparisonData(data) && attempt < REWRITE_REPORT_RETRY_LIMIT) {
            retryTimer = window.setTimeout(
              () => loadRewriteReport(attempt + 1),
              REWRITE_REPORT_RETRY_DELAY_MS
            );
          }
        })
        .catch(() => {
          if (cancelled) return;
          if (attempt < REWRITE_REPORT_RETRY_LIMIT) {
            retryTimer = window.setTimeout(
              () => loadRewriteReport(attempt + 1),
              REWRITE_REPORT_RETRY_DELAY_MS
            );
            return;
          }
          setRewriteResultReport(null);
          setRewriteResultSummary(null);
        });
    };

    loadRewriteReport();

    return () => {
      cancelled = true;
      if (retryTimer) {
        window.clearTimeout(retryTimer);
      }
    };
  }, [rewriteJob, report?.rewrite]);

  const activeRewriteForTimer = rewriteJob || report?.rewrite;
  const currentRewrite = activeRewriteForTimer;
  const rewriteInProgress = isRewriteActive(currentRewrite?.status);
  const hasCompletedRewrite = currentRewrite?.status === 'completed';
  const hasRewriteResult = hasCompletedRewrite && Boolean(currentRewrite?.id);
  // Manual editing stays available after a completed rewrite — the user refines the
  // rewritten draft (or, before any rewrite, their original submission) and re-scans,
  // which spins up a fresh /report/{id}. Only an in-flight rewrite locks editing, to
  // avoid editing against a moving baseline.
  const canEditSubmittedDraft = !rewriteInProgress;
  // Don't let the user open the editor until its baseline is actually loaded: the
  // original text is always ready, but after a rewrite we wait for its report to load
  // so we seed the rewritten text (not the original) — otherwise a click right after a
  // rewrite would edit the original by mistake. If a completed rewrite carries no
  // final_text (e.g. the original was preserved), editingRewriteDraft stays false and
  // the editor falls back to the original submission rather than going dead.
  const submittedEditorReady = canEditSubmittedDraft && (!hasRewriteResult || Boolean(rewriteResultReport));
  // On-page "Manual Rewrite / Correction" entry buttons show only BEFORE a rewrite.
  // Once a rewrite exists, that entry point lives on the /rewrite page, which routes
  // back here with ?edit=1 to auto-open this same (rewrite-aware) editor — so we hide
  // the on-page buttons after a rewrite while the editor itself stays reachable.
  const showSubmittedEditEntry = submittedEditorReady && !hasRewriteResult;

  // Auto-open the editor when arriving from the /rewrite page's "Manual Rewrite /
  // Correction" button (/report/{id}?edit=1). Must live ABOVE the loading/error/!report
  // early returns below so the hook is called unconditionally (rules of hooks). Wait
  // until the editor baseline is ready (submittedEditorReady gates on the rewrite report
  // having loaded, so the seed effect has loaded the rewritten text), open the modal
  // once via the stable openSubmittedEditor callback, then strip the param so a
  // refresh/close won't reopen it.
  useEffect(() => {
    if (searchParams.get('edit') !== '1' || autoOpenedEditorRef.current) return;
    if (!submittedEditorReady) return;
    autoOpenedEditorRef.current = true;
    openSubmittedEditor();
    const next = new URLSearchParams(searchParams);
    next.delete('edit');
    setSearchParams(next, { replace: true });
  }, [searchParams, submittedEditorReady, openSubmittedEditor, setSearchParams]);
  const rewriteTimerActive = rewriteLoading || rewriteInProgress;

  useEffect(() => {
    if (canEditSubmittedDraft || !submittedEditorOpen) return;
    closeSubmittedEditor();
  }, [canEditSubmittedDraft, closeSubmittedEditor, submittedEditorOpen]);

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
  const originalScanContributionSummary = getScanContributionSummary(originalComparisonScan);
  const originalContributionOverride = buildRewriteContributionOverride(rewriteResultSummary, 'original') || originalScanContributionSummary;
  const originalAiComponents = originalComparisonBadge.ai_components || {};
  const originalWritingComponents = originalComparisonBadge.writing_components || {};
  addScoreProfileFeature(authorshipFeatures, 'ai_likelihood', originalComparisonAiScore);
  addScoreProfileFeature(authorshipFeatures, 'topk_pattern_raw', originalAiComponents.topk_pattern_raw ?? originalAiComponents.topk_pattern);
  addScoreProfileFeature(authorshipFeatures, 'topk_calibrated_risk', originalAiComponents.topk_calibrated_risk);
  addScoreProfileFeature(authorshipFeatures, 'adjusted_ai_risk', originalContributionOverride?.rawAdjustedAiRisk ?? originalContributionOverride?.adjustedAiRisk);
  addScoreProfileFeature(authorshipFeatures, 'calibrated_ai_risk', originalContributionOverride?.adjustedAiRisk);
  addScoreProfileFeature(authorshipFeatures, 'human_anchor_score', originalContributionOverride?.humanContribution);
  addScoreProfileFeature(authorshipFeatures, 'human_anchor_discount', originalContributionOverride?.humanAnchorDiscount);
  addScoreProfileFeature(authorshipFeatures, 'calibration_confidence', originalContributionOverride?.calibrationConfidence);
  addScoreProfileFeature(authorshipFeatures, 'reporting_suppression', originalContributionOverride?.reportingSuppression);
  addScoreProfileFeature(authorshipFeatures, 'grounding_quality_risk', groundingQualityComposite(originalWritingComponents));
  addScoreProfileFeature(authorshipFeatures, 'citation_grounding_risk', originalWritingComponents.source_grounding_risk ?? originalWritingComponents.unsupported_claim_risk ?? originalWritingComponents.citation_grounding_risk);
  const transformationSignals = buildTransformationSignals(authorshipFeatures, transformationSignalMetadata);
  const transformationSummary = transformation
    ? mergeTransformationSummary(
      buildTransformationSummary(authorshipFeatures, transformationSignals, originalContributionOverride, t),
      originalScanContributionSummary
    )
    : null;
  const rewrittenScan = getRewrittenDetectScan(rewriteResultReport) || {};
  const rewrittenBadge = rewrittenScan.ai_risk_badge || {};
  const rewrittenAiScore = rewrittenScan.ai_score ?? rewrittenBadge.ai_likelihood_score ?? rewriteResultSummary?.rewrite_risk ?? null;
  // Honest, external-facing expectation. Prefer the rewritten content's estimate (that is what users
  // cross-check on third-party sites); fall back to the main report badge for plain scans.
  const rewrittenTransformation = rewrittenBadge.transformation_classification || null;
  const rewrittenTransformationSignalMetadata = getScanTransformationSignals(rewrittenScan);
  const rewrittenTransformationFeatureFallbacks = transformationSignalFeatureMap(rewrittenTransformationSignalMetadata);
  const rewrittenAuthorshipFeatures = {
    ...rewrittenTransformationFeatureFallbacks,
    ...(rewrittenTransformation?.features || {}),
  };
  const rewrittenScanContributionSummary = getScanContributionSummary(rewrittenScan);
  const rewrittenContributionOverride = buildRewriteContributionOverride(rewriteResultSummary, 'rewritten') || rewrittenScanContributionSummary;
  const rewrittenAiComponents = rewrittenBadge.ai_components || {};
  const rewrittenWritingComponents = rewrittenBadge.writing_components || {};
  addScoreProfileFeature(rewrittenAuthorshipFeatures, 'ai_likelihood', rewrittenAiScore);
  addScoreProfileFeature(rewrittenAuthorshipFeatures, 'topk_pattern_raw', rewrittenAiComponents.topk_pattern_raw ?? rewrittenAiComponents.topk_pattern);
  addScoreProfileFeature(rewrittenAuthorshipFeatures, 'topk_calibrated_risk', rewrittenAiComponents.topk_calibrated_risk);
  addScoreProfileFeature(rewrittenAuthorshipFeatures, 'adjusted_ai_risk', rewrittenContributionOverride?.rawAdjustedAiRisk ?? rewrittenContributionOverride?.adjustedAiRisk);
  addScoreProfileFeature(rewrittenAuthorshipFeatures, 'calibrated_ai_risk', rewrittenContributionOverride?.adjustedAiRisk);
  addScoreProfileFeature(rewrittenAuthorshipFeatures, 'human_anchor_score', rewrittenContributionOverride?.humanContribution);
  addScoreProfileFeature(rewrittenAuthorshipFeatures, 'human_anchor_discount', rewrittenContributionOverride?.humanAnchorDiscount);
  addScoreProfileFeature(rewrittenAuthorshipFeatures, 'calibration_confidence', rewrittenContributionOverride?.calibrationConfidence);
  addScoreProfileFeature(rewrittenAuthorshipFeatures, 'reporting_suppression', rewrittenContributionOverride?.reportingSuppression);
  addScoreProfileFeature(rewrittenAuthorshipFeatures, 'grounding_quality_risk', groundingQualityComposite(rewrittenWritingComponents));
  addScoreProfileFeature(rewrittenAuthorshipFeatures, 'citation_grounding_risk', rewrittenWritingComponents.source_grounding_risk ?? rewrittenWritingComponents.unsupported_claim_risk ?? rewrittenWritingComponents.citation_grounding_risk);
  const rewrittenTransformationSignals = buildTransformationSignals(
    rewrittenAuthorshipFeatures,
    rewrittenTransformationSignalMetadata
  );
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
  const rewrittenWritingScore = rewrittenScan.writing_score ?? rewrittenBadge.writing_quality_score ?? null;
  // Once a rewrite's report has loaded, the hero header reflects the REWRITTEN draft
  // instead of the original submission. Gated on rewriteResultReport (not just
  // hasRewriteResult) so the numbers never flicker original→rewritten mid-load. The
  // tier shown is the rewrite's real tier — it usually stays flagged — and the header
  // is labeled "rewritten" (rewrittenHeroView) so a lower number can't read as a
  // clean/final verdict (expose-the-ugly-side: no green-washing).
  const rewrittenHeroView = hasRewriteResult && Boolean(rewriteResultReport);
  // Derive the rewritten tier from the SAME DraftProof band the comparison panel shows
  // (aiLikelihoodBands), not rewrittenBadge.tier — that field is empty on the rewritten
  // scan, so `|| report.tier` was silently falling back to the ORIGINAL tier (e.g. red /
  // "Critical Risk") under a "rewritten" header, contradicting the amber band shown below.
  const rewrittenBandTierKey = (aiLikelihoodBands(rewrittenBadge).draftproof?.tier || '').toLowerCase();
  const rewrittenTierKey = rewrittenBandTierKey || rewrittenBadge.tier || report.tier;
  const heroReport = rewrittenHeroView
    ? {
      ...report,
      tier: rewrittenTierKey,
      word_count: countWords(rewriteResultReport?.final_text) || report.word_count,
    }
    : report;
  const heroTier = rewrittenHeroView ? (TIER_CONFIG[rewrittenTierKey] || tier) : tier;
  // Additive Submission-risk view for the hero (null when the diagnosis abstained,
  // or when an older report predates the field — the hero then leads as before).
  const heroSubmissionRisk = submissionRisk(rewrittenHeroView ? rewrittenBadge : originalComparisonBadge);
  // The merged hero card must draw EVERY section from the same scan's badge. Only `sr`
  // used to swap on rewrittenHeroView, so the card mixed the ORIGINAL headline/breakdown
  // with the REWRITTEN "where the risk sits" rows (and the original deep-scan proportion
  // in the explainer) — three scans' numbers on one card.
  const heroBadge = rewrittenHeroView ? rewrittenBadge : badge;
  const calibratedAuthorshipRisk = clampPercent(authorshipFeatures.calibrated_ai_risk);
  const topkPatternScore = clampPercent(originalComparisonBadge.ai_components?.topk_pattern_raw ?? originalComparisonBadge.ai_components?.topk_pattern);
  const topkCalibratedRisk = clampPercent(originalComparisonBadge.ai_components?.topk_calibrated_risk);
  const rewrittenCalibratedAuthorshipRisk = clampPercent(rewrittenAuthorshipFeatures.calibrated_ai_risk)
    ?? calibratedReportAiScore(rewrittenAiScore);
  const rewrittenTopkPatternScore = clampPercent(rewrittenBadge.ai_components?.topk_pattern_raw ?? rewrittenBadge.ai_components?.topk_pattern);
  const rewrittenTopkCalibratedRisk = clampPercent(rewrittenBadge.ai_components?.topk_calibrated_risk);
  const rewrittenDocumentContext = getScanDocumentContext(rewrittenScan);
  const storedAuthorshipRating = badge.authorship_rating || deriveAuthorshipRatingFallback(
    aiScore,
    badge.tier || report.tier,
    writingScore,
    badge.ai_components,
    badge.writing_components
  ) || {};
  // When the backend produced a DeBERTa-authoritative rating, use it directly — the frontend's
  // deriveCalibratedAuthorshipRating recomputes from perplexity components (calibratedAuthorshipRisk),
  // which would override the DeBERTa score with a perplexity one (~33% Moderate vs 18.75 amber).
  const isDebertaAuthoritative = badge.signal_source === 'deberta_authoritative';
  const authorshipRating = translateAuthorshipRating(
    isDebertaAuthoritative
      ? (badge.authorship_rating || storedAuthorshipRating)
      : (deriveCalibratedAuthorshipRating(
          calibratedAuthorshipRisk,
          topkPatternScore,
          topkCalibratedRisk,
          authorshipFeatures,
          originalDocumentContext,
          originalComparisonBadge.ai_components?.topk_calibration_eligible
        ) || storedAuthorshipRating),
    t,
  );
  const authorshipTone = getAuthorshipTone(authorshipRating);
  const authorshipRatingFullLabel = authorshipRating.label || badge.authorship_rating_label || null;
  const authorshipRatingLabel = authorshipRating.short_label || authorshipRatingFullLabel;
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
  // The seal verdict/tone/detail come from the detector band (rewriteDetectorVerdict). The
  // underlying authorship classification is retained as the verdict's hover-tooltip (fullLabel).
  const rewrittenAuthorshipRatingFullLabel = rewrittenAuthorshipRating.label || rewrittenBadge.authorship_rating_label || null;
  const rewrittenRequiresAuthorReview = requiresRewriteAuthorReview(rewriteResultSummary);
  const rewrittenRequiresExternalReview = requiresRewriteExternalReview(rewriteResultSummary);
  // The rewritten rating VERDICT tracks the detector reality (external/Turnitin band), not the
  // authorship "GOOD" -- users read "GOOD" as "Turnitin-safe", which no rewrite can honestly promise.
  const rewrittenDetectorVerdict = rewriteDetectorVerdict(aiLikelihoodBands(rewrittenBadge).external?.band, t);
  const manualReviewTone = { color: '#92400e', bg: '#fffbeb' };
  // Column rating chips rate BOTH sides on the V7 fused scale (same band labels and
  // palette as the scan card's verdict chip) so original vs rewritten is one comparable
  // vocabulary. Previously the original said an authorship "Good" while the rewritten
  // said a detector-band "Detector risk low" — different scales, and "Good" read as the
  // better outcome even when the fused score improved (live feedback 2026-07-07).
  // Palette mirrors .merged-verdict-chip.is-<tier> in 06-report-overview.css.
  const FUSED_TIER_TONES = {
    green: { color: '#15803d', bg: '#dcfce7' },
    amber: { color: '#b45309', bg: '#fef3c7' },
    orange: { color: '#c2410c', bg: '#ffedd5' },
    red: { color: '#b91c1c', bg: '#fee2e2' },
  };
  const fusedColumnRatingBadge = (variantBadge, caption) => {
    const ta = variantBadge?.tier_authority;
    if (!ta || typeof ta.fused_score !== 'number') return null;
    const tierKey = String(variantBadge.tier || '').toLowerCase();
    const band = TIER_TO_BAND[tierKey];
    if (!band) return null;
    return {
      caption,
      label: `${t(`report.authorshipBreakdown.fusedHeadline.bands.${band}`)} · ${Math.round(ta.fused_score)}%`,
      fullLabel: t('report.authorshipBreakdown.fusedHeadline.evidence', {
        composite: Math.round(ta.composite_score ?? 0),
        deepScan: Math.round((ta.proportion || 0) * 100),
      }),
      tone: FUSED_TIER_TONES[tierKey],
    };
  };
  const originalColumnRatingBadge = fusedColumnRatingBadge(
    originalComparisonBadge,
    t('report.transformation.originalRating'),
  ) || {
    caption: t('report.transformation.originalRating'),
    label: authorshipRatingLabel,
    fullLabel: authorshipRatingFullLabel,
    tone: authorshipTone,
  };
  const rewrittenColumnRatingBadge = rewrittenRequiresAuthorReview
    ? {
      caption: t('report.transformation.rewrittenRating'),
      label: t('rewritePage.reviewRequired'),
      fullLabel: t('rewritePage.authorReviewTitle'),
      tone: manualReviewTone,
    }
    : rewrittenRequiresExternalReview
    ? {
      caption: t('report.transformation.rewrittenRating'),
      label: t('rewritePage.reviewRequired'),
      fullLabel: t('rewritePage.externalReviewTitle'),
      tone: manualReviewTone,
    }
    : fusedColumnRatingBadge(rewrittenBadge, t('report.transformation.rewrittenRating')) || {
      caption: t('report.transformation.rewrittenRating'),
      label: rewrittenDetectorVerdict.label,
      fullLabel: rewrittenAuthorshipRatingFullLabel || rewrittenDetectorVerdict.label,
      tone: rewrittenDetectorVerdict.tone,
    };
  const issueCounts = { critical: 0, high: 0, medium: 0, low: 0, info: 0 };
  issues.forEach((iss) => { if (issueCounts[iss.severity] !== undefined) issueCounts[iss.severity]++; });
  const normalizedReport = { ...report, issues };
  const submittedContent = buildSubmittedContentModel(normalizedReport);
  // Per-paragraph severity heatmap bar (finding-tier-weighted density, proportional width).
  const paragraphSeverityBar = buildParagraphSeverityBar(submittedContent.paragraphs, submittedContent.sentenceIssueTags);
  const selectedParagraph = (
    submittedContent.paragraphs.find((paragraph) => paragraph.id === selectedParagraphId) ||
    submittedContent.paragraphs.find((paragraph) => paragraph.signals.length > 0) ||
    null
  );

  // The API nests the scan report under results_json (only ai_risk_badge etc. are hoisted to top level).
  const authorshipEvidence = report?.results_json?.authorship_evidence || report?.authorship_evidence || null;
  // Stylometric-consistency "Writing-style outliers" panel data (poc/report/
  // consistency_panel.py's compose_consistency_display — a TOP-LEVEL report
  // key, unlike claim_graph_display which nests under authorship_evidence).
  // null when DRAFTPROOF_CONSISTENCY is off / no paragraph was flagged / older
  // report — ConsistencyRisk renders nothing in that case.
  const consistencyDisplay = report?.results_json?.consistency_display || report?.consistency_display || null;
  // Defence-readiness check (Task 8): reuse the SAME question array the read-only
  // CriticalThinkingControl panel below already reads from badge.critical_thinking_control
  // — one source of truth, no separate fetch/derivation. DefenceCheck itself renders
  // nothing when this is empty or when the backing GET 404s (DRAFTPROOF_DEFENCE_CHECK off).
  const defenceQuestions = Array.isArray(badge?.critical_thinking_control?.questions)
    ? badge.critical_thinking_control.questions
    : [];

  const selectAndScrollParagraph = (paragraphId) => {
    setSelectedParagraphId(paragraphId);
    if (!submittedDocumentRef.current) return;
    const container = submittedDocumentRef.current;
    const btn = container.querySelector(`[data-paragraph-id="${paragraphId}"]`);
    if (!btn) return;
    const containerRect = container.getBoundingClientRect();
    const btnRect = btn.getBoundingClientRect();
    const targetScrollTop = container.scrollTop + (btnRect.top - containerRect.top) - 16;
    container.scrollTo({ top: targetScrollTop, behavior: 'smooth' });
  };
  const lockAndScrollParagraph = (paragraphId) => {
    setLockedParagraphId(paragraphId);
    selectAndScrollParagraph(paragraphId);
  };
  const previewParagraph = (paragraphId) => {
    if (lockedParagraphId) return;
    setSelectedParagraphId(paragraphId);
  };
  const highlightedParagraphs = submittedContent.paragraphs.filter((paragraph) => paragraph.signals.length > 0);
  const selectedHighlightIndex = highlightedParagraphs.findIndex((paragraph) => paragraph.id === selectedParagraph?.id);
  const selectAdjacentHighlightedParagraph = (direction) => {
    if (!highlightedParagraphs.length) return;
    const currentIndex = selectedHighlightIndex >= 0 ? selectedHighlightIndex : 0;
    const nextIndex = (currentIndex + direction + highlightedParagraphs.length) % highlightedParagraphs.length;
    lockAndScrollParagraph(highlightedParagraphs[nextIndex].id);
  };
  // Editor-panel guidance: DeBERTa-native (the paragraph model's readerSummary/recommendation),
  // NOT the perplexity-fed LLM explanation. The learned classifier is the sole methodology for
  // this section; the old explanation (reader_summary/main_issue/why_flagged/rewrite_hint over
  // perplexity findings) leaked "predictable, generic phrasing" advice next to a DeBERTa highlight.
  const selectedReaderSummary = selectedParagraph?.readerSummary || '';
  const selectedRecommendation = selectedParagraph?.recommendation || '';
  // Per-paragraph Critical Thinking tag (deterministic; from the scan report).
  const selectedCriticalThinking = (() => {
    const rows = badge?.critical_thinking_control?.paragraphs;
    if (!Array.isArray(rows) || !selectedParagraph?.id) return null;
    return rows.find((row) => row.paragraph_id === selectedParagraph.id) || null;
  })();
  const selectedSecondarySignals = Array.isArray(selectedParagraph?.signals)
    ? selectedParagraph.signals.filter((signal) => signal && signal.key !== selectedParagraph.primarySignal?.key)
    : [];
  const originalSubmittedText = submittedContentToText(submittedContent);
  // In rewrite mode the editor's baseline is the rewritten draft, so "changed" and the
  // tracked-changes view are measured against it (not the original submission).
  const submittedBaselineText = editingRewriteDraft ? rewriteFinalText : originalSubmittedText;
  const submittedDraftChanged = submittedDraftText !== submittedBaselineText;
  const submittedTrackedDiff = submittedEditorOpen
    ? buildTrackedDiff(submittedBaselineText, submittedDraftText)
    : [];
  const affectedParagraphs = highlightedParagraphs;
  const selectedParagraphDraftStatus = selectedParagraph?.text && submittedDraftText.includes(selectedParagraph.text)
    ? t('report.submitted.editor.paragraphUnchanged')
    : t('report.submitted.editor.paragraphEdited');
  const selectedSignalStrength = clampPercent(selectedParagraph?.primarySignal?.score);
  const submittedHighlightRange = selectedParagraph?.id ? submittedHighlightRanges[selectedParagraph.id] : null;
  const submittedEditorHighlightParts = highlightedEditorParts(submittedDraftText, submittedHighlightRange);
  const submittedDraftWordCount = countWords(submittedDraftText);
  const submittedDraftTokensRequired = paidScanTokens(submittedDraftWordCount);
  const repairSummary = buildRepairSummary({
    report,
    submittedContent,
    authorshipEvidence,
    transformationSummary,
    status: authorshipRatingLabel,
    pattern: transformation,
    t,
  });
  const fixFirstItems = buildFixFirstItems({ submittedContent, t });
  // Scale the repair-plan framing to the tier: a Low/green report reads as
  // "polish", not "repair" (owner review 2026-07-05).
  const fixFirstLowTone = ['green', 'low'].includes(String(report.tier || '').toLowerCase());
  // The "Repair Summary" band and "Repair Plan" checklist describe the ORIGINAL
  // submission's findings. Once a rewrite has completed, the rewrite completion band
  // and the (rewrite-aware) editor drive the flow, so the original-content guidance is
  // stale — suppress it rather than show risks that no longer match what's on screen.
  const showOriginalRepairGuidance = !hasRewriteResult;

  const resolveSubmittedParagraphRange = (paragraph, existingRanges = submittedHighlightRanges) => {
    if (!paragraph?.id || !paragraph.text) return null;
    return (
      findTextRange(submittedDraftText, paragraph.text) ||
      existingRanges[paragraph.id]
    );
  };

  const buildSubmittedHighlightRanges = (existingRanges = submittedHighlightRanges) => {
    const nextRanges = { ...(existingRanges || {}) };
    affectedParagraphs.forEach((paragraph) => {
      if (nextRanges[paragraph.id]) return;
      const range = resolveSubmittedParagraphRange(paragraph, nextRanges);
      if (range) nextRanges[paragraph.id] = { ...range, segmentId: paragraph.id };
    });
    return nextRanges;
  };

  const focusParagraphInSubmittedEditor = (paragraph) => {
    const editor = submittedEditorRef.current;
    if (!editor || !paragraph?.text) return;
    const range = resolveSubmittedParagraphRange(paragraph);
    setSubmittedHighlightRanges((current) => {
      if (!range) return current;
      return { ...current, [paragraph.id]: { ...range, segmentId: paragraph.id } };
    });
    editor.focus();
    if (range) {
      editor.setSelectionRange(range.start, range.start);
    }
  };

  const openSubmittedEditorForParagraph = (paragraph = selectedParagraph) => {
    const targetParagraph = paragraph?.signals?.length
      ? paragraph
      : affectedParagraphs[0] || paragraph;

    openSubmittedEditor();
    setSubmittedRescanError(null);

    if (targetParagraph?.id) {
      setSelectedParagraphId(targetParagraph.id);
    }

    setSubmittedHighlightRanges((current) => {
      const hydratedRanges = buildSubmittedHighlightRanges(current);
      if (!targetParagraph?.id) return hydratedRanges;

      const range = resolveSubmittedParagraphRange(targetParagraph, hydratedRanges);
      return range
        ? {
          ...hydratedRanges,
          [targetParagraph.id]: { ...range, segmentId: targetParagraph.id },
        }
        : hydratedRanges;
    });

    if (targetParagraph) {
      requestAnimationFrame(() => focusParagraphInSubmittedEditor(targetParagraph));
    }
  };


  const syncSubmittedHighlightScroll = () => {
    if (!submittedEditorRef.current || !submittedHighlightRef.current) return;
    submittedHighlightRef.current.scrollTop = submittedEditorRef.current.scrollTop;
    submittedHighlightRef.current.scrollLeft = submittedEditorRef.current.scrollLeft;
    scheduleSubmittedCaretUpdate();
  };

  const copySelectedParagraphGuidance = async () => {
    if (!selectedParagraph?.primarySignal) return;
    // DeBERTa-native guidance only (reader summary + recommendation). No perplexity text.
    const parts = [
      selectedReaderSummary ? `${t('report.submitted.readerSummary')}: ${selectedReaderSummary}` : '',
      selectedRecommendation ? `${t('report.submitted.recommendation')}: ${selectedRecommendation}` : '',
    ].filter(Boolean);
    await navigator.clipboard?.writeText(parts.join('\n'));
  };

  const copySubmittedTrackedChanges = async () => {
    if (!submittedTrackedDiff.length) return;
    const plainText = trackedDiffToPlainText(submittedTrackedDiff);
    const html = trackedDiffToHtml(submittedTrackedDiff);
    clearSubmittedTrackedCopyTimer();

    try {
      if (navigator.clipboard?.write && window.ClipboardItem) {
        await navigator.clipboard.write([
          new window.ClipboardItem({
            'text/html': new Blob([html], { type: 'text/html' }),
            'text/plain': new Blob([plainText], { type: 'text/plain' }),
          }),
        ]);
      } else if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(plainText);
      } else {
        throw new Error('Clipboard API unavailable');
      }

      setSubmittedTrackedCopyStatus('copied');
      submittedTrackedCopyTimerRef.current = window.setTimeout(() => {
        setSubmittedTrackedCopyStatus('idle');
        submittedTrackedCopyTimerRef.current = null;
      }, 1800);
    } catch {
      setSubmittedTrackedCopyStatus('error');
      submittedTrackedCopyTimerRef.current = window.setTimeout(() => {
        setSubmittedTrackedCopyStatus('idle');
        submittedTrackedCopyTimerRef.current = null;
      }, 2200);
    }
  };

  const translateSubmittedSelection = async () => {
    const editor = submittedEditorRef.current;
    if (!editor) return;
    const start = editor.selectionStart;
    const end = editor.selectionEnd;
    const selected = (start != null && end != null && end > start)
      ? submittedDraftText.slice(start, end)
      : '';
    if (!selected.trim()) {
      setSubmittedTranslateError(t('report.submitted.editor.translateSelectFirst'));
      return;
    }
    setSubmittedTranslateBusy(true);
    setSubmittedTranslateError(null);
    try {
      const { data } = await translateText(selected, { target: 'en' });
      const translated = (data?.text || '').trim();
      if (!translated) {
        setSubmittedTranslateError(t('report.submitted.editor.translateError'));
        return;
      }
      const previous = submittedDraftText;
      const nextText = previous.slice(0, start) + translated + previous.slice(end);
      setSubmittedPreTranslateText(previous);
      setSubmittedHighlightRanges((ranges) => adjustHighlightedRanges(ranges, previous, nextText));
      setSubmittedDraftText(nextText);
      setSubmittedRescanError(null);
      window.requestAnimationFrame(() => {
        const node = submittedEditorRef.current;
        if (!node) return;
        node.focus({ preventScroll: true });
        const caret = start + translated.length;
        node.setSelectionRange(caret, caret);
        scheduleSubmittedCaretUpdate();
      });
    } catch (err) {
      setSubmittedTranslateError(t('report.submitted.editor.translateError'));
    } finally {
      setSubmittedTranslateBusy(false);
    }
  };

  const undoSubmittedTranslate = () => {
    if (submittedPreTranslateText == null) return;
    const restore = submittedPreTranslateText;
    setSubmittedHighlightRanges((ranges) => adjustHighlightedRanges(ranges, submittedDraftText, restore));
    setSubmittedDraftText(restore);
    setSubmittedPreTranslateText(null);
    setSubmittedTranslateError(null);
  };

  const resetSubmittedDraft = async () => {
    setSubmittedDraftText(submittedBaselineText);
    setSubmittedHighlightRanges((current) => {
      if (!selectedParagraph?.text || !Object.keys(current || {}).length) return {};
      const range = findTextRange(submittedBaselineText, selectedParagraph.text);
      return range ? { [selectedParagraph.id]: { ...range, segmentId: selectedParagraph.id } } : {};
    });
    setSubmittedDraftStatus('idle');
    setSubmittedDraftUpdatedAt(null);
    setSubmittedRescanError(null);
    await deleteReportDraft(submittedDraftStorageKey);
  };

  // After a rewrite, the hero "Download PDF" serves the REWRITTEN PDF (the rewrite job's
  // generated copy) rather than the original scan's PDF. Mirrors the /rewrite page's
  // download: getRewriteDownload returns a signed { url } opened in a new tab.
  const handleDownloadRewrittenPdf = async () => {
    const rewriteId = currentRewrite?.id;
    if (!rewriteId) return;
    const downloadWindow = window.open('about:blank', '_blank');
    if (downloadWindow) downloadWindow.opener = null;
    try {
      const { data } = await getRewriteDownload(rewriteId, 'pdf');
      if (data?.url) {
        if (downloadWindow) downloadWindow.location.replace(data.url);
        else window.location.assign(data.url);
      } else {
        downloadWindow?.close();
      }
    } catch {
      downloadWindow?.close();
    }
  };

  const rescanSubmittedDraft = async () => {
    const text = submittedDraftText.trim();
    if (!text) {
      setSubmittedRescanError(t('report.submitted.editor.emptyDraft'));
      return;
    }
    if (balance !== null && balance < submittedDraftTokensRequired) {
      setSubmittedRescanNeedsTokens(true);
      setSubmittedRescanError(null);
      setSubmittedRescanStatus(null);
      return;
    }

    setSubmittedRescanBusy(true);
    setSubmittedRescanError(null);
    setSubmittedRescanStatus(t('report.submitted.editor.rescanQueueing'));

    try {
      const { data: scan } = await startScanWithText(text);
      setSubmittedRescanStatus(t('report.submitted.editor.rescanProcessing'));

      if (scan.status === 'completed') {
        refreshBalance?.();
        navigate(`/report/${scan.id}`);
        return;
      }

      for (let i = 0; i < RESCAN_MAX_POLLS; i += 1) {
        await sleep(RESCAN_POLL_INTERVAL);
        const { data } = await getScanStatus(scan.id);
        if (data.status === 'completed') {
          refreshBalance?.();
          navigate(`/report/${scan.id}`);
          return;
        }
        if (data.status === 'failed') {
          throw new Error(data.error || t('report.submitted.editor.rescanFailed'));
        }
        if (data.progress_message) {
          setSubmittedRescanStatus(data.progress_message);
        }
      }

      throw new Error(t('report.submitted.editor.rescanTimedOut'));
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || t('report.submitted.editor.rescanFailed');
      const httpStatus = err.response?.status;
      const isAuthExpired = httpStatus === 401 || (
        httpStatus === 403 &&
        String(msg).toLowerCase().includes('not authenticated')
      );
      const isInsufficient = (httpStatus === 400 || httpStatus === 402) && (
        String(msg).toLowerCase().includes('insufficient') ||
        String(msg).toLowerCase().includes('no credit account') ||
        String(msg).toLowerCase().includes('purchase')
      );
      if (isAuthExpired) {
        sessionStorage.setItem('auth_next', `/report/${id}`);
        await logout?.();
        navigate('/signin?error=Session expired. Please sign in again.', { replace: true });
        return;
      }
      if (isInsufficient) {
        setSubmittedRescanNeedsTokens(true);
      } else {
        setSubmittedRescanError(msg);
      }
      setSubmittedRescanStatus(null);
      setSubmittedRescanBusy(false);
    }
  };

  const hasAIFindings = issues.some(i =>
    i.category === 'ai_generation' ||
    i.scanner === 'ai_generation' ||
    i.signal_category === 'authorship_risk' ||
    i.actionability === 'auto_rewrite_candidate'
  );
  const submittedWordCount = Number.isFinite(Number(report.word_count)) ? Number(report.word_count) : null;
  const rewriteTokenCost = Number.isFinite(Number(report.rewrite_token_cost)) ? Number(report.rewrite_token_cost) : null;
  const rewriteTokenEstimate = submittedWordCount != null && rewriteTokenCost != null
    ? t('report.rewrite.tokenEstimate', {
      tokens: t('common.token', { count: rewriteTokenCost }),
      words: submittedWordCount.toLocaleString(locale),
    })
    : null;
  const hasRewriteSignalComparison = Boolean(
    hasRewriteResult &&
    hasRewriteComparisonData(rewriteResultReport) &&
    (rewrittenTransformation || rewrittenTransformationSummary || rewrittenAiScore != null)
  );
  const transformationOriginalScore = hasRewriteSignalComparison
    ? (rewriteResultSummary?.original_ai_authorship ?? rewriteResultSummary?.original_risk ?? originalComparisonAiScore)
    : originalComparisonAiScore;
  const transformationRewrittenScore = hasRewriteSignalComparison
    ? (rewriteResultSummary?.rewritten_ai_authorship ?? rewriteResultSummary?.rewrite_risk ?? rewrittenAiScore)
    : rewrittenAiScore;
  const reportAllowsRewrite = report.can_start_rewrite ?? hasAIFindings;
  const canStartRewrite = reportAllowsRewrite === true && !hasRewriteResult;
  const showNoRewriteableNotice = reportAllowsRewrite === false && !hasRewriteResult && !rewriteInProgress && !rewriteLoading && !rewriteError;
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
    if (rewriteLoading || rewriteCanceling || hasRewriteResult) return;
    if (rewriteInProgress && currentRewrite?.id) {
      setRewriteStartedHere(true);
      setRewriteError(null);
      setRewriteSseUnavailable(false);
      watchedRewriteIdsRef.current.add(currentRewrite.id);
      if (!connectRewriteEvents(currentRewrite.id)) {
        await pollRewriteStatus(currentRewrite.id);
      }
      return;
    }
    requestBrowserNotificationPermission();
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
      if (data.id) {
        watchedRewriteIdsRef.current.add(data.id);
      }
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
      <ConfirmDialog
        open={submittedRescanNeedsTokens}
        title={t('scan.notEnoughTitle')}
        message={t('scan.notEnoughMessage')}
        confirmLabel={t('scan.buyTokens')}
        onConfirm={() => navigate('/buy')}
        onCancel={() => setSubmittedRescanNeedsTokens(false)}
      />
      <div className="container">
        <ReportHero
          t={t}
          locale={locale}
          report={heroReport}
          tier={heroTier}
          isRewrittenView={rewrittenHeroView}
          onDownloadRewrittenPdf={handleDownloadRewrittenPdf}
          canStartRewrite={canStartRewrite}
          rewriteLoading={rewriteLoading}
          rewriteCanceling={rewriteCanceling}
          rewriteInProgress={rewriteInProgress}
          rewriteTokenEstimate={rewriteTokenEstimate}
          currentRewrite={currentRewrite}
          onRewrite={handleRewrite}
          onCancelRewrite={handleCancelRewrite}
          repairSummary={showOriginalRepairGuidance ? repairSummary : null}
          repairMainRiskLabel={t('report.repairSummary.mainRisk')}
          repairActionLabel={t('report.submitted.editor.editDraft')}
          repairActionHint={t('report.repairSummary.editDraftHint')}
          onRepairAction={showSubmittedEditEntry ? () => openSubmittedEditorForParagraph() : null}
          mergedCard={
            <MergedAuthorshipRisk
              t={t}
              breakdown={(heroBadge && heroBadge.authorship_breakdown) || null}
              sr={heroSubmissionRisk}
              authoritativeTier={heroBadge.tier || heroReport.tier}
              tierAuthority={(heroBadge && heroBadge.tier_authority) || null}
              headlineConfidence={(heroBadge && heroBadge.headline_confidence) || null}
              claimGraphDisplay={(authorshipEvidence && authorshipEvidence.claim_graph_display) || null}
              hideVerdictChip
            />
          }
        />
        {/* Stylometric-consistency "Writing-style outliers" panel (advisory,
            informational-only — Phase 1, poc/detect/consistency.py). Renders
            nothing when consistencyDisplay is null (flag off / no paragraph
            flagged / older report). Standalone (not nested in
            MergedAuthorshipRisk like the claim-graph panel) since this signal
            is independent of the authorship/tier badge. */}
        <ConsistencyRisk display={consistencyDisplay} />
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
                  <span>{t('report.rewrite.emailPdfProgress')}</span>
                </div>
              )}
            </div>
          </div>
        )}
        {showNoRewriteableNotice && (
          <section className="rewrite-status-alert">
            <strong>{t('report.rewrite.noRewriteableTitle')}</strong>
            <p>{t('report.rewrite.noRewriteableMessage')}</p>
          </section>
        )}

        {/* V7-centered layout (owner decision 2026-07-04): only the rewrite-comparison
            scorecard stays above the fold (it's the point of a rewrite re-scan). The
            plain writing-signal-pattern scorecard, summary bar, and score profile all
            live in the collapsed Advanced signals section after the document view. */}
        {hasRewriteSignalComparison && transformation && transformationSignals.length > 0 && (
          <section className="report-overview-card is-rewrite-comparison" aria-label={t('report.overview')}>
            {/* After a rewrite the scorecard already shows original-vs-rewritten, so the
                separate "Original scan baseline" bar is redundant — omit it. */}
            <TransformationScorecard
              t={t}
              report={report}
              hasRewriteSignalComparison={hasRewriteSignalComparison}
              rewrittenBadge={rewrittenBadge}
              rewrittenAiScore={rewrittenAiScore}
              originalComparisonBadge={originalComparisonBadge}
              aiScore={aiScore}
              transformation={transformation}
              transformationSignals={transformationSignals}
              transformationSummary={transformationSummary}
              transformationOriginalScore={transformationOriginalScore}
              originalColumnRatingBadge={originalColumnRatingBadge}
              rewrittenTransformation={rewrittenTransformation}
              rewrittenTransformationSummary={rewrittenTransformationSummary}
              transformationRewrittenScore={transformationRewrittenScore}
              rewrittenColumnRatingBadge={rewrittenColumnRatingBadge}
              rewriteResultSummary={rewriteResultSummary}
            />
          </section>
        )}

        {showOriginalRepairGuidance && (
          <FixFirstChecklist
            items={fixFirstItems}
            onSelectParagraph={lockAndScrollParagraph}
            title={t(fixFirstLowTone ? 'report.whatToFixFirst.titleLow' : 'report.whatToFixFirst.title')}
            kicker={t(fixFirstLowTone ? 'report.whatToFixFirst.kickerLow' : 'report.whatToFixFirst.kicker')}
            intro={t(fixFirstLowTone ? 'report.whatToFixFirst.introLow' : 'report.whatToFixFirst.intro')}
          />
        )}

        <RewriteCompletionBand
          hasRewriteResult={hasRewriteResult}
          rewriteOutcome={rewriteOutcome}
          rewriteBandTitle={rewriteBandTitle}
          rewriteBandDetail={rewriteBandDetail}
          rewriteResultSummary={rewriteResultSummary}
          currentRewrite={currentRewrite}
        />

        {/* allow-hardcode: CSS classNames in JSX markup — UI layout, not a scoring oracle */}
        {submittedContent.paragraphs.length > 0 && (
          <SignalHighlights
            submittedContent={submittedContent}
            selectedParagraph={selectedParagraph}
            selectedParagraphId={selectedParagraphId}
            highlightedParagraphs={highlightedParagraphs}
            paragraphSeverityBar={paragraphSeverityBar}
            selectedCriticalThinking={selectedCriticalThinking}
            evidenceLevels={(heroBadge && heroBadge.authorship_evidence_levels) || null}
            showSubmittedEditEntry={showSubmittedEditEntry}
            onSelectParagraph={lockAndScrollParagraph}
            onPreviewParagraph={previewParagraph}
            onAdjacent={selectAdjacentHighlightedParagraph}
            onEditParagraph={openSubmittedEditorForParagraph}
            onCopyGuidance={copySelectedParagraphGuidance}
          />
        )}
        {/* Advanced-signals drawer removed (owner decision 2026-07-04): the
            writing-signal scorecard, score-profile, authenticity dashboard, and
            DeBERTa second-opinion tile were old-methodology internals / redundant
            with the V7 panel + deep-scan (the second-opinion tile stopped being a
            distinct detector after the highlights switched to deep-scan). Critical
            Thinking is the one V7-native, actionable, non-redundant panel, so it's
            promoted to a visible section here rather than buried in a drawer. */}
        <CriticalThinkingControl badge={badge} t={t} />
        {/* Interactive counterpart to the read-only panel above: student answers a
            flagged question, an LLM judge scores it. Renders nothing when the
            DRAFTPROOF_DEFENCE_CHECK flag is off or there are no questions. */}
        <DefenceCheck scanId={id} questions={defenceQuestions} t={t} />
        {submittedEditorOpen && (
          <div className={`submitted-editor-backdrop${submittedEditorClosing ? ' is-closing' : ''}`} role="dialog" aria-modal="true" aria-label={t('report.submitted.editor.title')}>
            <div className="submitted-editor-sheet">
              <button
                type="button"
                className="submitted-editor-close-button"
                aria-label={t('report.submitted.editor.close')}
                title={t('report.submitted.editor.close')}
                onClick={() => closeSubmittedEditor()}
                disabled={submittedRescanBusy}
              >
                X
              </button>
              <div className="submitted-editor-head">
                <div>
                  <span className="submitted-content-kicker">{t('report.submitted.editor.kicker')}</span>
                  <h2>{t(editingRewriteDraft ? 'report.submitted.editor.rewriteTitle' : 'report.submitted.editor.title')}</h2>
                  <p>{t(editingRewriteDraft ? 'report.submitted.editor.rewriteNotice' : 'report.submitted.editor.priorScanNotice')}</p>
                </div>
                <div className="submitted-editor-actions">
                  <span className={`submitted-save-state is-${submittedDraftStatus}`}>
                    {submittedDraftStatus === 'saving'
                      ? t('report.submitted.editor.saving')
                      : submittedDraftStatus === 'saved'
                        ? t('report.submitted.editor.saved', {
                          value: submittedDraftUpdatedAt ? formatDate(submittedDraftUpdatedAt, locale) : t('common.lastUpdated'),
                        })
                        : submittedDraftStatus === 'error'
                          ? t('report.submitted.editor.saveError')
                          : t('report.submitted.editor.noDraft')}
                  </span>
                  <button
                    type="button"
                    className="btn btn-ghost"
                    onClick={() => closeSubmittedEditor()}
                    disabled={submittedRescanBusy}
                  >
                    {t('report.submitted.editor.close')}
                  </button>
                </div>
              </div>

              <div className="submitted-editor-grid">
                <section className="submitted-editor-main" aria-label={t('report.submitted.editor.documentEditor')}>
                  <div className="submitted-editor-toolbar">
                    <div>
                      <strong>{t('report.submitted.editor.documentEditor')}</strong>
                      <span>{submittedDraftChanged ? t('report.submitted.editor.changed') : t('report.submitted.editor.unchanged')}</span>
                    </div>
                    <div className="submitted-editor-toolbar-actions">
                      <button
                        type="button"
                        className="btn btn-secondary submitted-translate-button"
                        onClick={translateSubmittedSelection}
                        disabled={submittedTranslateBusy || submittedRescanBusy}
                        title={t('report.submitted.editor.translateNote')}
                      >
                        <svg className="cta-edit-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true" focusable="false">
                          <path d="M4 5h7M7.5 5v1.5M9.5 5c0 4-2.5 7-5.5 8.5M6 9c.8 2 2.6 3.6 5 4.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                          <path d="M13 19l3.2-8h.6L20 19M14 16.5h5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                        {submittedTranslateBusy
                          ? t('report.submitted.editor.translating')
                          : t('report.submitted.editor.translateCnEn')}
                      </button>
                      {submittedPreTranslateText != null && (
                        <button
                          type="button"
                          className="btn btn-ghost btn-small submitted-translate-undo"
                          onClick={undoSubmittedTranslate}
                          disabled={submittedTranslateBusy || submittedRescanBusy}
                        >
                          {t('report.submitted.editor.undoTranslate')}
                        </button>
                      )}
                      <button
                        type="button"
                        className="btn btn-ghost"
                        onClick={resetSubmittedDraft}
                        disabled={!submittedDraftChanged || submittedRescanBusy}
                      >
                        {t('report.submitted.editor.discardDraft')}
                      </button>
                      <button
                        type="button"
                        className="btn btn-primary"
                        onClick={rescanSubmittedDraft}
                        disabled={submittedRescanBusy || !submittedDraftText.trim()}
                      >
                        {submittedRescanBusy ? t('report.submitted.editor.rescanning') : t('report.submitted.editor.rescanDraft')}
                      </button>
                      <span className="submitted-rescan-token-note">
                        {t('scan.word', { count: submittedDraftWordCount })}
                        {' · '}
                        {t('scan.tokensRequired', { count: submittedDraftTokensRequired })}
                      </span>
                    </div>
                  </div>
                  <div className={`submitted-translate-tip${submittedTranslateError ? ' is-error' : ''}`}>
                    <svg className="cta-edit-icon" width="15" height="15" viewBox="0 0 24 24" fill="none" aria-hidden="true" focusable="false">
                      <path d="M4 5h7M7.5 5v1.5M9.5 5c0 4-2.5 7-5.5 8.5M6 9c.8 2 2.6 3.6 5 4.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                      <path d="M13 19l3.2-8h.6L20 19M14 16.5h5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                    <span>
                      {submittedTranslateError
                        ? submittedTranslateError
                        : submittedPreTranslateText != null
                          ? t('report.submitted.editor.translateNote')
                          : t('report.submitted.editor.translateHint')}
                    </span>
                  </div>
                  <div className="submitted-editor-textarea-wrap">
                    <div
                      ref={submittedHighlightRef}
                      className="submitted-editor-highlight-layer"
                      aria-hidden="true"
                    >
                      {submittedEditorHighlightParts.map((part, index) => (
                        <span key={`${part.type}-${index}`} className={`submitted-editor-highlight-${part.type}`}>
                          {part.text}
                        </span>
                      ))}
                      {'\n'}
                    </div>
                    <textarea
                      ref={submittedEditorRef}
                      className="submitted-editor-textarea"
                      value={submittedDraftText}
                      onChange={(event) => {
                        const nextText = event.target.value;
                        setSubmittedHighlightRanges((ranges) => adjustHighlightedRanges(ranges, submittedDraftText, nextText));
                        setSubmittedDraftText(nextText);
                        setSubmittedRescanError(null);
                        setSubmittedPreTranslateText(null);
                        clearSubmittedTrackedCopyTimer();
                        setSubmittedTrackedCopyStatus('idle');
                        scheduleSubmittedCaretUpdate();
                      }}
                      onClick={scheduleSubmittedCaretUpdate}
                      onFocus={scheduleSubmittedCaretUpdate}
                      onKeyUp={scheduleSubmittedCaretUpdate}
                      onScroll={syncSubmittedHighlightScroll}
                      onSelect={scheduleSubmittedCaretUpdate}
                      onBlur={hideSubmittedCaret}
                      spellCheck="true"
                    />
                    <div
                      className={`submitted-editor-custom-caret${submittedCaret.visible ? ' is-visible' : ''}`}
                      style={{
                        height: `${submittedCaret.height}px`,
                        transform: `translate(${submittedCaret.left}px, ${submittedCaret.top}px)`,
                      }}
                      aria-hidden="true"
                    />
                  </div>
                  {(submittedRescanStatus || submittedRescanError) && (
                    <div className={`submitted-rescan-status${submittedRescanError ? ' is-error' : ''}`}>
                      {submittedRescanError || submittedRescanStatus}
                    </div>
                  )}
                  <div className="submitted-tracked-preview" aria-label={t('report.submitted.editor.trackedPreview')}>
                    <div className="submitted-tracked-head">
                      <div className="submitted-tracked-title">
                        <strong>{t('report.submitted.editor.trackedPreview')}</strong>
                        <span>{t('report.submitted.editor.trackedPreviewBody')}</span>
                      </div>
                      <button
                        type="button"
                        className={`submitted-tracked-copy-button${submittedTrackedCopyStatus === 'copied' ? ' is-copied' : ''}${submittedTrackedCopyStatus === 'error' ? ' has-error' : ''}`}
                        onClick={copySubmittedTrackedChanges}
                        disabled={!submittedTrackedDiff.length}
                      >
                        {submittedTrackedCopyStatus === 'copied'
                          ? t('report.submitted.editor.copiedTrackedChanges')
                          : submittedTrackedCopyStatus === 'error'
                            ? t('report.submitted.editor.copyTrackedChangesFailed')
                            : t('report.submitted.editor.copyTrackedChanges')}
                      </button>
                    </div>
                    <div className="submitted-tracked-body">
                      {submittedTrackedDiff.map((part, index) => (
                        <span key={`${part.type}-${index}`} className={`submitted-diff-${part.type}`}>
                          {part.text}
                        </span>
                      ))}
                    </div>
                  </div>
                </section>

                <aside className="submitted-affected-panel" aria-label={t('report.submitted.editor.affectedParagraphs')}>
                  <div className="submitted-affected-head">
                    <span>{t('report.submitted.editor.affectedParagraphs')}</span>
                    <strong>{affectedParagraphs.length}</strong>
                  </div>
                  <div className="submitted-affected-list">
                    {affectedParagraphs.map((paragraph) => {
                      const signal = paragraph.primarySignal || paragraph.signals[0];
                      const isSelected = selectedParagraph?.id === paragraph.id;
                      const isEdited = Boolean(paragraph.text) && !submittedDraftText.includes(paragraph.text);
                      return (
                        <button
                          key={`affected-${paragraph.id}`}
                          type="button"
                          className={`submitted-affected-item${isSelected ? ' is-selected' : ''}${isEdited ? ' is-edited' : ''}`}
                          title={isEdited
                            ? t('report.submitted.editor.paragraphEdited')
                            : t('report.submitted.editor.paragraphUnchanged')}
                          onClick={() => {
                            setSelectedParagraphId(paragraph.id);
                            focusParagraphInSubmittedEditor(paragraph);
                          }}
                        >
                          <span className="submitted-affected-meta">
                            <span className="submitted-affected-meta-left">
                              <span>{paragraph.sentence_id}</span>
                              {signal.tier && (
                                <span className={`submitted-sev-badge is-${signal.tier}`}>
                                  {t(`report.severities.${signal.tier}`, { defaultValue: signal.tier })}
                                </span>
                              )}
                            </span>
                            {isEdited && (
                              <span
                                className="submitted-affected-check"
                                aria-label={t('report.submitted.editor.paragraphEdited')}
                              >
                                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" aria-hidden="true" focusable="false">
                                  <path d="M5 12.5l4.2 4.2L19 7" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
                                </svg>
                              </span>
                            )}
                          </span>
                          <strong>{signalLabel(signal.key, signal.label, t)}</strong>
                          {paragraph.flaggedSentences?.length > 0 && (
                            <span className="submitted-affected-count">
                              {t('report.submitted.paragraphSignals', { count: paragraph.flaggedSentences.length })}
                            </span>
                          )}
                          <em>{paragraph.text}</em>
                        </button>
                      );
                    })}
                  </div>

                  <div className="submitted-editor-detail">
                    {selectedParagraph?.primarySignal ? (
                      <>
                        <span className="submitted-panel-kicker">{selectedParagraph.sentence_id}</span>
                        <h3>{signalLabel(selectedParagraph.primarySignal.key, selectedParagraph.primarySignal.label, t)}</h3>
                        {(selectedParagraph.primarySignal.tier || selectedParagraph.flaggedSentences?.length > 0) && (
                          <div className="submitted-affected-detail-meta">
                            {selectedParagraph.primarySignal.tier && (
                              <span className={`submitted-sev-badge is-${selectedParagraph.primarySignal.tier}`}>
                                {t(`report.severities.${selectedParagraph.primarySignal.tier}`, { defaultValue: selectedParagraph.primarySignal.tier })}
                              </span>
                            )}
                            {selectedParagraph.flaggedSentences?.length > 0 && (
                              <span className="submitted-affected-count">
                                {t('report.submitted.paragraphSignals', { count: selectedParagraph.flaggedSentences.length })}
                              </span>
                            )}
                          </div>
                        )}
                        {/* Lead with the specific flagged sentences + fixes (mirrors the
                            "Flagged paragraphs" cards) rather than re-dumping the whole
                            paragraph — the full text is already editable on the left.
                            Falls back to the paragraph text when a signal has no
                            per-sentence flagged evidence. */}
                        {selectedParagraph.flaggedSentences?.length > 0 ? (
                          <div className="submitted-editor-flagged">
                            <span className="submitted-panel-kicker">{t('report.submitted.flaggedSentences')}</span>
                            <ul className="deberta-evidence-list">
                              {selectedParagraph.flaggedSentences.map((sent) => (
                                <li key={sent.sentence_id}>
                                  {sent.tier && (
                                    <span className={`deberta-evidence-tier is-${sent.tier}`}>
                                      {t(`report.severities.${sent.tier}`, { defaultValue: sent.tier })}
                                    </span>
                                  )}
                                  <span className="deberta-evidence-text">{sent.text}</span>
                                  {sent.suggestion && (
                                    <span className="deberta-evidence-suggestion">{sent.suggestion}</span>
                                  )}
                                </li>
                              ))}
                            </ul>
                          </div>
                        ) : (
                          <div className="submitted-editor-sentence">
                            <span>{t('report.submitted.editor.affectedParagraph')}</span>
                            <p>{selectedParagraph.text}</p>
                          </div>
                        )}
                        <SubmittedSignalGauge
                          selectedSignalStrength={selectedSignalStrength}
                          selectedParagraph={selectedParagraph}
                          submittedDraftText={submittedDraftText}
                        />
                        <div className="submitted-panel-meta">
                          <span>{selectedParagraphDraftStatus}</span>
                        </div>
                        <div className="submitted-editor-signal">
                          <span>{t('report.submitted.editor.signal')}</span>
                          <p>{selectedReaderSummary}</p>
                        </div>
                        {selectedRecommendation && (
                          <div className="submitted-panel-note">
                            <span>{t('report.submitted.recommendation')}</span>
                            <p>{selectedRecommendation}</p>
                          </div>
                        )}
                        {selectedSecondarySignals.length > 0 && (
                          <div className="submitted-also-detected">
                            <span className="submitted-also-detected-head">{t('report.submitted.alsoDetected')}</span>
                            <ul>
                              {selectedSecondarySignals.map((signal) => {
                                const advice = signal.recommendation
                                  || signalDescription(signal.key, signal.description, t);
                                const strength = clampPercent(signal.score);
                                return (
                                  <li key={signal.key}>
                                    <div className="submitted-also-detected-label">
                                      <strong>{signalLabel(signal.key, signal.label, t)}</strong>
                                      {strength != null && <span>{Math.round(strength)}%</span>}
                                    </div>
                                    {advice && <p>{advice}</p>}
                                  </li>
                                );
                              })}
                            </ul>
                          </div>
                        )}
                      </>
                    ) : (
                      <p>{t('report.submitted.mapReadyBody')}</p>
                    )}
                  </div>
                </aside>
              </div>
            </div>
          </div>
        )}

      </div>
    </main>
  );
}
