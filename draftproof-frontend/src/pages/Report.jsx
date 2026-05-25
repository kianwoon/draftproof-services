import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate, useParams, Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { getReport, createRewrite, cancelRewrite, getRewriteStatus, getRewriteReport, getScanStatus, startScanWithText } from '../api/draftproofApi';
import ErrorReload from '../components/ErrorReload';
import ConfirmDialog from '../components/ConfirmDialog';
import { useAuth } from '../context/AuthContext';
import { deleteReportDraft, getReportDraft, saveReportDraft } from '../utils/reportDraftStorage';
import { countWords, scanTokensRequired } from '../utils/scanBilling';
import {
  requestBrowserNotificationPermission,
  showBrowserNotification,
} from '../utils/browserNotifications';
import RewriteNoticeDialog from './report/RewriteNoticeDialog';
import {
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
  buildPairedTransformationSignals,
  groupTransformationSignals,
  getTransformationSignalImprovement,
  transformationSignalFeatureMap,
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
  mergeScanSummary,
  getScanDocumentContext,
  getScanTransformationSignals,
  getScanContributionSummary,
  mergeTransformationSummary,
  buildRewriteResultSummary,
  buildRewriteContributionOverride,
  buildSubmittedContentModel,
  requiresRewriteAuthorReview,
  requiresRewriteExternalReview,
  isRewriteActive,
  normalizeRewriteProgressMessage,
  normalizeRewriteJob,
  formatElapsed,
  getRewriteProgressDetail,
  isReviewOnlyRewriteMessage,
  buildRewriteEventsUrl,
} from './report/reportHelpers';

const RESCAN_POLL_INTERVAL = 3000;
const RESCAN_MAX_POLLS = 200;
const SUBMITTED_EDITOR_TRANSITION_MS = 240;
const REWRITE_REPORT_RETRY_LIMIT = 8;
const REWRITE_REPORT_RETRY_DELAY_MS = 1500;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function addScoreProfileFeature(features, key, value) {
  if (features[key] != null) return;
  const percent = clampPercent(value);
  if (percent != null) {
    features[key] = percent;
  }
}

function submittedContentToText(model) {
  return (model?.paragraphs || [])
    .map((paragraph) => paragraph.segments.map((segment) => segment.text).join(' ').trim())
    .filter(Boolean)
    .join('\n\n');
}

function tokenizeTrackedText(text) {
  return String(text || '').match(/\s+|[A-Za-z0-9]+|[^\sA-Za-z0-9]/g) || [];
}

function compactDiffParts(parts) {
  const compacted = [];
  parts.forEach((part) => {
    if (!part?.text) return;
    const previous = compacted[compacted.length - 1];
    if (previous?.type === part.type) {
      previous.text += part.text;
    } else {
      compacted.push({ ...part });
    }
  });
  return compacted;
}

function lcsTokenDiff(originalTokens, currentTokens) {
  const originalLength = originalTokens.length;
  const currentLength = currentTokens.length;
  const matrix = Array.from({ length: originalLength + 1 }, () => Array(currentLength + 1).fill(0));

  for (let i = originalLength - 1; i >= 0; i -= 1) {
    for (let j = currentLength - 1; j >= 0; j -= 1) {
      matrix[i][j] = originalTokens[i] === currentTokens[j]
        ? matrix[i + 1][j + 1] + 1
        : Math.max(matrix[i + 1][j], matrix[i][j + 1]);
    }
  }

  const parts = [];
  let i = 0;
  let j = 0;
  while (i < originalLength && j < currentLength) {
    if (originalTokens[i] === currentTokens[j]) {
      parts.push({ type: 'equal', text: originalTokens[i] });
      i += 1;
      j += 1;
    } else if (matrix[i + 1][j] >= matrix[i][j + 1]) {
      parts.push({ type: 'delete', text: originalTokens[i] });
      i += 1;
    } else {
      parts.push({ type: 'insert', text: currentTokens[j] });
      j += 1;
    }
  }

  while (i < originalLength) {
    parts.push({ type: 'delete', text: originalTokens[i] });
    i += 1;
  }
  while (j < currentLength) {
    parts.push({ type: 'insert', text: currentTokens[j] });
    j += 1;
  }

  return compactDiffParts(parts);
}

function charDiff(originalText, currentText) {
  return lcsTokenDiff(Array.from(originalText || ''), Array.from(currentText || ''));
}

function refineReplacementParts(parts) {
  const refined = [];
  for (let i = 0; i < parts.length; i += 1) {
    const current = parts[i];
    const next = parts[i + 1];
    const currentIsWord = current?.type === 'delete' && /^[A-Za-z0-9]+$/.test(current.text || '');
    const nextIsWord = next?.type === 'insert' && /^[A-Za-z0-9]+$/.test(next.text || '');
    if (currentIsWord && nextIsWord) {
      refined.push(...charDiff(current.text, next.text));
      i += 1;
    } else {
      refined.push(current);
    }
  }
  return compactDiffParts(refined);
}

function buildTrackedDiff(originalText, currentText) {
  if (originalText === currentText) {
    return [{ type: 'equal', text: currentText }];
  }

  const originalTokens = tokenizeTrackedText(originalText);
  const currentTokens = tokenizeTrackedText(currentText);

  return refineReplacementParts(lcsTokenDiff(originalTokens, currentTokens));
}

function findTextRange(haystack, needle) {
  if (!haystack || !needle) return null;
  const start = haystack.indexOf(needle);
  if (start < 0) return null;
  return { start, end: start + needle.length };
}

function changedTextRange(previousText, nextText) {
  let start = 0;
  const previousLength = previousText.length;
  const nextLength = nextText.length;

  while (
    start < previousLength &&
    start < nextLength &&
    previousText[start] === nextText[start]
  ) {
    start += 1;
  }

  let previousEnd = previousLength;
  let nextEnd = nextLength;
  while (
    previousEnd > start &&
    nextEnd > start &&
    previousText[previousEnd - 1] === nextText[nextEnd - 1]
  ) {
    previousEnd -= 1;
    nextEnd -= 1;
  }

  return { start, previousEnd, nextEnd, delta: nextEnd - previousEnd };
}

function adjustHighlightedRange(range, previousText, nextText) {
  if (!range) return null;
  const change = changedTextRange(previousText, nextText);
  const nextLength = nextText.length;

  if (change.previousEnd <= range.start) {
    return {
      ...range,
      start: Math.max(0, Math.min(nextLength, range.start + change.delta)),
      end: Math.max(0, Math.min(nextLength, range.end + change.delta)),
    };
  }

  if (change.start >= range.end) {
    return {
      ...range,
      start: Math.max(0, Math.min(nextLength, range.start)),
      end: Math.max(0, Math.min(nextLength, range.end)),
    };
  }

  const start = Math.max(0, Math.min(nextLength, Math.min(range.start, change.start)));
  const end = Math.max(
    start,
    Math.min(nextLength, Math.max(range.end + change.delta, change.nextEnd))
  );

  return { ...range, start, end };
}

function adjustHighlightedRanges(ranges, previousText, nextText) {
  return Object.fromEntries(
    Object.entries(ranges || {})
      .map(([segmentId, range]) => [segmentId, adjustHighlightedRange(range, previousText, nextText)])
      .filter(([, range]) => range && range.end > range.start)
  );
}

function mapOriginalRangeToCurrent(originalText, currentText, range) {
  if (!range || range.end <= range.start) return null;
  if (originalText === currentText) return range;

  const parts = lcsTokenDiff(tokenizeTrackedText(originalText), tokenizeTrackedText(currentText));
  let originalOffset = 0;
  let currentOffset = 0;
  let mappedStart = null;
  let mappedEnd = null;

  const include = (start, end = start) => {
    mappedStart = mappedStart == null ? start : Math.min(mappedStart, start);
    mappedEnd = mappedEnd == null ? end : Math.max(mappedEnd, end);
  };

  parts.forEach((part) => {
    const length = part.text.length;
    if (part.type === 'equal') {
      const originalEnd = originalOffset + length;
      const overlapStart = Math.max(range.start, originalOffset);
      const overlapEnd = Math.min(range.end, originalEnd);
      if (overlapStart < overlapEnd) {
        include(
          currentOffset + (overlapStart - originalOffset),
          currentOffset + (overlapEnd - originalOffset)
        );
      }
      originalOffset = originalEnd;
      currentOffset += length;
      return;
    }

    if (part.type === 'delete') {
      const originalEnd = originalOffset + length;
      if (Math.max(range.start, originalOffset) < Math.min(range.end, originalEnd)) {
        include(currentOffset);
      }
      originalOffset = originalEnd;
      return;
    }

    if (part.type === 'insert') {
      if (originalOffset >= range.start && originalOffset <= range.end) {
        include(currentOffset, currentOffset + length);
      }
      currentOffset += length;
    }
  });

  if (mappedStart == null || mappedEnd == null) return null;
  const start = Math.max(0, Math.min(currentText.length, mappedStart));
  const end = Math.max(start, Math.min(currentText.length, mappedEnd));
  return end > start ? { start, end } : null;
}

function buildOriginalSegmentRanges(originalText, segments) {
  const ranges = {};
  let cursor = 0;
  (segments || []).forEach((segment) => {
    if (!segment?.id || !segment.text) return;
    let start = originalText.indexOf(segment.text, cursor);
    if (start < 0) start = originalText.indexOf(segment.text);
    if (start < 0) return;
    const end = start + segment.text.length;
    ranges[segment.id] = { start, end, segmentId: segment.id };
    cursor = end;
  });
  return ranges;
}

function highlightedEditorParts(text, range) {
  if (!range || range.end <= range.start) return [{ type: 'plain', text }];
  const start = Math.max(0, Math.min(text.length, range.start));
  const end = Math.max(start, Math.min(text.length, range.end));

  return [
    { type: 'plain', text: text.slice(0, start) },
    { type: 'selected', text: text.slice(start, end) },
    { type: 'plain', text: text.slice(end) },
  ].filter((part) => part.text);
}

export default function Report() {
  const { id } = useParams();
  const navigate = useNavigate();
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
  const [activeProfileTab, setActiveProfileTab] = useState(null);
  const [submittedEditorOpen, setSubmittedEditorOpen] = useState(false);
  const [submittedEditorClosing, setSubmittedEditorClosing] = useState(false);
  const [submittedDraftText, setSubmittedDraftText] = useState('');
  const [submittedDraftLoaded, setSubmittedDraftLoaded] = useState(false);
  const [submittedDraftStatus, setSubmittedDraftStatus] = useState('idle');
  const [submittedDraftUpdatedAt, setSubmittedDraftUpdatedAt] = useState(null);
  const [submittedRescanBusy, setSubmittedRescanBusy] = useState(false);
  const [submittedRescanStatus, setSubmittedRescanStatus] = useState(null);
  const [submittedRescanError, setSubmittedRescanError] = useState(null);
  const [submittedHighlightRanges, setSubmittedHighlightRanges] = useState({});
  const rewritePollRef = useRef(null);
  const rewriteEventSourceRef = useRef(null);
  const rewriteTimerStartRef = useRef(null);
  const watchedRewriteIdsRef = useRef(new Set());
  const notifiedRewriteIdsRef = useRef(new Set());
  const submittedEditorRef = useRef(null);
  const submittedHighlightRef = useRef(null);
  const submittedEditorCloseTimerRef = useRef(null);

  const clearSubmittedEditorCloseTimer = useCallback(() => {
    if (submittedEditorCloseTimerRef.current) {
      window.clearTimeout(submittedEditorCloseTimerRef.current);
      submittedEditorCloseTimerRef.current = null;
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
        ? message
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
    setActiveProfileTab(null);
  }, [id]);

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
    setSubmittedHighlightRanges({});
  }, [id, closeSubmittedEditor]);

  useEffect(() => () => clearSubmittedEditorCloseTimer(), [clearSubmittedEditorCloseTimer]);

  useEffect(() => {
    if (!report) return undefined;

    let cancelled = false;
    const reportIssues = Array.isArray(report.issues) ? report.issues : [];
    const model = buildSubmittedContentModel({ ...report, issues: reportIssues });
    const originalText = submittedContentToText(model);

    setSubmittedDraftLoaded(false);
    setSubmittedDraftText(originalText);
    setSubmittedDraftStatus('idle');
    setSubmittedDraftUpdatedAt(null);

    getReportDraft(id)
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
  }, [id, report?.id]);

  useEffect(() => {
    if (!report || !submittedDraftLoaded) return undefined;

    const reportIssues = Array.isArray(report.issues) ? report.issues : [];
    const model = buildSubmittedContentModel({ ...report, issues: reportIssues });
    const originalText = submittedContentToText(model);
    if (submittedDraftText === originalText) {
      setSubmittedDraftStatus('idle');
      setSubmittedDraftUpdatedAt(null);
      deleteReportDraft(id).catch(() => {});
      return undefined;
    }

    setSubmittedDraftStatus('saving');
    const timer = window.setTimeout(() => {
      saveReportDraft(id, submittedDraftText)
        .then((draft) => {
          setSubmittedDraftStatus('saved');
          setSubmittedDraftUpdatedAt(draft?.updatedAt || null);
        })
        .catch(() => {
          setSubmittedDraftStatus('error');
        });
    }, 650);

    return () => window.clearTimeout(timer);
  }, [id, report?.id, submittedDraftLoaded, submittedDraftText]);

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
  const canEditSubmittedDraft = !hasRewriteResult && !rewriteInProgress;
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
  const rewrittenRequiresAuthorReview = requiresRewriteAuthorReview(rewriteResultSummary);
  const rewrittenRequiresExternalReview = requiresRewriteExternalReview(rewriteResultSummary);
  const manualReviewTone = { color: '#92400e', bg: '#fffbeb' };
  const originalColumnRatingBadge = {
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
    : {
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
  const originalSubmittedText = submittedContentToText(submittedContent);
  const submittedDraftChanged = submittedDraftText !== originalSubmittedText;
  const submittedTrackedDiff = submittedEditorOpen
    ? buildTrackedDiff(originalSubmittedText, submittedDraftText)
    : [];
  const affectedSegments = submittedContent.segments.filter((segment) => segment.signals.length > 0);
  const originalAffectedRanges = buildOriginalSegmentRanges(originalSubmittedText, affectedSegments);
  const selectedSentenceDraftStatus = selectedSegment?.text && submittedDraftText.includes(selectedSegment.text)
    ? t('report.submitted.editor.sentenceUnchanged')
    : t('report.submitted.editor.sentenceEdited');
  const selectedSignalStrength = clampPercent(selectedSegment?.primarySignal?.score);
  const submittedHighlightRange = selectedSegment?.id ? submittedHighlightRanges[selectedSegment.id] : null;
  const submittedEditorHighlightParts = highlightedEditorParts(submittedDraftText, submittedHighlightRange);
  const submittedDraftWordCount = countWords(submittedDraftText);
  const submittedDraftTokensRequired = scanTokensRequired(submittedDraftWordCount);

  const resolveSubmittedSegmentRange = (segment, existingRanges = submittedHighlightRanges) => {
    if (!segment?.id || !segment.text) return null;
    return (
      findTextRange(submittedDraftText, segment.text) ||
      existingRanges[segment.id] ||
      mapOriginalRangeToCurrent(originalSubmittedText, submittedDraftText, originalAffectedRanges[segment.id])
    );
  };

  const buildSubmittedHighlightRanges = (existingRanges = submittedHighlightRanges) => {
    const nextRanges = { ...(existingRanges || {}) };
    affectedSegments.forEach((segment) => {
      if (nextRanges[segment.id]) return;
      const range = resolveSubmittedSegmentRange(segment, nextRanges);
      if (range) nextRanges[segment.id] = { ...range, segmentId: segment.id };
    });
    return nextRanges;
  };

  const renderSubmittedSignalGauge = () => {
    if (selectedSignalStrength == null || !selectedSegment?.primarySignal) return null;
    const value = Math.round(selectedSignalStrength);
    return (
      <div
        className="submitted-signal-gauge"
        style={{
          '--signal-color': selectedSegment.primarySignal.color || '#b45309',
          '--signal-strength': `${value}%`,
        }}
        aria-label={t('report.submitted.signalStrength', { value })}
      >
        <div className="submitted-signal-gauge-head">
          <span>{t('report.submitted.signalStrengthLabel')}</span>
          <strong>{value}%</strong>
        </div>
        <div className="submitted-signal-gauge-track" aria-hidden="true">
          <span />
        </div>
      </div>
    );
  };

  const focusSentenceInSubmittedEditor = (segment) => {
    const editor = submittedEditorRef.current;
    if (!editor || !segment?.text) return;
    const range = resolveSubmittedSegmentRange(segment);
    setSubmittedHighlightRanges((current) => {
      if (!range) return current;
      return { ...current, [segment.id]: { ...range, segmentId: segment.id } };
    });
    editor.focus();
    if (range) {
      editor.setSelectionRange(range.start, range.start);
    }
  };

  const syncSubmittedHighlightScroll = () => {
    if (!submittedEditorRef.current || !submittedHighlightRef.current) return;
    submittedHighlightRef.current.scrollTop = submittedEditorRef.current.scrollTop;
    submittedHighlightRef.current.scrollLeft = submittedEditorRef.current.scrollLeft;
  };

  const resetSubmittedDraft = async () => {
    setSubmittedDraftText(originalSubmittedText);
    setSubmittedHighlightRanges((current) => {
      if (!selectedSegment?.text || !Object.keys(current || {}).length) return {};
      const range = findTextRange(originalSubmittedText, selectedSegment.text);
      return range ? { [selectedSegment.id]: { ...range, segmentId: selectedSegment.id } } : {};
    });
    setSubmittedDraftStatus('idle');
    setSubmittedDraftUpdatedAt(null);
    setSubmittedRescanError(null);
    await deleteReportDraft(id);
  };

  const rescanSubmittedDraft = async () => {
    const text = submittedDraftText.trim();
    if (!text) {
      setSubmittedRescanError(t('report.submitted.editor.emptyDraft'));
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
      setSubmittedRescanError(err.response?.data?.detail || err.message || t('report.submitted.editor.rescanFailed'));
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
  const sealReferenceScore = hasRewriteSignalComparison
    ? rewrittenCalibratedAuthorshipRisk
    : calibratedAuthorshipRisk;
  const sealDisplayReferenceScore = hasRewriteSignalComparison
    ? calibratedReportAiScore(transformationRewrittenScore)
    : sealReferenceScore;
  const sealAiSignalStamp = getAiSignalStamp(sealDisplayReferenceScore ?? sealReferenceScore, t);
  const rewriteBelowReferenceBand = hasRewriteSignalComparison && sealDisplayReferenceScore != null && sealDisplayReferenceScore < 20;
  const displayedRewrittenColumnRatingBadge = rewriteBelowReferenceBand
    ? {
      ...rewrittenColumnRatingBadge,
      label: sealAiSignalStamp.label,
      fullLabel: sealAiSignalStamp.label,
      tone: sealAiSignalStamp.tone,
    }
    : rewrittenColumnRatingBadge;
  const sealRatingBadge = hasRewriteSignalComparison ? displayedRewrittenColumnRatingBadge : originalColumnRatingBadge;
  const sealTone = sealRatingBadge.tone || sealAiSignalStamp.tone;
  const sealRatingCaption = sealRatingBadge.caption || (hasRewriteSignalComparison ? t('report.transformation.rewrittenAiSignal') : t('report.transformation.aiSignal'));
  const sealRatingLabel = sealRatingBadge.label || sealAiSignalStamp.label;
  const sealAuthorshipDetail = rewrittenRequiresAuthorReview
    ? t('rewritePage.authorReviewTitle')
    : rewrittenRequiresExternalReview
    ? t('rewritePage.externalReviewTitle')
    : formatAuthorshipSealDetailWithReference(
      hasRewriteSignalComparison ? rewrittenAuthorshipSealDetail : authorshipSealDetail,
      sealDisplayReferenceScore ?? sealReferenceScore,
      t
    );
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
          <span className="report-stat-value" style={{ color: tier.color }}>{formatMetricPercent(calibratedReportAiScore(rawAuthorshipSignal), 0)}</span>
          <span className="report-stat-label">{t('report.summary.rawAiSignal')}</span>
        </div>
      )}
      {writingScore != null && (
        <div className="report-stat">
          <span className="report-stat-value" style={{ color: '#6366f1' }}>{formatMetricPercent(writingScore, 0)}</span>
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

  const renderTransformationDetails = (variant, pattern, summary, variantAiScore, ratingBadge = null) => {
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
          <em>{formatMetricPercent(calibratedReportAiScore(variantAiScore), 0)}</em>
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
      </div>
    );
  };

  const renderSignalGaugeStrip = (summary) => {
    if (!summary) return null;

    const signalGauges = [
      {
        key: 'aiRisk',
        label: t('report.transformation.smartSignals.aiRisk'),
        value: summary.adjustedAiRisk,
        tone: summary.adjustedAiRisk <= 20 ? 'positive' : summary.adjustedAiRisk <= 35 ? 'warning' : 'danger',
      },
      {
        key: 'humanAnchor',
        label: t('report.transformation.smartSignals.humanAnchor'),
        value: summary.humanAnchorDiscount,
        tone: 'positive',
      },
      {
        key: 'confidence',
        label: t('report.transformation.smartSignals.confidence'),
        value: summary.calibrationConfidence,
        tone: 'info',
      },
      {
        key: 'suppression',
        label: t('report.transformation.smartSignals.suppression'),
        value: summary.reportingSuppression,
        tone: 'neutral',
      },
    ].filter((item) => Number.isFinite(Number(item.value)));

    if (signalGauges.length === 0) return null;

    return (
      <div className="transformation-signal-gauge-strip" aria-label={t('report.transformation.smartSignalsLabel')}>
        {signalGauges.map((item) => {
          const value = Math.round(clampPercent(Number(item.value)) ?? 0);
          return (
            <div
              key={item.key}
              className={`transformation-signal-gauge is-${item.tone}`}
              role="meter"
              aria-label={item.label}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={value}
              aria-valuetext={t('report.transformation.smartSignalsValue', { label: item.label, value })}
            >
              <svg viewBox="0 0 92 50" aria-hidden="true" focusable="false">
                <path
                  className="transformation-signal-gauge-track"
                  d="M12 42 A34 34 0 0 1 80 42"
                  pathLength="100"
                />
                <path
                  className="transformation-signal-gauge-fill"
                  d="M12 42 A34 34 0 0 1 80 42"
                  pathLength="100"
                  strokeDasharray={`${value} 100`}
                />
              </svg>
              <strong>{value}%</strong>
              <span>{item.label}</span>
            </div>
          );
        })}
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
        {!hasRewriteSignalComparison && renderSignalGaugeStrip(transformationSummary)}
        <div
          className="transformation-authorship-seal"
          style={{
            '--rating-color': sealTone.color,
            '--rating-bg': sealTone.bg,
          }}
        >
          <span>{sealRatingCaption}</span>
          <strong title={sealRatingBadge.fullLabel || sealRatingLabel}>
            {sealRatingLabel}
          </strong>
          <em>
            {sealAuthorshipDetail}
          </em>
        </div>
      </div>
      <div className="transformation-chart">
        {hasRewriteSignalComparison ? (
          <div className="transformation-comparison-grid">
            {renderTransformationDetails('original', transformation, transformationSummary, transformationOriginalScore, originalColumnRatingBadge)}
            {renderTransformationDetails('rewritten', rewrittenTransformation, rewrittenTransformationSummary, transformationRewrittenScore, displayedRewrittenColumnRatingBadge)}
          </div>
        ) : (
          renderTransformationDetails('original', transformation, transformationSummary, transformationOriginalScore)
        )}
        {Array.isArray(transformation.evidence) && transformation.evidence.length > 0 && (
          <div className="transformation-evidence">
            {transformation.evidence.slice(0, 3).map((item) => (
              <span key={item}>{evidenceLabel(item, t)}</span>
            ))}
          </div>
        )}
        <p className="transformation-reference-note">{t('report.transformation.turnitinReferenceNote')}</p>
      </div>
    </section>
  ) : null;

  const scoreProfilePairs = transformationSignals.length > 0
    ? buildPairedTransformationSignals(
      transformationSignals,
      hasRewriteSignalComparison ? rewrittenTransformationSignals : []
    )
    : [];
  const scoreProfileGroups = groupTransformationSignals(scoreProfilePairs).map((group) => translatedGroup(group, t));
  const scoreProfileSummaryGroups = scoreProfileGroups.slice(0, 3).map((group) => {
    const signalEntries = group.signals.map((pair) => {
      const current = hasRewriteSignalComparison ? (pair.rewritten || pair.original) : pair.original;
      const baseline = hasRewriteSignalComparison ? pair.original : null;
      return {
        pair,
        current,
        improvement: getTransformationSignalImprovement(current, baseline),
      };
    });
    const improvedCount = signalEntries.filter((entry) => entry.improvement).length;
    const topEntry = [...signalEntries].sort((a, b) => Number(b.current?.value || 0) - Number(a.current?.value || 0))[0];
    return {
      ...group,
      improvedCount,
      topSignal: topEntry?.current ? translatedSignal(topEntry.current, t) : null,
      topValue: topEntry?.current?.value,
    };
  });
  const scoreProfileSummaryText = hasRewriteSignalComparison
    ? t('report.scoreProfile.summaryRewrite')
    : transformationSummary?.summary || t('report.scoreProfile.summaryOriginal');
  const scoreProfileTabIds = scoreProfileSummaryGroups.map((group) => group.id);
  const currentActiveTab = scoreProfileTabIds.includes(activeProfileTab)
    ? activeProfileTab
    : scoreProfileTabIds[0];
  const activeGroup = scoreProfileGroups.find((g) => g.id === currentActiveTab);
  const focusScoreProfileTab = (tabId) => {
    if (!tabId) return;
    requestAnimationFrame(() => {
      document.getElementById(`score-profile-tab-${tabId}`)?.focus();
    });
  };
  const handleScoreProfileTabKeyDown = (event, groupId) => {
    const currentIndex = scoreProfileTabIds.indexOf(groupId);
    if (currentIndex < 0) return;

    const lastIndex = scoreProfileTabIds.length - 1;
    const nextByKey = {
      ArrowRight: currentIndex === lastIndex ? 0 : currentIndex + 1,
      ArrowDown: currentIndex === lastIndex ? 0 : currentIndex + 1,
      ArrowLeft: currentIndex === 0 ? lastIndex : currentIndex - 1,
      ArrowUp: currentIndex === 0 ? lastIndex : currentIndex - 1,
      Home: 0,
      End: lastIndex,
    };
    if (!(event.key in nextByKey)) return;

    event.preventDefault();
    const nextTabId = scoreProfileTabIds[nextByKey[event.key]];
    setActiveProfileTab(nextTabId);
    focusScoreProfileTab(nextTabId);
  };

  const scoreProfileSection = scoreProfileGroups.length > 0 ? (
    <section className="score-profile-section" aria-label={t('report.scoreProfile.sectionLabel')}>
      <div className="score-profile-head">
        <div>
          <span className="score-profile-kicker">{t('report.scoreProfile.kicker')}</span>
          <h2>{t('report.scoreProfile.title')}</h2>
          <p>{scoreProfileSummaryText}</p>
        </div>
        <div className="score-profile-stat">
          <strong>{scoreProfilePairs.length}</strong>
          <span>{t('report.scoreProfile.trackedSignals')}</span>
        </div>
      </div>
      <div
        className="score-profile-summary-grid"
        role="tablist"
        aria-label={t('report.scoreProfile.sectionLabel')}
      >
        {scoreProfileSummaryGroups.map((group) => {
          const isActive = group.id === currentActiveTab;
          return (
            <button
              key={group.id}
              type="button"
              role="tab"
              id={`score-profile-tab-${group.id}`}
              aria-selected={isActive}
              aria-controls={`score-profile-panel-${group.id}`}
              tabIndex={isActive ? 0 : -1}
              onClick={() => setActiveProfileTab(group.id)}
              onKeyDown={(event) => handleScoreProfileTabKeyDown(event, group.id)}
              className={`score-profile-summary-card is-${group.id}${isActive ? ' is-active' : ''}`}
            >
              <span>{group.label}</span>
              <strong>{group.topSignal?.label || t('report.scoreProfile.noSignal')}</strong>
              <p>{group.description || t('report.scoreProfile.groupFallback')}</p>
              <div className="score-profile-summary-meta">
                {group.topValue != null && (
                  <em>{t('report.scoreProfile.leadingSignal', { value: Math.round(group.topValue) })}</em>
                )}
                {hasRewriteSignalComparison && (
                  <em>{t('report.scoreProfile.improvedSignals', { count: group.improvedCount })}</em>
                )}
              </div>
            </button>
          );
        })}
      </div>

      {activeGroup && (
        <div
          className="score-profile-focused-detail"
          key={activeGroup.id}
          id={`score-profile-panel-${activeGroup.id}`}
          role="tabpanel"
          tabIndex={0}
          aria-labelledby={`score-profile-tab-${activeGroup.id}`}
        >
          <section className={`score-profile-group is-${activeGroup.id}`}>
            <div className="score-profile-group-head">
              <div>
                <h3>{activeGroup.label}</h3>
                {activeGroup.description && <p>{activeGroup.description}</p>}
              </div>
              <span>{t('report.transformation.signals', { count: activeGroup.signals.length })}</span>
            </div>
            <div className="score-profile-bars">
              {activeGroup.signals.map((pair) => {
                const originalSignal = pair.original ? translatedSignal(pair.original, t) : null;
                const rewrittenSignal = pair.rewritten ? translatedSignal(pair.rewritten, t) : null;
                const currentSignal = hasRewriteSignalComparison ? (rewrittenSignal || originalSignal) : originalSignal;
                const improvement = getTransformationSignalImprovement(pair.rewritten, pair.original);
                const signalColor = currentSignal?.color || originalSignal?.color || pair.color || '#0f766e';
                return (
                  <div key={pair.key} className="score-profile-row">
                     <div className="score-profile-row-head">
                       <span title={currentSignal?.description || pair.description}>{currentSignal?.label || pair.label}</span>
                       <strong>{currentSignal?.value != null ? formatMetricPercent(currentSignal.value, 0) : t('report.transformation.notPresent')}</strong>
                     </div>
                     <div className="score-profile-track" aria-hidden="true">
                       {originalSignal?.value != null && hasRewriteSignalComparison && (
                         <i className="score-profile-fill is-original" style={{ width: `${originalSignal.value}%` }} />
                       )}
                       {currentSignal?.value != null && (
                         <i
                           className="score-profile-fill is-current"
                           style={{ width: `${currentSignal.value}%`, '--score-profile-color': signalColor }}
                         />
                       )}
                     </div>
                     <div className="score-profile-row-foot">
                       {hasRewriteSignalComparison && originalSignal?.value != null && rewrittenSignal?.value != null ? (
                         <span>{t('report.scoreProfile.originalToRewritten', {
                           original: Math.round(originalSignal.value),
                           rewritten: Math.round(rewrittenSignal.value),
                         })}</span>
                       ) : (
                         <span>{currentSignal?.description}</span>
                       )}
                       {improvement && (
                         <em>{t('report.transformation.improvedFromTo', {
                           from: Math.round(improvement.from),
                           to: Math.round(improvement.to),
                         })}</em>
                       )}
                     </div>
                  </div>
                );
              })}
            </div>
          </section>
        </div>
      )}
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
        <span>{formatMetricPercent(calibratedReportAiScore(rewriteResultSummary?.original_ai_authorship ?? rewriteResultSummary?.original_risk), 0)}</span>
        <small>{t('report.rewrite.aiBefore')}</small>
      </div>
      <div className="rewrite-summary-stat">
        <span>{formatMetricPercent(calibratedReportAiScore(rewriteResultSummary?.rewritten_ai_authorship ?? rewriteResultSummary?.rewrite_risk), 0)}</span>
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
                <div className="rewrite-action-group">
                  <button
                    type="button"
                    className="rewrite-btn"
                    onClick={handleRewrite}
                    disabled={rewriteLoading || rewriteCanceling}
                  >
                    {rewriteLoading ? t('report.rewrite.starting') : rewriteInProgress ? t('report.rewrite.resume') : t('report.rewrite.rewriteAiSections')}
                  </button>
                  {rewriteTokenEstimate && (
                    <strong className="rewrite-token-estimate">{rewriteTokenEstimate}</strong>
                  )}
                  <span>{t('report.rewrite.emailPdfNotice')}</span>
                </div>
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
                  <span>{t('report.rewrite.emailPdfProgress')}</span>
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
              transformationScorecard
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
              <div className="submitted-content-actions">
                <div className="submitted-content-count">
                  <strong>{submittedContent.highlightedCount}</strong>
                  <span>{t('report.submitted.highlightedSections')}</span>
                </div>
                {canEditSubmittedDraft && (
                  <button
                    type="button"
                    className="btn btn-secondary submitted-edit-button"
                    onClick={() => {
                      openSubmittedEditor();
                      setSubmittedRescanError(null);
                      setSubmittedHighlightRanges((current) => buildSubmittedHighlightRanges(current));
                    }}
                  >
                    {t('report.submitted.editor.editDraft')}
                  </button>
                )}
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
                    {renderSubmittedSignalGauge()}
                    <div className="submitted-panel-meta">
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
                      <h2>{t('report.submitted.editor.title')}</h2>
                      <p>{t('report.submitted.editor.priorScanNotice')}</p>
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
                            {submittedDraftTokensRequired > 0
                              ? t('scan.tokensRequired', { count: submittedDraftTokensRequired })
                              : t('scan.freeScan')}
                          </span>
                        </div>
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
                          }}
                          onScroll={syncSubmittedHighlightScroll}
                          spellCheck="true"
                        />
                      </div>
                      {(submittedRescanStatus || submittedRescanError) && (
                        <div className={`submitted-rescan-status${submittedRescanError ? ' is-error' : ''}`}>
                          {submittedRescanError || submittedRescanStatus}
                        </div>
                      )}
                      <div className="submitted-tracked-preview" aria-label={t('report.submitted.editor.trackedPreview')}>
                        <div className="submitted-tracked-head">
                          <strong>{t('report.submitted.editor.trackedPreview')}</strong>
                          <span>{t('report.submitted.editor.trackedPreviewBody')}</span>
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

                    <aside className="submitted-affected-panel" aria-label={t('report.submitted.editor.affectedSentences')}>
                      <div className="submitted-affected-head">
                        <span>{t('report.submitted.editor.affectedSentences')}</span>
                        <strong>{affectedSegments.length}</strong>
                      </div>
                      <div className="submitted-affected-list">
                        {affectedSegments.map((segment) => {
                          const signal = segment.primarySignal || segment.signals[0];
                          const isSelected = selectedSegment?.id === segment.id;
                          return (
                            <button
                              key={`affected-${segment.id}`}
                              type="button"
                              className={`submitted-affected-item${isSelected ? ' is-selected' : ''}`}
                              onClick={() => {
                                setSelectedSegmentId(segment.id);
                                focusSentenceInSubmittedEditor(segment);
                              }}
                            >
                              <span>{segment.sentence_id}</span>
                              <strong>{signalLabel(signal.key, signal.label, t)}</strong>
                              <em>{segment.text}</em>
                            </button>
                          );
                        })}
                      </div>

                      <div className="submitted-editor-detail">
                        {selectedSegment?.primarySignal ? (
                          <>
                            <span className="submitted-panel-kicker">{selectedSegment.sentence_id}</span>
                            <h3>{signalLabel(selectedSegment.primarySignal.key, selectedSegment.primarySignal.label, t)}</h3>
                            <div className="submitted-editor-sentence">
                              <span>{t('report.submitted.editor.affectedSentence')}</span>
                              <p>{selectedSegment.text}</p>
                            </div>
                            {renderSubmittedSignalGauge()}
                            <div className="submitted-panel-meta">
                              <span>{selectedSentenceDraftStatus}</span>
                              {selectedSegment.primarySignal.tier && (
                                <span>{t('report.submitted.priority', { value: t(`report.severities.${selectedSegment.primarySignal.tier}`, { defaultValue: selectedSegment.primarySignal.tier }) })}</span>
                              )}
                            </div>
                            <div className="submitted-editor-signal">
                              <span>{t('report.submitted.editor.signal')}</span>
                              <p>{signalDescription(selectedSegment.primarySignal.key, selectedSegment.primarySignal.description, t)}</p>
                            </div>
                            {selectedSegment.primarySignal.recommendation && (
                              <div className="submitted-panel-note">
                                <span>{t('report.submitted.recommendation')}</span>
                                <p>{selectedSegment.primarySignal.recommendation}</p>
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
          </section>
        )}

        {scoreProfileSection}

      </div>
    </main>
  );
}
