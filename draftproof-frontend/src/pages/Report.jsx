import { useCallback, useEffect, useRef, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getReport, createRewrite, getRewriteStatus, getRewriteReport, buildApiEventUrl } from '../api/draftproofApi';
import ErrorReload from '../components/ErrorReload';

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

function formatDate(iso) {
  if (!iso) return '';
  return new Date(iso).toLocaleDateString('en-SG', { day: 'numeric', month: 'long', year: 'numeric', hour: '2-digit', minute: '2-digit' });
}

function pct(value) {
  if (value == null || Number.isNaN(Number(value))) return null;
  return `${(Number(value) * 100).toFixed(0)}%`;
}

function formatMetricPercent(value) {
  if (value == null || Number.isNaN(Number(value))) return '—';
  const number = Number(value);
  const percent = Math.abs(number) <= 1 ? number * 100 : number;
  return `${percent.toFixed(1)}%`;
}

function formatSignedDelta(original, next) {
  if (original == null || next == null) return '—';
  const delta = Number(next) - Number(original);
  if (Number.isNaN(delta)) return '—';
  if (delta > 0) return `+${delta}`;
  return `${delta}`;
}

function countRewriteFindings(findings) {
  if (!findings || typeof findings !== 'object') return null;
  return ['critical', 'high', 'medium', 'low', 'info'].reduce((total, tier) => {
    const rows = findings[tier];
    return total + (Array.isArray(rows) ? rows.length : 0);
  }, 0);
}

function buildRewriteResultSummary(rewriteReport) {
  const summary = rewriteReport?.summary || rewriteReport?.rewrite_summary || {};
  const detectScores = summary.detect_scores || {};
  const originalScan = summary.detect_scan_original || {};
  const rewrittenScan = summary.detect_scan_rewritten || {};
  const originalBadge = originalScan.ai_risk_badge || {};
  const rewrittenBadge = rewrittenScan.ai_risk_badge || {};
  const originalFindings = detectScores.original_findings ?? countRewriteFindings(originalScan.findings);
  const rewrittenFindings = detectScores.rewritten_findings ?? countRewriteFindings(rewrittenScan.findings);
  const changedSentences = (rewriteReport?.sentence_comparison || []).filter(
    (row) => String(row.orig_sentence || '').trim() !== String(row.new_sentence || '').trim()
  ).length;

  return {
    original_risk: detectScores.original_ai ?? originalBadge.ai_likelihood_score ?? summary.original_risk,
    rewrite_risk: detectScores.rewritten_ai ?? rewrittenBadge.ai_likelihood_score ?? summary.final_risk,
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

function findingEvidenceSummary(issue) {
  if (issue.evidence?.summary) return issue.evidence.summary;
  if (typeof issue.evidence === 'string') return issue.evidence;
  return '';
}

function formatRewriteStatus(status) {
  if (status === 'pending') return 'Queued';
  if (status === 'processing') return 'Rewriting AI sections';
  if (status === 'retrying') return 'Retrying rewrite';
  if (status === 'completed') return 'Rewrite complete';
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
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedIssue, setExpandedIssue] = useState(null);
  const [rewriteJob, setRewriteJob] = useState(null);
  const [rewriteLoading, setRewriteLoading] = useState(false);
  const [rewriteError, setRewriteError] = useState(null);
  const [rewriteStartedHere, setRewriteStartedHere] = useState(false);
  const [rewriteSseUnavailable, setRewriteSseUnavailable] = useState(false);
  const [rewriteNotice, setRewriteNotice] = useState(null);
  const [rewriteResultSummary, setRewriteResultSummary] = useState(null);
  const rewritePollRef = useRef(null);
  const rewriteEventSourceRef = useRef(null);

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
    if (normalizedJob?.status && normalizedJob.status !== 'failed') {
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
      if (data.status === 'completed') {
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
      return undefined;
    }

    let cancelled = false;
    getRewriteReport(completedRewrite.id)
      .then(({ data }) => {
        if (cancelled) return;
        setRewriteResultSummary(buildRewriteResultSummary(data));
      })
      .catch(() => {
        if (!cancelled) setRewriteResultSummary(null);
      });

    return () => {
      cancelled = true;
    };
  }, [rewriteJob, report?.rewrite]);

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
  const issueCounts = { critical: 0, high: 0, medium: 0, low: 0, info: 0 };
  report.issues.forEach((iss) => { if (issueCounts[iss.severity] !== undefined) issueCounts[iss.severity]++; });

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
  const canStartRewrite = hasAIFindings && !hasRewriteResult;
  const rewriteProgress = currentRewrite
    ? Math.max(0, Math.min(100, Number(currentRewrite.progress_percent) || (rewriteInProgress ? 5 : hasCompletedRewrite ? 100 : 0)))
    : 0;
  const rewriteProgressMessage = normalizeRewriteProgressMessage(
    currentRewrite?.progress_message,
    currentRewrite?.status
  );
  const showRewriteProgress = !hasRewriteResult && (
    rewriteStartedHere || rewriteInProgress || rewriteLoading || rewriteError
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

  return (
    <main className="dash-shell">
      <RewriteNoticeDialog
        open={Boolean(rewriteNotice)}
        title={rewriteNotice?.title}
        message={rewriteNotice?.message}
        onClose={() => setRewriteNotice(null)}
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
          {(canStartRewrite || rewriteLoading || rewriteInProgress) && (
            <button
              type="button"
              className="rewrite-btn"
              onClick={handleRewrite}
              disabled={rewriteLoading}
            >
              {rewriteLoading ? 'Starting rewrite...' : rewriteInProgress ? 'Resume Rewrite' : 'Rewrite AI Sections'}
            </button>
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
            </div>
          </div>
        )}

        {/* Summary bar */}
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
          {Object.entries(issueCounts).filter(([, v]) => v > 0).map(([sev, count]) => {
            const sc = SEVERITY_CONFIG[sev];
            return (
              <div key={sev} className="report-stat">
                <span className="report-stat-value" style={{ color: sc.color }}>{count}</span>
                <span className="report-stat-label">{sc.label}</span>
              </div>
            );
          })}
          {aiScore != null && (
            <div className="report-stat">
              <span className="report-stat-value" style={{ color: tier.color }}>{Number(aiScore).toFixed(2)}%</span>
              <span className="report-stat-label">AI Score</span>
            </div>
          )}
          {writingScore != null && (
            <div className="report-stat">
              <span className="report-stat-value" style={{ color: '#6366f1' }}>{Number(writingScore).toFixed(2)}%</span>
              <span className="report-stat-label">Writing Score</span>
            </div>
          )}
        </div>

        {hasRewriteResult && (
          <div className="report-rewrite-summary-bar">
            <div className="rewrite-summary-icon" aria-hidden="true">
              <span>
                <svg width="42" height="42" viewBox="0 0 42 42" fill="none">
                  <circle cx="21" cy="21" r="15" fill="currentColor"/>
                  <path d="M14 21.5l4.5 4.5L28.5 16" stroke="#fff" strokeWidth="3.2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </span>
            </div>
            <div className="rewrite-summary-main">
              <span className="rewrite-summary-kicker">Rewrite complete</span>
              <strong>AI sections rewritten</strong>
            </div>
            <div className="rewrite-summary-stat">
              <span>{formatMetricPercent(rewriteResultSummary?.original_risk)}</span>
              <small>Original risk</small>
            </div>
            <div className="rewrite-summary-stat">
              <span>{formatMetricPercent(rewriteResultSummary?.rewrite_risk)}</span>
              <small>Rewrite risk</small>
            </div>
            <div className="rewrite-summary-stat">
              <span>{formatSignedDelta(rewriteResultSummary?.original_findings, rewriteResultSummary?.rewritten_findings)}</span>
              <small>Findings change</small>
            </div>
            <div className="rewrite-summary-stat">
              <span>{rewriteResultSummary?.changed_sentences ?? '—'}</span>
              <small>Sentences changed</small>
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
        )}

        {/* Download links */}
        {report.report_pdf_url && (
          <div className="report-downloads">
            <a href={report.report_pdf_url} target="_blank" rel="noopener noreferrer" className="btn btn-primary">
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none" style={{ marginRight: 6 }}>
                <path d="M3 10v2.5A1.5 1.5 0 004.5 14h7a1.5 1.5 0 001.5-1.5V10M8 2v8M5 7l3 3 3-3" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              Download PDF
            </a>
          </div>
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
