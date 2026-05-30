import { useEffect, useRef, useState } from 'react';
import { Link, Navigate, useParams } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { getRewriteStatus, getRewriteReport, getRewriteDownload } from '../api/draftproofApi';
import ErrorReload from '../components/ErrorReload';
import { useAuth } from '../context/AuthContext';
import {
  requiresRewriteAuthorReview,
  requiresRewriteExternalReview,
} from './report/reportHelpers';

function normalizeSentence(value) {
  return String(value || '').replace(/\s+/g, ' ').trim();
}

function countWords(value) {
  const normalized = String(value || '').trim();
  return normalized ? normalized.split(/\s+/).length : 0;
}

function tokenizeDiffText(text) {
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

function buildSplitDiff(originalText, rewrittenText) {
  if (!originalText || !rewrittenText || normalizeSentence(originalText) === normalizeSentence(rewrittenText)) {
    return null;
  }

  const parts = lcsTokenDiff(tokenizeDiffText(originalText), tokenizeDiffText(rewrittenText));
  return {
    original: parts.filter((part) => part.type !== 'insert'),
    rewritten: parts.filter((part) => part.type !== 'delete'),
  };
}

function renderDiffParts(parts, side) {
  return (parts || []).map((part, index) => {
    const className = part.type === 'equal'
      ? 'rewrite-diff-equal'
      : side === 'original'
        ? 'rewrite-diff-delete'
        : 'rewrite-diff-insert';
    return (
      <span key={`${side}-${part.type}-${index}`} className={className}>
        {part.text}
      </span>
    );
  });
}

function renderPlaceholderText(value) {
  return String(value || '').split(/(\[[^\[\]]+\])/g).map((part, index) => {
    if (!part) return null;
    if (/^\[[^\[\]]+\]$/.test(part)) {
      return <mark key={`${part}-${index}`} className="rewrite-placeholder">{part}</mark>;
    }
    return <span key={`${part}-${index}`}>{part}</span>;
  });
}

export default function Rewrite() {
  const { rewriteId } = useParams();
  const { refreshBalance } = useAuth();
  const { t } = useTranslation();
  const [rewrite, setRewrite] = useState(null);
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(Boolean(rewriteId));
  const [error, setError] = useState(null);
  const [copyStatus, setCopyStatus] = useState('idle');
  const originalDiffRef = useRef(null);
  const rewrittenDiffRef = useRef(null);
  const diffScrollSyncingRef = useRef(false);

  useEffect(() => {
    if (!rewriteId) return undefined;

    let cancelled = false;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const { data: status } = await getRewriteStatus(rewriteId);
        if (cancelled) return;
        setRewrite(status);
        if (status.status !== 'completed') {
          setError(status.status === 'failed' ? (status.error || t('rewritePage.failed')) : t('rewritePage.incomplete'));
          return;
        }

        const { data: rewriteReport } = await getRewriteReport(rewriteId);
        if (cancelled) return;
        setReport(rewriteReport);
        refreshBalance();
      } catch (err) {
        if (!cancelled) {
          setError(err.response?.data?.detail || t('rewritePage.loadFailed'));
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [rewriteId, refreshBalance, t]);

  if (!rewriteId) {
    return <Navigate to="/reports" replace />;
  }

  const handleDownload = async (fmt) => {
    const downloadWindow = window.open('about:blank', '_blank');
    if (downloadWindow) {
      downloadWindow.opener = null;
    }

    try {
      const { data } = await getRewriteDownload(rewriteId, fmt);
      if (data.url) {
        if (downloadWindow) {
          downloadWindow.location.replace(data.url);
        } else {
          window.location.assign(data.url);
        }
      } else {
        downloadWindow?.close();
        setError(t('rewritePage.downloadUnavailable'));
      }
    } catch (err) {
      downloadWindow?.close();
      setError(err.response?.data?.detail || t('rewritePage.downloadFailed'));
    }
  };

  const handleCopyRewrittenDocument = async () => {
    if (!report?.final_text) return;

    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(report.final_text);
      } else {
        copyTextFallback(report.final_text, t('rewritePage.copyCommandFailed'));
      }
      setCopyStatus('copied');
      window.setTimeout(() => setCopyStatus('idle'), 1800);
    } catch {
      setCopyStatus('error');
      window.setTimeout(() => setCopyStatus('idle'), 2200);
    }
  };

  const syncDiffScroll = (source) => {
    const sourceEl = source === 'original' ? originalDiffRef.current : rewrittenDiffRef.current;
    const targetEl = source === 'original' ? rewrittenDiffRef.current : originalDiffRef.current;
    if (!sourceEl || !targetEl || diffScrollSyncingRef.current) return;

    const sourceTopMax = sourceEl.scrollHeight - sourceEl.clientHeight;
    const targetTopMax = targetEl.scrollHeight - targetEl.clientHeight;
    const sourceLeftMax = sourceEl.scrollWidth - sourceEl.clientWidth;
    const targetLeftMax = targetEl.scrollWidth - targetEl.clientWidth;
    const topRatio = sourceTopMax > 0 ? sourceEl.scrollTop / sourceTopMax : 0;
    const leftRatio = sourceLeftMax > 0 ? sourceEl.scrollLeft / sourceLeftMax : 0;

    diffScrollSyncingRef.current = true;
    targetEl.scrollTop = targetTopMax > 0 ? targetTopMax * topRatio : 0;
    targetEl.scrollLeft = targetLeftMax > 0 ? targetLeftMax * leftRatio : 0;
    window.requestAnimationFrame(() => {
      diffScrollSyncingRef.current = false;
    });
  };

  const summary = report?.summary || report?.rewrite_summary || {};
  const mitigationPlan = summary.mitigation_plan || report?.mitigation_plan || {};
  const markedSuggestions = (
    mitigationPlan.marked_content_suggestions ||
    summary.marked_content_suggestions ||
    report?.marked_content_suggestions ||
    []
  ).filter(Boolean);
  const manualSuggestions = (summary.manual_suggestions || report?.manual_suggestions || []).filter(Boolean);
  const authorProxyContext = summary.author_proxy_context || report?.author_proxy_context || {};
  const authorshipEvidence = report?.authorship_evidence || summary.authorship_evidence || null;
  const authorReviewCards = (
    summary.author_review_cards ||
    authorProxyContext.review_cards ||
    report?.author_review_cards ||
    []
  ).filter(Boolean);
  const finalizationSummary = {
    ...summary,
    status: report?.status || summary.status || '',
    no_text_change: summary.no_text_change === true || report?.no_text_change === true,
    author_proxy_context: authorProxyContext,
  };
  const requiresAuthorReview = requiresRewriteAuthorReview(finalizationSummary);
  const requiresExternalReview = requiresRewriteExternalReview(finalizationSummary);
  const requiresManualReview = requiresAuthorReview || requiresExternalReview;
  const outcome = requiresAuthorReview
    ? 'rewrite_candidate_generated_needs_author_review'
    : requiresExternalReview
    ? 'rewrite_candidate_generated_needs_external_review'
    : summary.outcome || (rewrite?.status === 'completed' ? 'completed' : rewrite?.status || '');
  const outcomeLabel = outcome
    ? t(`rewritePage.outcomes.${outcome}`, { defaultValue: outcome.replaceAll('_', ' ') })
    : '';
  const outcomeTone = requiresManualReview
    ? { background: '#fffbeb', color: '#92400e', borderColor: '#fde68a' }
    : { background: '#f0fdf4', color: '#15803d', borderColor: '#bbf7d0' };
  const scanId = rewrite?.scan_id;
  const rewrittenWordCount = countWords(report?.final_text);
  const originalText = report?.original_text || summary.original_text || '';
  const documentDiff = buildSplitDiff(originalText, report?.final_text || '');

  return (
    <main className="dash-shell">
      <div className="container">
        <Link to={scanId ? `/report/${scanId}` : '/reports'} className="report-back">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M10 3L5 8l5 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          {t('rewritePage.back')}
        </Link>

        <div className="report-hero" style={{ marginTop: 16 }}>
          <div className="report-hero-info">
            <div className="report-eyebrow">{t('rewritePage.eyebrow')}</div>
            <h1>{t('rewritePage.title')}</h1>
          </div>
          {outcome && (
            <div
              className="report-hero-tier"
              style={{ background: outcomeTone.background, border: `1px solid ${outcomeTone.borderColor}` }}
            >
              <span style={{ color: outcomeTone.color, fontWeight: 700 }}>
                {outcomeLabel}
              </span>
            </div>
          )}
        </div>

        {loading && (
          <div className="report-loading">
            <div className="report-pulse" />
            <p>{t('rewritePage.loading')}</p>
          </div>
        )}

        {error && <ErrorReload message={error} />}

        {requiresManualReview && report?.final_text && (
          <section className="rewrite-status-alert">
            <strong>{t(requiresAuthorReview ? 'rewritePage.authorReviewTitle' : 'rewritePage.externalReviewTitle')}</strong>
            <p>{t(requiresAuthorReview ? 'rewritePage.authorReviewCopy' : 'rewritePage.externalReviewCopy')}</p>
          </section>
        )}

        {documentDiff && (
          <section className="rewrite-diff-section" aria-label={t('rewritePage.compareChanges')}>
            <div className="rewrite-diff-heading">
              <div>
                <span className="rewrite-review-kicker">{t('rewritePage.compareKicker')}</span>
                <h3>{t('rewritePage.compareChanges')}</h3>
              </div>
              <div className="rewrite-diff-legend" aria-hidden="true">
                <span className="is-delete">{t('rewritePage.removedText')}</span>
                <span className="is-insert">{t('rewritePage.addedText')}</span>
              </div>
            </div>
            <p className="rewrite-review-copy">{t('rewritePage.compareChangesCopy')}</p>
            <div className="rewrite-diff-grid">
              <article className="rewrite-diff-pane">
                <div className="rewrite-diff-pane-head">
                  <span>{t('rewritePage.original')}</span>
                  <strong>{t('rewritePage.word', { count: countWords(originalText) })}</strong>
                </div>
                <div
                  ref={originalDiffRef}
                  className="rewrite-diff-body"
                  onScroll={() => syncDiffScroll('original')}
                >
                  {renderDiffParts(documentDiff.original, 'original')}
                </div>
              </article>
              <article className="rewrite-diff-pane">
                <div className="rewrite-diff-pane-head">
                  <span>{t('rewritePage.rewritten')}</span>
                  <strong>{t('rewritePage.word', { count: rewrittenWordCount })}</strong>
                </div>
                <div
                  ref={rewrittenDiffRef}
                  className="rewrite-diff-body"
                  onScroll={() => syncDiffScroll('rewritten')}
                >
                  {renderDiffParts(documentDiff.rewritten, 'rewritten')}
                </div>
              </article>
            </div>
          </section>
        )}

        {authorshipEvidence && (
          <section className="rewrite-review-section">
            <div className="rewrite-review-heading">
              <div>
                <span className="rewrite-review-kicker">{t('authorshipEvidence.scanKicker')}</span>
                <h3>{t('authorshipEvidence.rewriteTitle')}</h3>
              </div>
            </div>
            <p className="rewrite-review-copy">{t('authorshipEvidence.rewriteCopy')}</p>
            {authorshipEvidence.preserved_ideas?.length > 0 && (
              <div className="rewrite-target-block">
                <span>{t('authorshipEvidence.preservedTitle')}</span>
                {authorshipEvidence.preserved_ideas.slice(0, 8).map((p, i) => <p key={i}>{p.text}</p>)}
              </div>
            )}
            {authorshipEvidence.present_markers?.length > 0 && (
              <div className="rewrite-target-block">
                <span>{t('authorshipEvidence.presentTitle')}</span>
                <ul className="signal-list">
                  {authorshipEvidence.present_markers.map((m, i) => <li key={`${m.signal}-${i}`}>{m.label}</li>)}
                </ul>
              </div>
            )}
            {authorshipEvidence.thin_signals?.length > 0 && (
              <div className="rewrite-addition-block">
                <span>{t('authorshipEvidence.thinTitle')}</span>
                <p className="rewrite-review-copy">{t('authorshipEvidence.thinCopy')}</p>
                <ul className="signal-list">
                  {authorshipEvidence.thin_signals.map((tn, i) => <li key={`${tn.signal}-${i}`}>{tn.action}</li>)}
                </ul>
              </div>
            )}
          </section>
        )}

        {report?.final_text && (
          <section className="rewritten-document-section">
            <p className="rewritten-document-remark">{t('rewritePage.rewrittenDocumentRemark')}</p>
            <div className="rewritten-document-heading">
              <div className="rewritten-document-title">
                <h3>{t('rewritePage.rewrittenDocument')}</h3>
                <span>{t('rewritePage.word', { count: rewrittenWordCount })}</span>
              </div>
              <button
                type="button"
                className={`copy-rewrite-btn${copyStatus === 'copied' ? ' is-copied' : ''}${copyStatus === 'error' ? ' has-error' : ''}`}
                onClick={handleCopyRewrittenDocument}
                aria-live="polite"
              >
                {copyStatus === 'copied' ? t('rewritePage.copied') : copyStatus === 'error' ? t('rewritePage.copyFailed') : t('rewritePage.copy')}
              </button>
            </div>
            <div className="rewritten-document-content">
              {report.final_text}
            </div>
          </section>
        )}

        {markedSuggestions.length > 0 && (
          <section className="rewrite-review-section">
            <div className="rewrite-review-heading">
              <div>
                <span className="rewrite-review-kicker">{t('rewritePage.reviewRequired')}</span>
                <h3>{t('rewritePage.suggestedAdditions')}</h3>
              </div>
              <span className="rewrite-review-count">{markedSuggestions.length}</span>
            </div>
            <p className="rewrite-review-copy">
              {t('rewritePage.suggestedAdditionsCopy')}
            </p>
            <div className="rewrite-suggestion-grid">
              {markedSuggestions.map((item, i) => (
                <article className="rewrite-suggestion-card" key={`${item.component || 'suggestion'}-${i}`}>
                  <div className="rewrite-suggestion-meta">
                    <span>{item.priority ? String(item.priority).replaceAll('_', ' ') : t('rewritePage.suggestion', { count: i + 1 })}</span>
                    <span>{item.where || t('rewritePage.flaggedText')}</span>
                  </div>
                  <h4>{item.title || t('rewritePage.suggestedReviewAddition')}</h4>
                  {(item.target_text || item.evidence) && (
                    <div className="rewrite-target-block">
                      <span>{t('rewritePage.targetText')}</span>
                      <p>{item.target_text || item.evidence}</p>
                    </div>
                  )}
                  <div className="rewrite-addition-block">
                    <span>{t('rewritePage.suggestedAddition')}</span>
                    <p>{renderPlaceholderText(item.suggested_addition)}</p>
                  </div>
                  {(item.why_it_helps || item.user_note) && (
                    <div className="rewrite-review-note">
                      {item.why_it_helps && <p>{item.why_it_helps}</p>}
                      {item.user_note && <p>{item.user_note}</p>}
                    </div>
                  )}
                </article>
              ))}
            </div>
          </section>
        )}

        {authorReviewCards.length > 0 && (
          <section className="rewrite-review-section">
            <div className="rewrite-review-heading">
              <div>
                <span className="rewrite-review-kicker">{t('rewritePage.reviewRequired')}</span>
                <h3>{t('rewritePage.authorReviewCards')}</h3>
              </div>
              <span className="rewrite-review-count">{authorReviewCards.length}</span>
            </div>
            <p className="rewrite-review-copy">
              {t('rewritePage.authorReviewCardsCopy')}
            </p>
            <div className="rewrite-suggestion-grid">
              {authorReviewCards.slice(0, 12).map((item, i) => (
                <article className="rewrite-suggestion-card" key={`${item.card_id || item.kind || 'author-review'}-${i}`}>
                  <div className="rewrite-suggestion-meta">
                    <span>{item.provenance ? String(item.provenance).replaceAll('_', ' ') : t('rewritePage.authorProxyDraft')}</span>
                    <span>{item.where || item.bucket || t('rewritePage.reviewManually')}</span>
                  </div>
                  <h4>{item.instruction || item.lever || t('rewritePage.authorTask')}</h4>
                  {item.target_text && (
                    <div className="rewrite-target-block">
                      <span>{t('rewritePage.targetText')}</span>
                      <p>{item.target_text}</p>
                    </div>
                  )}
                  {item.user_input_needed && (
                    <div className="rewrite-addition-block">
                      <span>{t('rewritePage.needed')}</span>
                      <p>{item.user_input_needed}</p>
                    </div>
                  )}
                  {item.author_task && (
                    <div className="rewrite-review-note"><p>{item.author_task}</p></div>
                  )}
                </article>
              ))}
            </div>
          </section>
        )}

        {manualSuggestions.length > 0 && (
          <section className="rewrite-review-section">
            <div className="rewrite-review-heading">
              <div>
                <span className="rewrite-review-kicker">{t('rewritePage.manualOptions')}</span>
                <h3>{t('rewritePage.manualSuggestions')}</h3>
              </div>
              <span className="rewrite-review-count">{manualSuggestions.length}</span>
            </div>
            <div className="rewrite-suggestion-grid">
              {manualSuggestions.slice(0, 12).map((item, i) => (
                <article className="rewrite-suggestion-card" key={`${item.finding_id || 'manual'}-${i}`}>
                  <div className="rewrite-suggestion-meta">
                    <span>{item.scanner_target || item.finding_type || t('rewritePage.suggestion', { count: i + 1 })}</span>
                    <span>{item.rejection_reason || t('rewritePage.reviewManually')}</span>
                  </div>
                  {item.original_sentence && (
                    <div className="rewrite-target-block">
                      <span>{t('rewritePage.original')}</span>
                      <p>{item.original_sentence}</p>
                    </div>
                  )}
                  {item.suggested_sentence && (
                    <div className="rewrite-addition-block">
                      <span>{t('rewritePage.suggestedSentence')}</span>
                      <p>{item.suggested_sentence}</p>
                    </div>
                  )}
                  {item.why_review_manually && (
                    <div className="rewrite-review-note"><p>{item.why_review_manually}</p></div>
                  )}
                </article>
              ))}
            </div>
          </section>
        )}

        {rewrite?.status === 'completed' && (
          <div className="report-downloads">
            <button type="button" className="btn btn-primary" onClick={() => handleDownload('pdf')}>
              {t('rewritePage.downloadPdf')}
            </button>
          </div>
        )}
      </div>
    </main>
  );
}

function copyTextFallback(text, errorMessage) {
  const textarea = document.createElement('textarea');
  textarea.value = text;
  textarea.setAttribute('readonly', '');
  textarea.style.position = 'fixed';
  textarea.style.top = '-9999px';
  textarea.style.left = '-9999px';
  document.body.appendChild(textarea);
  textarea.select();
  const copied = document.execCommand('copy');
  document.body.removeChild(textarea);
  if (!copied) throw new Error(errorMessage || 'copy failed');
}
