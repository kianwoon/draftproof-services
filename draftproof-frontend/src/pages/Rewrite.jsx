import { useEffect, useRef, useState } from 'react';
import { Link, Navigate, useParams } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { getRewriteStatus, getRewriteReport, getRewriteDownload } from '../api/draftproofApi';
import ErrorReload from '../components/ErrorReload';
import RewriteDraftEditor from './report/RewriteDraftEditor';
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

function sanitizeSpans(spans, length) {
  return (Array.isArray(spans) ? spans : [])
    .map((s) => (Array.isArray(s) ? [Number(s[0]), Number(s[1])] : null))
    .filter((s) => s && Number.isInteger(s[0]) && Number.isInteger(s[1]) && s[0] >= 0 && s[1] <= length && s[0] < s[1]);
}

// Render `text` with two exact, offset-based highlight layers: actionable HIGH top-k sentences
// (shaded) and actionable predictable word runs (underlined). Spans are [start, end] char ranges
// from scanner offsets; overlapping layers compose. Pure offset segmentation -- no string matching.
function bracketSpansByKind(spans, length, wanted) {
  return (Array.isArray(spans) ? spans : [])
    .filter((b) => b && b.kind === wanted && Number.isInteger(b.start) && Number.isInteger(b.end)
      && b.start >= 0 && b.end <= length && b.start < b.end)
    .map((b) => [b.start, b.end]);
}

function renderTopkHighlights(text, sentenceSpans, wordSpans, bracketSpans) {
  const source = String(text || '');
  const n = source.length;
  const sents = sanitizeSpans(sentenceSpans, n);
  const words = sanitizeSpans(wordSpans, n);
  const improved = bracketSpansByKind(bracketSpans, n, 'improved');  // green: rewrite improved the span
  const kept = bracketSpansByKind(bracketSpans, n, 'kept');          // amber: kept span; user should edit it
  if (!sents.length && !words.length && !improved.length && !kept.length) return source;

  const cuts = new Set([0, n]);
  [...sents, ...words, ...improved, ...kept].forEach(([s, e]) => { cuts.add(s); cuts.add(e); });
  const points = Array.from(cuts).filter((p) => p >= 0 && p <= n).sort((a, b) => a - b);
  const covers = (spans, a, b) => spans.some(([s, e]) => s <= a && e >= b);

  const nodes = [];
  for (let k = 0; k < points.length - 1; k += 1) {
    const a = points[k];
    const b = points[k + 1];
    if (a >= b) continue;
    const seg = source.slice(a, b);
    const inSentence = covers(sents, a, b);
    const inWord = covers(words, a, b);
    const inImproved = covers(improved, a, b);
    const inKept = covers(kept, a, b);
    if (!inSentence && !inWord && !inImproved && !inKept) {
      nodes.push(<span key={`tk-${a}`}>{seg}</span>);
    } else {
      const cls = `topk-mark${inSentence ? ' is-sentence' : ''}${inWord ? ' is-word' : ''}`
        + `${inImproved ? ' is-improved' : ''}${inKept ? ' is-kept' : ''}`;
      nodes.push(<mark key={`tk-${a}`} className={cls}>{seg}</mark>);
    }
  }
  return nodes;
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
  const [editorOpen, setEditorOpen] = useState(false);
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
  // Prod nests the rewrite summary under report.summary, so the hoisted estimate lands there;
  // the poc/bare path exposes it top-level. Check both (mirrors authorshipEvidence above).
  const externalEstimate = report?.external_detector_estimate || summary?.external_detector_estimate || null;
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
  // Exact top-k highlight spans for the rewritten document (actionable HIGH sentences + word runs).
  const topkHighlights = summary?.predictability_highlights || report?.predictability_highlights || null;
  const topkSentenceSpans = topkHighlights?.actionable_sentences || topkHighlights?.sentences || [];
  const topkWordSpans = topkHighlights?.actionable_words || topkHighlights?.words || [];
  const hasTopkHighlights = (topkSentenceSpans.length || topkWordSpans.length) && report?.final_text;
  // bracket-grounding colour spans: kind 'improved' -> green, 'kept' -> amber
  const bracketSpans = summary?.bracket_grounding_spans || report?.bracket_grounding_spans || [];
  // honest register/polish coaching (NOT a score lever) -- backend supplies the note + selected lines
  const registerCoaching = summary?.register_coaching || report?.register_coaching || null;
  // worked teaching examples: generic claim -> grounded version -> why (the "teacher works the problem")
  const workedExamples = (summary?.predictability_showcase || report?.predictability_showcase || []).filter(
    (it) => it && it.sentence && it.suggestion,
  );
  const hasBracketHighlights = Boolean(bracketSpans.length && report?.final_text);
  const hasDocHighlights = Boolean((hasTopkHighlights || hasBracketHighlights) && report?.final_text);

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
            {authorshipEvidence.strengthen_examples?.length > 0 && (
              <div className="rewrite-target-block">
                <span>{t('authorshipEvidence.strengthenExamplesTitle')}</span>
                {authorshipEvidence.strengthen_examples.slice(0, 5).map((s, i) => (
                  <p key={i}>{s}</p>
                ))}
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
              <div className="rewritten-document-actions">
                {scanId && (
                  <button
                    type="button"
                    className="manual-correction-btn"
                    onClick={() => setEditorOpen(true)}
                  >
                    {t('rewritePage.manualCorrection')}
                  </button>
                )}
                <button
                  type="button"
                  className={`copy-rewrite-btn${copyStatus === 'copied' ? ' is-copied' : ''}${copyStatus === 'error' ? ' has-error' : ''}`}
                  onClick={handleCopyRewrittenDocument}
                  aria-live="polite"
                >
                  {copyStatus === 'copied' ? t('rewritePage.copied') : copyStatus === 'error' ? t('rewritePage.copyFailed') : t('rewritePage.copy')}
                </button>
              </div>
            </div>
            {hasTopkHighlights ? (
              <p className="topk-legend">
                <mark className="topk-mark is-sentence">{t('rewritePage.topk.legendSentence')}</mark>
                <mark className="topk-mark is-word">{t('rewritePage.topk.legendWord')}</mark>
                <span>{t('rewritePage.topk.legendNote')}</span>
              </p>
            ) : null}
            {hasBracketHighlights ? (
              <p className="topk-legend">
                <mark className="topk-mark is-improved">{t('rewritePage.bracketLegend.improved')}</mark>
                <mark className="topk-mark is-kept">{t('rewritePage.bracketLegend.kept')}</mark>
                <span>{t('rewritePage.bracketLegend.note')}</span>
              </p>
            ) : null}
            <div className="rewritten-document-content">
              {hasDocHighlights
                ? renderTopkHighlights(report.final_text, topkSentenceSpans, topkWordSpans, bracketSpans)
                : report.final_text}
            </div>
          </section>
        )}

        {workedExamples.length > 0 && (
          <section className="rewrite-review-section">
            <div className="rewrite-review-heading">
              <div>
                <span className="rewrite-review-kicker">{t('rewritePage.workedExamples.kicker')}</span>
                <h3>{t('rewritePage.workedExamples.heading')}</h3>
              </div>
              <span className="rewrite-review-count">{workedExamples.length}</span>
            </div>
            <p className="rewrite-review-copy">{t('rewritePage.workedExamples.copy')}</p>
            <div className="rewrite-suggestion-grid">
              {workedExamples.map((item, i) => (
                <article className="rewrite-suggestion-card" key={`worked-${i}`}>
                  <div className="rewrite-target-block">
                    <span>{t('rewritePage.workedExamples.generalClaim')}</span>
                    <p>{item.sentence}</p>
                  </div>
                  <div className="rewrite-addition-block">
                    <span>{t('rewritePage.workedExamples.moreGrounded')}</span>
                    <p>{item.suggestion}</p>
                  </div>
                  {item.why && (
                    <div className="rewrite-review-note"><p>{t('rewritePage.workedExamples.why')}: {item.why}</p></div>
                  )}
                </article>
              ))}
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

        {registerCoaching && (registerCoaching.offenders?.length > 0 || registerCoaching.worked_contrast || registerCoaching.rhythm_even) && (
          <section className="rewrite-review-section">
            <div className="rewrite-review-heading">
              <div>
                <span className="rewrite-review-kicker">{t('rewritePage.registerCoaching.kicker')}</span>
                <h3>{t('rewritePage.registerCoaching.heading')}</h3>
              </div>
            </div>
            {registerCoaching.note && (
              <p className="rewrite-review-copy">{registerCoaching.note}</p>
            )}
            {registerCoaching.worked_contrast?.polished?.text && registerCoaching.worked_contrast?.plain?.text && (
              <article className="rewrite-suggestion-card">
                <h4>{t('rewritePage.registerCoaching.contrastHeading')}</h4>
                <div className="rewrite-target-block">
                  <span>{t('rewritePage.registerCoaching.polishedLabel')}</span>
                  <p>{registerCoaching.worked_contrast.polished.text}</p>
                </div>
                <div className="rewrite-addition-block">
                  <span>{t('rewritePage.registerCoaching.plainLabel')}</span>
                  <p>{registerCoaching.worked_contrast.plain.text}</p>
                </div>
              </article>
            )}
            {registerCoaching.offenders?.length > 0 && (
              <div className="rewrite-suggestion-grid">
                {registerCoaching.offenders.slice(0, 4).map((item, i) => (
                  <article className="rewrite-suggestion-card" key={`register-${i}`}>
                    <div className="rewrite-suggestion-meta">
                      <span>{t('rewritePage.registerCoaching.offendersHeading')}</span>
                    </div>
                    <div className="rewrite-target-block">
                      <p>{item.text}</p>
                    </div>
                  </article>
                ))}
              </div>
            )}
            {registerCoaching.rhythm_even && (
              <p className="rewrite-review-copy">{t('rewritePage.registerCoaching.rhythmNote')}</p>
            )}
          </section>
        )}

        {rewrite?.status === 'completed' && (
          <div className="report-downloads">
            <button type="button" className="btn btn-primary" onClick={() => handleDownload('pdf')}>
              {t('rewritePage.downloadPdf')}
            </button>
            <p className="report-download-retention">{t('rewritePage.retentionNotice')}</p>
          </div>
        )}
      </div>
      {editorOpen && report?.final_text && scanId && (
        <RewriteDraftEditor
          storageKey={`${scanId}:rewrite:${rewriteId}`}
          baselineText={report.final_text}
          workedExamples={workedExamples}
          bracketSpans={bracketSpans}
          onClose={() => setEditorOpen(false)}
        />
      )}
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
