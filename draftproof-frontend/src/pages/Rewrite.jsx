import { useEffect, useState } from 'react';
import { Link, Navigate, useParams } from 'react-router-dom';
import { getRewriteStatus, getRewriteReport, getRewriteDownload, getDetectJson } from '../api/draftproofApi';
import ErrorReload from '../components/ErrorReload';
import { useAuth } from '../context/AuthContext';

function normalizeSentence(value) {
  return String(value || '').replace(/\s+/g, ' ').trim();
}

function countWords(value) {
  const normalized = String(value || '').trim();
  return normalized ? normalized.split(/\s+/).length : 0;
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
  const [rewrite, setRewrite] = useState(null);
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(Boolean(rewriteId));
  const [error, setError] = useState(null);
  const [copyStatus, setCopyStatus] = useState('idle');

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
          setError(status.status === 'failed' ? (status.error || 'Rewrite failed') : 'Rewrite is not complete yet.');
          return;
        }

        const { data: rewriteReport } = await getRewriteReport(rewriteId);
        if (cancelled) return;
        setReport(rewriteReport);
        refreshBalance();
      } catch (err) {
        if (!cancelled) {
          setError(err.response?.data?.detail || 'Failed to load rewrite result');
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [rewriteId, refreshBalance]);

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
        setError('Download not available yet. Please try again in a moment.');
      }
    } catch (err) {
      downloadWindow?.close();
      setError(err.response?.data?.detail || 'Download failed. Please try again.');
    }
  };

  const handleDownloadDetectJson = async () => {
    try {
      const { data } = await getDetectJson(rewriteId);
      if (data.url) {
        window.open(data.url, '_blank');
      } else {
        setError('Detect scan JSON not available.');
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to download detect scan JSON.');
    }
  };

  const handleCopyRewrittenDocument = async () => {
    if (!report?.final_text) return;

    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(report.final_text);
      } else {
        copyTextFallback(report.final_text);
      }
      setCopyStatus('copied');
      window.setTimeout(() => setCopyStatus('idle'), 1800);
    } catch {
      setCopyStatus('error');
      window.setTimeout(() => setCopyStatus('idle'), 2200);
    }
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
  const sentenceRows = (report?.sentence_comparison || []).filter(
    (row) => normalizeSentence(row.orig_sentence) !== normalizeSentence(row.new_sentence)
  );
  const outcome = summary.outcome || (rewrite?.status === 'completed' ? 'completed' : rewrite?.status || '');
  const scanId = rewrite?.scan_id;
  const rewrittenWordCount = countWords(report?.final_text);

  return (
    <main className="dash-shell">
      <div className="container">
        <Link to={scanId ? `/report/${scanId}` : '/reports'} className="report-back">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M10 3L5 8l5 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          Back to Report
        </Link>

        <div className="report-hero" style={{ marginTop: 16 }}>
          <div className="report-hero-info">
            <div className="report-eyebrow">Rewrite Report</div>
            <h1>AI Section Rewrite</h1>
          </div>
          {outcome && (
            <div className="report-hero-tier" style={{ background: '#f0fdf4' }}>
              <span style={{ color: '#15803d', fontWeight: 700 }}>
                {outcome.replaceAll('_', ' ')}
              </span>
            </div>
          )}
        </div>

        {loading && (
          <div className="report-loading">
            <div className="report-pulse" />
            <p>Loading rewrite result...</p>
          </div>
        )}

        {error && <ErrorReload message={error} />}

        {report?.final_text && (
          <section className="rewritten-document-section">
            <div className="rewritten-document-heading">
              <div className="rewritten-document-title">
                <h3>Rewritten Document</h3>
                <span>{rewrittenWordCount.toLocaleString()} word{rewrittenWordCount === 1 ? '' : 's'}</span>
              </div>
              <button
                type="button"
                className={`copy-rewrite-btn${copyStatus === 'copied' ? ' is-copied' : ''}${copyStatus === 'error' ? ' has-error' : ''}`}
                onClick={handleCopyRewrittenDocument}
                aria-live="polite"
              >
                {copyStatus === 'copied' ? 'Copied' : copyStatus === 'error' ? 'Copy failed' : 'Copy'}
              </button>
            </div>
            <div className="rewritten-document-content">
              {report.final_text}
            </div>
          </section>
        )}

        {sentenceRows.length > 0 && (
          <div style={{ margin: '24px 0' }}>
            <h3>Sentence Changes ({sentenceRows.length})</h3>
            <div className="reports-table-wrap">
              <table className="reports-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Original</th>
                    <th>Rewritten</th>
                  </tr>
                </thead>
                <tbody>
                  {sentenceRows.map((row, i) => (
                    <tr key={`${row.index || i}-${i}`}>
                      <td>{row.index ?? i + 1}</td>
                      <td>{row.orig_sentence || '-'}</td>
                      <td>{row.new_sentence || '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {markedSuggestions.length > 0 && (
          <section className="rewrite-review-section">
            <div className="rewrite-review-heading">
              <div>
                <span className="rewrite-review-kicker">Review Required</span>
                <h3>Suggested Additions For Review</h3>
              </div>
              <span className="rewrite-review-count">{markedSuggestions.length}</span>
            </div>
            <p className="rewrite-review-copy">
              These suggestions are not inserted automatically. Bracketed text marks details the user must verify, replace, or remove before using.
            </p>
            <div className="rewrite-suggestion-grid">
              {markedSuggestions.map((item, i) => (
                <article className="rewrite-suggestion-card" key={`${item.component || 'suggestion'}-${i}`}>
                  <div className="rewrite-suggestion-meta">
                    <span>{item.priority ? String(item.priority).replaceAll('_', ' ') : `Suggestion ${i + 1}`}</span>
                    <span>{item.where || 'Flagged text'}</span>
                  </div>
                  <h4>{item.title || 'Suggested review addition'}</h4>
                  {(item.target_text || item.evidence) && (
                    <div className="rewrite-target-block">
                      <span>Target text</span>
                      <p>{item.target_text || item.evidence}</p>
                    </div>
                  )}
                  <div className="rewrite-addition-block">
                    <span>Suggested addition</span>
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

        {manualSuggestions.length > 0 && (
          <section className="rewrite-review-section">
            <div className="rewrite-review-heading">
              <div>
                <span className="rewrite-review-kicker">Manual Options</span>
                <h3>Manual Suggestions</h3>
              </div>
              <span className="rewrite-review-count">{manualSuggestions.length}</span>
            </div>
            <div className="rewrite-suggestion-grid">
              {manualSuggestions.slice(0, 12).map((item, i) => (
                <article className="rewrite-suggestion-card" key={`${item.finding_id || 'manual'}-${i}`}>
                  <div className="rewrite-suggestion-meta">
                    <span>{item.scanner_target || item.finding_type || `Suggestion ${i + 1}`}</span>
                    <span>{item.rejection_reason || 'Review manually'}</span>
                  </div>
                  {item.original_sentence && (
                    <div className="rewrite-target-block">
                      <span>Original</span>
                      <p>{item.original_sentence}</p>
                    </div>
                  )}
                  {item.suggested_sentence && (
                    <div className="rewrite-addition-block">
                      <span>Suggested sentence</span>
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
              Download PDF
            </button>
            <button type="button" className="btn btn-secondary" onClick={() => handleDownload('txt')}>
              Download Rewritten Text
            </button>
            <button type="button" className="btn btn-secondary" onClick={handleDownloadDetectJson}>
              Download Detect Scan JSON
            </button>
            <button type="button" className="btn btn-secondary" onClick={() => handleDownload('log')}>
              Download Rewrite Debug Log
            </button>
          </div>
        )}
      </div>
    </main>
  );
}

function copyTextFallback(text) {
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
  if (!copied) throw new Error('Copy command failed');
}
