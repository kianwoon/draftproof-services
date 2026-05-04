import { useEffect, useState } from 'react';
import { Link, Navigate, useParams } from 'react-router-dom';
import { getRewriteStatus, getRewriteReport, getRewriteDownload, getDetectJson, regenerateRewriteReport } from '../api/draftproofApi';
import ErrorReload from '../components/ErrorReload';
import { useAuth } from '../context/AuthContext';

function normalizeSentence(value) {
  return String(value || '').replace(/\s+/g, ' ').trim();
}

export default function Rewrite() {
  const { rewriteId } = useParams();
  const { refreshBalance } = useAuth();
  const [rewrite, setRewrite] = useState(null);
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(Boolean(rewriteId));
  const [error, setError] = useState(null);

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
    try {
      if (fmt === 'pdf') {
        const { data: regen } = await regenerateRewriteReport(rewriteId);
        if (regen?.status && regen.status !== 'completed') {
          setError('Rewrite PDF is being regenerated. Please try the download again in a moment.');
          return;
        }
      }
      const { data } = await getRewriteDownload(rewriteId, fmt);
      if (data.url) {
        window.open(data.url, '_blank');
      } else {
        setError('Download not available yet. Please try again in a moment.');
      }
    } catch (err) {
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

  const summary = report?.summary || report?.rewrite_summary || {};
  const sentenceRows = (report?.sentence_comparison || []).filter(
    (row) => normalizeSentence(row.orig_sentence) !== normalizeSentence(row.new_sentence)
  );
  const outcome = summary.outcome || (rewrite?.status === 'completed' ? 'completed' : rewrite?.status || '');
  const scanId = rewrite?.scan_id;

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
          <div style={{ margin: '24px 0' }}>
            <h3>Rewritten Document</h3>
            <div style={{
              background: '#f8fafc',
              border: '1px solid #e2e8f0',
              borderRadius: 12,
              padding: 20,
              whiteSpace: 'pre-wrap',
              fontFamily: 'Georgia, serif',
              lineHeight: 1.7,
              maxHeight: 600,
              overflow: 'auto',
            }}>
              {report.final_text}
            </div>
          </div>
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
