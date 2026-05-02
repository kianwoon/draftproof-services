import { useEffect, useState, useRef, useCallback } from 'react';
import { useParams, useSearchParams, Link } from 'react-router-dom';
import { getRewriteStatus, getRewriteReport, getRewriteDownload, createRewrite, getScanStatus } from '../api/draftproofApi';
import ErrorReload from '../components/ErrorReload';

export default function Rewrite() {
  const { id: scanId } = useParams();
  const [params] = useSearchParams();
  const [rewrite, setRewrite] = useState(null);
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const pollRef = useRef(null);

  const pollStatus = useCallback(async (rewriteId) => {
    try {
      const { data } = await getRewriteStatus(rewriteId);
      setRewrite(data);
      if (data.status === 'completed') {
        setLoading(false);
        const { data: rpt } = await getRewriteReport(rewriteId).catch(() => ({ data: null }));
        setReport(rpt);
      } else if (data.status === 'failed') {
        setLoading(false);
        setError(data.error || 'Rewrite failed');
      }
    } catch {
      setLoading(false);
      setError('Failed to check rewrite status');
    }
  }, []);

  useEffect(() => {
    let rid = params.get('rid');
    const start = async () => {
      try {
        if (!rid) {
          try {
            const { data } = await createRewrite(scanId);
            rid = data.id;
            setRewrite(data);
          } catch (err) {
            if (err.response?.status === 409) {
              // Rewrite already exists — find it and poll instead
              const { data: scanData } = await getScanStatus(scanId).catch(() => ({ data: null }));
              const existingId = scanData?.rewrite?.id;
              if (existingId) {
                rid = existingId;
                setRewrite(scanData.rewrite);
              } else {
                throw err;
              }
            } else {
              throw err;
            }
          }
        }
        await pollStatus(rid);
      } catch (err) {
        setLoading(false);
        setError(err.response?.data?.detail || 'Failed to start rewrite');
      }
    };
    start();
  }, [scanId, params, pollStatus]);

  useEffect(() => {
    if (!rewrite || rewrite.status === 'completed' || rewrite.status === 'failed') {
      if (pollRef.current) clearInterval(pollRef.current);
      return;
    }
    pollRef.current = setInterval(() => pollStatus(rewrite.id), 3000);
    return () => clearInterval(pollRef.current);
  }, [rewrite?.status, rewrite?.id, pollStatus]);

  const handleRewriteAgain = async () => {
    setLoading(true);
    setError(null);
    setReport(null);
    try {
      const { data } = await createRewrite(scanId);
      setRewrite(data);
      await pollStatus(data.id);
    } catch (err) {
      setLoading(false);
      setError(err.response?.data?.detail || 'Failed to start rewrite');
    }
  };

  const handleDownload = async (fmt) => {
    if (!rewrite) return;
    try {
      const { data } = await getRewriteDownload(rewrite.id, fmt);
      if (data.url) {
        window.open(data.url, '_blank');
      } else {
        setError('Download not available yet. Please try again in a moment.');
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Download failed. Please try again.');
    }
  };

  if (loading && (!rewrite || rewrite.status === 'pending' || rewrite.status === 'processing')) return (
    <main className="dash-shell">
      <div className="container">
        <div className="report-loading">
          <div className="report-pulse" />
          <p>Rewriting your document...</p>
          <p style={{ color: '#64748b', fontSize: '14px' }}>This may take 1-3 minutes</p>
        </div>
      </div>
    </main>
  );

  if (error) return (
    <main className="dash-shell">
      <div className="container">
        <Link to={`/report/${scanId}`} className="report-back">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M10 3L5 8l5 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          Back to Report
        </Link>
        <ErrorReload message={error} />
      </div>
    </main>
  );

  const summary = report?.summary || report?.rewrite_summary || {};
  const origRisk = summary.original_risk ?? 0;
  const finalRisk = summary.final_risk ?? 0;
  const converged = summary.converged ?? false;

  return (
    <main className="dash-shell">
      <div className="container">
        <Link to={`/report/${scanId}`} className="report-back">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M10 3L5 8l5 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          Back to Report
        </Link>

        <div className="report-hero" style={{ marginTop: '16px' }}>
          <div className="report-hero-info">
            <div className="report-eyebrow">Rewrite Report</div>
            <h1>AI Section Rewrite</h1>
          </div>
          <div className="report-hero-tier" style={{
            background: converged ? '#f0fdf4' : '#fffbeb',
          }}>
            <span style={{ color: converged ? '#22c55e' : '#f59e0b', fontWeight: 600 }}>
              {converged ? 'Converged' : 'Partially Improved'}
            </span>
          </div>
        </div>

        {/* Before/After */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', margin: '24px 0' }}>
          <div style={{ padding: '20px', borderRadius: '12px', background: '#fef2f2' }}>
            <h4 style={{ margin: '0 0 8px', color: '#ef4444' }}>Before</h4>
            <div style={{ fontSize: '28px', fontWeight: 700 }}>{(origRisk * 100).toFixed(1)}%</div>
            <div style={{ color: '#64748b', fontSize: '13px' }}>Risk Score</div>
          </div>
          <div style={{ padding: '20px', borderRadius: '12px', background: '#f0fdf4' }}>
            <h4 style={{ margin: '0 0 8px', color: '#22c55e' }}>After</h4>
            <div style={{ fontSize: '28px', fontWeight: 700 }}>{(finalRisk * 100).toFixed(1)}%</div>
            <div style={{ color: '#64748b', fontSize: '13px' }}>Risk Score</div>
          </div>
        </div>

        {/* Rewritten text */}
        {report?.final_text && (
          <div style={{ margin: '24px 0' }}>
            <h3>Rewritten Document</h3>
            <div style={{
              background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '12px',
              padding: '20px', whiteSpace: 'pre-wrap', fontFamily: 'Georgia, serif',
              lineHeight: 1.7, maxHeight: '600px', overflow: 'auto',
            }}>
              {report.final_text}
            </div>
          </div>
        )}

        {/* Sentence comparison */}
        {report?.sentence_comparison?.length > 0 && (
          <div style={{ margin: '24px 0' }}>
            <h3>Sentence Changes ({report.sentence_comparison.length})</h3>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ padding: '8px', textAlign: 'left' }}>#</th>
                    <th style={{ padding: '8px', textAlign: 'left' }}>Original</th>
                    <th style={{ padding: '8px', textAlign: 'left' }}>Rewritten</th>
                  </tr>
                </thead>
                <tbody>
                  {report.sentence_comparison.map((sc, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px', color: '#64748b' }}>{i + 1}</td>
                      <td style={{ padding: '8px' }}>{sc.orig_sentence || '-'}</td>
                      <td style={{ padding: '8px', color: '#22c55e' }}>{sc.new_sentence || '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Downloads */}
        {rewrite?.status === 'completed' && (
          <div style={{ margin: '24px 0', display: 'flex', gap: '12px' }}>
            <button onClick={() => handleDownload('pdf')} style={dlBtnStyle}>
              Download PDF
            </button>
            <button onClick={() => handleDownload('txt')} style={dlBtnStyle}>
              Download Rewritten Text
            </button>
            <button onClick={handleRewriteAgain} style={{ ...dlBtnStyle, background: '#6366f1' }}>
              Rewrite Again
            </button>
          </div>
        )}
      </div>
    </main>
  );
}

const dlBtnStyle = {
  padding: '10px 20px', borderRadius: '8px', background: '#1e293b', color: '#fff',
  border: 'none', cursor: 'pointer', fontSize: '14px', fontWeight: 600,
};
