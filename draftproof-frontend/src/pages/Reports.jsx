import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { listScans, deleteScan } from '../api/draftproofApi';
import ErrorReload from '../components/ErrorReload';
import ConfirmDialog from '../components/ConfirmDialog';

const PAGE_SIZE = 10;

const STATUS_MAP = {
  pending:   { label: 'Pending', tone: 'neutral' },
  processing:{ label: 'Scanning', tone: 'active' },
  completed: { label: 'Completed', tone: 'positive' },
  failed:    { label: 'Failed', tone: 'negative' },
};

const TIER_LABELS = {
  low: 'Low',
  moderate: 'Moderate',
  high: 'High',
  green: 'Low',
  amber: 'Moderate',
  orange: 'High',
  red: 'Critical',
};

const TIER_TONES = {
  low: 'positive',
  moderate: 'warning',
  high: 'negative',
  green: 'positive',
  amber: 'warning',
  orange: 'warning',
  red: 'negative',
};

function formatDate(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toLocaleDateString('en-SG', { day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' });
}

export default function Reports() {
  const [scans, setScans] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [total, setTotal] = useState(0);
  const [deletingId, setDeletingId] = useState(null);
  const [confirmTarget, setConfirmTarget] = useState(null);

  const handleDelete = async (scanId) => {
    setDeletingId(scanId);
    setConfirmTarget(null);
    try {
      await deleteScan(scanId);
      setScans((prev) => prev.filter((s) => s.id !== scanId));
      setTotal((prev) => prev - 1);
    } catch {
      alert('Failed to delete. Please try again.');
    } finally {
      setDeletingId(null);
    }
  };

  useEffect(() => {
    const ac = new AbortController();
    setLoading(true);
    listScans(page, PAGE_SIZE, { signal: ac.signal })
      .then(({ data }) => {
        setScans(data.items);
        setTotalPages(data.pages);
        setTotal(data.total);
      })
      .catch((err) => {
        if (err.name === 'AbortError' || err.code === 'ERR_CANCELED') return;
        setError(err.response?.data?.detail || 'Failed to load reports');
      })
      .finally(() => setLoading(false));
    return () => ac.abort();
  }, [page]);

  if (loading) {
    return (
      <main className="app-page reports-page-shell">
        <div className="container">
          <div className="reports-loading">
            <div className="reports-spinner" />
            <p>Loading your reports...</p>
          </div>
        </div>
      </main>
    );
  }

  if (error) {
    return (
      <main className="app-page reports-page-shell">
        <div className="container">
          <ErrorReload message={error} />
        </div>
      </main>
    );
  }

  return (
    <main className="app-page reports-page-shell">
      <div className="container">
        <section className="app-hero app-hero-dark reports-hero">
          <div>
            <p className="eyebrow">Report library</p>
            <h1>Your reports</h1>
            <p>
              Review past scans, open completed reports, and keep a clean trail
              of what was checked before submission.
            </p>
          </div>
          <div className="reports-hero-actions">
            <div className="app-hero-stat">
              <span>Total scans</span>
              <strong>{total}</strong>
              <small>{totalPages > 1 ? `Page ${page} of ${totalPages}` : 'Report archive'}</small>
            </div>
            <Link to="/scan" className="btn btn-ghost">
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                <path d="M2 4.5A2.5 2.5 0 014.5 2h7A2.5 2.5 0 0114 4.5v7a2.5 2.5 0 01-2.5 2.5h-7A2.5 2.5 0 012 11.5v-7z" stroke="currentColor" strokeWidth="1.4"/>
                <path d="M5 8h6M8 5v6" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"/>
              </svg>
              New scan
            </Link>
          </div>
        </section>

        {scans.length === 0 ? (
          <div className="reports-empty">
            <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
              <rect x="6" y="4" width="36" height="40" rx="4" stroke="var(--line)" strokeWidth="2"/>
              <path d="M16 18h16M16 24h12M16 30h8" stroke="var(--line)" strokeWidth="2" strokeLinecap="round"/>
            </svg>
            <h3>No reports yet</h3>
            <p>Run your first scan to see results here.</p>
            <Link to="/scan" className="btn btn-primary">Start Scanning</Link>
          </div>
        ) : (
          <div className="reports-table-wrap">
            <table className="reports-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Status</th>
                  <th>Review tier</th>
                  <th>AI signal</th>
                  <th>Writing</th>
                  <th>Findings</th>
                  <th>Words</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {scans.map((scan) => {
                  const st = STATUS_MAP[scan.status] || STATUS_MAP.pending;
                  const tierTone = TIER_TONES[scan.tier] || 'neutral';
                  return (
                    <tr key={scan.id}>
                      <td className="td-date">{formatDate(scan.created_at)}</td>
                      <td>
                        <span className={`status-badge status-badge-${st.tone}`}>
                          {st.label}
                        </span>
                      </td>
                      <td>
                        {scan.tier ? (
                          <span className={`tier-badge tier-badge-${tierTone}`}>
                            {TIER_LABELS[scan.tier] || scan.tier}
                          </span>
                        ) : '—'}
                      </td>
                      <td className="td-score">
                        {scan.ai_score != null ? (
                          <strong className={`score-value score-value-${tierTone}`}>{scan.ai_score.toFixed(1)}%</strong>
                        ) : '—'}
                      </td>
                      <td className="td-score">
                        {scan.writing_score != null ? (
                          <strong className="score-value score-value-positive">{scan.writing_score.toFixed(1)}%</strong>
                        ) : '—'}
                      </td>
                      <td className="td-findings">{scan.finding_count ?? '—'}</td>
                      <td className="td-words">{scan.word_count?.toLocaleString() ?? '—'}</td>
                      <td>
                        <div className="td-actions">
                          {scan.status === 'completed' && (
                            <Link to={`/report/${scan.id}`} className="btn btn-secondary btn-small">
                              View
                            </Link>
                          )}
                          {scan.status !== 'processing' && (
                            <button
                              className="btn btn-small btn-delete"
                              disabled={deletingId === scan.id}
                              onClick={() => setConfirmTarget(scan.id)}
                              title="Delete report"
                            >
                              {deletingId === scan.id ? '…' : 'Delete'}
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {totalPages > 1 && (
          <div className="pagination">
            <button
              className="btn btn-secondary btn-small"
              disabled={page === 1}
              onClick={() => setPage(p => p - 1)}
            >
              Previous
            </button>
            <span className="pagination-info">
              {(() => {
                const pages = [];
                const maxVisible = 7;
                if (totalPages <= maxVisible) {
                  for (let i = 1; i <= totalPages; i++) pages.push(i);
                } else {
                  pages.push(1);
                  let start = Math.max(2, page - 2);
                  let end = Math.min(totalPages - 1, page + 2);
                  if (page <= 3) end = Math.min(5, totalPages - 1);
                  if (page >= totalPages - 2) start = Math.max(totalPages - 4, 2);
                  if (start > 2) pages.push('...');
                  for (let i = start; i <= end; i++) pages.push(i);
                  if (end < totalPages - 1) pages.push('...');
                  pages.push(totalPages);
                }
                return pages.map((p, i) =>
                  p === '...' ? (
                    <span key={`ell${i}`} className="pagination-ellipsis">…</span>
                  ) : (
                    <button
                      key={p}
                      className={`pagination-btn${p === page ? ' active' : ''}`}
                      onClick={() => setPage(p)}
                    >
                      {p}
                    </button>
                  )
                );
              })()}
            </span>
            <button
              className="btn btn-secondary btn-small"
              disabled={page === totalPages}
              onClick={() => setPage(p => p + 1)}
            >
              Next
            </button>
          </div>
        )}
      </div>

      <ConfirmDialog
        open={confirmTarget !== null}
        title="Delete this report?"
        message="This report will be permanently deleted and cannot be recovered. Make sure you have downloaded or saved it first."
        confirmLabel="Delete permanently"
        onConfirm={() => handleDelete(confirmTarget)}
        onCancel={() => setConfirmTarget(null)}
      />
    </main>
  );
}
