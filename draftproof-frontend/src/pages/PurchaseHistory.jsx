import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { getPurchaseHistory } from '../api/draftproofApi';
import ErrorReload from '../components/ErrorReload';

const PAGE_SIZE = 5;

const STATUS_MAP = {
  paid:      { label: 'Paid',      color: '#16a34a', bg: '#f0fdf4' },
  pending:   { label: 'Pending',   color: '#94a3b8', bg: '#f1f5f9' },
  failed:    { label: 'Failed',    color: '#dc2626', bg: '#fef2f2' },
  completed: { label: 'Completed', color: '#16a34a', bg: '#f0fdf4' },
};

function formatDate(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toLocaleDateString('en-SG', { day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' });
}

function formatAmount(cents, currency) {
  return `${(cents / 100).toFixed(2)} ${currency || 'USD'}`;
}

export default function PurchaseHistory() {
  const [payments, setPayments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [total, setTotal] = useState(0);
  const navigate = useNavigate();

  useEffect(() => {
    const ac = new AbortController();
    setLoading(true);
    getPurchaseHistory(page, PAGE_SIZE, { signal: ac.signal })
      .then(({ data }) => {
        setPayments(data.items);
        setTotalPages(data.pages);
        setTotal(data.total);
      })
      .catch((err) => {
        if (err.name === 'AbortError' || err.code === 'ERR_CANCELED') return;
        setError(err.response?.data?.detail || 'Failed to load history');
      })
      .finally(() => setLoading(false));
    return () => ac.abort();
  }, [page]);

  if (loading) {
    return (
      <main className="dash-shell">
        <div className="container">
          <div className="reports-loading">
            <div className="reports-spinner" />
            <p>Loading purchase history...</p>
          </div>
        </div>
      </main>
    );
  }

  if (error) {
    return (
      <main className="dash-shell">
        <div className="container">
          <ErrorReload message={error} />
        </div>
      </main>
    );
  }

  return (
    <main className="dash-shell">
      <div className="container">
        <div className="reports-header">
          <div>
            <h1>Purchase History</h1>
            <p className="reports-subtitle">
              {total} transaction{total !== 1 ? 's' : ''}
            </p>
          </div>
          <button onClick={() => navigate('/buy')} className="btn btn-primary">
            Buy Tokens
          </button>
        </div>

        {payments.length === 0 ? (
          <div className="reports-empty">
            <p>No purchases yet.</p>
            <button className="btn btn-primary" onClick={() => navigate('/buy')}>
              Buy your first tokens
            </button>
          </div>
        ) : (
          <div className="reports-table-wrap" style={{ maxWidth: 960 }}>
            <table className="reports-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Tokens</th>
                  <th>Amount</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {payments.map((p) => {
                  const st = STATUS_MAP[p.status] || { label: p.status, color: '#64748b', bg: '#f8fafc' };
                  return (
                    <tr key={p.id}>
                      <td>{formatDate(p.created_at)}</td>
                      <td className="td-findings">{p.tokens}</td>
                      <td>{formatAmount(p.amount_cents, p.currency)}</td>
                      <td>
                        <span className="status-badge" style={{ color: st.color, backgroundColor: st.bg }}>
                          {st.label}
                        </span>
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
    </main>
  );
}
