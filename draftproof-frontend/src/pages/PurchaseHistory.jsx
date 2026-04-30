import { useEffect, useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { getPurchaseHistory } from '../api/draftproofApi';
import ErrorReload from '../components/ErrorReload';

const PAGE_SIZE = 10;

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
  return `${(cents / 100).toFixed(2)} ${currency || 'SGD'}`;
}

export default function PurchaseHistory() {
  const [payments, setPayments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [page, setPage] = useState(1);
  const navigate = useNavigate();

  const totalPages = Math.ceil(payments.length / PAGE_SIZE);
  const pagePayments = useMemo(
    () => payments.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE),
    [payments, page],
  );

  useEffect(() => { if (page > totalPages && totalPages > 0) setPage(1); }, [page, totalPages]);

  useEffect(() => {
    getPurchaseHistory()
      .then(({ data }) => setPayments(data))
      .catch((err) => setError(err.response?.data?.detail || 'Failed to load history'))
      .finally(() => setLoading(false));
  }, []);

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

  const totalTokens = payments.reduce((sum, p) => sum + (p.tokens || 0), 0);
  const totalSpent = payments.reduce((sum, p) => sum + (p.amount_cents || 0), 0);

  return (
    <main className="dash-shell">
      <div className="container">
        <div className="reports-header">
          <div>
            <h1>Purchase History</h1>
            <p className="reports-subtitle">
              {payments.length} transaction{payments.length !== 1 ? 's' : ''} — {totalTokens} tokens purchased, {formatAmount(totalSpent, 'SGD')} total
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
                {pagePayments.map((p) => {
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
              {Array.from({ length: totalPages }, (_, i) => i + 1).map(p => (
                <button
                  key={p}
                  className={`pagination-btn${p === page ? ' active' : ''}`}
                  onClick={() => setPage(p)}
                >
                  {p}
                </button>
              ))}
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
