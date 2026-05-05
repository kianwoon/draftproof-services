import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { getPurchaseHistory } from '../api/draftproofApi';
import ErrorReload from '../components/ErrorReload';

const PAGE_SIZE = 5;

const STATUS_MAP = {
  paid:      { label: 'Paid', tone: 'positive' },
  pending:   { label: 'Pending', tone: 'neutral' },
  failed:    { label: 'Failed', tone: 'negative' },
  completed: { label: 'Completed', tone: 'positive' },
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
      <main className="app-page history-page-shell">
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
      <main className="app-page history-page-shell">
        <div className="container">
          <ErrorReload message={error} />
        </div>
      </main>
    );
  }

  return (
    <main className="app-page history-page-shell">
      <div className="container">
        <section className="app-hero app-hero-dark reports-hero">
          <div>
            <p className="eyebrow">Billing</p>
            <h1>Purchase history</h1>
            <p>
              Track token purchases, payment status, and billing activity for
              your DraftProof account.
            </p>
          </div>
          <div className="reports-hero-actions">
            <div className="app-hero-stat">
              <span>Transactions</span>
              <strong>{total}</strong>
              <small>{totalPages > 1 ? `Page ${page} of ${totalPages}` : 'Billing archive'}</small>
            </div>
            <button onClick={() => navigate('/buy')} className="btn btn-ghost">
              Buy tokens
            </button>
          </div>
        </section>

        {payments.length === 0 ? (
          <div className="reports-empty">
            <p>No purchases yet.</p>
            <button className="btn btn-primary" onClick={() => navigate('/buy')}>
              Buy your first tokens
            </button>
          </div>
        ) : (
          <div className="reports-table-wrap history-table-wrap">
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
                  const st = STATUS_MAP[p.status] || { label: p.status, tone: 'neutral' };
                  return (
                    <tr key={p.id}>
                      <td className="td-date">{formatDate(p.created_at)}</td>
                      <td><strong className="history-token-count">{p.tokens}</strong></td>
                      <td className="history-amount">{formatAmount(p.amount_cents, p.currency)}</td>
                      <td>
                        <span className={`status-badge status-badge-${st.tone}`}>
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
