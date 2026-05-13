import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { getPurchaseHistory } from '../api/draftproofApi';
import ErrorReload from '../components/ErrorReload';
import CodeTexture from '../components/CodeTexture';

const PAGE_SIZE = 5;

const STATUS_TONES = {
  paid: 'positive',
  pending: 'neutral',
  failed: 'negative',
  completed: 'positive',
};

function formatDate(iso, locale) {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toLocaleDateString(locale, { day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' });
}

function formatAmount(cents, currency) {
  return `${(cents / 100).toFixed(2)} ${currency || 'USD'}`;
}

export default function PurchaseHistory() {
  const { t, i18n } = useTranslation();
  const [payments, setPayments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [total, setTotal] = useState(0);
  const navigate = useNavigate();
  const locale = i18n.resolvedLanguage?.startsWith('zh') ? 'zh-CN' : 'en-SG';

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
        setError(err.response?.data?.detail || t('history.loadFailed'));
      })
      .finally(() => setLoading(false));
    return () => ac.abort();
  }, [page, t]);

  if (loading) {
    return (
      <main className="app-page history-page-shell">
        <div className="container">
          <div className="reports-loading">
            <div className="reports-spinner" />
            <p>{t('history.loading')}</p>
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
          <CodeTexture id="historyHero" />
          <div>
            <p className="eyebrow">{t('history.billing')}</p>
            <h1>{t('history.title')}</h1>
            <p>{t('history.body')}</p>
          </div>
          <div className="reports-hero-actions">
            <div className="app-hero-stat">
              <span>{t('history.transactions')}</span>
              <strong>{total}</strong>
              <small>{totalPages > 1 ? t('common.pageOf', { page, totalPages }) : t('history.archive')}</small>
            </div>
            <button onClick={() => navigate('/buy')} className="btn btn-ghost">
              {t('history.buyTokens')}
            </button>
          </div>
        </section>

        {payments.length === 0 ? (
          <div className="reports-empty">
            <p>{t('history.empty')}</p>
            <button className="btn btn-primary" onClick={() => navigate('/buy')}>
              {t('history.firstTokens')}
            </button>
          </div>
        ) : (
          <div className="reports-table-wrap history-table-wrap">
            <table className="reports-table">
              <thead>
                <tr>
                  <th>{t('history.date')}</th>
                  <th>{t('history.tokens')}</th>
                  <th>{t('history.amount')}</th>
                  <th>{t('history.status')}</th>
                </tr>
              </thead>
              <tbody>
                {payments.map((p) => {
                  const tone = STATUS_TONES[p.status] || 'neutral';
                  return (
                    <tr key={p.id}>
                      <td className="td-date">{formatDate(p.created_at, locale)}</td>
                      <td><strong className="history-token-count">{p.tokens}</strong></td>
                      <td className="history-amount">{formatAmount(p.amount_cents, p.currency)}</td>
                      <td>
                        <span className={`status-badge status-badge-${tone}`}>
                          {t(`history.statuses.${p.status}`, { defaultValue: p.status })}
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
              {t('common.previous')}
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
              {t('common.next')}
            </button>
          </div>
        )}
      </div>
    </main>
  );
}
