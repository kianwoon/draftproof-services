import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { getPurchaseHistory } from '../api/draftproofApi';
import ErrorReload from '../components/ErrorReload';
import CodeTexture from '../components/CodeTexture';
import { TOKEN_CURRENCY_CODE } from '../pricingConfig';

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

function formatDateParts(iso, locale) {
  if (!iso) return { date: '—', time: '' };
  const d = new Date(iso);
  return {
    date: d.toLocaleDateString(locale, { day: 'numeric', month: 'short', year: 'numeric' }),
    time: d.toLocaleTimeString(locale, { hour: '2-digit', minute: '2-digit' }),
  };
}

function formatAmount(cents, currency) {
  const parsedCents = Number(cents);
  if (!Number.isFinite(parsedCents)) return '—';
  const currencyCode = String(currency || TOKEN_CURRENCY_CODE).toUpperCase();
  return `${currencyCode} $${(parsedCents / 100).toFixed(2)}`;
}

function summarizeAmounts(payments) {
  const totalsByCurrency = new Map();
  payments.forEach((payment) => {
    const cents = Number(payment.amount_cents);
    if (!Number.isFinite(cents)) return;
    const currencyCode = String(payment.currency || TOKEN_CURRENCY_CODE).toUpperCase();
    totalsByCurrency.set(currencyCode, (totalsByCurrency.get(currencyCode) || 0) + cents);
  });
  if (totalsByCurrency.size === 0) return '—';
  return Array.from(totalsByCurrency.entries())
    .map(([currencyCode, cents]) => formatAmount(cents, currencyCode))
    .join(' · ');
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
  const visibleTokenTotal = payments.reduce((sum, payment) => sum + (Number(payment.tokens) || 0), 0);
  const latestPurchase = payments[0] ? formatDate(payments[0].created_at, locale) : '—';
  const summaryStats = [
    { label: t('history.summary.totalPurchases'), value: total, note: t('history.summary.allTime') },
    { label: t('history.summary.visibleTokens'), value: t('common.token', { count: visibleTokenTotal }), note: t('history.summary.currentPage') },
    { label: t('history.summary.visibleSpend'), value: summarizeAmounts(payments), note: t('history.summary.currentPage') },
    { label: t('history.summary.latestPurchase'), value: latestPurchase, note: t('history.summary.recentActivity') },
  ];

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

        {payments.length > 0 && (
          <section className="history-summary-strip" aria-label={t('history.summary.label')}>
            {summaryStats.map((stat) => (
              <div className="history-summary-item" key={stat.label}>
                <span>{stat.label}</span>
                <strong>{stat.value}</strong>
                <small>{stat.note}</small>
              </div>
            ))}
          </section>
        )}

        {payments.length === 0 ? (
          <div className="reports-empty history-empty">
            <span className="history-empty-kicker">{t('history.emptyKicker')}</span>
            <h3>{t('history.empty')}</h3>
            <p>{t('history.emptyBody')}</p>
            <button className="btn btn-primary" onClick={() => navigate('/buy')}>
              {t('history.firstTokens')}
            </button>
          </div>
        ) : (
          <section className="history-ledger" aria-label={t('history.transactions')}>
            <div className="history-ledger-note">
              <span>{t('history.ledgerNote')}</span>
            </div>
            <div className="reports-table-wrap history-table-wrap">
              <table className="reports-table history-table">
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
                    const dateParts = formatDateParts(p.created_at, locale);
                    const currencyCode = String(p.currency || TOKEN_CURRENCY_CODE).toUpperCase();
                    const isLegacyCurrency = currencyCode !== TOKEN_CURRENCY_CODE;
                    return (
                      <tr key={p.id}>
                        <td className="td-date">
                          <span>{dateParts.date}</span>
                          {dateParts.time && <small>{dateParts.time}</small>}
                        </td>
                        <td><strong className="history-token-count">+{t('common.token', { count: p.tokens })}</strong></td>
                        <td className="history-amount">
                          <span>{formatAmount(p.amount_cents, p.currency)}</span>
                          {isLegacyCurrency && <small>{t('history.legacyCurrency')}</small>}
                        </td>
                        <td className="history-status-cell">
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
            <div className="history-card-list">
              {payments.map((p) => {
                const tone = STATUS_TONES[p.status] || 'neutral';
                const dateParts = formatDateParts(p.created_at, locale);
                const currencyCode = String(p.currency || TOKEN_CURRENCY_CODE).toUpperCase();
                const isLegacyCurrency = currencyCode !== TOKEN_CURRENCY_CODE;
                return (
                  <article className="history-card-row" key={p.id}>
                    <div className="history-card-main">
                      <div>
                        <span className="history-card-date">{dateParts.date}</span>
                        {dateParts.time && <small>{dateParts.time}</small>}
                      </div>
                      <span className={`status-badge status-badge-${tone}`}>
                        {t(`history.statuses.${p.status}`, { defaultValue: p.status })}
                      </span>
                    </div>
                    <div className="history-card-meta">
                      <div>
                        <span>{t('history.tokens')}</span>
                        <strong className="history-token-count">+{t('common.token', { count: p.tokens })}</strong>
                      </div>
                      <div>
                        <span>{t('history.amount')}</span>
                        <strong>{formatAmount(p.amount_cents, p.currency)}</strong>
                        {isLegacyCurrency && <small>{t('history.legacyCurrency')}</small>}
                      </div>
                    </div>
                  </article>
                );
              })}
            </div>
          </section>
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
