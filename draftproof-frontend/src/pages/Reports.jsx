import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { listScans, deleteScan } from '../api/draftproofApi';
import ErrorReload from '../components/ErrorReload';
import ConfirmDialog from '../components/ConfirmDialog';
import CodeTexture from '../components/CodeTexture';

const PAGE_SIZE = 10;

const TIER_TONES = {
  low: 'positive',
  moderate: 'warning',
  high: 'negative',
  green: 'positive',
  amber: 'warning',
  orange: 'warning',
  red: 'negative',
};

function formatDate(iso, locale) {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toLocaleDateString(locale, { day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' });
}

export default function Reports() {
  const { t, i18n } = useTranslation();
  const [scans, setScans] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [total, setTotal] = useState(0);
  const [deletingId, setDeletingId] = useState(null);
  const [confirmTarget, setConfirmTarget] = useState(null);
  const locale = i18n.resolvedLanguage?.startsWith('zh') ? 'zh-CN' : 'en-SG';

  const handleDelete = async (scanId) => {
    setDeletingId(scanId);
    setConfirmTarget(null);
    try {
      await deleteScan(scanId);
      setScans((prev) => prev.filter((s) => s.id !== scanId));
      setTotal((prev) => prev - 1);
    } catch {
      alert(t('reports.deleteFailed'));
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
        setError(err.response?.data?.detail || t('reports.loadFailed'));
      })
      .finally(() => setLoading(false));
    return () => ac.abort();
  }, [page, t]);

  if (loading) {
    return (
      <main className="app-page reports-page-shell">
        <div className="container">
          <div className="reports-loading">
            <div className="reports-spinner" />
            <p>{t('reports.loading')}</p>
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
          <CodeTexture id="reportsHero" />
          <div>
            <p className="eyebrow">{t('reports.library')}</p>
            <h1>{t('reports.title')}</h1>
            <p>{t('reports.body')}</p>
          </div>
          <div className="reports-hero-actions">
            <div className="app-hero-stat">
              <span>{t('reports.totalScans')}</span>
              <strong>{total}</strong>
              <small>{totalPages > 1 ? t('common.pageOf', { page, totalPages }) : t('reports.archive')}</small>
            </div>
            <Link to="/scan" className="btn btn-ghost">
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                <path d="M2 4.5A2.5 2.5 0 014.5 2h7A2.5 2.5 0 0114 4.5v7a2.5 2.5 0 01-2.5 2.5h-7A2.5 2.5 0 012 11.5v-7z" stroke="currentColor" strokeWidth="1.4"/>
                <path d="M5 8h6M8 5v6" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"/>
              </svg>
              {t('reports.newScan')}
            </Link>
          </div>
        </section>

        {scans.length === 0 ? (
          <div className="reports-empty">
            <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
              <rect x="6" y="4" width="36" height="40" rx="4" stroke="var(--line)" strokeWidth="2"/>
              <path d="M16 18h16M16 24h12M16 30h8" stroke="var(--line)" strokeWidth="2" strokeLinecap="round"/>
            </svg>
            <h3>{t('reports.emptyTitle')}</h3>
            <p>{t('reports.emptyBody')}</p>
            <Link to="/scan" className="btn btn-primary">{t('reports.startScanning')}</Link>
          </div>
        ) : (
          <div className="reports-table-wrap">
            <table className="reports-table">
              <thead>
                <tr>
                  <th>{t('reports.date')}</th>
                  <th>{t('reports.status')}</th>
                  <th>{t('reports.reviewTier')}</th>
                  <th>{t('reports.aiSignal')}</th>
                  <th>{t('reports.writing')}</th>
                  <th>{t('reports.findings')}</th>
                  <th>{t('reports.words')}</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {scans.map((scan) => {
                  const statusTone = scan.status === 'processing' ? 'active' : scan.status === 'completed' ? 'positive' : scan.status === 'failed' ? 'negative' : 'neutral';
                  const tierTone = TIER_TONES[scan.tier] || 'neutral';
                  return (
                    <tr key={scan.id}>
                      <td className="td-date">{formatDate(scan.created_at, locale)}</td>
                      <td>
                        <span className={`status-badge status-badge-${statusTone}`}>
                          {t(`reports.statuses.${scan.status}`, { defaultValue: scan.status || t('reports.statuses.pending') })}
                        </span>
                      </td>
                      <td>
                        {scan.tier ? (
                          <span className={`tier-badge tier-badge-${tierTone}`}>
                            {t(`reports.tiers.${scan.tier}`, { defaultValue: scan.tier })}
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
                              {t('reports.view')}
                            </Link>
                          )}
                          {scan.status !== 'processing' && (
                            <button
                              className="btn btn-small btn-delete"
                              disabled={deletingId === scan.id}
                              onClick={() => setConfirmTarget(scan.id)}
                              title={t('reports.deleteTitle')}
                            >
                              {deletingId === scan.id ? '…' : t('reports.delete')}
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

      <ConfirmDialog
        open={confirmTarget !== null}
        title={t('reports.confirmTitle')}
        message={t('reports.confirmMessage')}
        confirmLabel={t('reports.confirmDelete')}
        onConfirm={() => handleDelete(confirmTarget)}
        onCancel={() => setConfirmTarget(null)}
      />
    </main>
  );
}
