import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { listScans, getPurchaseHistory } from '../api/draftproofApi';

// allow-hardcode: the strings below are SVG `d` path geometry for the three nav
// icons (report / clock / key), copied verbatim from the former Dashboard. They
// are presentational vector data, not a content detect-list or matching oracle.

// Right-rail navigation for the unified workspace (formerly the Dashboard
// side-stack). Lives next to the scan form so reports, purchases, and API keys
// stay one click away without a separate dashboard page. Self-contained: it
// fetches its own lightweight count badges (only the `total` field is needed).
export default function WorkspaceNav() {
  const { t } = useTranslation();
  const [reportCount, setReportCount] = useState(null);
  const [purchaseCount, setPurchaseCount] = useState(null);

  useEffect(() => {
    const ac = new AbortController();
    listScans(1, 1, { signal: ac.signal })
      .then(({ data }) => setReportCount(data.total))
      .catch(() => {});
    getPurchaseHistory(1, 1, { signal: ac.signal })
      .then(({ data }) => setPurchaseCount(data.total))
      .catch(() => {});
    return () => ac.abort();
  }, []);

  return (
    <aside className="dash-side-stack" aria-label={t('dashboard.actions')}>
      <Link to="/reports" className="dash-small-card">
        <span className="dash-action-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
            <path d="M7 3.8h7.2L18 7.6v12.6H7V3.8Z" />
            <path d="M14 3.8v4h4M9.5 11h5M9.5 14h5M9.5 17h3" />
          </svg>
        </span>
        <div>
          <h3>{t('dashboard.viewReports')}</h3>
          <p>{t('dashboard.reportsBody')}</p>
        </div>
        {reportCount > 0 && (
          <span
            className="dash-card-count"
            aria-label={t('dashboard.reportsCount', { count: reportCount })}
          >
            {reportCount}
          </span>
        )}
      </Link>

      <Link to="/history" className="dash-small-card">
        <span className="dash-action-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
            <path d="M12 6v6l3.5 2" />
            <path d="M20 12a8 8 0 1 1-2.35-5.65" />
            <path d="M20 4v5h-5" />
          </svg>
        </span>
        <div>
          <h3>{t('dashboard.purchaseHistory')}</h3>
          <p>{t('dashboard.historyBody')}</p>
        </div>
        {purchaseCount > 0 && (
          <span
            className="dash-card-count"
            aria-label={t('dashboard.purchasesCount', { count: purchaseCount })}
          >
            {purchaseCount}
          </span>
        )}
      </Link>

      <Link to="/api-keys" className="dash-small-card">
        <span className="dash-action-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
            <circle cx="8" cy="15" r="3.2" />
            <path d="M10.3 12.7 19 4M16 7l2.5 2.5M14 9l2 2" />
          </svg>
        </span>
        <div>
          <h3>{t('dashboard.apiKeysTitle')}</h3>
          <p>{t('dashboard.apiKeysBody')}</p>
        </div>
      </Link>
    </aside>
  );
}
