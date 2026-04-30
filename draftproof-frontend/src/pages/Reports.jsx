import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { listScans } from '../api/draftproofApi';
import ErrorReload from '../components/ErrorReload';

const STATUS_MAP = {
  pending:   { label: 'Pending',   color: '#94a3b8', bg: '#f1f5f9' },
  processing:{ label: 'Scanning',  color: '#2563eb', bg: '#eff6ff' },
  completed: { label: 'Completed', color: '#16a34a', bg: '#f0fdf4' },
  failed:    { label: 'Failed',    color: '#dc2626', bg: '#fef2f2' },
};

const TIER_COLORS = {
  low: '#22c55e',
  moderate: '#f59e0b',
  high: '#ef4444',
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

  useEffect(() => {
    listScans()
      .then(({ data }) => setScans(data))
      .catch((err) => setError(err.response?.data?.detail || 'Failed to load reports'))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <main className="dash-shell">
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
            <h1>Your Reports</h1>
            <p className="reports-subtitle">{scans.length} scan{scans.length !== 1 ? 's' : ''} completed</p>
          </div>
          <Link to="/scan" className="btn btn-primary">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" style={{ marginRight: 6 }}>
              <path d="M2 4.5A2.5 2.5 0 014.5 2h7A2.5 2.5 0 0114 4.5v7a2.5 2.5 0 01-2.5 2.5h-7A2.5 2.5 0 012 11.5v-7z" stroke="currentColor" strokeWidth="1.4"/>
              <path d="M5 8h6M8 5v6" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"/>
            </svg>
            New Scan
          </Link>
        </div>

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
                  <th>Risk Level</th>
                  <th>Findings</th>
                  <th>Words</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {scans.map((scan) => {
                  const st = STATUS_MAP[scan.status] || STATUS_MAP.pending;
                  const tierColor = TIER_COLORS[scan.tier];
                  return (
                    <tr key={scan.id}>
                      <td className="td-date">{formatDate(scan.created_at)}</td>
                      <td>
                        <span className="status-badge" style={{ color: st.color, background: st.bg }}>
                          {st.label}
                        </span>
                      </td>
                      <td>
                        {scan.tier ? (
                          <span className="tier-badge" style={{ color: tierColor, borderColor: tierColor }}>
                            {scan.tier}
                          </span>
                        ) : '—'}
                      </td>
                      <td className="td-findings">{scan.finding_count ?? '—'}</td>
                      <td className="td-words">{scan.word_count?.toLocaleString() ?? '—'}</td>
                      <td>
                        {scan.status === 'completed' ? (
                          <Link to={`/report/${scan.id}`} className="btn btn-secondary btn-small">
                            View
                          </Link>
                        ) : (
                          <span className="muted-link">—</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </main>
  );
}
