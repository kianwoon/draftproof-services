import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getReport } from '../api/draftproofApi';
import ErrorReload from '../components/ErrorReload';

const TIER_CONFIG = {
  low:      { label: 'Low Risk',      color: '#22c55e', bg: '#f0fdf4', icon: 'M12 15.5l-3-3 1.4-1.4L12 12.6l4.6-4.6L18 9.5z' },
  moderate: { label: 'Moderate Risk',  color: '#f59e0b', bg: '#fffbeb', icon: 'M12 9v4M12 15h.01' },
  high:     { label: 'High Risk',      color: '#ef4444', bg: '#fef2f2', icon: 'M12 9v4M12 15h.01M4.93 4.93l14.14 14.14' },
};

const SEVERITY_CONFIG = {
  critical: { color: '#dc2626', bg: '#fef2f2', label: 'CRITICAL' },
  high:     { color: '#ef4444', bg: '#fef2f2', label: 'HIGH' },
  medium:   { color: '#f59e0b', bg: '#fffbeb', label: 'MEDIUM' },
  low:      { color: '#22c55e', bg: '#f0fdf4', label: 'LOW' },
  info:     { color: '#3b82f6', bg: '#eff6ff', label: 'INFO' },
};

function formatDate(iso) {
  if (!iso) return '';
  return new Date(iso).toLocaleDateString('en-SG', { day: 'numeric', month: 'long', year: 'numeric', hour: '2-digit', minute: '2-digit' });
}

export default function Report() {
  const { id } = useParams();
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedIssue, setExpandedIssue] = useState(null);

  useEffect(() => {
    getReport(id)
      .then(({ data }) => setReport(data))
      .catch((err) => setError(err.response?.data?.detail || 'Failed to load report'))
      .finally(() => setLoading(false));
  }, [id]);

  if (loading) return (
    <main className="dash-shell">
      <div className="container">
        <div className="report-loading">
          <div className="report-pulse" />
          <p>Analyzing your report...</p>
        </div>
      </div>
    </main>
  );

  if (error) return (
    <main className="dash-shell">
      <div className="container"><ErrorReload message={error} /></div>
    </main>
  );

  if (!report) return (
    <main className="dash-shell">
      <div className="container"><p>Report not found.</p></div>
    </main>
  );

  const tier = TIER_CONFIG[report.tier] || TIER_CONFIG.moderate;
  const issueCounts = { critical: 0, high: 0, medium: 0, low: 0, info: 0 };
  report.issues.forEach((iss) => { if (issueCounts[iss.severity] !== undefined) issueCounts[iss.severity]++; });

  return (
    <main className="dash-shell">
      <div className="container">
        {/* Back link */}
        <Link to="/reports" className="report-back">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M10 3L5 8l5 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          Back to Reports
        </Link>

        {/* Report header */}
        <div className="report-hero">
          <div className="report-hero-info">
            <div className="report-eyebrow">Analysis Report</div>
            <h1>{report.document_name}</h1>
            {report.created_at && <p className="report-meta">{formatDate(report.created_at)}</p>}
          </div>
          <div className="report-hero-tier" style={{ background: tier.bg }}>
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke={tier.color} strokeWidth="2" strokeLinecap="round">
              <path d={tier.icon} />
              <circle cx="12" cy="12" r="10" />
            </svg>
            <span style={{ color: tier.color }}>{tier.label}</span>
          </div>
        </div>

        {/* Summary bar */}
        <div className="report-summary-bar">
          <div className="report-stat">
            <span className="report-stat-value">{report.issues.length}</span>
            <span className="report-stat-label">Total Findings</span>
          </div>
          {Object.entries(issueCounts).filter(([, v]) => v > 0).map(([sev, count]) => {
            const sc = SEVERITY_CONFIG[sev];
            return (
              <div key={sev} className="report-stat">
                <span className="report-stat-value" style={{ color: sc.color }}>{count}</span>
                <span className="report-stat-label">{sc.label}</span>
              </div>
            );
          })}
          <div className="report-stat">
            <span className="report-stat-value">{report.tier || '—'}</span>
            <span className="report-stat-label">Risk Tier</span>
          </div>
        </div>

        {/* Download links */}
        {(report.report_pdf_url || report.report_md_url) && (
          <div className="report-downloads">
            {report.report_pdf_url && (
              <a href={report.report_pdf_url} target="_blank" rel="noopener noreferrer" className="btn btn-primary">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none" style={{ marginRight: 6 }}>
                  <path d="M3 10v2.5A1.5 1.5 0 004.5 14h7a1.5 1.5 0 001.5-1.5V10M8 2v8M5 7l3 3 3-3" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
                Download PDF
              </a>
            )}
            {report.report_md_url && (
              <a href={report.report_md_url} target="_blank" rel="noopener noreferrer" className="btn btn-secondary">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none" style={{ marginRight: 6 }}>
                  <path d="M3 2h10v12H3zM6 5h4M6 7.5h4M6 10h2" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/>
                </svg>
                View Markdown
              </a>
            )}
          </div>
        )}

        {/* Findings list */}
        {report.issues.length > 0 ? (
          <div className="report-findings">
            <h2>Findings</h2>
            <div className="findings-list">
              {report.issues.map((issue, i) => {
                const sc = SEVERITY_CONFIG[issue.severity] || SEVERITY_CONFIG.info;
                const isExpanded = expandedIssue === i;
                const hasScores = issue.score != null || issue.top10_ratio != null;
                return (
                  <div
                    key={issue.id || i}
                    className={`finding-card${isExpanded ? ' expanded' : ''}`}
                    onClick={() => setExpandedIssue(isExpanded ? null : i)}
                    style={{ borderLeftColor: sc.color }}
                  >
                    <div className="finding-header">
                      <span className="finding-severity" style={{ color: sc.color, background: sc.bg }}>
                        {sc.label}
                      </span>
                      <span className="finding-number">#{i + 1}</span>
                      {issue.title && <span className="finding-title-tag">{issue.title.replace(/_/g, ' ')}</span>}
                      {issue.location && <span className="finding-location">{issue.location}</span>}
                      <svg
                        className="finding-chevron"
                        width="14" height="14" viewBox="0 0 14 14" fill="none"
                        style={{ transform: isExpanded ? 'rotate(180deg)' : 'rotate(0)', transition: 'transform .2s' }}
                      >
                        <path d="M3 5l4 4 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                      </svg>
                    </div>
                    <p className="finding-desc">{issue.description}</p>
                    {isExpanded && (
                      <div className="finding-detail" onClick={(e) => e.stopPropagation()}>
                        {issue.scanner && (
                          <div className="finding-meta-row">
                            <span className="finding-meta-label">Scanner</span>
                            <span className="finding-meta-value">{issue.scanner}</span>
                          </div>
                        )}
                        {issue.category && (
                          <div className="finding-meta-row">
                            <span className="finding-meta-label">Category</span>
                            <span className="finding-meta-value">{issue.category}</span>
                          </div>
                        )}
                        {issue.signal_category && (
                          <div className="finding-meta-row">
                            <span className="finding-meta-label">Signal</span>
                            <span className="finding-meta-value">{issue.signal_category.replace(/_/g, ' ')}</span>
                          </div>
                        )}
                        {issue.actionability && (
                          <div className="finding-meta-row">
                            <span className="finding-meta-label">Action</span>
                            <span className={`finding-action-badge finding-action-${issue.actionability}`}>
                              {issue.actionability.replace(/_/g, ' ')}
                            </span>
                          </div>
                        )}
                        {hasScores && (
                          <div className="finding-scores">
                            {issue.score != null && (
                              <div className="finding-score-item">
                                <span className="finding-score-label">Risk Score</span>
                                <div className="finding-score-bar">
                                  <div className="finding-score-fill" style={{ width: `${Math.min(issue.score * 100, 100)}%`, background: sc.color }} />
                                </div>
                                <span className="finding-score-value">{(issue.score * 100).toFixed(0)}%</span>
                              </div>
                            )}
                            {issue.top10_ratio != null && (
                              <div className="finding-score-item">
                                <span className="finding-score-label">Top-10 Predictability</span>
                                <div className="finding-score-bar">
                                  <div className="finding-score-fill" style={{ width: `${Math.min(issue.top10_ratio * 100, 100)}%`, background: '#8b5cf6' }} />
                                </div>
                                <span className="finding-score-value">{(issue.top10_ratio * 100).toFixed(0)}%</span>
                              </div>
                            )}
                          </div>
                        )}
                        {issue.evidence && (
                          <div className="finding-evidence">
                            <span className="finding-meta-label">Evidence</span>
                            {issue.evidence.summary && <p>{issue.evidence.summary}</p>}
                            {issue.evidence.sentence && (
                              <blockquote className="finding-quote">&ldquo;{issue.evidence.sentence}&rdquo;</blockquote>
                            )}
                          </div>
                        )}
                        {issue.recommendation && (
                          <div className="finding-recommendation">
                            <span className="finding-meta-label">Recommendation</span>
                            <p>{issue.recommendation}</p>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        ) : (
          <div className="report-clean">
            <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
              <circle cx="24" cy="24" r="20" stroke="#22c55e" strokeWidth="2"/>
              <path d="M16 24l5 5 11-11" stroke="#22c55e" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            <h3>No issues found</h3>
            <p>Your document looks clean. No findings were detected.</p>
          </div>
        )}

        {/* Results JSON (collapsible) */}
        {report.results_json && (
          <details className="report-raw-details">
            <summary>Raw Analysis Data</summary>
            <pre className="report-raw-json">{JSON.stringify(report.results_json, null, 2)}</pre>
          </details>
        )}
      </div>
    </main>
  );
}
