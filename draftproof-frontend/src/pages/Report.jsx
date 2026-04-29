import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { getReport } from '../api/draftproofApi';

const TIER_COLORS = {
  low: '#22c55e',
  moderate: '#f59e0b',
  high: '#ef4444',
};

export default function Report() {
  const { id } = useParams();
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    getReport(id)
      .then(({ data }) => setReport(data))
      .catch((err) => setError(err.response?.data?.detail || 'Failed to load report'))
      .finally(() => setLoading(false));
  }, [id]);

  if (loading) return <div className="container"><p>Loading report...</p></div>;
  if (error) return <div className="container"><p className="error">{error}</p></div>;
  if (!report) return <div className="container"><p>Report not found.</p></div>;

  const tierColor = TIER_COLORS[report.tier] || '#888';

  return (
    <div className="container report-page">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h1>Report: {report.document_name}</h1>
        {report.tier && (
          <span style={{
            padding: '4px 12px',
            borderRadius: '12px',
            background: tierColor,
            color: '#fff',
            fontWeight: 600,
            fontSize: '14px',
            textTransform: 'uppercase',
          }}>
            {report.tier} risk
          </span>
        )}
      </div>

      <div style={{ margin: '20px 0' }}>
        {report.report_pdf_url && (
          <a href={report.report_pdf_url} target="_blank" rel="noopener noreferrer"
            className="btn btn-primary" style={{ marginRight: '12px' }}>
            View PDF Report
          </a>
        )}
        {report.report_md_url && (
          <a href={report.report_md_url} target="_blank" rel="noopener noreferrer"
            className="btn btn-secondary">
            View Markdown
          </a>
        )}
      </div>

      {report.issues.length > 0 && (
        <div className="report-preview">
          <h2>Findings ({report.issues.length})</h2>
          {report.issues.map((issue, i) => (
            <div key={issue.id || i} className="issue-card">
              <span className={`severity ${issue.severity}`}>{issue.severity}</span>
              <p>{issue.description}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
