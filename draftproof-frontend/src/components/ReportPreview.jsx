import { getSuggestion, applySuggestion } from '../api/draftproofApi';
import { useState } from 'react';

export default function ReportPreview({ issues }) {
  const [expandedId, setExpandedId] = useState(null);
  const [suggestion, setSuggestion] = useState(null);
  const [error, setError] = useState(null);

  const handleExpand = async (issueId) => {
    if (expandedId === issueId) {
      setExpandedId(null);
      return;
    }
    setExpandedId(issueId);
    setError(null);
    try {
      const { data } = await getSuggestion(issueId);
      setSuggestion(data);
    } catch {
      setError('Failed to load suggestion.');
    }
  };

  const handleApply = async (issueId, suggestionId) => {
    try {
      await applySuggestion(issueId, suggestionId);
      setExpandedId(null);
    } catch {
      setError('Failed to apply suggestion.');
    }
  };

  return (
    <div className="report-preview">
      <h2>Scan Results</h2>
      {error && <div className="alert alert-error">{error}</div>}
      {issues.length === 0 && <p>No issues found.</p>}
      {issues.map((issue) => (
        <div key={issue.id} className="issue-card">
          <div className="issue-header" onClick={() => handleExpand(issue.id)}>
            <span className={`severity ${issue.severity}`}>{issue.severity}</span>
            <p>{issue.description}</p>
          </div>
          {expandedId === issue.id && suggestion && (
            <div className="suggestion">
              <p><strong>Suggestion:</strong> {suggestion.text}</p>
              <button
                className="btn btn-sm"
                onClick={() => handleApply(issue.id, suggestion.id)}
              >
                Apply
              </button>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
