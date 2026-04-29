import { getSuggestion, applySuggestion } from '../api/draftproofApi';
import { useState } from 'react';

export default function ReportPreview({ issues }) {
  const [expandedId, setExpandedId] = useState(null);
  const [suggestion, setSuggestion] = useState(null);

  const handleExpand = async (issueId) => {
    if (expandedId === issueId) {
      setExpandedId(null);
      return;
    }
    setExpandedId(issueId);
    const { data } = await getSuggestion(issueId);
    setSuggestion(data);
  };

  const handleApply = async (issueId, suggestionId) => {
    await applySuggestion(issueId, suggestionId);
    setExpandedId(null);
  };

  return (
    <div className="report-preview">
      <h2>Scan Results</h2>
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
