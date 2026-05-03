import { useEffect, useState, useRef, useCallback } from 'react';
import { useParams, useSearchParams, Link } from 'react-router-dom';
import { getRewriteStatus, getRewriteReport, getRewriteDownload, getDetectJson, createRewrite, getScanStatus } from '../api/draftproofApi';
import ErrorReload from '../components/ErrorReload';
import { useAuth } from '../context/AuthContext';

export default function Rewrite() {
  const { id: scanId } = useParams();
  const [params] = useSearchParams();
  const { refreshBalance } = useAuth();
  const [rewrite, setRewrite] = useState(null);
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const pollRef = useRef(null);

  const pollStatus = useCallback(async (rewriteId) => {
    try {
      const { data } = await getRewriteStatus(rewriteId);
      setRewrite(data);
      if (data.status === 'completed') {
        setLoading(false);
        const { data: rpt } = await getRewriteReport(rewriteId).catch(() => ({ data: null }));
        setReport(rpt);
        refreshBalance();
      } else if (data.status === 'failed') {
        setLoading(false);
        setError(data.error || 'Rewrite failed');
      }
    } catch {
      setLoading(false);
      setError('Failed to check rewrite status');
    }
  }, []);

  useEffect(() => {
    let rid = params.get('rid');
    const start = async () => {
      try {
        if (!rid) {
          try {
            const { data } = await createRewrite(scanId);
            rid = data.id;
            setRewrite(data);
          } catch (err) {
            if (err.response?.status === 409) {
              // Rewrite already exists — find it and poll instead
              const { data: scanData } = await getScanStatus(scanId).catch(() => ({ data: null }));
              const existingId = scanData?.rewrite?.id;
              if (existingId) {
                rid = existingId;
                setRewrite(scanData.rewrite);
              } else {
                throw err;
              }
            } else {
              throw err;
            }
          }
        }
        await pollStatus(rid);
      } catch (err) {
        setLoading(false);
        setError(err.response?.data?.detail || 'Failed to start rewrite');
      }
    };
    start();
  }, [scanId, params, pollStatus]);

  useEffect(() => {
    if (!rewrite || rewrite.status === 'completed' || rewrite.status === 'failed') {
      if (pollRef.current) clearInterval(pollRef.current);
      return;
    }
    pollRef.current = setInterval(() => pollStatus(rewrite.id), 5000);
    return () => clearInterval(pollRef.current);
  }, [rewrite?.status, rewrite?.id, pollStatus]);

  const handleDownload = async (fmt) => {
    if (!rewrite) return;
    try {
      const { data } = await getRewriteDownload(rewrite.id, fmt);
      if (data.url) {
        window.open(data.url, '_blank');
      } else {
        setError('Download not available yet. Please try again in a moment.');
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Download failed. Please try again.');
    }
  };

  const handleDownloadDetectJson = async () => {
    if (!rewrite) return;
    try {
      const { data } = await getDetectJson(rewrite.id);
      if (data.url) {
        window.open(data.url, '_blank');
      } else {
        setError('Detect scan JSON not available.');
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to download detect scan JSON.');
    }
  };

  if (loading && (!rewrite || rewrite.status === 'pending' || rewrite.status === 'processing')) return (
    <main className="dash-shell">
      <div className="container">
        <div className="report-loading">
          <div className="report-pulse" />
          <p>Rewriting your document...</p>
          <p style={{ color: '#64748b', fontSize: '14px' }}>This may take 1-3 minutes</p>
        </div>
      </div>
    </main>
  );

  if (error) return (
    <main className="dash-shell">
      <div className="container">
        <Link to={`/report/${scanId}`} className="report-back">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M10 3L5 8l5 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          Back to Report
        </Link>
        <ErrorReload message={error} />
      </div>
    </main>
  );

  const summary = report?.summary || report?.rewrite_summary || {};
  const converged = summary.converged ?? false;
  const outcome = summary.outcome || '';
  const hasChangedSentences = (report?.sentence_comparison || []).some(
    (sc) => (sc.orig_sentence || '') !== (sc.new_sentence || '')
  );
  const noTextChange = Boolean(summary.no_text_change) ||
    ((summary.passes_completed ?? 0) === 0 && !hasChangedSentences);

  // Detect scan comparison (full pipeline scores)
  const origScan = summary.detect_scan_original || {};
  const reportedNewScan = summary.detect_scan_rewritten || {};
  const newScan = noTextChange ? origScan : reportedNewScan;

  const origBadge = origScan.ai_risk_badge || {};
  const newBadge = newScan.ai_risk_badge || {};

  const origAI = (origBadge.ai_likelihood_score ?? summary.detect_ai_likelihood ?? 0);
  const newAI = newBadge.ai_likelihood_score ?? 0;
  const origWQ = (origBadge.writing_quality_score ?? summary.detect_writing_quality ?? 0);
  const newWQ = newBadge.writing_quality_score ?? 0;
  const origFindings = origScan.findings || {};
  const newFindings = newScan.findings || {};
  const tiers = ['critical', 'high', 'medium', 'low'];
  const countF = (fdict) => tiers.reduce((sum, t) => sum + (fdict[t]?.length || 0), 0);
  const origTotal = countF(origFindings);
  const newTotal = countF(newFindings);

  // Fallback to internal metrics if detect scan not available
  const hasScanComparison = origBadge.ai_likelihood_score != null || newBadge.ai_likelihood_score != null;
  const origRisk = summary.original_risk ?? 0;
  const finalRisk = summary.final_risk ?? 0;

  const aiImproved = hasScanComparison && newAI < origAI - 0.05;
  const aiWorse = hasScanComparison && newAI > origAI + 0.05;
  const qualityImproved = hasScanComparison && newWQ < origWQ - 0.05;
  const findingsImproved = hasScanComparison && newTotal < origTotal;
  const findingsWorse = hasScanComparison && newTotal > origTotal;
  const rollbackApplied = Boolean(summary.rollback_applied) || outcome === 'rejected_for_drift';
  const finalLooksOriginal = hasScanComparison
    ? Math.abs(newAI - origAI) <= 0.05 && Math.abs(newWQ - origWQ) <= 0.05 && newTotal === origTotal
    : false;
  const originalPreserved = noTextChange || (rollbackApplied && finalLooksOriginal);

  // Derive outcome label from the final full detect scan, not local rewrite attempts.
  const improvedWithReview = !originalPreserved && (aiImproved || qualityImproved) && findingsWorse;
  const mixedResult = !originalPreserved && !improvedWithReview && (aiImproved || qualityImproved || findingsImproved) && findingsWorse;
  const regressed = !originalPreserved && !mixedResult && (
    aiWorse ||
    outcome === 'floor_reached' ||
    (findingsWorse && !aiImproved && !qualityImproved)
  );
  const improved = !originalPreserved && !regressed && !mixedResult && (
    outcome === 'improved' ||
    outcome === 'partially_improved' ||
    ((aiImproved || qualityImproved || findingsImproved) && !findingsWorse)
  );
  const outcomeLabel = noTextChange
    ? 'Author Input Needed'
    : originalPreserved
      ? 'Original Preserved'
      : improvedWithReview
        ? 'Improved With Review'
      : mixedResult
        ? 'Mixed Result'
    : improved
        ? 'Revision Improved'
        : converged
          ? 'Revision Complete'
          : 'Review Needed';
  const outcomeTone = improved || improvedWithReview || converged ? 'positive' : noTextChange || originalPreserved || mixedResult || regressed ? 'guided' : 'neutral';
  const outcomeColor = outcomeTone === 'positive' ? '#15803d' : outcomeTone === 'guided' ? '#92400e' : '#475569';
  const outcomeBg = outcomeTone === 'positive' ? '#ecfdf5' : outcomeTone === 'guided' ? '#fffbeb' : '#f8fafc';
  const resultMessage = noTextChange
    ? 'DraftProof found revision opportunities, but the main issues need evidence, examples, or source context from the author.'
    : originalPreserved
      ? 'The attempted rewrite was not kept because the final scan did not improve.'
      : improvedWithReview
        ? 'AI likelihood and writing-quality risk improved. Review the new findings before keeping the final output.'
      : mixedResult
        ? 'AI likelihood improved, but total findings increased. Review the revision plan before keeping the final output.'
      : improved
        ? 'The final output reduced at least one measured risk signal.'
        : 'Review the revision plan below before making another pass.';
  const mitigation = summary.mitigation_plan || {};
  const mitigationCounts = mitigation.counts || {};
  const mitigationMode = (mitigation.primary_mode || '').replaceAll('_', ' ');
  const badgeDrivers = mitigation.component_drivers || [];
  const referencePatterns = mitigation.reference_patterns || [];
  const finalPreserved = originalPreserved;
  const revisionCards = [
    [
      'Sentence Targets',
      'auto_rewrite',
      finalPreserved
        ? 'Sentence-level targets were identified, but no sentence edits were kept in the final output.'
        : 'Sentence-level targets identified for detector-gated editing.',
    ],
    ['Needs Evidence', 'needs_source_or_example', 'Claims that need author examples, citations, or more concrete context.'],
    ['Structure Work', 'structure_guidance', 'Paragraph or section changes that should be revised manually.'],
    ['Review Only', 'review_only', 'Signals worth checking, but not suitable for automatic rewrite.'],
    ['Protected', 'protected', 'Quoted or sensitive text that should remain unchanged.'],
  ];
  const formatDriver = (name) => (name || '')
    .replaceAll('_risk', '')
    .replaceAll('_', ' ')
    .replace(/\b\w/g, (ch) => ch.toUpperCase());
  const renderReferenceText = (text) => String(text || '')
    .split(/(\[[^\[\]]+\])/g)
    .map((part, i) => (
      part.startsWith('[') && part.endsWith(']')
        ? (
          <mark key={i} style={{
            background: '#fef08a',
            color: '#1f2937',
            padding: '0 3px',
            borderRadius: '3px',
            boxDecorationBreak: 'clone',
            WebkitBoxDecorationBreak: 'clone',
          }}>
            {part}
          </mark>
        )
        : part
    ));

  return (
    <main className="dash-shell">
      <div className="container">
        <Link to={`/report/${scanId}`} className="report-back">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M10 3L5 8l5 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          Back to Report
        </Link>

        <div className="report-hero" style={{ marginTop: '16px' }}>
          <div className="report-hero-info">
            <div className="report-eyebrow">Rewrite Report</div>
            <h1>AI Section Rewrite</h1>
          </div>
          <div className="report-hero-tier" style={{
            background: outcomeBg,
          }}>
            <span style={{ color: outcomeColor, fontWeight: 600 }}>
              {outcomeLabel}
            </span>
          </div>
        </div>

        {/* Before/After */}
        {hasScanComparison ? (
          <>
            <div style={{
              margin: '0 0 18px',
              padding: '14px 16px',
              borderRadius: '10px',
              border: '1px solid #fde68a',
              background: '#fffbeb',
              color: '#78350f',
              fontSize: '14px',
              lineHeight: 1.55,
            }}>
              {resultMessage}
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', margin: '24px 0' }}>
              <div style={{ padding: '20px', borderRadius: '10px', background: '#fff', border: '1px solid #e2e8f0' }}>
                <h4 style={{ margin: '0 0 8px', color: '#475569' }}>Original Scan</h4>
                <div style={{ fontSize: '28px', fontWeight: 700 }}>{origAI.toFixed(1)}%</div>
                <div style={{ color: '#64748b', fontSize: '13px' }}>AI Likelihood</div>
                <div style={{ marginTop: '8px', fontSize: '13px', color: '#64748b' }}>
                  Findings: <strong>{origTotal}</strong>
                </div>
              </div>
              <div style={{ padding: '20px', borderRadius: '10px', background: '#fff', border: `1px solid ${outcomeTone === 'positive' ? '#bbf7d0' : '#fde68a'}` }}>
                <h4 style={{ margin: '0 0 8px', color: outcomeColor }}>Final Output</h4>
                <div style={{ fontSize: '28px', fontWeight: 700 }}>{newAI.toFixed(1)}%</div>
                <div style={{ color: '#64748b', fontSize: '13px' }}>AI Likelihood</div>
                <div style={{ marginTop: '8px', fontSize: '13px', color: '#64748b' }}>
                  Findings: <strong>{newTotal}</strong>
                </div>
              </div>
            </div>

            {/* Detailed comparison table */}
            <div style={{ margin: '16px 0' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ padding: '8px', textAlign: 'left' }}>Metric</th>
                    <th style={{ padding: '8px', textAlign: 'center' }}>Original</th>
                    <th style={{ padding: '8px', textAlign: 'center' }}>Final Output</th>
                    <th style={{ padding: '8px', textAlign: 'center' }}>Change</th>
                  </tr>
                </thead>
                <tbody>
                  <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px', fontWeight: 600 }}>AI Likelihood</td>
                    <td style={{ padding: '8px', textAlign: 'center' }}>{origAI.toFixed(1)}%</td>
                    <td style={{ padding: '8px', textAlign: 'center' }}>{newAI.toFixed(1)}%</td>
                    <td style={{ padding: '8px', textAlign: 'center', color: newAI < origAI ? '#22c55e' : newAI > origAI ? '#ef4444' : '#64748b' }}>
                      {(newAI - origAI).toFixed(1)}%
                    </td>
                  </tr>
                  <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px', fontWeight: 600 }}>Writing Quality</td>
                    <td style={{ padding: '8px', textAlign: 'center' }}>{origWQ.toFixed(1)}%</td>
                    <td style={{ padding: '8px', textAlign: 'center' }}>{newWQ.toFixed(1)}%</td>
                    <td style={{ padding: '8px', textAlign: 'center', color: newWQ < origWQ ? '#22c55e' : newWQ > origWQ ? '#ef4444' : '#64748b' }}>
                      {(newWQ - origWQ).toFixed(1)}%
                    </td>
                  </tr>
                  {tiers.map(t => {
                    const oc = origFindings[t]?.length || 0;
                    const nc = newFindings[t]?.length || 0;
                    return (
                      <tr key={t} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '8px', fontWeight: 600 }}>{t.charAt(0).toUpperCase() + t.slice(1)}</td>
                        <td style={{ padding: '8px', textAlign: 'center' }}>{oc}</td>
                        <td style={{ padding: '8px', textAlign: 'center' }}>{nc}</td>
                        <td style={{ padding: '8px', textAlign: 'center', color: nc < oc ? '#22c55e' : nc > oc ? '#ef4444' : '#64748b' }}>
                          {nc - oc === 0 ? '—' : `${nc - oc}`}
                        </td>
                      </tr>
                    );
                  })}
                  <tr style={{ borderBottom: '2px solid #e2e8f0', fontWeight: 700 }}>
                    <td style={{ padding: '8px' }}>Total Findings</td>
                    <td style={{ padding: '8px', textAlign: 'center' }}>{origTotal}</td>
                    <td style={{ padding: '8px', textAlign: 'center' }}>{newTotal}</td>
                    <td style={{ padding: '8px', textAlign: 'center', color: newTotal < origTotal ? '#22c55e' : newTotal > origTotal ? '#ef4444' : '#64748b' }}>
                      {newTotal - origTotal === 0 ? '—' : newTotal - origTotal}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', margin: '24px 0' }}>
            <div style={{ padding: '20px', borderRadius: '12px', background: '#fef2f2' }}>
              <h4 style={{ margin: '0 0 8px', color: '#ef4444' }}>Before</h4>
              <div style={{ fontSize: '28px', fontWeight: 700 }}>{(origRisk * 100).toFixed(1)}%</div>
              <div style={{ color: '#64748b', fontSize: '13px' }}>Risk Score</div>
            </div>
            <div style={{ padding: '20px', borderRadius: '12px', background: '#f0fdf4' }}>
              <h4 style={{ margin: '0 0 8px', color: '#22c55e' }}>After</h4>
              <div style={{ fontSize: '28px', fontWeight: 700 }}>{(finalRisk * 100).toFixed(1)}%</div>
              <div style={{ color: '#64748b', fontSize: '13px' }}>Risk Score</div>
            </div>
          </div>
        )}

        {mitigation.primary_mode && (
          <div style={{ margin: '24px 0' }}>
            <h3>Revision Plan</h3>
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
              gap: '12px',
              marginTop: '12px',
            }}>
              {revisionCards.map(([label, key, help]) => (
                <div key={key} style={{
                  border: '1px solid #e2e8f0',
                  borderRadius: '8px',
                  padding: '14px',
                  background: '#fff',
                }}>
                  <div style={{ fontSize: '12px', color: '#64748b', textTransform: 'uppercase', letterSpacing: '.04em' }}>{label}</div>
                  <div style={{ fontSize: '24px', fontWeight: 700, marginTop: '4px' }}>{mitigationCounts[key] || 0}</div>
                  <div style={{ marginTop: '6px', fontSize: '12px', color: '#64748b', lineHeight: 1.45 }}>{help}</div>
                </div>
              ))}
            </div>
            <div style={{ marginTop: '12px', color: '#475569', fontSize: '14px' }}>
              Recommended path: <strong style={{ textTransform: 'capitalize' }}>{mitigationMode || 'guided revision'}</strong>
            </div>
            {badgeDrivers.length > 0 && (
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px', marginTop: '12px' }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ padding: '8px', textAlign: 'left' }}>Revision Focus</th>
                    <th style={{ padding: '8px', textAlign: 'center' }}>Signal Strength</th>
                    <th style={{ padding: '8px', textAlign: 'left' }}>Suggested Action</th>
                  </tr>
                </thead>
                <tbody>
                  {badgeDrivers.slice(0, 6).map((driver, i) => (
                    <tr key={`${driver.component}-${i}`} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px' }}>{formatDriver(driver.component)}</td>
                      <td style={{ padding: '8px', textAlign: 'center' }}>{Number(driver.score || 0).toFixed(1)}%</td>
                      <td style={{ padding: '8px' }}>{driver.mitigation}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
            {referencePatterns.length > 0 && (
              <div style={{ marginTop: '22px' }}>
                <h4 style={{ margin: '0 0 8px', fontSize: '16px' }}>Reference Revision Examples</h4>
                <div style={{
                  padding: '12px 14px',
                  borderRadius: '8px',
                  border: '1px solid #dbeafe',
                  background: '#eff6ff',
                  color: '#1e3a8a',
                  fontSize: '13px',
                  lineHeight: 1.5,
                  marginBottom: '12px',
                }}>
                  For learning and revision guidance only. Do not submit these patterns as-is; replace placeholders with the author's own evidence, source, and context.
                </div>
                <div style={{ display: 'grid', gap: '12px' }}>
                  {referencePatterns.slice(0, 4).map((pattern, i) => (
                    <div key={`${pattern.component || pattern.focus}-${i}`} style={{
                      border: '1px solid #e2e8f0',
                      borderRadius: '8px',
                      background: '#fff',
                      padding: '14px',
                    }}>
                      <div style={{ fontSize: '13px', fontWeight: 700, color: '#0f172a', marginBottom: '8px' }}>
                        {pattern.focus || 'Revision pattern'}
                      </div>
                      {pattern.flagged_excerpt && (
                        <blockquote style={{
                          margin: '0 0 10px',
                          padding: '10px 12px',
                          borderLeft: '3px solid #facc15',
                          background: '#fefce8',
                          color: '#3f3f46',
                          fontSize: '13px',
                          lineHeight: 1.55,
                        }}>
                          “{pattern.flagged_excerpt}”
                        </blockquote>
                      )}
                      {pattern.instead_of && (
                        <div style={{ fontSize: '13px', color: '#64748b', marginBottom: '8px', lineHeight: 1.55 }}>
                          <strong style={{ color: '#475569' }}>Instead of:</strong> {renderReferenceText(pattern.instead_of)}
                        </div>
                      )}
                      <div style={{
                        fontSize: '13px',
                        color: '#0f172a',
                        lineHeight: 1.6,
                        padding: '10px 12px',
                        borderRadius: '6px',
                        background: '#f8fafc',
                        border: '1px solid #e2e8f0',
                        marginBottom: '8px',
                      }}>
                        <strong>Try this pattern:</strong> {renderReferenceText(pattern.try_pattern)}
                      </div>
                      {pattern.why && (
                        <div style={{ fontSize: '12px', color: '#64748b', lineHeight: 1.5 }}>
                          {pattern.why}
                        </div>
                      )}
                      {pattern.application_note && (
                        <div style={{ marginTop: '8px', fontSize: '12px', color: '#475569', lineHeight: 1.5 }}>
                          <strong>How to apply:</strong> {pattern.application_note}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Rewritten text */}
        {report?.final_text && (
          <div style={{ margin: '24px 0' }}>
            <h3>Rewritten Document</h3>
            <div style={{
              background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '12px',
              padding: '20px', whiteSpace: 'pre-wrap', fontFamily: 'Georgia, serif',
              lineHeight: 1.7, maxHeight: '600px', overflow: 'auto',
            }}>
              {report.final_text}
            </div>
          </div>
        )}

        {/* Sentence comparison */}
        {report?.sentence_comparison?.length > 0 && (
          <div style={{ margin: '24px 0' }}>
            <h3>Sentence Changes ({report.sentence_comparison.length})</h3>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ padding: '8px', textAlign: 'left' }}>#</th>
                    <th style={{ padding: '8px', textAlign: 'left' }}>Original</th>
                    <th style={{ padding: '8px', textAlign: 'left' }}>Rewritten</th>
                  </tr>
                </thead>
                <tbody>
                  {report.sentence_comparison.map((sc, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px', color: '#64748b' }}>{i + 1}</td>
                      <td style={{ padding: '8px' }}>{sc.orig_sentence || '-'}</td>
                      <td style={{ padding: '8px', color: '#22c55e' }}>{sc.new_sentence || '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Downloads */}
        {rewrite?.status === 'completed' && (
          <div style={{ margin: '24px 0', display: 'flex', gap: '12px' }}>
            <button onClick={() => handleDownload('pdf')} style={dlBtnStyle}>
              Download PDF
            </button>
            <button onClick={() => handleDownload('txt')} style={dlBtnStyle}>
              Download Rewritten Text
            </button>
            <button onClick={handleDownloadDetectJson} style={dlBtnStyle}>
              Download Detect Scan JSON
            </button>
          </div>
        )}
      </div>
    </main>
  );
}

const dlBtnStyle = {
  padding: '10px 20px', borderRadius: '8px', background: '#1e293b', color: '#fff',
  border: 'none', cursor: 'pointer', fontSize: '14px', fontWeight: 600,
};
