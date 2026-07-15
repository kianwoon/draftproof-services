import { useState, useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { signalClassName, signalLabel, signalDescription } from './reportHelpers';
import ParagraphSeverityBar from '../../components/ParagraphSeverityBar';
import EditPencilIcon from './EditPencilIcon';

// Flagged finding sentences render in RED FONT (owner decision 2026-07-04): they
// must stay visible on EVERY document — including Low-Risk (green) ones — so the
// reader sees exactly which sentences the deep-scan flagged. Severity graduates by
// red shade + font weight (80-89 < 90-98 < >=99), not by document tier.

export default function SignalHighlights({
  submittedContent, selectedParagraph, selectedParagraphId, highlightedParagraphs,
  paragraphSeverityBar,
  selectedCriticalThinking,
  showSubmittedEditEntry, onSelectParagraph, onPreviewParagraph, onAdjacent,
  onEditParagraph, onCopyGuidance,
}) {
  const { t } = useTranslation();
  const [tab, setTab] = useState('document'); // 'document' | 'issues' — default to the full read
  const [openId, setOpenId] = useState(highlightedParagraphs[0]?.id ?? null);
  const toggleCard = (id) => setOpenId((cur) => (cur === id ? null : id));
  const issuesRef = useRef(null);
  useEffect(() => {
    if (tab !== 'issues' || !selectedParagraph?.id) return;
    setOpenId(selectedParagraph.id);
    const el = issuesRef.current?.querySelector(`[data-issue-id="${selectedParagraph.id}"]`);
    if (el) {
      const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      el.scrollIntoView({ block: 'nearest', behavior: prefersReduced ? 'auto' : 'smooth' });
    }
  }, [selectedParagraph?.id, tab]);

  if (!submittedContent?.paragraphs?.length) return null;

  // Per-sentence issue-tag underline layer (server composer: poc/report/
  // sentence_issue_tags.py). Single source of truth for the colored issue
  // underlines: red = reads as AI, amber = weak grounding, purple = reasoning
  // jump. Null on older reports / clean docs -> the view renders exactly as
  // before (byte-identical). AI's red treatment stays on the existing heatmap
  // span; this layer ADDS the amber/purple underlines (stacked) + a per-sentence
  // popover listing every issue with its one-line fix.
  const issueTags = submittedContent.sentenceIssueTags || null;
  const issueTagsBySid = new Map(Object.entries(issueTags?.sentences || {}));
  const ISSUE_COLOR_HEX = { red: '#dc2626', amber: '#f59e0b', purple: '#7c3aed' };
  const issueLabel = (code) => t(`report.submitted.issueTags.${code}`, { defaultValue: code });
  const issueFix = (tag) => tag.fix_text || issueLabel(tag.fix_code);

  // Gated band -> severity class. The band is VERDICT-GATED by the backend
  // (report.py::_gate_heatmap_bands): 'high' stays a red flag; 'review' is a muted review
  // candidate (a >=0.999 sentence inside a document whose calibrated verdict reads clean —
  // the detector saturates on ~24% of human sentences, so it is NOT a red verdict). NEVER
  // re-derive from the raw score, which stays 100 on muted 'review' sentences.
  const debertaSeverityClass = (deb) => {
    const band = String(deb?.title || '').replace('deberta_', '');
    if (band === 'high') return 'is-severity-critical';    // red — high-confidence AI, doc corroborates
    if (band === 'review') return 'is-severity-review';    // muted amber — possible AI, review candidate
    return '';                                             // clean / gated-out — no flag
  };

  const FullDocument = (
    <div className="submitted-document" aria-label={t('report.submitted.documentText')}>
      {submittedContent.paragraphs.map((paragraph) => {
        const isSelected = selectedParagraphId === paragraph.id;
        // Render per-sentence so each is colored by its OWN DeBERTa score (graduated severity),
        // not one flat color for the whole paragraph. Map sentence_id -> its DeBERTa signal.
        const debertaBySid = new Map();
        paragraph.segments.forEach((segment) => {
          const deb = (segment.signals || []).find((sg) => sg.key === 'ai_signal_deberta');
          if (deb) debertaBySid.set(segment.sentence_id, deb);
        });
        const hasAny = debertaBySid.size > 0;
        if (!hasAny) {
          return (
            <p key={paragraph.id}>
              <button type="button" data-paragraph-id={paragraph.id}
                className={`submitted-clean-paragraph${isSelected ? ' is-selected' : ''}`}
                onMouseEnter={() => onPreviewParagraph(paragraph.id)}
                onFocus={() => onPreviewParagraph(paragraph.id)}
                onClick={() => { onSelectParagraph(paragraph.id); setTab('issues'); }}>
                {paragraph.text}
              </button>
            </p>
          );
        }
        return (
          <p key={paragraph.id} className="submitted-paragraph-heatmap">
            <button type="button" data-paragraph-id={paragraph.id}
              className={`submitted-paragraph-heatmap-btn${isSelected ? ' is-selected' : ''}`}
              onMouseEnter={() => onPreviewParagraph(paragraph.id)}
              onFocus={() => onPreviewParagraph(paragraph.id)}
              onClick={() => { onSelectParagraph(paragraph.id); setTab('issues'); }}>
              {paragraph.segments.map((segment, i) => {
                const deb = debertaBySid.get(segment.sentence_id);
                const sev = deb ? debertaSeverityClass(deb) : '';
                const tags = issueTagsBySid.get(segment.sentence_id) || [];
                // Only grounding/reasoning add a NEW underline here — AI's red is
                // already on the heatmap span below (backward-compat).
                const extraTags = tags.filter((tg) => tg.type !== 'ai');
                // Byte-identical path: nothing new to mark -> render exactly as before.
                if (extraTags.length === 0) {
                  if (!sev) {
                    return <span key={i} className="submitted-sentence-plain">{segment.text} </span>;
                  }
                  return (
                    <span key={i}
                      className={`submitted-sentence-highlight ${sev}`}
                      title={signalDescription(deb.key, deb.description, t)}>
                      {segment.text}
                    </span>
                  );
                }
                // NEW issue-underline layer: stack the grounding/reasoning underlines
                // beneath the sentence (layered background gradients, so 2+ issues each
                // show their own colored line — nothing hidden) and attach a popover
                // listing every issue on this sentence with its one-line fix.
                const markStyle = {
                  backgroundImage: extraTags
                    .map((tg) => `linear-gradient(${ISSUE_COLOR_HEX[tg.color]}, ${ISSUE_COLOR_HEX[tg.color]})`).join(', '),
                  backgroundSize: extraTags.map(() => '100% 2px').join(', '),
                  backgroundPosition: extraTags.map((_, idx) => `0 calc(100% - ${idx * 3}px)`).join(', '),
                  backgroundRepeat: 'no-repeat',
                  paddingBottom: `${extraTags.length * 3 + 2}px`,
                };
                const titleText = tags.map((tg) => `${issueLabel(tg.label_code)} — ${issueFix(tg)}`).join('\n');
                return (
                  <span key={i} className="submitted-sentence-issue" style={markStyle} title={titleText}>
                    {sev
                      ? <span className={`submitted-sentence-highlight ${sev}`}>{segment.text}</span>
                      : segment.text}
                    {' '}
                    <span className="submitted-sentence-issue-pop" role="note">
                      {tags.map((tg, ti) => (
                        <span key={ti} className="submitted-sentence-issue-pop-row">
                          <span className={`submitted-issue-swatch is-${tg.color}`} aria-hidden="true" />
                          <span className="submitted-issue-pop-label">{issueLabel(tg.label_code)}</span>
                          <span className="submitted-issue-pop-fix">{issueFix(tg)}</span>
                        </span>
                      ))}
                    </span>
                  </span>
                );
              })}
            </button>
          </p>
        );
      })}
    </div>
  );

  return (
    <section className="submitted-content-review" aria-label={t('report.submitted.sectionLabel')}>
      <div className="submitted-content-head">
        <div>
          <span className="submitted-content-kicker">{t('report.submitted.kicker')}</span>
          <h2>{t('report.submitted.title')}</h2>
        </div>
        <div className="submitted-content-actions">
          <div className="submitted-content-count">
            <strong>{submittedContent.highlightedCount}</strong>
            <span>{t('report.submitted.highlightedSections')}</span>
          </div>
        </div>
      </div>

      {paragraphSeverityBar?.length > 0 && (
        <ParagraphSeverityBar bar={paragraphSeverityBar} selectedId={selectedParagraph?.id} onSelect={onSelectParagraph} />
      )}

      {submittedContent.legend?.length > 0 && (
        <div className="submitted-signal-legend" aria-label={t('report.submitted.legend')}>
          {submittedContent.legend.slice(0, 6).map((signal) => {
            // Legend label reflects which detector actually produced these highlights on THIS
            // report: the V7 deep-scan detector (same one the panel headlines) when
            // signalSource === 'deep_scan', else the existing fakespot copy, byte-unchanged.
            const label = signal.key === 'ai_signal_deberta' && submittedContent.signalSource === 'deep_scan'
              ? t('report.signals.labels.ai_signal_deep_scan', { defaultValue: signalLabel(signal.key, signal.label, t) })
              : signalLabel(signal.key, signal.label, t);
            return (
              <span key={signal.key}
                className={`submitted-signal-chip signal-style-${signalClassName(signal.key)}`}
                style={{ '--signal-color': signal.color }}>
                <i aria-hidden="true" />{label}<strong>{signal.count}</strong>
              </span>
            );
          })}
        </div>
      )}

      <div className="submitted-tabs" role="tablist" aria-label={t('report.submitted.title')}>
        <button type="button" role="tab" id="sh-tab-issues" aria-selected={tab === 'issues'}
          aria-controls="sh-panel-issues"
          className={`submitted-tab${tab === 'issues' ? ' is-active' : ''}`}
          onClick={() => setTab('issues')}>
          {t('report.submitted.tabIssues', {
            count: highlightedParagraphs.length,
            total: submittedContent.paragraphs.length,
          })}
        </button>
        <button type="button" role="tab" id="sh-tab-document" aria-selected={tab === 'document'}
          aria-controls="sh-panel-document"
          className={`submitted-tab${tab === 'document' ? ' is-active' : ''}`}
          onClick={() => setTab('document')}>
          {t('report.submitted.tabDocument')}
        </button>
      </div>

      {tab === 'document' && (
        <div role="tabpanel" id="sh-panel-document" aria-labelledby="sh-tab-document">
          {issueTags && (
            <div className="submitted-issue-legend" aria-label={t('report.submitted.issueTags.legendTitle')}>
              <span className="submitted-issue-legend-title">{t('report.submitted.issueTags.legendTitle')}</span>
              {issueTags.legend.map((row) => (
                <span key={row.color} className="submitted-issue-legend-item">
                  <span className={`submitted-issue-swatch is-${row.color}`} aria-hidden="true" />
                  {issueLabel(row.label_code)}
                </span>
              ))}
              {issueTags.document_level?.length > 0 && (
                <p className="submitted-issue-doc-note">
                  {issueTags.document_level.map((d, i) => (
                    <span key={i} className="submitted-issue-doc-note-row">
                      <span className={`submitted-issue-swatch is-${d.color}`} aria-hidden="true" />
                      <span><strong>{issueLabel(d.label_code)}</strong> — {issueFix(d)}</span>
                    </span>
                  ))}
                </p>
              )}
            </div>
          )}
          {FullDocument}
        </div>
      )}
      {tab === 'issues' && (
        highlightedParagraphs.length === 0 ? (
          <div className="submitted-issues-empty" role="tabpanel" id="sh-panel-issues" aria-labelledby="sh-tab-issues">
            <h3>{t('report.submitted.issuesEmptyTitle')}</h3>
            <p>{t('report.submitted.issuesEmptyBody')}</p>
          </div>
        ) : (
          <div className="submitted-issues" ref={issuesRef} role="tabpanel" id="sh-panel-issues" aria-labelledby="sh-tab-issues">
            {highlightedParagraphs.map((paragraph, index) => {
              const signal = paragraph.primarySignal;
              const isOpen = openId === paragraph.id;
              const tier = signal?.tier;
              return (
                <article key={paragraph.id}
                  data-issue-id={paragraph.id}
                  className={`issue-card signal-style-${signalClassName(signal?.key)}${isOpen ? ' is-open' : ''}`}
                  style={{ '--signal-color': signal?.color }}>
                  <button type="button" className="issue-card-head"
                    aria-expanded={isOpen}
                    onClick={() => { toggleCard(paragraph.id); onSelectParagraph(paragraph.id); }}
                    onMouseEnter={() => onPreviewParagraph(paragraph.id)}>
                    <span className="issue-card-main">
                      <span className="issue-card-chips">
                        <em className="issue-card-num" aria-label={t('report.submitted.position', { current: index + 1, total: highlightedParagraphs.length })}>
                          {t('report.submitted.positionShort', { current: index + 1, total: highlightedParagraphs.length })}
                        </em>
                        {tier && <em className={`issue-chip issue-chip-tier is-${tier}`}>{t(`report.severities.${tier}`, { defaultValue: tier })}</em>}
                        {signal && <em className="issue-chip">{signalLabel(signal.key, signal.label, t)}</em>}
                        <em className="issue-chip">{t('report.submitted.paragraphSignals', { count: paragraph.signalCount || paragraph.signals.length })}</em>
                      </span>
                    </span>
                    <span className="issue-card-caret" aria-hidden="true">{isOpen ? '▾' : '▸'}</span>
                  </button>
                  {isOpen && (
                    <div className="issue-card-body">
                      {selectedParagraph?.id === paragraph.id ? (
                        <>
                          {selectedParagraph.readerSummary && (
                            <p className="issue-card-summary">{selectedParagraph.readerSummary}</p>
                          )}
                          {/* Fallback only: shown when NO flagged sentence has a tailored suggestion. */}
                          {!selectedParagraph.hasSentenceSuggestions && selectedParagraph.recommendation && (
                            <div className="issue-action">
                              <span className="issue-action-label">{t('report.submitted.recommendation')}</span>
                              <p>{selectedParagraph.recommendation}</p>
                            </div>
                          )}
                          {selectedCriticalThinking && (
                            <div className="issue-action issue-action-thinking">
                              <span className="issue-action-label">{t('report.submitted.criticalThinking')}</span>
                              <p>
                                <strong>{t(`report.criticalThinking.dimensions.${selectedCriticalThinking.dimension}.label`)}</strong>
                                {' — '}
                                {t(`report.criticalThinking.dimensions.${selectedCriticalThinking.dimension}.action`)}
                              </p>
                            </div>
                          )}

                          {selectedParagraph.flaggedSentences?.length > 0 && (
                            <div className="issue-action issue-action-evidence">
                              <span className="issue-action-label">{t('report.submitted.flaggedSentences')}</span>
                              <ul className="deberta-evidence-list">
                                {selectedParagraph.flaggedSentences.map((sent) => (
                                  <li key={sent.sentence_id}>
                                    {/* The per-sentence classifier score saturates (~100 on any flagged
                                        unit) and is NOT a calibrated probability — printing it as a raw
                                        "%" contradicts the de-escalated verdict (e.g. 100% on a green/LOW
                                        doc). Show the calibrated tier word the verdict-gate already set. */}
                                    {sent.tier && (
                                      <span className={`deberta-evidence-tier is-${sent.tier}`}>
                                        {t(`report.severities.${sent.tier}`, { defaultValue: sent.tier })}
                                      </span>
                                    )}
                                    <span className="deberta-evidence-text">{sent.text}</span>
                                    {sent.suggestion && (
                                      <span className="deberta-evidence-suggestion">{sent.suggestion}</span>
                                    )}
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}

                          <div className="issue-card-foot">
                            <div className="issue-card-foot-left">
                              {showSubmittedEditEntry && (
                                <button type="button" className="btn btn-primary btn-small" onClick={() => onEditParagraph(paragraph)}>
                                  <EditPencilIcon />{t('report.submitted.editParagraph')}
                                </button>
                              )}
                              <button type="button" className="btn btn-ghost btn-small" onClick={onCopyGuidance}>
                                {t('report.submitted.copyGuidance')}
                              </button>
                            </div>
                            <div className="issue-card-nav">
                              <button type="button" className="btn btn-secondary btn-small"
                                disabled={highlightedParagraphs.length < 2}
                                onClick={() => { onAdjacent(-1); }}>
                                {t('report.submitted.previousIssue')}
                              </button>
                              <button type="button" className="btn btn-secondary btn-small"
                                disabled={highlightedParagraphs.length < 2}
                                onClick={() => { onAdjacent(1); }}>
                                {t('report.submitted.nextIssue')}
                              </button>
                            </div>
                          </div>
                        </>
                      ) : (
                        <p className="issue-card-summary">{t('report.submitted.noSignal')}</p>
                      )}
                    </div>
                  )}
                </article>
              );
            })}
          </div>
        )
      )}
    </section>
  );
}
