import { useState, useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { signalClassName, signalLabel, signalDescription } from './reportHelpers';
import ParagraphSeverityBar from '../../components/ParagraphSeverityBar';
import EditPencilIcon from './EditPencilIcon';

export default function SignalHighlights({
  submittedContent, selectedParagraph, selectedParagraphId, highlightedParagraphs,
  paragraphSeverityBar,
  selectedCriticalThinking,
  showSubmittedEditEntry, onSelectParagraph, onPreviewParagraph, onAdjacent,
  onEditParagraph, onCopyGuidance,
}) {
  const { t } = useTranslation();
  const [tab, setTab] = useState('issues'); // 'issues' | 'document'
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

  const FullDocument = (
    <div className="submitted-document" aria-label={t('report.submitted.documentText')}>
      {submittedContent.paragraphs.map((paragraph) => {
        const signal = paragraph.primarySignal;
        const isSelected = selectedParagraphId === paragraph.id;
        if (!signal) {
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
          <p key={paragraph.id}>
            <button type="button" data-paragraph-id={paragraph.id}
              className={`submitted-highlight submitted-paragraph-highlight signal-style-${signalClassName(signal.key)}${isSelected ? ' is-selected' : ''}`}
              style={{ '--signal-color': signal.color }}
              title={signalDescription(signal.key, signal.description, t)}
              onMouseEnter={() => onPreviewParagraph(paragraph.id)}
              onFocus={() => onPreviewParagraph(paragraph.id)}
              onClick={() => { onSelectParagraph(paragraph.id); setTab('issues'); }}>
              {paragraph.text}
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
          {showSubmittedEditEntry && (
            <button type="button" className="btn btn-secondary submitted-edit-button"
              onClick={() => onEditParagraph()}>
              <EditPencilIcon />{t('report.submitted.editor.editDraft')}
            </button>
          )}
        </div>
      </div>

      {paragraphSeverityBar?.length > 0 && (
        <ParagraphSeverityBar bar={paragraphSeverityBar} selectedId={selectedParagraph?.id} onSelect={onSelectParagraph} />
      )}

      {submittedContent.legend?.length > 0 && (
        <div className="submitted-signal-legend" aria-label={t('report.submitted.legend')}>
          {submittedContent.legend.slice(0, 6).map((signal) => (
            <span key={signal.key}
              className={`submitted-signal-chip signal-style-${signalClassName(signal.key)}`}
              style={{ '--signal-color': signal.color }}>
              <i aria-hidden="true" />{signalLabel(signal.key, signal.label, t)}<strong>{signal.count}</strong>
            </span>
          ))}
        </div>
      )}

      <div className="submitted-tabs" role="tablist" aria-label={t('report.submitted.title')}>
        <button type="button" role="tab" id="sh-tab-issues" aria-selected={tab === 'issues'}
          aria-controls="sh-panel-issues"
          className={`submitted-tab${tab === 'issues' ? ' is-active' : ''}`}
          onClick={() => setTab('issues')}>
          {t('report.submitted.tabIssues', { count: highlightedParagraphs.length })}
        </button>
        <button type="button" role="tab" id="sh-tab-document" aria-selected={tab === 'document'}
          aria-controls="sh-panel-document"
          className={`submitted-tab${tab === 'document' ? ' is-active' : ''}`}
          onClick={() => setTab('document')}>
          {t('report.submitted.tabDocument')}
        </button>
      </div>

      {tab === 'document' && (
        <div role="tabpanel" id="sh-panel-document" aria-labelledby="sh-tab-document">{FullDocument}</div>
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
              const snippet = (paragraph.text || '').slice(0, 160);
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
                      <span className="issue-card-snippet">{snippet}{paragraph.text.length > 160 ? '…' : ''}</span>
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
                          {selectedParagraph.recommendation && (
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
                                {selectedParagraph.flaggedSentences.slice(0, 3).map((sent) => (
                                  <li key={sent.sentence_id}>
                                    <span className="deberta-evidence-score">{Math.round(sent.score)}%</span>
                                    <span className="deberta-evidence-text">{sent.text}</span>
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
