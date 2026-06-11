// allow-hardcode: this is a presentational React component — the repeated string
// literals are CSS class names (submitted-editor-*, btn btn-ghost) and i18n message
// keys, i.e. UI markup, not a content detect-list or scoring/matching oracle. It
// mirrors the editor sheet markup in Report.jsx so both share identical styling.
import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { startScanWithText, translateText, getScanStatus } from '../../api/draftproofApi';
import { getReportDraft, saveReportDraft, deleteReportDraft } from '../../utils/reportDraftStorage';
import { countWords, scanTokensRequired } from '../../utils/scanBilling';
import { useAuth } from '../../context/AuthContext';
import { formatDate } from './reportHelpers';
import { buildTrackedDiff, trackedDiffToPlainText, trackedDiffToHtml } from './trackedDiff';
import { keptSentences, findSentenceRange, highlightParts } from '../../utils/bracketSpans';

const TRANSITION_MS = 480;
const RESCAN_POLL_INTERVAL = 3000;
const RESCAN_MAX_POLLS = 200;
const sleep = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));

// In-place "Manual Rewrite / Correction" editor for the /rewrite page. Edits the
// rewritten draft (baselineText = rewrite final_text), autosaves to the namespaced
// draft key, and re-scans the edited text into a fresh /report/{id}. Mirrors the
// submitted-draft editor sheet in Report.jsx (reuses the same submitted-editor-* CSS),
// minus the original-scan highlight overlay and affected-paragraph panel — neither maps
// onto rewritten text, so rewrite mode never showed them.
//
// The right panel is amber-first: bracketSpans carries the rewrite's green/amber spans,
// from which we extract the AMBER ('kept') sentences — the ones the rewrite could not
// safely improve, i.e. the user's to ground. We surface them as an "edit these" checklist
// above the "how to ground a claim" worked examples. We do NOT highlight inline in the
// textarea: the span offsets are valid only against the unedited baseline and would drift
// on the first keystroke.
export default function RewriteDraftEditor({ storageKey, baselineText, workedExamples = [], bracketSpans = [], onClose }) {
  const { t, i18n } = useTranslation();
  const locale = i18n.language;
  const navigate = useNavigate();
  const { balance, refreshBalance } = useAuth();

  const editorRef = useRef(null);
  const highlightRef = useRef(null);
  const closeTimerRef = useRef(null);
  const trackedCopyTimerRef = useRef(null);

  const [draftText, setDraftText] = useState(baselineText || '');
  const [draftLoaded, setDraftLoaded] = useState(false);
  const [draftStatus, setDraftStatus] = useState('idle');
  const [draftUpdatedAt, setDraftUpdatedAt] = useState(null);
  const [closing, setClosing] = useState(false);
  const [rescanBusy, setRescanBusy] = useState(false);
  const [rescanStatus, setRescanStatus] = useState(null);
  const [rescanError, setRescanError] = useState(null);
  const [needsTokens, setNeedsTokens] = useState(false);
  const [translateBusy, setTranslateBusy] = useState(false);
  const [translateError, setTranslateError] = useState(null);
  const [preTranslateText, setPreTranslateText] = useState(null);
  const [trackedCopyStatus, setTrackedCopyStatus] = useState('idle');

  const draftChanged = draftText !== baselineText;
  const guideExamples = Array.isArray(workedExamples) ? workedExamples.filter(Boolean) : [];
  const hasGuide = guideExamples.length > 0;
  // Amber ('kept') sentences the rewrite left for the user to ground themselves.
  const amberSentences = keptSentences(baselineText, bracketSpans);
  const hasAmber = amberSentences.length > 0;
  const hasPanel = hasGuide || hasAmber;
  // Re-find each amber sentence in the CURRENT draft so highlights survive edits
  // elsewhere and clear once the sentence itself is rewritten. `index` ties each
  // overlay span (and the right-panel card) together for click-to-jump.
  const amberMatches = amberSentences
    .map((sentence, index) => ({ sentence, index, range: findSentenceRange(draftText, sentence) }))
    .filter((m) => m.range);
  const amberRangeByIndex = new Map(amberMatches.map((m) => [m.index, m.range]));
  const editorHighlightParts = highlightParts(
    draftText,
    amberMatches.map((m) => ({ ...m.range, index: m.index })),
  );
  const trackedDiff = buildTrackedDiff(baselineText, draftText);
  const draftWordCount = countWords(draftText);
  const draftTokensRequired = scanTokensRequired(draftWordCount);

  const clearCloseTimer = () => {
    if (closeTimerRef.current) {
      window.clearTimeout(closeTimerRef.current);
      closeTimerRef.current = null;
    }
  };
  const clearTrackedCopyTimer = () => {
    if (trackedCopyTimerRef.current) {
      window.clearTimeout(trackedCopyTimerRef.current);
      trackedCopyTimerRef.current = null;
    }
  };

  const handleClose = () => {
    if (rescanBusy) return;
    clearCloseTimer();
    const reduceMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    if (reduceMotion) {
      onClose?.();
      return;
    }
    setClosing(true);
    closeTimerRef.current = window.setTimeout(() => {
      closeTimerRef.current = null;
      onClose?.();
    }, TRANSITION_MS);
  };

  // Seed from the rewritten baseline, then hydrate any saved local draft.
  useEffect(() => {
    let cancelled = false;
    setDraftLoaded(false);
    setDraftText(baselineText || '');
    setDraftStatus('idle');
    setDraftUpdatedAt(null);

    getReportDraft(storageKey)
      .then((draft) => {
        if (cancelled) return;
        if (draft?.text) {
          setDraftText(draft.text);
          setDraftStatus('saved');
          setDraftUpdatedAt(draft.updatedAt || null);
        }
        setDraftLoaded(true);
      })
      .catch(() => {
        if (!cancelled) {
          setDraftLoaded(true);
          setDraftStatus('error');
        }
      });

    return () => { cancelled = true; };
  }, [storageKey, baselineText]);

  // Debounced autosave; delete the draft once it matches the baseline again.
  useEffect(() => {
    if (!draftLoaded) return undefined;
    if (draftText === baselineText) {
      setDraftStatus('idle');
      setDraftUpdatedAt(null);
      deleteReportDraft(storageKey).catch(() => {});
      return undefined;
    }
    setDraftStatus('saving');
    const timer = window.setTimeout(() => {
      saveReportDraft(storageKey, draftText)
        .then((draft) => {
          setDraftStatus('saved');
          setDraftUpdatedAt(draft?.updatedAt || null);
        })
        .catch(() => setDraftStatus('error'));
    }, 650);
    return () => window.clearTimeout(timer);
  }, [draftText, draftLoaded, baselineText, storageKey]);

  // Escape closes (unless a re-scan is in flight); clean up timers on unmount.
  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.key === 'Escape' && !rescanBusy) handleClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rescanBusy]);

  useEffect(() => () => {
    clearCloseTimer();
    clearTrackedCopyTimer();
  }, []);

  const handleChange = (event) => {
    setDraftText(event.target.value);
    setRescanError(null);
    setPreTranslateText(null);
    clearTrackedCopyTimer();
    setTrackedCopyStatus('idle');
  };

  // Keep the (pointer-events: none) highlight overlay aligned with the textarea.
  const syncHighlightScroll = () => {
    const editor = editorRef.current;
    const layer = highlightRef.current;
    if (!editor || !layer) return;
    layer.scrollTop = editor.scrollTop;
    layer.scrollLeft = editor.scrollLeft;
  };

  // Click an amber card -> select that sentence in the editor and scroll to it.
  // No-op once the sentence has been edited away (its range no longer matches).
  const jumpToAmber = (index) => {
    const range = amberRangeByIndex.get(index);
    const editor = editorRef.current;
    if (!range || !editor) return;
    editor.focus({ preventScroll: true });
    editor.setSelectionRange(range.start, range.end);
    // Measure against the overlay span (identical metrics) and centre it.
    const span = highlightRef.current?.querySelector(`[data-amber-index="${index}"]`);
    const layer = highlightRef.current;
    if (span && layer) {
      const target = span.offsetTop - layer.clientHeight / 2 + span.offsetHeight / 2;
      editor.scrollTop = Math.max(0, target);
    }
    syncHighlightScroll();
  };

  const resetDraft = async () => {
    setDraftText(baselineText || '');
    setDraftStatus('idle');
    setDraftUpdatedAt(null);
    setRescanError(null);
    setPreTranslateText(null);
    await deleteReportDraft(storageKey).catch(() => {});
  };

  const translateSelection = async () => {
    const editor = editorRef.current;
    if (!editor) return;
    const start = editor.selectionStart;
    const end = editor.selectionEnd;
    const selected = (start != null && end != null && end > start) ? draftText.slice(start, end) : '';
    if (!selected.trim()) {
      setTranslateError(t('report.submitted.editor.translateSelectFirst'));
      return;
    }
    setTranslateBusy(true);
    setTranslateError(null);
    try {
      const { data } = await translateText(selected, { target: 'en' });
      const translated = (data?.text || '').trim();
      if (!translated) {
        setTranslateError(t('report.submitted.editor.translateError'));
        return;
      }
      const previous = draftText;
      const nextText = previous.slice(0, start) + translated + previous.slice(end);
      setPreTranslateText(previous);
      setDraftText(nextText);
      setRescanError(null);
      window.requestAnimationFrame(() => {
        const node = editorRef.current;
        if (!node) return;
        node.focus({ preventScroll: true });
        const caret = start + translated.length;
        node.setSelectionRange(caret, caret);
      });
    } catch {
      setTranslateError(t('report.submitted.editor.translateError'));
    } finally {
      setTranslateBusy(false);
    }
  };

  const undoTranslate = () => {
    if (preTranslateText == null) return;
    setDraftText(preTranslateText);
    setPreTranslateText(null);
    setTranslateError(null);
  };

  const copyTrackedChanges = async () => {
    const plainText = trackedDiffToPlainText(trackedDiff);
    const html = trackedDiffToHtml(trackedDiff);
    clearTrackedCopyTimer();
    try {
      if (navigator.clipboard?.write && window.ClipboardItem) {
        await navigator.clipboard.write([
          new window.ClipboardItem({
            'text/html': new Blob([html], { type: 'text/html' }),
            'text/plain': new Blob([plainText], { type: 'text/plain' }),
          }),
        ]);
      } else if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(plainText);
      } else {
        throw new Error('Clipboard API unavailable');
      }
      setTrackedCopyStatus('copied');
      trackedCopyTimerRef.current = window.setTimeout(() => {
        setTrackedCopyStatus('idle');
        trackedCopyTimerRef.current = null;
      }, 1800);
    } catch {
      setTrackedCopyStatus('error');
      trackedCopyTimerRef.current = window.setTimeout(() => {
        setTrackedCopyStatus('idle');
        trackedCopyTimerRef.current = null;
      }, 2200);
    }
  };

  const rescanDraft = async () => {
    const text = draftText.trim();
    if (!text) {
      setRescanError(t('report.submitted.editor.emptyDraft'));
      return;
    }
    if (balance !== null && balance < draftTokensRequired) {
      setNeedsTokens(true);
      setRescanError(null);
      setRescanStatus(null);
      return;
    }

    setRescanBusy(true);
    setRescanError(null);
    setRescanStatus(t('report.submitted.editor.rescanQueueing'));

    try {
      const { data: scan } = await startScanWithText(text);
      setRescanStatus(t('report.submitted.editor.rescanProcessing'));

      if (scan.status === 'completed') {
        refreshBalance?.();
        navigate(`/report/${scan.id}`);
        return;
      }

      for (let i = 0; i < RESCAN_MAX_POLLS; i += 1) {
        await sleep(RESCAN_POLL_INTERVAL);
        const { data } = await getScanStatus(scan.id);
        if (data.status === 'completed') {
          refreshBalance?.();
          navigate(`/report/${scan.id}`);
          return;
        }
        if (data.status === 'failed') {
          throw new Error(data.error || t('report.submitted.editor.rescanFailed'));
        }
        if (data.progress_message) {
          setRescanStatus(data.progress_message);
        }
      }
      throw new Error(t('report.submitted.editor.rescanTimedOut'));
    } catch (err) {
      setRescanError(err?.response?.data?.detail || err?.message || t('report.submitted.editor.rescanFailed'));
      setRescanStatus(null);
      setRescanBusy(false);
    }
  };

  return (
    <div
      className={`submitted-editor-backdrop${closing ? ' is-closing' : ''}`}
      role="dialog"
      aria-modal="true"
      aria-label={t('report.submitted.editor.rewriteTitle')}
    >
      <div className="submitted-editor-sheet">
        <button
          type="button"
          className="submitted-editor-close-button"
          aria-label={t('report.submitted.editor.close')}
          title={t('report.submitted.editor.close')}
          onClick={handleClose}
          disabled={rescanBusy}
        >
          X
        </button>
        <div className="submitted-editor-head">
          <div>
            <span className="submitted-content-kicker">{t('report.submitted.editor.kicker')}</span>
            <h2>{t('report.submitted.editor.rewriteTitle')}</h2>
            <p>{t('report.submitted.editor.rewriteNotice')}</p>
          </div>
          <div className="submitted-editor-actions">
            <span className={`submitted-save-state is-${draftStatus}`}>
              {draftStatus === 'saving'
                ? t('report.submitted.editor.saving')
                : draftStatus === 'saved'
                  ? t('report.submitted.editor.saved', {
                    value: draftUpdatedAt ? formatDate(draftUpdatedAt, locale) : t('common.lastUpdated'),
                  })
                  : draftStatus === 'error'
                    ? t('report.submitted.editor.saveError')
                    : t('report.submitted.editor.noDraft')}
            </span>
            <button type="button" className="btn btn-ghost" onClick={handleClose} disabled={rescanBusy}>
              {t('report.submitted.editor.close')}
            </button>
          </div>
        </div>

        <div className={`submitted-editor-grid${hasPanel ? '' : ' is-solo'}`}>
          <section className="submitted-editor-main" aria-label={t('report.submitted.editor.documentEditor')}>
            <div className="submitted-editor-toolbar">
              <div>
                <strong>{t('report.submitted.editor.documentEditor')}</strong>
                <span>{draftChanged ? t('report.submitted.editor.changed') : t('report.submitted.editor.unchanged')}</span>
              </div>
              <div className="submitted-editor-toolbar-actions">
                <button
                  type="button"
                  className="btn btn-secondary submitted-translate-button"
                  onClick={translateSelection}
                  disabled={translateBusy || rescanBusy}
                  title={t('report.submitted.editor.translateNote')}
                >
                  <svg className="cta-edit-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true" focusable="false">
                    <path d="M4 5h7M7.5 5v1.5M9.5 5c0 4-2.5 7-5.5 8.5M6 9c.8 2 2.6 3.6 5 4.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                    <path d="M13 19l3.2-8h.6L20 19M14 16.5h5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  {translateBusy ? t('report.submitted.editor.translating') : t('report.submitted.editor.translateCnEn')}
                </button>
                {preTranslateText != null && (
                  <button
                    type="button"
                    className="btn btn-ghost btn-small submitted-translate-undo"
                    onClick={undoTranslate}
                    disabled={translateBusy || rescanBusy}
                  >
                    {t('report.submitted.editor.undoTranslate')}
                  </button>
                )}
                <button
                  type="button"
                  className="btn btn-ghost"
                  onClick={resetDraft}
                  disabled={!draftChanged || rescanBusy}
                >
                  {t('report.submitted.editor.discardDraft')}
                </button>
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={rescanDraft}
                  disabled={rescanBusy || !draftText.trim()}
                >
                  {rescanBusy ? t('report.submitted.editor.rescanning') : t('report.submitted.editor.rescanDraft')}
                </button>
                <span className="submitted-rescan-token-note">
                  {t('scan.word', { count: draftWordCount })}
                  {' · '}
                  {draftTokensRequired > 0
                    ? t('scan.tokensRequired', { count: draftTokensRequired })
                    : t('scan.freeScan')}
                </span>
              </div>
            </div>
            <div className={`submitted-translate-tip${translateError ? ' is-error' : ''}`}>
              <svg className="cta-edit-icon" width="15" height="15" viewBox="0 0 24 24" fill="none" aria-hidden="true" focusable="false">
                <path d="M4 5h7M7.5 5v1.5M9.5 5c0 4-2.5 7-5.5 8.5M6 9c.8 2 2.6 3.6 5 4.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                <path d="M13 19l3.2-8h.6L20 19M14 16.5h5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              <span>
                {translateError
                  ? translateError
                  : preTranslateText != null
                    ? t('report.submitted.editor.translateNote')
                    : t('report.submitted.editor.translateHint')}
              </span>
            </div>
            <div className="submitted-editor-textarea-wrap">
              {hasAmber && (
                <div ref={highlightRef} className="submitted-editor-highlight-layer" aria-hidden="true">
                  {editorHighlightParts.map((part, index) => (
                    <span
                      key={`${part.type}-${index}`}
                      className={`submitted-editor-highlight-${part.type}`}
                      data-amber-index={part.type === 'selected' ? part.index : undefined}
                    >
                      {part.text}
                    </span>
                  ))}
                  {'\n'}
                </div>
              )}
              <textarea
                ref={editorRef}
                className="submitted-editor-textarea"
                value={draftText}
                onChange={handleChange}
                onScroll={hasAmber ? syncHighlightScroll : undefined}
                spellCheck="true"
              />
            </div>
            {(rescanStatus || rescanError) && (
              <div className={`submitted-rescan-status${rescanError ? ' is-error' : ''}`}>
                {rescanError || rescanStatus}
              </div>
            )}
            {needsTokens && (
              <div className="submitted-rescan-status is-error">
                <span>{t('scan.notEnoughMessage')}</span>
                {' '}
                <button type="button" className="btn btn-ghost btn-small" onClick={() => navigate('/buy')}>
                  {t('scan.buyTokens')}
                </button>
              </div>
            )}
            <div className="submitted-tracked-preview" aria-label={t('report.submitted.editor.trackedPreview')}>
              <div className="submitted-tracked-head">
                <div className="submitted-tracked-title">
                  <strong>{t('report.submitted.editor.trackedPreview')}</strong>
                  <span>{t('report.submitted.editor.trackedPreviewBody')}</span>
                </div>
                <button
                  type="button"
                  className={`submitted-tracked-copy-button${trackedCopyStatus === 'copied' ? ' is-copied' : ''}${trackedCopyStatus === 'error' ? ' has-error' : ''}`}
                  onClick={copyTrackedChanges}
                  disabled={!trackedDiff.length}
                >
                  {trackedCopyStatus === 'copied'
                    ? t('report.submitted.editor.copiedTrackedChanges')
                    : trackedCopyStatus === 'error'
                      ? t('report.submitted.editor.copyTrackedChangesFailed')
                      : t('report.submitted.editor.copyTrackedChanges')}
                </button>
              </div>
              <div className="submitted-tracked-body">
                {trackedDiff.map((part, index) => (
                  <span key={`${part.type}-${index}`} className={`submitted-diff-${part.type}`}>
                    {part.text}
                  </span>
                ))}
              </div>
            </div>
          </section>

          {hasPanel && (
            <aside
              className="submitted-editor-guide"
              aria-label={hasAmber
                ? t('report.submitted.editor.amberGuideHeading')
                : t('rewritePage.workedExamples.heading')}
            >
              <div className="submitted-editor-guide-head">
                <div>
                  <span>{hasAmber
                    ? t('report.submitted.editor.amberGuideKicker')
                    : t('rewritePage.workedExamples.kicker')}</span>
                  <strong className="submitted-editor-guide-heading">{hasAmber
                    ? t('report.submitted.editor.amberGuideHeading')
                    : t('rewritePage.workedExamples.heading')}</strong>
                </div>
                <span className="submitted-editor-guide-count">{hasAmber ? amberSentences.length : guideExamples.length}</span>
              </div>
              <p className="submitted-editor-guide-copy">{hasAmber
                ? t('report.submitted.editor.amberGuideCopy')
                : t('rewritePage.workedExamples.copy')}</p>
              <div className="submitted-editor-guide-list">
                {hasAmber && (
                  <>
                    <p className="submitted-editor-guide-subhead">{t('report.submitted.editor.amberListHeading')}</p>
                    {amberSentences.map((sentence, i) => {
                      const matched = amberRangeByIndex.has(i);
                      return (
                        <article
                          className={`rewrite-suggestion-card is-amber${matched ? ' is-clickable' : ' is-resolved'}`}
                          key={`amber-${i}`}
                          role={matched ? 'button' : undefined}
                          tabIndex={matched ? 0 : undefined}
                          onClick={matched ? () => jumpToAmber(i) : undefined}
                          onKeyDown={matched ? (e) => {
                            if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); jumpToAmber(i); }
                          } : undefined}
                          title={matched ? t('report.submitted.editor.amberJumpHint') : t('report.submitted.editor.amberResolved')}
                        >
                          <div className="rewrite-target-block">
                            <span>{matched
                              ? t('report.submitted.editor.amberItemLabel')
                              : t('report.submitted.editor.amberResolved')}</span>
                            <p>{sentence}</p>
                          </div>
                        </article>
                      );
                    })}
                  </>
                )}
                {hasGuide && (
                  <>
                    {hasAmber && (
                      <p className="submitted-editor-guide-subhead">{t('rewritePage.workedExamples.heading')}</p>
                    )}
                    {guideExamples.map((item, i) => (
                      <article className="rewrite-suggestion-card" key={`guide-${i}`}>
                        <div className="rewrite-target-block">
                          <span>{t('rewritePage.workedExamples.generalClaim')}</span>
                          <p>{item.sentence}</p>
                        </div>
                        <div className="rewrite-addition-block">
                          <span>{t('rewritePage.workedExamples.moreGrounded')}</span>
                          <p>{item.suggestion}</p>
                        </div>
                        {item.why && (
                          <div className="rewrite-review-note"><p>{t('rewritePage.workedExamples.why')}: {item.why}</p></div>
                        )}
                      </article>
                    ))}
                  </>
                )}
              </div>
            </aside>
          )}
        </div>
      </div>
    </div>
  );
}
