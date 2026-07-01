// draftproof-frontend/src/pages/report/DebertaSignal.jsx
// Additive second-opinion AI signal: a separate detector's read on the document, shown
// alongside the composite for comparison. Renders nothing if the badge has no
// ai_signal_deberta (flag off / older report). STRICTLY ADVISORY — never feeds the tier or
// any gate.
//
// SCHEMA (v2, threshold-proportion): the document is broken into sentence windows; a
// sentence "flags" if its detector score >= 0.99 (high-confidence AI band). signal_pct =
// the % of sentences that flag. Below 20% (the reliability floor) there is NO verdict —
// only the flagged passages are shown. >= 20% shows a band (amber/orange/red — never green:
// a signal that fires is never declared "safe"). This mirrors how mature detectors treat
// the low-reliability band.
//
// LEGACY v1 data (model_version "deberta_signal_v1", the old calibrated score) is rendered
// via a minimal fallback so old reports in R2 don't break — but those reports predate this
// component's flagged-passage detail.
import { TIER_CONFIG } from './reportHelpers';

const V1_BANDS = { green: true, amber: true, orange: true, red: true }; // v1 traffic-light bands

function FlaggedPassages({ t, passages }) {
  if (!passages || !passages.length) return null;
  return (
    <div className="deberta-flagged-list">
      <span className="deberta-flagged-list-label">{t('report.debertaSignal.reviewPassages')}</span>
      <ul>
        {passages.map((p) => (
          <li key={p.sentence_id} className="deberta-flagged-item">
            <span className="deberta-flagged-score">{Math.round((p.score || 0) * 100)}%</span>
            <span className="deberta-flagged-text">{p.text}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

// Legacy v1 (calibrated score) fallback — old reports in R2 still carry this shape.
function V1Signal({ t, signal, compositeTier }) {
  const band = V1_BANDS[signal.band] ? signal.band : null;
  const cfg = (band && TIER_CONFIG[band]) || TIER_CONFIG.moderate;
  const compositeKey = String(compositeTier || '').toLowerCase();
  const agree = !!(band && band === compositeKey);
  return (
    <section className="deberta-signal is-legacy" aria-label={t('report.debertaSignal.ariaLabel')}
      style={{ background: cfg.bg, borderColor: cfg.color }}>
      <div className="deberta-head">
        <h3>{t('report.debertaSignal.title')}</h3>
        {band && <span className="deberta-chip" style={{ background: cfg.color }}>{cfg.label}</span>}
      </div>
      <div className="deberta-score-row">
        <strong className="deberta-score" style={{ color: cfg.color }}>
          {typeof signal.score === 'number' ? Math.round(signal.score) : '—'}%
        </strong>
        <span className="deberta-score-label">{t('report.debertaSignal.scoreLabel')}</span>
      </div>
      <p className="deberta-note">
        {agree
          ? t('report.debertaSignal.noteAgree')
          : t('report.debertaSignal.noteDisagree')}
      </p>
    </section>
  );
}

export default function DebertaSignal({ t, signal, compositeTier }) {
  if (!signal) return null;

  // Legacy v1 reports (calibrated score schema). Render a minimal tile and stop.
  if (signal.model_version !== 'deberta_signal_v2') {
    return <V1Signal t={t} signal={signal} compositeTier={compositeTier} />;
  }

  // Unavailable (too-short / model load or inference failed): muted note, no tile body.
  if (signal.available === false) {
    return (
      <section className="deberta-signal is-na" aria-label={t('report.debertaSignal.ariaLabel')}>
        <h3>{t('report.debertaSignal.title')}</h3>
        <p className="deberta-na-note">
          {t('report.debertaSignal.unavailable')}{signal.caveat ? ` — ${signal.caveat}` : ''}
        </p>
      </section>
    );
  }

  const pct = typeof signal.signal_pct === 'number' ? signal.signal_pct : null;
  const band = signal.band; // insufficient | amber | orange | red
  const aboveFloor = band && band !== 'insufficient';
  const cfg = (band && TIER_CONFIG[band]) || TIER_CONFIG.moderate; // insufficient -> moderate grey-ish
  const flagged = Array.isArray(signal.flagged_passages) ? signal.flagged_passages : [];

  // Composite comparison — only meaningful above the floor.
  const compositeKey = String(compositeTier || '').toLowerCase();
  const compositeAlsoFlags = !!(compositeKey && compositeKey !== 'green');

  return (
    <section
      className={`deberta-signal ${aboveFloor ? '' : 'is-insufficient'}`}
      aria-label={t('report.debertaSignal.ariaLabel')}
      style={aboveFloor ? { background: cfg.bg, borderColor: cfg.color } : undefined}
    >
      <div className="deberta-head">
        <h3>{t('report.debertaSignal.title')}</h3>
        {aboveFloor && <span className="deberta-chip" style={{ background: cfg.color }}>{cfg.label}</span>}
      </div>

      {aboveFloor ? (
        // ABOVE FLOOR — verdict band + proportion + flagged passages.
        <>
          <div className="deberta-score-row">
            <strong className="deberta-score" style={{ color: cfg.color }}>
              {pct === null ? '—' : pct}%
            </strong>
            <span className="deberta-score-label">{t('report.debertaSignal.scoreLabel')}</span>
          </div>
          <p className="deberta-caveat">
            {t('report.debertaSignal.aboveFloor', { pct: (pct === null ? '—' : pct) })}
          </p>
          <FlaggedPassages t={t} passages={flagged} />
          <p className="deberta-note">
            {compositeAlsoFlags
              ? t('report.debertaSignal.noteAgree')
              : t('report.debertaSignal.noteDisagree')}
          </p>
        </>
      ) : (
        // BELOW FLOOR — no verdict. Show the flagged passages (if any) + the floor caveat.
        // Even one flagged passage is worth surfacing for review, but we do NOT show a band.
        <>
          <p className="deberta-caveat">
            {t('report.debertaSignal.passagesFlaggedForReview',
              { n: signal.sentences_flagged || 0, total: signal.sentences_scored || 0 })}
          </p>
          <FlaggedPassages t={t} passages={flagged} />
          <p className="deberta-note">{t('report.debertaSignal.belowFloor')}</p>
        </>
      )}
    </section>
  );
}
