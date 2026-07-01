// draftproof-frontend/src/pages/report/DebertaSignal.jsx
// Additive second-opinion AI signal: an off-the-shelf DeBERTa-class detector's score, shown
// side-by-side with the composite for comparison. Renders nothing if the badge has no
// ai_signal_deberta (flag off / older report). STRICTLY ADVISORY — never feeds the tier or
// any gate. Band uses the SAME green/amber/orange/red legend + cutoffs as the composite so the
// two are directly comparable; when they disagree that is the whole point of a second opinion.
import { TIER_CONFIG } from './reportHelpers';

export default function DebertaSignal({ t, signal, compositeTier }) {
  if (!signal) return null;

  const compositeKey = String(compositeTier || '').toLowerCase();

  // Unavailable (too-short / model load or inference failed): muted note, not the tile.
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

  const band = signal.band; // green|amber|orange|red (same legend as the composite)
  const cfg = (band && TIER_CONFIG[band]) || TIER_CONFIG.moderate;
  const score = typeof signal.score === 'number' ? Math.round(signal.score) : null;
  const bandLabel = t(`report.tiers.${band}`, { defaultValue: cfg.label });
  const compositeLabel = t(`report.tiers.${compositeKey}`, { defaultValue: (TIER_CONFIG[compositeKey] || TIER_CONFIG.moderate).label });
  const agree = !!(band && band === compositeKey);

  return (
    <section
      className="deberta-signal"
      aria-label={t('report.debertaSignal.ariaLabel')}
      style={{ background: cfg.bg, borderColor: cfg.color }}
    >
      <div className="deberta-head">
        <h3>{t('report.debertaSignal.title')}</h3>
        <span className="deberta-chip" style={{ background: cfg.color }}>{bandLabel}</span>
      </div>
      <div className="deberta-score-row">
        <strong className="deberta-score" style={{ color: cfg.color }}>
          {score === null ? '—' : score}%
        </strong>
        <span className="deberta-score-label">{t('report.debertaSignal.scoreLabel')}</span>
      </div>
      <p className="deberta-caveat">
        {signal.calibrated ? t('report.debertaSignal.calibrated') : t('report.debertaSignal.raw')}
      </p>
      <p className="deberta-note">
        {agree
          ? t('report.debertaSignal.noteAgree', { band: bandLabel })
          : t('report.debertaSignal.noteDisagree', { composite: compositeLabel, deberta: bandLabel })}
      </p>
    </section>
  );
}
