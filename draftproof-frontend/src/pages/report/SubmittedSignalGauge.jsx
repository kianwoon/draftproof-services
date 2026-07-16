import { useTranslation } from 'react-i18next';

export default function SubmittedSignalGauge({
  selectedSignalStrength,
  selectedParagraph,
  submittedDraftText,
}) {
  const { t } = useTranslation();
  if (selectedSignalStrength == null || !selectedParagraph?.primarySignal) return null;
  const value = Math.round(selectedSignalStrength);
  const stale = Boolean(selectedParagraph?.text) && !submittedDraftText.includes(selectedParagraph.text);
  return (
    <div
      className={`submitted-signal-gauge${stale ? ' is-stale' : ''}`}
      style={{
        '--signal-color': selectedParagraph.primarySignal.color || '#b45309',
        '--signal-strength': `${value}%`,
      }}
      aria-label={t('report.submitted.signalStrength', { value })}
    >
      <div className="submitted-signal-gauge-head">
        <span>{t('report.submitted.signalStrengthLabel')}</span>
        <strong>{value}%</strong>
      </div>
      <div className="submitted-signal-gauge-track" aria-hidden="true">
        <span />
      </div>
    </div>
  );
}
