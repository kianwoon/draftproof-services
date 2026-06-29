// draftproof-frontend/src/pages/report/AuthenticityDashboard.jsx
// Additive multi-dimension authenticity view. Renders nothing if the badge has no
// authenticity_dashboard (flag off / older report). Tiles never claim independence;
// the AI confidence interval is labelled tentative.
const TILES = ['learning_ownership', 'grounding', 'citation_quality', 'reasoning_consistency'];

function Tile({ t, keyName, tile }) {
  const score = tile && typeof tile.score === 'number' ? Math.round(tile.score) : null;
  return (
    <div className={`authn-tile ${score === null ? 'is-na' : ''}`}>
      <span className="authn-tile-label">{t(`report.authenticityDashboard.tiles.${keyName}`)}</span>
      <strong className="authn-tile-score">{score === null ? t('report.authenticityDashboard.na') : score}</strong>
      {tile && tile.caveat && <em className="authn-tile-caveat">{tile.caveat}</em>}
    </div>
  );
}

export default function AuthenticityDashboard({ t, dashboard }) {
  if (!dashboard) return null;
  const ai = dashboard.ai_assistance || {};
  const overall = dashboard.overall;
  return (
    <div className="authenticity-dashboard" aria-label={t('report.authenticityDashboard.ariaLabel')}>
      <h3>{t('report.authenticityDashboard.title')}</h3>
      <div className="authn-grid">
        {TILES.map((k) => <Tile key={k} t={t} keyName={k} tile={dashboard[k]} />)}
        <div className="authn-tile authn-ai">
          <span className="authn-tile-label">{t('report.authenticityDashboard.tiles.ai_assistance')}</span>
          <strong className="authn-tile-score">{ai.band ? t(`report.authenticityDashboard.bands.${ai.band}`) : t('report.authenticityDashboard.na')}</strong>
          {ai.ci && <em className="authn-tile-caveat">{t('report.authenticityDashboard.ciTentative', { low: Math.round(ai.ci.low), high: Math.round(ai.ci.high) })}</em>}
        </div>
      </div>
      {overall && (
        <p className="authn-overall">
          {t('report.authenticityDashboard.overall')}: <strong className={`is-${(overall.band || '').toLowerCase()}`}>{t(`report.authenticityDashboard.bands.${overall.band}`)}</strong>
        </p>
      )}
      <p className="authn-note">{t('report.authenticityDashboard.note')}</p>
    </div>
  );
}
