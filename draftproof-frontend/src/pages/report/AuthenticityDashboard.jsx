// draftproof-frontend/src/pages/report/AuthenticityDashboard.jsx
// Additive multi-dimension authenticity view. Renders nothing if the badge has no
// authenticity_dashboard (flag off / older report). Tiles never claim independence;
// the AI confidence interval is labelled tentative.
//
// POLARITY: score tiles are on the AUTHENTICITY scale (higher = better → good);
// band tiles (overall / ai_assistance) use the RISK vocabulary (Low = good,
// High = bad). One `tone` (good|fair|weak|na) drives all accent color so the two
// scales can never be cross-colored. Cuts mirror the composer _AUTH_HIGH_MIN (66)
// / _AUTH_MED_MIN (38).

// Authenticity-axis cuts (higher score = more authentic). Must mirror the composer
// (_AUTH_HIGH_MIN=66 / _AUTH_MED_MIN=38 in poc/detect/authenticity_dashboard.py).
const AUTH_GOOD_MIN = 66;
const AUTH_FAIR_MIN = 38;

const TILES = ['learning_ownership', 'grounding', 'citation_quality', 'reasoning_consistency'];

// RISK vocabulary → tone. Low = good (green), High = bad (red).
const BAND_TONE = { Low: 'good', Medium: 'fair', Moderate: 'fair', High: 'weak' };

function scoreTone(score) {
  if (typeof score !== 'number') return 'na';
  return score >= AUTH_GOOD_MIN ? 'good' : score >= AUTH_FAIR_MIN ? 'fair' : 'weak';
}

function Tile({ t, keyName, tile }) {
  const score = tile && typeof tile.score === 'number' ? Math.round(tile.score) : null;
  const tone = scoreTone(tile && tile.score);
  return (
    <div className={`authn-tile is-${tone}`}>
      <span className="authn-tile-label">{t(`report.authenticityDashboard.tiles.${keyName}`)}</span>
      <strong className="authn-tile-score">{score === null ? t('report.authenticityDashboard.na') : score}</strong>
      {tile && tile.caveat && <em className="authn-tile-caveat">{t(`report.authenticityDashboard.caveats.${keyName}`)}</em>}
    </div>
  );
}

export default function AuthenticityDashboard({ t, dashboard }) {
  if (!dashboard) return null;
  const ai = dashboard.ai_assistance || {};
  const overall = dashboard.overall;
  const aiTone = (ai.band && BAND_TONE[ai.band]) || 'na';
  const overallTone = (overall && overall.band && BAND_TONE[overall.band]) || 'na';
  return (
    <section className="authenticity-dashboard" aria-label={t('report.authenticityDashboard.ariaLabel')}>
      <div className="authn-head">
        <span className="authn-eyebrow">
          <h3>{t('report.authenticityDashboard.title')}</h3>
        </span>
        {overall && (
          <span className={`authn-overall-chip is-${overallTone}`}>
            <span className="dot" />
            {t('report.authenticityDashboard.overall')}: {t(`report.authenticityDashboard.bands.${overall.band}`)}
          </span>
        )}
      </div>
      <div className="authn-grid">
        {TILES.map((k) => <Tile key={k} t={t} keyName={k} tile={dashboard[k]} />)}
        <div className={`authn-tile authn-ai is-${aiTone}`}>
          <span className="authn-tile-label">{t('report.authenticityDashboard.tiles.ai_assistance')}</span>
          <strong className="authn-tile-score">{ai.band ? t(`report.authenticityDashboard.bands.${ai.band}`) : t('report.authenticityDashboard.na')}</strong>
          {ai.ci ? (
            <em className="authn-tile-caveat">{t('report.authenticityDashboard.ciTentative', { low: Math.round(ai.ci.low), high: Math.round(ai.ci.high) })}</em>
          ) : ai.band ? (
            <em className="authn-tile-caveat">{t('report.authenticityDashboard.ciUnavailable')}</em>
          ) : null}
        </div>
      </div>
      <p className="authn-note">{t('report.authenticityDashboard.note')}</p>
    </section>
  );
}
