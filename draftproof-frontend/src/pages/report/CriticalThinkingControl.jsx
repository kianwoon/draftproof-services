import {
  criticalThinkingControl,
  CRITICAL_THINKING_DIMENSIONS,
  CRITICAL_THINKING_CONTROL_ENABLED,
} from './reportHelpers';

// allow-hardcode: numeric->colour ramp, NOT a content/text matcher. Maps a 0-100
// control score to a hex colour at the band thresholds in
// poc/detect/critical_thinking.py (_band): 80 / 60 / 40 / 20. No document text
// is ever compared here.
function controlColor(control) {
  if (control >= 80) return '#15803d'; // strong control
  if (control >= 60) return '#65a30d'; // acceptable
  if (control >= 40) return '#d97706'; // weak
  if (control >= 20) return '#ea580c'; // high dependency
  return '#dc2626'; // very high dependency
}

// Critical Thinking Control — additive, non-gating, findings-first. Matches the
// report card design system (.critical-thinking-* in 06-report-overview.css,
// mirroring .score-profile-section / .fix-first-card). Leads with the weakest
// dimension, shows per-dimension control bars (weakest first), and the overall
// score as a secondary head stat. Renders nothing when absent or when the
// diagnosis abstained (insufficient_data).
export default function CriticalThinkingControl({ badge, t }) {
  if (!CRITICAL_THINKING_CONTROL_ENABLED) return null;
  const ctc = criticalThinkingControl(badge);
  if (!ctc) return null;

  const score = Math.round(ctc.score);
  const bandColor = controlColor(score);
  const dims = ctc.dimensions || {};
  const leadKey = ctc.lead_dimension;

  // Most-to-improve first: lowest control leads.
  const ordered = CRITICAL_THINKING_DIMENSIONS
    .filter((k) => dims[k] && dims[k].control != null)
    .sort((a, b) => dims[a].control - dims[b].control);

  // Phase-2 LLM-judged extras (present only when the kill-switch is on). These are
  // QUALITATIVE review flags — kept distinct from the scored bars because they
  // are NOT part of the control number.
  const llm = ctc.llm_dimensions || null;
  const highlights = Array.isArray(ctc.highlights) ? ctc.highlights : [];
  const llmRows = llm
    ? ['alternative_comparison', 'reflection']
        .filter((k) => llm[k] && llm[k].score != null)
        .map((k) => ({ key: k, score: Math.round(llm[k].score) }))
    : [];

  return (
    <section className="critical-thinking-section" aria-label={t('report.criticalThinking.title')}>
      <div className="critical-thinking-head">
        <div className="critical-thinking-head-main">
          <span className="critical-thinking-kicker">{t('report.criticalThinking.title')}</span>
          {leadKey ? (
            <>
              <h2>{t('report.criticalThinking.leadPrefix')}: {t(`report.criticalThinking.dimensions.${leadKey}.label`)}</h2>
              <p>
                {t(`report.criticalThinking.dimensions.${leadKey}.action`)}
                {ctc.caveat ? ` (${t('report.criticalThinking.tentative')})` : ''}
              </p>
            </>
          ) : (
            <h2>{t('report.criticalThinking.strongHeading')}</h2>
          )}
        </div>
        <div className="critical-thinking-stat">
          <strong style={{ color: bandColor }}>
            {score}<span className="critical-thinking-stat-max">/100</span>
          </strong>
          <span>{t(`report.criticalThinking.bands.${ctc.band}`)}</span>
        </div>
      </div>

      <div className="critical-thinking-body">
        {/* Per-dimension control bars (higher = stronger control). */}
        <div className="critical-thinking-bars">
          {ordered.map((k) => {
            const control = Math.round(dims[k].control);
            const isLead = k === leadKey;
            const color = controlColor(control);
            return (
              <div key={k} className="critical-thinking-bar-row">
                <div className="critical-thinking-bar-label">
                  <span className={isLead ? 'is-lead' : undefined}>
                    {t(`report.criticalThinking.dimensions.${k}.label`)}
                  </span>
                  <span style={{ color }}>{control}</span>
                </div>
                <div className="critical-thinking-bar-track">
                  <div className="critical-thinking-bar-fill" style={{ width: `${control}%`, background: color }} />
                </div>
              </div>
            );
          })}
        </div>

        {/* Phase-2 qualitative review flags (LLM-judged; NOT part of the score). */}
        {llmRows.length > 0 && (
          <div className="critical-thinking-flags">
            <span className="critical-thinking-subhead">{t('report.criticalThinking.reviewFlagsTitle')}</span>
            {llmRows.map(({ key, score: s }) => (
              <div key={key} className="critical-thinking-flag-row">
                <span>{t(`report.criticalThinking.llmDimensions.${key}`)}</span>
                <span style={{ color: controlColor(s) }}>{s}/100</span>
              </div>
            ))}
          </div>
        )}

        {/* Phase-2 sentence-level coaching highlights. */}
        {highlights.length > 0 && (
          <div className="critical-thinking-highlights">
            {highlights.map((h, i) => (
              <div key={`${h.paragraph_id}-${i}`} className="critical-thinking-highlight">
                <span className="critical-thinking-highlight-label">
                  {t(`report.criticalThinking.highlightLabels.${h.label}`, { defaultValue: h.label })}
                </span>
                <p className="critical-thinking-highlight-quote">“{h.sentence}”</p>
                {h.fix_instruction && (
                  <p className="critical-thinking-highlight-fix">{h.fix_instruction}</p>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
