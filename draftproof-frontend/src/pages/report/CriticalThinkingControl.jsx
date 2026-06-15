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

// Critical Thinking Control — additive, non-gating, findings-first.
// Leads with the weakest dimension (what to think harder about), shows per-
// dimension control bars (weakest first), and the overall score as a secondary
// badge. Reads report.ai_risk_badge.critical_thinking_control; renders nothing
// when absent or when the diagnosis abstained (insufficient_data).
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
  // QUALITATIVE review flags — kept visually distinct from the scored bars above
  // because they are NOT part of the control number.
  const llm = ctc.llm_dimensions || null;
  const highlights = Array.isArray(ctc.highlights) ? ctc.highlights : [];
  const llmRows = llm
    ? ['alternative_comparison', 'reflection']
        .filter((k) => llm[k] && llm[k].score != null)
        .map((k) => ({ key: k, score: Math.round(llm[k].score) }))
    : [];

  return (
    <section className="critical-thinking-control" style={{ marginTop: '20px' }}>
      <div className="ai-likelihood-caption" style={{ fontWeight: 600 }}>
        {t('report.criticalThinking.title')}
      </div>

      {/* Findings-first: lead with the weakest dimension to work on. */}
      {leadKey && (
        <div style={{ marginTop: '8px', marginBottom: '14px' }}>
          <div style={{ fontWeight: 500 }}>
            {t('report.criticalThinking.leadPrefix')}: {t(`report.criticalThinking.dimensions.${leadKey}.label`)}
          </div>
          <div style={{ color: 'var(--color-text-secondary, #64748b)' }}>
            {t(`report.criticalThinking.dimensions.${leadKey}.action`)}
            {ctc.caveat ? ` (${t('report.criticalThinking.tentative')})` : ''}
          </div>
        </div>
      )}

      {/* Per-dimension control bars (higher = stronger control). */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {ordered.map((k) => {
          const control = Math.round(dims[k].control);
          const isLead = k === leadKey;
          const color = controlColor(control);
          return (
            <div key={k} className="ai-likelihood-bar-row">
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', marginBottom: '2px' }}>
                <span style={{ fontWeight: isLead ? 600 : 400 }}>
                  {t(`report.criticalThinking.dimensions.${k}.label`)}
                </span>
                <span style={{ color, fontWeight: isLead ? 600 : 400 }}>{control}</span>
              </div>
              <div style={{ height: '10px', background: 'var(--color-background-secondary, #eef0f2)', borderRadius: '5px', overflow: 'hidden' }}>
                <div style={{ width: `${control}%`, height: '100%', background: color, borderRadius: '5px' }} />
              </div>
            </div>
          );
        })}
      </div>

      {/* allow-hardcode: inline CSS styles + i18n key lookups in JSX markup — UI layout, not a scoring/text-matching oracle. */}
      {/* Phase-2 qualitative review flags (LLM-judged; NOT part of the score). */}
      {llmRows.length > 0 && (
        <div style={{ marginTop: '14px' }}>
          <div style={{ fontSize: '13px', color: 'var(--color-text-secondary, #64748b)', marginBottom: '6px' }}>
            {t('report.criticalThinking.reviewFlagsTitle')}
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            {llmRows.map(({ key, score }) => (
              <div key={key} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px' }}>
                <span>{t(`report.criticalThinking.llmDimensions.${key}`)}</span>
                <span style={{ color: controlColor(score) }}>{score}/100</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Phase-2 sentence-level coaching highlights. */}
      {highlights.length > 0 && (
        <div style={{ marginTop: '14px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {highlights.map((h, i) => (
            <div key={`${h.paragraph_id}-${i}`} style={{ borderLeft: '3px solid #d97706', paddingLeft: '10px' }}>
              <div style={{ fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.04em', color: 'var(--color-text-secondary, #64748b)' }}>
                {t(`report.criticalThinking.highlightLabels.${h.label}`, { defaultValue: h.label })}
              </div>
              <div style={{ fontStyle: 'italic', margin: '2px 0' }}>“{h.sentence}”</div>
              {h.fix_instruction && (
                <div style={{ fontSize: '13px', color: 'var(--color-text-secondary, #64748b)' }}>{h.fix_instruction}</div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Secondary: overall score badge (the number never leads). */}
      <div style={{ marginTop: '14px', display: 'flex', alignItems: 'baseline', gap: '8px' }}>
        <span style={{ color: 'var(--color-text-secondary, #64748b)', fontSize: '13px' }}>
          {t('report.criticalThinking.scoreLabel')}:
        </span>
        <span style={{ color: bandColor, fontWeight: 600 }}>{score}/100</span>
        <span style={{ color: 'var(--color-text-secondary, #64748b)', fontSize: '13px' }}>
          — {t(`report.criticalThinking.bands.${ctc.band}`)}
        </span>
      </div>
    </section>
  );
}
