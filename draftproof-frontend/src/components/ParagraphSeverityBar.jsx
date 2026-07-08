import { useTranslation } from 'react-i18next';
import { SEVERITY_CONFIG, debertaSeverityColor } from '../pages/report/reportHelpers';

// allow-hardcode: the string literals below are i18n translation KEYS (report.severityBar.*) and CSS
// class names, not a detect/scoring/matching word-list. Paragraph severity is computed by the
// content-agnostic buildParagraphSeverityBar() in report/reportHelpers.js; this only renders it.
//
// Per-paragraph severity heatmap. Each segment is one paragraph; width is proportional to the
// paragraph's length, hue is its worst finding tier, and opacity reflects finding-severity DENSITY
// (concentration). Clean paragraphs render as a neutral grey. Hover shows a tooltip only (no click).
// The `bar` prop comes from buildParagraphSeverityBar() in report/reportHelpers.js.
export default function ParagraphSeverityBar({ bar, selectedId = null, onSelect = null }) {
  const { t } = useTranslation();
  if (!Array.isArray(bar) || bar.length === 0) return null;
  const interactive = typeof onSelect === 'function';

  return (
    <div className="paragraph-severity">
      <span className="paragraph-severity-caption">{t('report.severityBar.caption')}</span>
      <div
        className="paragraph-severity-bar"
        role={interactive ? 'group' : 'img'}
        aria-label={t('report.severityBar.ariaLabel', { count: bar.length })}
      >
        {bar.map((segment) => {
          const clean = segment.findingCount === 0;
          // A verdict-gated 'review' candidate renders the SAME muted amber as the full-document
          // underline (.is-severity-review), so the bar agrees with "Read full document". It must
          // win BEFORE the tier fallback, which would otherwise paint it 'low'-tier green.
          const REVIEW_COLOR = '#c99a3b';
          // Prefer DeBERTa severity color (matches the full-document heatmap scale) when a
          // DeBERTa score is present; fall back to the tier color for non-DeBERTa reports.
          const debertaColor = debertaSeverityColor(segment.maxDebertaScore || 0);
          const tierColor = segment.reviewBand
            ? REVIEW_COLOR
            : (debertaColor || SEVERITY_CONFIG[segment.topTier]?.color || '#94a3b8');
          // Density -> opacity: faint at low concentration, solid at the doc's densest paragraph.
          const opacity = clean ? 1 : 0.35 + 0.65 * Math.min(1, Math.max(0, segment.intensity));
          const tierLabel = segment.topTier
            ? t(`report.severityBar.tier.${segment.topTier}`, { defaultValue: segment.topTier })
            : '';
          const title = clean
            ? t('report.severityBar.tooltipClean', { index: segment.index })
            : segment.reviewBand
            ? t('report.severityBar.tooltipReview', {
                index: segment.index,
                defaultValue: 'Sentence {{index}} — flagged for review (reads as human overall)',
              })
            : t('report.severityBar.tooltip', {
                index: segment.index,
                count: segment.findingCount,
                tier: tierLabel,
              });
          const className = `paragraph-severity-seg${(selectedId === segment.id || selectedId === segment.paragraphId) ? ' is-selected' : ''}`;
          const style = {
            width: `${segment.widthPct}%`,
            backgroundColor: clean ? '#22c55e' : tierColor,
            opacity,
          };
          if (interactive) {
            return (
              <button
                key={segment.id || segment.index}
                type="button"
                className={className}
                style={style}
                title={title}
                aria-label={title}
                onClick={() => onSelect(segment.paragraphId || segment.id)}
              />
            );
          }
          return (
            <span
              key={segment.id || segment.index}
              className={className}
              style={style}
              title={title}
            />
          );
        })}
      </div>
      <div className="paragraph-severity-meta">
        <span>{t('report.severityBar.legendLow')}</span>
        <span className="paragraph-severity-gradient" aria-hidden="true" />
        <span>{t('report.severityBar.legendHigh')}</span>
      </div>
    </div>
  );
}
