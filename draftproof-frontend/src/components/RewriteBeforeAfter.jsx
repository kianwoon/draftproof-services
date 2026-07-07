// Presentational before/after card for the rewrite marketing surfaces (the /rewrite
// page and the landing teaser). Pure display — takes strings, renders a two-row diff
// with an optional "suggested addition you replace" marker. No i18n inside: callers
// pass already-translated strings so this stays a dumb, reusable unit.
export function RewriteBeforeAfter({ before, after, marker, beforeLabel, afterLabel }) {
  return (
    <div className="rewrite-ba">
      <div className="rewrite-ba-row rewrite-ba-before">
        <span className="rewrite-ba-label">{beforeLabel}</span>
        {before}
      </div>
      <div className="rewrite-ba-row rewrite-ba-after">
        <span className="rewrite-ba-label">{afterLabel}</span>
        {after}
        {marker && <span className="rewrite-ba-marker">{marker}</span>}
      </div>
    </div>
  );
}
