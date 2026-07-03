/**
 * Pure text/range/score helpers for the Report page.
 *
 * Extracted from Report.jsx. Every function here is pure — no React, no
 * closure over component state. They take explicit args and return values.
 */
import { clampPercent } from './reportHelpers';

export const RESCAN_POLL_INTERVAL = 3000;
export const RESCAN_MAX_POLLS = 200;
export const SUBMITTED_EDITOR_TRANSITION_MS = 480;
export const REWRITE_REPORT_RETRY_LIMIT = 8;
export const REWRITE_REPORT_RETRY_DELAY_MS = 1500;

export function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function addScoreProfileFeature(features, key, value) {
  if (features[key] != null) return;
  const percent = clampPercent(value);
  if (percent != null) {
    features[key] = percent;
  }
}

// Stable grounding-quality headline: mean of the grounding signals the rewrite actually MOVES
// (lived-detail, broad-claim, unsupported-claim). Deliberately EXCLUDES the binary citation signal
// (source_grounding_risk) and the flat citation_weakness — those swing/don't move and live in the
// separate "Citation grounding" row. Returns null if no component is available.
export function groundingQualityComposite(writingComponents) {
  const wc = writingComponents || {};
  const parts = ['lived_detail_risk', 'broad_claim_risk', 'unsupported_claim_risk']
    .map((k) => clampPercent(wc[k]))
    .filter((v) => v != null);
  if (!parts.length) return null;
  return parts.reduce((sum, v) => sum + v, 0) / parts.length;
}

export function submittedContentToText(model) {
  return (model?.paragraphs || [])
    .map((paragraph) => paragraph.text || paragraph.segments.map((segment) => segment.text).join(' ').trim())
    .filter(Boolean)
    .join('\n\n');
}

export function findTextRange(haystack, needle) {
  if (!haystack || !needle) return null;
  const start = haystack.indexOf(needle);
  if (start < 0) return null;
  return { start, end: start + needle.length };
}

export function changedTextRange(previousText, nextText) {
  let start = 0;
  const previousLength = previousText.length;
  const nextLength = nextText.length;

  while (
    start < previousLength &&
    start < nextLength &&
    previousText[start] === nextText[start]
  ) {
    start += 1;
  }

  let previousEnd = previousLength;
  let nextEnd = nextLength;
  while (
    previousEnd > start &&
    nextEnd > start &&
    previousText[previousEnd - 1] === nextText[nextEnd - 1]
  ) {
    previousEnd -= 1;
    nextEnd -= 1;
  }

  return { start, previousEnd, nextEnd, delta: nextEnd - previousEnd };
}

export function adjustHighlightedRange(range, previousText, nextText) {
  if (!range) return null;
  const change = changedTextRange(previousText, nextText);
  const nextLength = nextText.length;

  if (change.previousEnd <= range.start) {
    return {
      ...range,
      start: Math.max(0, Math.min(nextLength, range.start + change.delta)),
      end: Math.max(0, Math.min(nextLength, range.end + change.delta)),
    };
  }

  if (change.start >= range.end) {
    return {
      ...range,
      start: Math.max(0, Math.min(nextLength, range.start)),
      end: Math.max(0, Math.min(nextLength, range.end)),
    };
  }

  const start = Math.max(0, Math.min(nextLength, Math.min(range.start, change.start)));
  const end = Math.max(
    start,
    Math.min(nextLength, Math.max(range.end + change.delta, change.nextEnd))
  );

  return { ...range, start, end };
}

export function adjustHighlightedRanges(ranges, previousText, nextText) {
  return Object.fromEntries(
    Object.entries(ranges || {})
      .map(([segmentId, range]) => [segmentId, adjustHighlightedRange(range, previousText, nextText)])
      .filter(([, range]) => range && range.end > range.start)
  );
}

export function buildOriginalSegmentRanges(originalText, segments) {
  const ranges = {};
  let cursor = 0;
  (segments || []).forEach((segment) => {
    if (!segment?.id || !segment.text) return;
    let start = originalText.indexOf(segment.text, cursor);
    if (start < 0) start = originalText.indexOf(segment.text);
    if (start < 0) return;
    const end = start + segment.text.length;
    ranges[segment.id] = { start, end, segmentId: segment.id };
    cursor = end;
  });
  return ranges;
}

export function highlightedEditorParts(text, range) {
  if (!range || range.end <= range.start) return [{ type: 'plain', text }];
  const start = Math.max(0, Math.min(text.length, range.start));
  const end = Math.max(start, Math.min(text.length, range.end));

  return [
    { type: 'plain', text: text.slice(0, start) },
    { type: 'selected', text: text.slice(start, end) },
    { type: 'plain', text: text.slice(end) },
  ].filter((part) => part.text);
}
