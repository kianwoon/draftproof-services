// Word/char-level tracked-changes diff used by the submitted-draft editor (Report.jsx)
// and the in-place rewrite editor (RewriteDraftEditor.jsx). Pure functions, no state.

function tokenizeTrackedText(text) {
  return String(text || '').match(/\s+|[A-Za-z0-9]+|[^\sA-Za-z0-9]/g) || [];
}

function compactDiffParts(parts) {
  const compacted = [];
  parts.forEach((part) => {
    if (!part?.text) return;
    const previous = compacted[compacted.length - 1];
    if (previous?.type === part.type) {
      previous.text += part.text;
    } else {
      compacted.push({ ...part });
    }
  });
  return compacted;
}

function lcsTokenDiff(originalTokens, currentTokens) {
  const originalLength = originalTokens.length;
  const currentLength = currentTokens.length;
  const matrix = Array.from({ length: originalLength + 1 }, () => Array(currentLength + 1).fill(0));

  for (let i = originalLength - 1; i >= 0; i -= 1) {
    for (let j = currentLength - 1; j >= 0; j -= 1) {
      matrix[i][j] = originalTokens[i] === currentTokens[j]
        ? matrix[i + 1][j + 1] + 1
        : Math.max(matrix[i + 1][j], matrix[i][j + 1]);
    }
  }

  const parts = [];
  let i = 0;
  let j = 0;
  while (i < originalLength && j < currentLength) {
    if (originalTokens[i] === currentTokens[j]) {
      parts.push({ type: 'equal', text: originalTokens[i] });
      i += 1;
      j += 1;
    } else if (matrix[i + 1][j] >= matrix[i][j + 1]) {
      parts.push({ type: 'delete', text: originalTokens[i] });
      i += 1;
    } else {
      parts.push({ type: 'insert', text: currentTokens[j] });
      j += 1;
    }
  }

  while (i < originalLength) {
    parts.push({ type: 'delete', text: originalTokens[i] });
    i += 1;
  }
  while (j < currentLength) {
    parts.push({ type: 'insert', text: currentTokens[j] });
    j += 1;
  }

  return compactDiffParts(parts);
}

function charDiff(originalText, currentText) {
  return lcsTokenDiff(Array.from(originalText || ''), Array.from(currentText || ''));
}

function refineReplacementParts(parts) {
  const refined = [];
  for (let i = 0; i < parts.length; i += 1) {
    const current = parts[i];
    const next = parts[i + 1];
    const currentIsWord = current?.type === 'delete' && /^[A-Za-z0-9]+$/.test(current.text || '');
    const nextIsWord = next?.type === 'insert' && /^[A-Za-z0-9]+$/.test(next.text || '');
    if (currentIsWord && nextIsWord) {
      refined.push(...charDiff(current.text, next.text));
      i += 1;
    } else {
      refined.push(current);
    }
  }
  return compactDiffParts(refined);
}

export function buildTrackedDiff(originalText, currentText) {
  if (originalText === currentText) {
    return [{ type: 'equal', text: currentText }];
  }

  const originalTokens = tokenizeTrackedText(originalText);
  const currentTokens = tokenizeTrackedText(currentText);

  return refineReplacementParts(lcsTokenDiff(originalTokens, currentTokens));
}

function escapeTrackedHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

export function trackedDiffToPlainText(parts) {
  return parts.map((part) => {
    if (part.type === 'insert') return `{+${part.text}+}`;
    if (part.type === 'delete') return `[-${part.text}-]`;
    return part.text;
  }).join('');
}

export function trackedDiffToHtml(parts) {
  const body = parts.map((part) => {
    const text = escapeTrackedHtml(part.text);
    if (part.type === 'insert') return `<ins>${text}</ins>`;
    if (part.type === 'delete') return `<del>${text}</del>`;
    return text;
  }).join('');
  return `<div>${body}</div>`;
}
