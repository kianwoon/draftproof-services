// allow-hardcode: presentational React component — CSS class names + JSX, not a scoring/matching oracle. All human-facing text + the declaration template come from i18n.
import { useState } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * Interactive AI-use declaration builder for the /ai-declaration page.
 * Pure client-side: assembles a declaration sentence from an i18n template
 * with {{tool}} / {{purpose}} / {{dateClause}} placeholders, then offers
 * copy-to-clipboard. Output is intentionally not part of the prerendered
 * (crawlable) content — the static templates below carry the SEO weight.
 */
export default function DeclarationGenerator() {
  const { t } = useTranslation();
  const g = (key) => t(`aiDeclaration.generator.${key}`);

  const [tool, setTool] = useState('');
  const [purpose, setPurpose] = useState('');
  const [date, setDate] = useState('');
  const [copied, setCopied] = useState(false);

  const dateClause = date.trim()
    ? g('dateClause').replace('{{date}}', date.trim())
    : '';
  const output = g('template')
    .replace('{{tool}}', tool.trim() || g('emptyTool'))
    .replace('{{purpose}}', purpose.trim() || g('emptyPurpose'))
    .replace('{{dateClause}}', dateClause);

  const handleCopy = async () => {
    let ok = false;
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(output);
        ok = true;
      }
    } catch {
      ok = false;
    }
    if (!ok) {
      // Fallback for contexts without the async clipboard API.
      const ta = document.createElement('textarea');
      ta.value = output;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      try { ok = document.execCommand('copy'); } catch { ok = false; }
      document.body.removeChild(ta);
    }
    if (ok) {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <section className="content-checker-section declaration-generator">
      <p className="eyebrow">{g('eyebrow')}</p>
      <h2>{g('title')}</h2>
      <p className="seo-section-lead">{g('subtitle')}</p>

      <div className="declaration-generator-grid">
        <div className="declaration-generator-fields">
          <label>
            <span>{g('toolLabel')}</span>
            <input
              type="text"
              value={tool}
              onChange={(e) => setTool(e.target.value)}
              placeholder={g('toolPlaceholder')}
            />
          </label>
          <label>
            <span>{g('purposeLabel')}</span>
            <input
              type="text"
              value={purpose}
              onChange={(e) => setPurpose(e.target.value)}
              placeholder={g('purposePlaceholder')}
            />
          </label>
          <label>
            <span>{g('dateLabel')}</span>
            <input
              type="text"
              value={date}
              onChange={(e) => setDate(e.target.value)}
              placeholder={g('datePlaceholder')}
            />
          </label>
        </div>

        <div className="declaration-generator-output">
          <span className="declaration-generator-output-label">{g('outputLabel')}</span>
          <pre className="seo-template-text">{output}</pre>
          <button type="button" className="btn btn-primary" onClick={handleCopy}>
            {copied ? g('copied') : g('copy')}
          </button>
        </div>
      </div>
    </section>
  );
}
