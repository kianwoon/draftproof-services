// allow-hardcode: this is a React view (JSX markup + i18n lookups + class names).
// All user-facing text comes from t('apiKeys.*'); there is no scoring/matching
// list here, only presentation.
import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import CodeTexture from '../components/CodeTexture';
import { listApiKeys, createApiKey, revokeApiKey } from '../api/draftproofApi';

// Render an install step, showing any `backtick`-wrapped path in monospace.
function withCodePaths(text) {
  return text.split('`').map((seg, i) => (i % 2 === 1 ? <code key={i}>{seg}</code> : seg));
}

function formatDate(iso, locale) {
  if (!iso) return null;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return null;
  return d.toLocaleDateString(locale, { day: 'numeric', month: 'short', year: 'numeric' });
}

export default function ApiKeys() {
  const { t, i18n } = useTranslation();
  const locale = i18n.resolvedLanguage?.startsWith('zh') ? 'zh-CN' : 'en-SG';

  const [keys, setKeys] = useState([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState(false);
  const [name, setName] = useState('');
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState(false);
  const [newKey, setNewKey] = useState(null); // {key, key_prefix} — shown once
  const [copied, setCopied] = useState(false);
  const [revokingId, setRevokingId] = useState(null);
  const [showInstall, setShowInstall] = useState(true);
  const [installOS, setInstallOS] = useState('word');
  // Full keys created THIS session, kept in memory only (never persisted) so a
  // freshly-created key can still be copied from the table after the modal closes.
  const [sessionKeys, setSessionKeys] = useState({});
  const [copiedRowId, setCopiedRowId] = useState(null);

  const load = async () => {
    setLoading(true);
    setLoadError(false);
    try {
      const { data } = await listApiKeys();
      setKeys(Array.isArray(data) ? data : []);
    } catch {
      setLoadError(true);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const handleCreate = async (e) => {
    e.preventDefault();
    if (creating) return;
    setCreating(true);
    setCreateError(false);
    try {
      const { data } = await createApiKey(name.trim() || undefined);
      setNewKey(data);
      setSessionKeys((prev) => ({ ...prev, [data.id]: data.key }));
      setCopied(false);
      setName('');
      await load();
    } catch {
      setCreateError(true);
    } finally {
      setCreating(false);
    }
  };

  const handleCopy = async () => {
    if (!newKey?.key) return;
    try {
      await navigator.clipboard.writeText(newKey.key);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  };

  const handleCopyRow = async (id) => {
    const key = sessionKeys[id];
    if (!key) return;
    try {
      await navigator.clipboard.writeText(key);
      setCopiedRowId(id);
      setTimeout(() => setCopiedRowId(null), 1500);
    } catch {
      // clipboard blocked — ignore
    }
  };

  const handleRevoke = async (id) => {
    // eslint-disable-next-line no-alert
    if (!window.confirm(t('apiKeys.revokeConfirm'))) return;
    setRevokingId(id);
    try {
      await revokeApiKey(id);
      await load();
    } catch {
      // eslint-disable-next-line no-alert
      window.alert(t('apiKeys.revokeError'));
    } finally {
      setRevokingId(null);
    }
  };

  const activeCount = keys.filter((k) => !k.revoked_at).length;

  return (
    <main className="app-page api-keys-page">
      <div className="container">
        <section className="app-hero app-hero-dark api-keys-hero">
          <CodeTexture id="apiKeysHero" />
          <div>
            <p className="eyebrow">{t('apiKeys.eyebrow')}</p>
            <h1>{t('apiKeys.title')}</h1>
            <p>{t('apiKeys.subtitle')}</p>
          </div>
          <div className="api-keys-hero-actions">
            <div className="app-hero-stat">
              <span>{t('apiKeys.activeLabel')}</span>
              <strong>{activeCount}</strong>
            </div>
          </div>
        </section>

        <div className="api-keys-addon">
          <button
            type="button"
            className="api-keys-addon-head"
            aria-expanded={showInstall}
            onClick={() => setShowInstall((v) => !v)}
          >
            <span className="api-keys-addon-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                <path d="M7 3.8h7.2L18 7.6v12.6H7V3.8Z" />
                <path d="M14 3.8v4h4" />
              </svg>
            </span>
            <span className="api-keys-addon-text">
              <strong>{t('apiKeys.addon.title')}</strong>
              <span>{t('apiKeys.addon.body')}</span>
            </span>
            <span className="api-keys-addon-cta">
              {showInstall ? t('apiKeys.addon.hide') : t('apiKeys.addon.cta')}
            </span>
          </button>
          {showInstall && (
            <div className="api-keys-addon-steps">
              <div className="api-keys-addon-tabs" role="tablist">
                <button
                  type="button"
                  className={`api-keys-addon-tab${installOS === 'word' ? ' is-active' : ''}`}
                  onClick={() => setInstallOS('word')}
                >
                  {t('apiKeys.addon.wordTab')}
                </button>
                <button
                  type="button"
                  className={`api-keys-addon-tab${installOS === 'gdocs' ? ' is-active' : ''}`}
                  onClick={() => setInstallOS('gdocs')}
                >
                  {t('apiKeys.addon.gdocsTab')}
                </button>
              </div>
              {installOS === 'word' ? (
                <>
                  <a className="btn btn-primary btn-small" href="/word-addin/manifest.xml" download="draftproof-manifest.xml">
                    {t('apiKeys.addon.download')}
                  </a>
                  <ol>
                    {t('apiKeys.addon.wordSteps', { returnObjects: true }).map((s, i) => (
                      <li key={i}>{withCodePaths(s)}</li>
                    ))}
                  </ol>
                </>
              ) : (
                <>
                  <ol>
                    {t('apiKeys.addon.gdocsSteps', { returnObjects: true }).map((s, i) => (
                      <li key={i}>{s}</li>
                    ))}
                  </ol>
                  <p className="api-keys-muted">{t('apiKeys.addon.gdocsNote')}</p>
                </>
              )}
            </div>
          )}
        </div>

        <section className="api-keys-panel">
          <form onSubmit={handleCreate} className="api-keys-create">
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder={t('apiKeys.createNamePlaceholder')}
              maxLength={80}
              aria-label={t('apiKeys.createNamePlaceholder')}
            />
            <button type="submit" className="btn btn-primary" disabled={creating}>
              {creating ? t('apiKeys.creating') : t('apiKeys.createButton')}
            </button>
          </form>
          <p className="api-keys-muted api-keys-copy-hint">{t('apiKeys.copyHint')}</p>
          {createError && (
            <p role="alert" className="api-keys-error">{t('apiKeys.createError')}</p>
          )}

          {loading ? (
            <p className="api-keys-muted">{t('apiKeys.loading')}</p>
          ) : loadError ? (
            <p role="alert" className="api-keys-error">{t('apiKeys.loadError')}</p>
          ) : keys.length === 0 ? (
            <p className="api-keys-muted">{t('apiKeys.empty')}</p>
          ) : (
            <div className="api-keys-table-wrap">
              <table className="api-keys-table">
                <thead>
                  <tr>
                    <th>{t('apiKeys.th.name')}</th>
                    <th>{t('apiKeys.th.key')}</th>
                    <th>{t('apiKeys.th.created')}</th>
                    <th>{t('apiKeys.th.lastUsed')}</th>
                    <th aria-hidden="true" />
                  </tr>
                </thead>
                <tbody>
                  {keys.map((k) => (
                    <tr key={k.id} className={k.revoked_at ? 'is-revoked' : ''}>
                      <td>{k.name}</td>
                      <td className="api-keys-mono">{k.key_prefix}…</td>
                      <td>{formatDate(k.created_at, locale) || '—'}</td>
                      <td>{formatDate(k.last_used_at, locale) || t('apiKeys.neverUsed')}</td>
                      <td className="api-keys-action-cell">
                        {!k.revoked_at && sessionKeys[k.id] && (
                          <button
                            type="button"
                            className="btn btn-secondary btn-small"
                            onClick={() => handleCopyRow(k.id)}
                          >
                            {copiedRowId === k.id ? t('apiKeys.copied') : t('apiKeys.copy')}
                          </button>
                        )}
                        {k.revoked_at ? (
                          <span className="api-keys-badge">{t('apiKeys.revokedBadge')}</span>
                        ) : (
                          <button
                            type="button"
                            className="btn btn-secondary btn-small"
                            onClick={() => handleRevoke(k.id)}
                            disabled={revokingId === k.id}
                          >
                            {revokingId === k.id ? t('apiKeys.revoking') : t('apiKeys.revoke')}
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>

      {newKey && (
        <div
          className="api-keys-modal-overlay"
          role="dialog"
          aria-modal="true"
          aria-label={t('apiKeys.modal.title')}
        >
          <div className="api-keys-modal">
            <h2>{t('apiKeys.modal.title')}</h2>
            <p className="api-keys-modal-warn">{t('apiKeys.modal.warning')}</p>
            <div className="api-keys-secret-row">
              <code className="api-keys-secret">{newKey.key}</code>
              <button
                type="button"
                className={`api-keys-copy-btn${copied ? ' is-copied' : ''}`}
                onClick={handleCopy}
                aria-label={copied ? t('apiKeys.modal.copied') : t('apiKeys.modal.copy')}
                title={copied ? t('apiKeys.modal.copied') : t('apiKeys.modal.copy')}
              >
                {copied ? (
                  <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                    <path d="m5 13 4 4 10-11" />
                  </svg>
                ) : (
                  <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                    <rect x="9" y="9" width="11" height="11" rx="2" />
                    <path d="M5 15V5a2 2 0 0 1 2-2h10" />
                  </svg>
                )}
              </button>
            </div>
            <p className="api-keys-copy-feedback" aria-live="polite">
              {copied ? t('apiKeys.modal.copied') : ''}
            </p>
            <p className="api-keys-muted">{t('apiKeys.modal.usageHint')}</p>
            <div className="api-keys-modal-actions">
              <button type="button" className="btn btn-primary" onClick={() => setNewKey(null)}>
                {t('apiKeys.modal.done')}
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
