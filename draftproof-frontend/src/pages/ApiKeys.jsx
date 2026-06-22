// allow-hardcode: this is a React view (JSX markup + inline CSS-var style
// fallbacks + i18n lookups). All user-facing text comes from t('apiKeys.*');
// there is no scoring/matching list here, only presentation.
import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { listApiKeys, createApiKey, revokeApiKey } from '../api/draftproofApi';

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

  const cellStyle = { padding: '0.55rem 0.5rem' };
  const mutedColor = 'var(--text-muted, #5b6472)';
  const borderColor = 'var(--border, #dde1e8)';

  return (
    <main
      className="container"
      style={{ paddingTop: 'calc(var(--header-h) + 3rem)', paddingBottom: '4rem', maxWidth: 880 }}
    >
      <h1 style={{ marginBottom: '0.5rem' }}>{t('apiKeys.title')}</h1>
      <p style={{ color: mutedColor, maxWidth: 660 }}>{t('apiKeys.subtitle')}</p>

      <form
        onSubmit={handleCreate}
        style={{ display: 'flex', gap: '0.5rem', margin: '1.5rem 0 0.5rem', flexWrap: 'wrap' }}
      >
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder={t('apiKeys.createNamePlaceholder')}
          maxLength={80}
          style={{
            flex: '1 1 260px', padding: '0.6rem 0.8rem', borderRadius: 8,
            border: `1px solid ${borderColor}`, font: 'inherit',
          }}
        />
        <button type="submit" className="btn" disabled={creating}>
          {creating ? t('apiKeys.creating') : t('apiKeys.createButton')}
        </button>
      </form>
      {createError && (
        <p role="alert" style={{ color: 'crimson' }}>{t('apiKeys.createError')}</p>
      )}

      <div style={{ marginTop: '1.5rem' }}>
        {loading ? (
          <p>{t('apiKeys.loading')}</p>
        ) : loadError ? (
          <p role="alert" style={{ color: 'crimson' }}>{t('apiKeys.loadError')}</p>
        ) : keys.length === 0 ? (
          <p style={{ color: mutedColor }}>{t('apiKeys.empty')}</p>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ textAlign: 'left', borderBottom: `1px solid ${borderColor}` }}>
                <th style={cellStyle}>{t('apiKeys.th.name')}</th>
                <th style={cellStyle}>{t('apiKeys.th.key')}</th>
                <th style={cellStyle}>{t('apiKeys.th.created')}</th>
                <th style={cellStyle}>{t('apiKeys.th.lastUsed')}</th>
                <th style={cellStyle} aria-hidden="true" />
              </tr>
            </thead>
            <tbody>
              {keys.map((k) => (
                <tr
                  key={k.id}
                  style={{ borderBottom: `1px solid ${borderColor}`, opacity: k.revoked_at ? 0.55 : 1 }}
                >
                  <td style={cellStyle}>{k.name}</td>
                  <td style={{ ...cellStyle, fontFamily: 'monospace' }}>{k.key_prefix}…</td>
                  <td style={cellStyle}>{formatDate(k.created_at, locale) || '—'}</td>
                  <td style={cellStyle}>{formatDate(k.last_used_at, locale) || t('apiKeys.neverUsed')}</td>
                  <td style={{ ...cellStyle, textAlign: 'right' }}>
                    {k.revoked_at ? (
                      <span style={{ color: mutedColor }}>{t('apiKeys.revokedBadge')}</span>
                    ) : (
                      <button
                        type="button"
                        className="btn btn-ghost btn-small"
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
        )}
      </div>

      {newKey && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label={t('apiKeys.modal.title')}
          style={{
            position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)', display: 'flex',
            alignItems: 'center', justifyContent: 'center', zIndex: 1000, padding: '1rem',
          }}
        >
          <div
            style={{
              background: 'var(--surface, #fff)', color: 'var(--text, #1a1f29)',
              borderRadius: 12, padding: '1.5rem', maxWidth: 540, width: '100%',
              boxShadow: '0 12px 40px rgba(0,0,0,0.25)',
            }}
          >
            <h2 style={{ marginTop: 0 }}>{t('apiKeys.modal.title')}</h2>
            <p style={{ color: 'crimson', fontWeight: 600 }}>{t('apiKeys.modal.warning')}</p>
            <code
              style={{
                display: 'block', wordBreak: 'break-all', background: 'var(--code-bg, #f4f5f8)',
                padding: '0.75rem', borderRadius: 8, margin: '0.75rem 0', fontSize: '0.9rem',
              }}
            >
              {newKey.key}
            </code>
            <p style={{ fontSize: '0.85rem', color: mutedColor }}>
              {t('apiKeys.modal.usageHint')}
            </p>
            <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'flex-end', marginTop: '1rem' }}>
              <button type="button" className="btn btn-ghost" onClick={handleCopy}>
                {copied ? t('apiKeys.modal.copied') : t('apiKeys.modal.copy')}
              </button>
              <button type="button" className="btn" onClick={() => setNewKey(null)}>
                {t('apiKeys.modal.done')}
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
