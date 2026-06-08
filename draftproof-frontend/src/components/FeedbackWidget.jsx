// allow-hardcode: the string literals here are inline CSS values and UI copy
// (button labels, placeholders) — presentational craft, not a scoring/matching
// oracle. No logic branches on these strings.
import { useEffect, useRef, useState } from 'react';
import { submitFeedback } from '../api/draftproofApi';
import { useAuth } from '../context/AuthContext';

const TURNSTILE_SRC = 'https://challenges.cloudflare.com/turnstile/v0/api.js';
const SITE_KEY = import.meta.env.VITE_TURNSTILE_SITE_KEY || '';

// Load the Turnstile script once, shared across opens. Resolves when the global
// `turnstile` object is ready so we can explicitly render into our modal's div.
let turnstilePromise = null;
function loadTurnstile() {
  if (window.turnstile) return Promise.resolve(window.turnstile);
  if (turnstilePromise) return turnstilePromise;
  turnstilePromise = new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = TURNSTILE_SRC;
    script.async = true;
    script.defer = true;
    script.onload = () => resolve(window.turnstile);
    script.onerror = () => reject(new Error('Failed to load Turnstile'));
    document.head.appendChild(script);
  });
  return turnstilePromise;
}

const EMPTY = { type: 'bug', title: '', body: '', email: '' };

export default function FeedbackWidget() {
  const { user } = useAuth();
  const [open, setOpen] = useState(false);
  const [form, setForm] = useState(EMPTY);
  const [token, setToken] = useState('');
  const [status, setStatus] = useState('idle'); // idle | submitting | success | error
  const [error, setError] = useState('');
  const [issueUrl, setIssueUrl] = useState('');
  const widgetRef = useRef(null);
  const widgetIdRef = useRef(null);

  // Render the Turnstile widget when the modal opens; clean it up on close.
  useEffect(() => {
    if (!open || !SITE_KEY) return;
    let cancelled = false;
    loadTurnstile()
      .then((turnstile) => {
        if (cancelled || !widgetRef.current) return;
        widgetIdRef.current = turnstile.render(widgetRef.current, {
          sitekey: SITE_KEY,
          callback: (tok) => setToken(tok),
          'expired-callback': () => setToken(''),
          'error-callback': () => setToken(''),
        });
      })
      .catch(() => setError('Could not load anti-bot verification. Please retry.'));
    return () => {
      cancelled = true;
      if (window.turnstile && widgetIdRef.current != null) {
        try { window.turnstile.remove(widgetIdRef.current); } catch { /* already gone */ }
      }
      widgetIdRef.current = null;
      setToken('');
    };
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e) => { if (e.key === 'Escape') close(); };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [open]);

  function close() {
    setOpen(false);
    setForm(EMPTY);
    setStatus('idle');
    setError('');
    setIssueUrl('');
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setError('');
    if (!SITE_KEY) { setError('Feedback is not configured (missing Turnstile site key).'); return; }
    if (!token) { setError('Please complete the verification below.'); return; }
    setStatus('submitting');
    try {
      const { data } = await submitFeedback({
        type: form.type,
        title: form.title.trim(),
        body: form.body.trim(),
        email: user ? undefined : (form.email.trim() || undefined),
        page_url: window.location.href,
        turnstile_token: token,
      });
      setIssueUrl(data?.url || '');
      setStatus('success');
    } catch (err) {
      setStatus('error');
      setError(err?.response?.data?.detail || 'Something went wrong. Please try again.');
      if (window.turnstile && widgetIdRef.current != null) {
        try { window.turnstile.reset(widgetIdRef.current); } catch { /* noop */ }
      }
      setToken('');
    }
  }

  return (
    <>
      <button
        type="button"
        className="feedback-fab"
        aria-label="Send feedback"
        onClick={() => setOpen(true)}
        style={{
          position: 'fixed', right: '20px', bottom: '20px', zIndex: 1000,
          padding: '10px 16px', borderRadius: '999px', border: 'none',
          background: '#111827', color: '#fff', fontSize: '14px', fontWeight: 600,
          cursor: 'pointer', boxShadow: '0 4px 14px rgba(0,0,0,0.25)',
        }}
      >
        💬 Feedback
      </button>

      {open && (
        <div className="modal-backdrop" onClick={close}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()} style={{ maxWidth: '440px', width: '92%' }}>
            {status === 'success' ? (
              <>
                <h3 className="modal-title">Thank you! 🙏</h3>
                <p className="modal-message">
                  Your {form.type === 'bug' ? 'bug report' : 'suggestion'} has been received.
                  {issueUrl ? ' We’re tracking it now.' : ''}
                </p>
                <div className="modal-actions">
                  <button className="btn btn-small btn-primary" onClick={close}>Close</button>
                </div>
              </>
            ) : (
              <form onSubmit={handleSubmit}>
                <h3 className="modal-title">Send feedback</h3>

                <div style={{ display: 'flex', gap: '8px', margin: '12px 0' }}>
                  {['bug', 'feature'].map((kind) => (
                    <button
                      type="button"
                      key={kind}
                      onClick={() => setForm((f) => ({ ...f, type: kind }))}
                      className={`btn btn-small ${form.type === kind ? 'btn-primary' : 'btn-secondary'}`}
                      style={{ flex: 1 }}
                    >
                      {kind === 'bug' ? '🐞 Report a bug' : '✨ Suggest a feature'}
                    </button>
                  ))}
                </div>

                <input
                  className="feedback-input"
                  placeholder={form.type === 'bug' ? 'Brief summary of the bug' : 'Your idea in one line'}
                  value={form.title}
                  maxLength={120}
                  onChange={(e) => setForm((f) => ({ ...f, title: e.target.value }))}
                  style={fieldStyle}
                  required
                />
                <textarea
                  className="feedback-input"
                  placeholder={form.type === 'bug'
                    ? 'What happened? What did you expect? Steps to reproduce help a lot.'
                    : 'What would you like DraftProof to do, and why?'}
                  value={form.body}
                  maxLength={5000}
                  rows={5}
                  onChange={(e) => setForm((f) => ({ ...f, body: e.target.value }))}
                  style={{ ...fieldStyle, resize: 'vertical' }}
                  required
                />
                {!user && (
                  <input
                    className="feedback-input"
                    type="email"
                    placeholder="Email (optional, for follow-up)"
                    value={form.email}
                    maxLength={254}
                    onChange={(e) => setForm((f) => ({ ...f, email: e.target.value }))}
                    style={fieldStyle}
                  />
                )}

                <div ref={widgetRef} style={{ marginTop: '12px', minHeight: '65px' }} />

                {error && <p style={{ color: '#b91c1c', fontSize: '13px', margin: '8px 0 0' }}>{error}</p>}

                <div className="modal-actions">
                  <button type="button" className="btn btn-secondary btn-small" onClick={close}>Cancel</button>
                  <button
                    type="submit"
                    className="btn btn-small btn-primary"
                    disabled={status === 'submitting'}
                  >
                    {status === 'submitting' ? 'Sending…' : 'Send'}
                  </button>
                </div>
              </form>
            )}
          </div>
        </div>
      )}
    </>
  );
}

const fieldStyle = {
  width: '100%', boxSizing: 'border-box', marginTop: '8px', padding: '9px 11px',
  border: '1px solid #d1d5db', borderRadius: '8px', fontSize: '14px', fontFamily: 'inherit',
};
