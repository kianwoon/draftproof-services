import { useState, useRef, useCallback, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { startScanWithText, getScanStatus, buildApiEventUrl } from '../api/draftproofApi';
import { useAuth } from '../context/AuthContext';
import ConfirmDialog from '../components/ConfirmDialog';
import CodeTexture from '../components/CodeTexture';
import { countWords, scanTokensRequired } from '../utils/scanBilling';

const POLL_INTERVAL = 3000;
const MAX_POLLS = 200; // 200 × 3s = 10 min max
const START_SCAN_TIMEOUT_MS = 20000;

export default function Scan() {
  const { t } = useTranslation();
  const [text, setText] = useState('');
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState(null);
  const [showProgress, setShowProgress] = useState(false);
  const [progressPercent, setProgressPercent] = useState(0);
  const [progressMessage, setProgressMessage] = useState(null);
  const [error, setError] = useState(null);
  const [serverError, setServerError] = useState(null);
  const [insufficientTokens, setInsufficientTokens] = useState(false);
  const [authExpired, setAuthExpired] = useState(false);
  const navigate = useNavigate();
  const { refreshBalance, balance, logout } = useAuth();
  const abortRef = useRef(null);
  const eventSourceRef = useRef(null);
  const wordCount = countWords(text);
  const tokensRequired = scanTokensRequired(wordCount);

  // Cancel in-flight polling on unmount
  useEffect(() => {
    const savedDraft = sessionStorage.getItem('draftproof_scan_draft');
    if (savedDraft) {
      setText(savedDraft);
      sessionStorage.removeItem('draftproof_scan_draft');
    }

    return () => {
      if (abortRef.current) {
        abortRef.current.abort();
      }
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  const redirectToSignIn = useCallback(async () => {
    if (text.trim()) {
      sessionStorage.setItem('draftproof_scan_draft', text);
    }
    sessionStorage.setItem('auth_next', '/scan');
    await logout?.();
    navigate('/signin?error=Session expired. Please sign in again.', { replace: true });
  }, [logout, navigate, text]);

  const handleAuthExpired = useCallback(() => {
    setBusy(false);
    setShowProgress(false);
    setProgressPercent(0);
    setProgressMessage(null);
    setStatus(null);
    setError(null);
    setServerError(null);
    setAuthExpired(true);
  }, [t]);

  const updateProgress = useCallback((data) => {
    const percent = Math.max(0, Math.min(100, Number(data.progress_percent) || 0));
    const message = data.progress_message || formatStatus(data.status, t);
    setProgressPercent(percent);
    setProgressMessage(message);
    setStatus(message);
  }, [t]);

  const pollUntilDone = useCallback(async (scanId, signal) => {
    for (let i = 0; i < MAX_POLLS; i++) {
      if (signal.aborted) return false;
      await sleep(POLL_INTERVAL);
      if (signal.aborted) return false;
      const { data } = await getScanStatus(scanId, { signal });
      updateProgress(data);
      if (data.status === 'completed') return true;
      if (data.status === 'failed') {
        setProgressMessage(t('scan.failed'));
        setServerError(t('scan.serverFailed'));
        return false;
      }
      if (data.status === 'retrying') {
        setStatus(t('scan.retrying'));
      }
    }
    setServerError(t('scan.timedOut'));
    return false;
  }, [t, updateProgress]);

  const waitForScanEvents = useCallback((scanId, signal) => {
    if (!window.EventSource) return Promise.resolve(null);

    return new Promise((resolve) => {
      const source = new EventSource(buildScanEventsUrl(scanId), { withCredentials: true });
      eventSourceRef.current = source;
      let settled = false;

      const finish = (result) => {
        if (settled) return;
        settled = true;
        signal.removeEventListener('abort', onAbort);
        source.close();
        if (eventSourceRef.current === source) eventSourceRef.current = null;
        resolve(result);
      };

      const onAbort = () => finish(false);
      signal.addEventListener('abort', onAbort, { once: true });

      source.addEventListener('progress', (event) => {
        let data;
        try {
          data = JSON.parse(event.data);
        } catch {
          finish(null);
          return;
        }
        updateProgress(data);
        if (data.status === 'completed') finish(true);
        if (data.status === 'failed') {
          setProgressMessage(t('scan.failed'));
          setServerError(t('scan.serverFailed'));
          finish(false);
        }
      });

      source.addEventListener('scan-error', () => {
        setProgressMessage(t('scan.failed'));
        setServerError(t('scan.serverFailed'));
        finish(false);
      });

      source.addEventListener('error', () => {
        finish(null);
      });
    });
  }, [t, updateProgress]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!text.trim()) {
      setError(t('scan.enterText'));
      setShowProgress(false);
      return;
    }
    if (balance !== null && balance < tokensRequired) {
      setInsufficientTokens(true);
      setShowProgress(false);
      return;
    }
    setBusy(true);
    setShowProgress(true);
    setError(null);
    setServerError(null);
    setProgressPercent(0);
    setProgressMessage(null);
    setStatus(t('scan.submitting'));

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      let scan;
      setStatus(t('scan.queueing'));
      setProgressPercent(3);
      setProgressMessage(t('scan.queueingProgress'));
      ({ data: scan } = await startScanWithText(text, {
        signal: controller.signal,
        timeout: START_SCAN_TIMEOUT_MS,
      }));

      setStatus(t('scan.scanning'));
      setProgressPercent(scan.progress_percent || 5);
      setProgressMessage(scan.progress_message || t('scan.queued'));
      let completed = await waitForScanEvents(scan.id, controller.signal);
      if (completed === null && !controller.signal.aborted) {
        completed = await pollUntilDone(scan.id, controller.signal);
      }
      if (completed) {
        refreshBalance();
        navigate(`/report/${scan.id}`);
      }
    } catch (err) {
      if (err.name === 'AbortError' || err.code === 'ERR_CANCELED') return;
      const isStartTimeout = err.code === 'ECONNABORTED' || (
        err.message && err.message.toLowerCase().includes('timeout')
      );
      if (isStartTimeout) {
        setShowProgress(false);
        setProgressMessage(null);
        setServerError(t('scan.timeoutMessage'));
        return;
      }
      const msg = err.response?.data?.detail || t('scan.failed');
      const httpStatus = err.response?.status;
      const isAuthExpired = httpStatus === 401 || (
        httpStatus === 403 &&
        String(msg).toLowerCase().includes('not authenticated')
      );
      const isInsufficient = httpStatus === 400 && (
        msg.toLowerCase().includes('insufficient') ||
        msg.toLowerCase().includes('no credit account') ||
        msg.toLowerCase().includes('purchase')
      );
      if (isAuthExpired) {
        handleAuthExpired();
      } else if (isInsufficient) {
        setShowProgress(false);
        setInsufficientTokens(true);
      } else if (httpStatus >= 400) {
        setProgressMessage(t('scan.failed'));
        setServerError(msg);
      } else {
        setProgressMessage(t('scan.failed'));
        setError(msg);
      }
    } finally {
      setBusy(false);
      setStatus(null);
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      abortRef.current = null;
    }
  };

  return (
    <main className="app-page scan-shell">
      <div className="container">
        <section className="app-hero app-hero-dark">
          <CodeTexture id="scanHero" />
          <div>
            <p className="eyebrow">{t('scan.eyebrow')}</p>
            <h1>{t('scan.title')}</h1>
            <p>{t('scan.body')}</p>
          </div>
          <div className="app-hero-stat">
            <span>{t('scan.balance')}</span>
            <strong>{balance === null ? t('common.checking') : t('common.token', { count: balance })}</strong>
            <small>{t('scan.freeThrough')}</small>
          </div>
        </section>

        <section className="scan-workspace">
          <form onSubmit={handleSubmit} className="scan-form">
            <label className="scan-label" htmlFor="scan-text">
              {t('scan.documentText')}
              <span>{t('scan.documentHelp')}</span>
            </label>
            <p className="scan-pricing-note">{t('scan.pricingNote')}</p>
            <textarea
              id="scan-text"
              className="scan-textarea"
              placeholder={t('scan.placeholder')}
              value={text}
              onChange={(e) => {
                setText(e.target.value);
                if (!busy) setShowProgress(false);
              }}
              rows={16}
            />
            <div className="scan-meta-row">
              <span>{t('scan.word', { count: wordCount })}</span>
              {wordCount > 0 && tokensRequired === 0 && (
                <strong>{t('scan.freeScan')}</strong>
              )}
              {tokensRequired > 0 && (
                <strong>{t('scan.tokensRequired', { count: tokensRequired })}</strong>
              )}
            </div>

            <button type="submit" className="btn btn-primary" disabled={busy}>
              {busy ? (status || t('scan.scanning')) : t('scan.start')}
            </button>
            <p className="scan-delivery-note">{t('scan.emailPdfNotice')}</p>

            {showProgress && (
              <div className="scan-progress" role="status" aria-live="polite">
                <div className="scan-progress-meta">
                  <span>{progressMessage || status || t('scan.scanning')}</span>
                  <span>{progressPercent}%</span>
                </div>
                <div
                  className="scan-progress-track"
                  role="progressbar"
                  aria-valuemin="0"
                  aria-valuemax="100"
                  aria-valuenow={progressPercent}
                >
                  <div
                    className="scan-progress-fill"
                    style={{ width: `${progressPercent}%` }}
                  />
                </div>
              </div>
            )}
          </form>

          <aside className="scan-side-panel" aria-label={t('scan.scopeLabel')}>
            <p className="eyebrow">{t('scan.reviewScope')}</p>
            <h2>{t('scan.checksTitle')}</h2>
            <ul>
              <li><span>1</span>{t('scan.check1')}</li>
              <li><span>2</span>{t('scan.check2')}</li>
              <li><span>3</span>{t('scan.check3')}</li>
              <li><span>4</span>{t('scan.check4')}</li>
            </ul>
          </aside>
        </section>

        {error && <p className="error">{error}</p>}
        {serverError && <p className="error">{serverError}</p>}

        <ConfirmDialog
          open={insufficientTokens}
          title={t('scan.notEnoughTitle')}
          message={t('scan.notEnoughMessage')}
          confirmLabel={t('scan.buyTokens')}
          onConfirm={() => navigate('/buy')}
          onCancel={() => setInsufficientTokens(false)}
        />

        <ConfirmDialog
          open={authExpired}
          title={t('scan.authTitle')}
          message={t('scan.authMessage')}
          confirmLabel={t('scan.signIn')}
          confirmClassName="btn-primary"
          hideCancel
          onConfirm={redirectToSignIn}
          onCancel={redirectToSignIn}
        />
      </div>
    </main>
  );
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function buildScanEventsUrl(scanId) {
  return buildApiEventUrl(`/scans/${scanId}/events`);
}

function formatStatus(status, t) {
  switch (status) {
    case 'pending': return t('scan.status.pending');
    case 'processing': return t('scan.status.processing');
    case 'retrying': return t('scan.status.retrying');
    case 'completed': return t('scan.status.completed');
    case 'failed': return t('scan.status.failed');
    default: return status || t('scan.scanning');
  }
}
