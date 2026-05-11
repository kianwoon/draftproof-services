import { useState, useRef, useCallback, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { startScanWithText, getScanStatus, buildApiEventUrl } from '../api/draftproofApi';
import { useAuth } from '../context/AuthContext';
import ConfirmDialog from '../components/ConfirmDialog';
import CodeTexture from '../components/CodeTexture';

const POLL_INTERVAL = 3000;
const MAX_POLLS = 200; // 200 × 3s = 10 min max
const START_SCAN_TIMEOUT_MS = 20000;
const FREE_SCAN_WORD_LIMIT = 300;
const START_SCAN_TIMEOUT_MESSAGE =
  'The scan server is restarting. Please try again in a moment.';

export default function Scan() {
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
  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
  const tokensRequired = wordCount > FREE_SCAN_WORD_LIMIT ? Math.max(1, Math.ceil(wordCount / 1000)) : 0;

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
  }, []);

  const updateProgress = useCallback((data) => {
    const percent = Math.max(0, Math.min(100, Number(data.progress_percent) || 0));
    const message = data.progress_message || formatStatus(data.status);
    setProgressPercent(percent);
    setProgressMessage(message);
    setStatus(message);
  }, []);

  const pollUntilDone = useCallback(async (scanId, signal) => {
    for (let i = 0; i < MAX_POLLS; i++) {
      if (signal.aborted) return false;
      await sleep(POLL_INTERVAL);
      if (signal.aborted) return false;
      const { data } = await getScanStatus(scanId, { signal });
      updateProgress(data);
      if (data.status === 'completed') return true;
      if (data.status === 'failed') {
        setProgressMessage('Scan failed');
        setServerError('Scan failed on the server');
        return false;
      }
      if (data.status === 'retrying') {
        setStatus('Retrying scan...');
      }
    }
    setServerError('Scan timed out');
    return false;
  }, [updateProgress]);

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
          setProgressMessage('Scan failed');
          setServerError('Scan failed on the server');
          finish(false);
        }
      });

      source.addEventListener('scan-error', () => {
        setProgressMessage('Scan failed');
        setServerError('Scan failed on the server');
        finish(false);
      });

      source.addEventListener('error', () => {
        finish(null);
      });
    });
  }, [updateProgress]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!text.trim()) {
      setError('Please enter some text');
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
    setStatus('Submitting...');

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      let scan;
      setStatus('Queuing scan...');
      setProgressPercent(3);
      setProgressMessage('Queuing scan');
      ({ data: scan } = await startScanWithText(text, {
        signal: controller.signal,
        timeout: START_SCAN_TIMEOUT_MS,
      }));

      setStatus('Scanning...');
      setProgressPercent(scan.progress_percent || 5);
      setProgressMessage(scan.progress_message || 'Scan queued');
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
        setServerError(START_SCAN_TIMEOUT_MESSAGE);
        return;
      }
      const msg = err.response?.data?.detail || 'Scan failed';
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
        setProgressMessage('Scan failed');
        setServerError(msg);
      } else {
        setProgressMessage('Scan failed');
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
            <p className="eyebrow">Pre-submission review</p>
            <h1>Scan your draft for fixable integrity signals.</h1>
            <p>
              Paste your text to review citation gaps, source grounding, generic
              phrasing, and authorship signals before submission.
            </p>
          </div>
          <div className="app-hero-stat">
            <span>Available balance</span>
            <strong>{balance === null ? 'Checking' : `${balance} token${balance === 1 ? '' : 's'}`}</strong>
            <small>Free through 300 words</small>
          </div>
        </section>

        <section className="scan-workspace">
          <form onSubmit={handleSubmit} className="scan-form">
            <label className="scan-label" htmlFor="scan-text">
              Document text
              <span>Paste plain text from your paper, report, or essay.</span>
            </label>
            <p className="scan-pricing-note">
              Scans with 300 words or fewer are free. Token billing starts at
              301 words.
            </p>
            <textarea
              id="scan-text"
              className="scan-textarea"
              placeholder="Paste your document text here..."
              value={text}
              onChange={(e) => {
                setText(e.target.value);
                if (!busy) setShowProgress(false);
              }}
              rows={16}
            />
            <div className="scan-meta-row">
              <span>{wordCount.toLocaleString()} word{wordCount !== 1 ? 's' : ''}</span>
              {wordCount > 0 && tokensRequired === 0 && (
                <strong>Free scan</strong>
              )}
              {tokensRequired > 0 && (
                <strong>
                  {tokensRequired} token{tokensRequired !== 1 ? 's' : ''} required
                </strong>
              )}
            </div>

            <button type="submit" className="btn btn-primary" disabled={busy}>
              {busy ? (status || 'Scanning...') : 'Start scan'}
            </button>

            {showProgress && (
              <div className="scan-progress" role="status" aria-live="polite">
                <div className="scan-progress-meta">
                  <span>{progressMessage || status || 'Scanning...'}</span>
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

          <aside className="scan-side-panel" aria-label="What DraftProof checks">
            <p className="eyebrow">Review scope</p>
            <h2>What gets checked</h2>
            <ul>
              <li><span>1</span>Citation gaps and unsupported claims</li>
              <li><span>2</span>Source fit against the claim being made</li>
              <li><span>3</span>Generic or boilerplate phrasing</li>
              <li><span>4</span>Review-only authorship signals</li>
            </ul>
          </aside>
        </section>

        {error && <p className="error">{error}</p>}
        {serverError && <p className="error">{serverError}</p>}

        <ConfirmDialog
          open={insufficientTokens}
          title="Not enough tokens"
          message="You don't have enough tokens to scan this document. Purchase more tokens to continue."
          confirmLabel="Buy tokens"
          onConfirm={() => navigate('/buy')}
          onCancel={() => setInsufficientTokens(false)}
        />

        <ConfirmDialog
          open={authExpired}
          title="Please sign in again"
          message="Your session has expired. Sign in again to continue your scan. Your pasted text will be restored when you return."
          confirmLabel="Sign in"
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

function formatStatus(status) {
  switch (status) {
    case 'pending': return 'Queued';
    case 'processing': return 'Scanning document';
    case 'retrying': return 'Retrying scan';
    case 'completed': return 'Scan complete';
    case 'failed': return 'Scan failed';
    default: return status || 'Scanning...';
  }
}
