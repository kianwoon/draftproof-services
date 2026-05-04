import { useState, useRef, useCallback, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { startScanWithText, getScanStatus } from '../api/draftproofApi';
import { useAuth } from '../context/AuthContext';
import ConfirmDialog from '../components/ConfirmDialog';

const POLL_INTERVAL = 3000;
const MAX_POLLS = 200; // 200 × 3s = 10 min max
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

export default function Scan() {
  const [text, setText] = useState('');
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState(null);
  const [progressPercent, setProgressPercent] = useState(0);
  const [progressMessage, setProgressMessage] = useState(null);
  const [error, setError] = useState(null);
  const [serverError, setServerError] = useState(null);
  const [insufficientTokens, setInsufficientTokens] = useState(false);
  const navigate = useNavigate();
  const { refreshBalance, balance } = useAuth();
  const abortRef = useRef(null);
  const eventSourceRef = useRef(null);
  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
  const tokensRequired = wordCount > 0 ? Math.max(1, Math.ceil(wordCount / 1000)) : 0;

  // Cancel in-flight polling on unmount
  useEffect(() => {
    return () => {
      if (abortRef.current) {
        abortRef.current.abort();
      }
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
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
          setServerError('Scan failed on the server');
          finish(false);
        }
      });

      source.addEventListener('scan-error', () => {
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
    if (balance !== null && balance < tokensRequired) {
      setInsufficientTokens(true);
      return;
    }
    setBusy(true);
    setError(null);
    setServerError(null);
    setProgressPercent(0);
    setProgressMessage(null);
    setStatus('Submitting...');

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      let scan;
      if (!text.trim()) { setError('Please enter some text'); setBusy(false); return; }
      setStatus('Queuing scan...');
      setProgressPercent(3);
      setProgressMessage('Queuing scan');
      ({ data: scan } = await startScanWithText(text));

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
      const msg = err.response?.data?.detail || 'Scan failed';
      const httpStatus = err.response?.status;
      if (httpStatus === 400 && msg.toLowerCase().includes('insufficient')) {
        setInsufficientTokens(true);
      } else if (httpStatus >= 400) {
        setServerError(msg);
      } else {
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
    <main className="dash-shell">
    <div className="container scan-page">
      <h1>Scan Document</h1>

      <form onSubmit={handleSubmit} className="scan-form">
        <>
        <textarea
          className="scan-textarea"
          placeholder="Paste your document text here..."
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={14}
        />
        <div className="word-count">
          {wordCount.toLocaleString()} word{wordCount !== 1 ? 's' : ''}
          {tokensRequired > 0 && (
            <span className="word-tokens">
              {' '}— {tokensRequired} token{tokensRequired !== 1 ? 's' : ''} required
              {tokensRequired > 1 && (
                <span className="word-limit"> (1 token per 1,000 words)</span>
              )}
            </span>
          )}
        </div>
        </>

        <button type="submit" className="btn btn-primary" disabled={busy}>
          {busy ? (status || 'Scanning...') : 'Start Scan'}
        </button>
        {busy && (
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
    </div>
    </main>
  );
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function buildScanEventsUrl(scanId) {
  const path = `/scans/${scanId}/events`;
  if (/^https?:\/\//i.test(API_BASE_URL)) {
    const base = new URL(API_BASE_URL);
    return `${base.origin}${base.pathname.replace(/\/$/, '')}${path}`;
  }
  return `${API_BASE_URL.replace(/\/$/, '')}${path}`;
}

function formatStatus(status) {
  if (status === 'pending') return 'Queued';
  if (status === 'processing') return 'Scanning';
  if (status === 'retrying') return 'Retrying scan';
  if (status === 'completed') return 'Scan complete';
  if (status === 'failed') return 'Scan failed';
  return 'Scanning';
}
