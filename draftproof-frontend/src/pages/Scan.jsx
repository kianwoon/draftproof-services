import { useState, useRef, useCallback, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { startScanWithText, getScanStatus } from '../api/draftproofApi';
import { useAuth } from '../context/AuthContext';
import ConfirmDialog from '../components/ConfirmDialog';

const POLL_INTERVAL = 3000;
const MAX_POLLS = 200; // 200 × 3s = 10 min max

export default function Scan() {
  const [text, setText] = useState('');
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [serverError, setServerError] = useState(null);
  const [insufficientTokens, setInsufficientTokens] = useState(false);
  const navigate = useNavigate();
  const { refreshBalance, balance } = useAuth();
  const abortRef = useRef(null);
  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
  const tokensRequired = wordCount > 0 ? Math.max(1, Math.ceil(wordCount / 1000)) : 0;

  // Cancel in-flight polling on unmount
  useEffect(() => {
    return () => {
      if (abortRef.current) {
        abortRef.current.abort();
      }
    };
  }, []);

  const pollUntilDone = useCallback(async (scanId, signal) => {
    for (let i = 0; i < MAX_POLLS; i++) {
      if (signal.aborted) return false;
      await sleep(POLL_INTERVAL);
      if (signal.aborted) return false;
      const { data } = await getScanStatus(scanId, { signal });
      setStatus(`Scanning... (${data.status})`);
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
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (balance !== null && balance < tokensRequired) {
      setInsufficientTokens(true);
      return;
    }
    setBusy(true);
    setError(null);
    setServerError(null);
    setStatus('Submitting...');

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      let scan;
      if (!text.trim()) { setError('Please enter some text'); setBusy(false); return; }
      setStatus('Queuing scan...');
      ({ data: scan } = await startScanWithText(text));

      setStatus('Scanning...');
      const completed = await pollUntilDone(scan.id, controller.signal);
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
