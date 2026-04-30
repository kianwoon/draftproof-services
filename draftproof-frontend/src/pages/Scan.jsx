import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { startScanWithText, getScanStatus } from '../api/draftproofApi';
import { useAuth } from '../context/AuthContext';
import ErrorReload from '../components/ErrorReload';

const POLL_INTERVAL = 3000;
const MAX_POLLS = 100;

export default function Scan() {
  const [text, setText] = useState('');
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [serverError, setServerError] = useState(null);
  const navigate = useNavigate();
  const { refreshBalance } = useAuth();
  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
  const tokensRequired = wordCount > 0 ? Math.max(1, Math.ceil(wordCount / 1000)) : 0;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    setServerError(null);
    setStatus('Submitting...');

    try {
      let scan;
      if (!text.trim()) { setError('Please enter some text'); setBusy(false); return; }
      setStatus('Queuing scan...');
      ({ data: scan } = await startScanWithText(text));

      setStatus('Scanning...');
      const completed = await pollUntilDone(scan.id);
      if (completed) {
        refreshBalance();
        navigate(`/report/${scan.id}`);
      }
    } catch (err) {
      const msg = err.response?.data?.detail || 'Scan failed';
      const status = err.response?.status;
      if (status >= 400) { setServerError(msg); } else { setError(msg); }
    } finally {
      setBusy(false);
      setStatus(null);
    }
  };

  const pollUntilDone = async (scanId) => {
    for (let i = 0; i < MAX_POLLS; i++) {
      await sleep(POLL_INTERVAL);
      const { data } = await getScanStatus(scanId);
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
  };

  return (
    <main className="dash-shell">
    <div className="container scan-page">
      <h1>Scan Document</h1>

      {/* TODO: restore Upload File tab when file parsing is ready */}

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
      {serverError && <ErrorReload message={serverError} />}
    </div>
    </main>
  );
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}
