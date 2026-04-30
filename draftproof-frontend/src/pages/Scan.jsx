import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { uploadDocument, uploadText, startScan, getScanStatus } from '../api/draftproofApi';

const POLL_INTERVAL = 3000;
const MAX_POLLS = 100;

export default function Scan() {
  const [tab, setTab] = useState('paste');
  const [text, setText] = useState('');
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const navigate = useNavigate();
  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    setStatus('Uploading...');

    try {
      let doc;
      if (tab === 'paste') {
        if (!text.trim()) { setError('Please enter some text'); setBusy(false); return; }
        ({ data: doc } = await uploadText(text));
      } else {
        if (!file) { setError('Please select a file'); setBusy(false); return; }
        const fd = new FormData();
        fd.append('file', file);
        ({ data: doc } = await uploadDocument(fd));
      }

      setStatus('Queuing scan...');
      const { data: scan } = await startScan(doc.id);

      setStatus('Scanning...');
      const completed = await pollUntilDone(scan.id);
      if (completed) {
        navigate(`/report/${scan.id}`);
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Scan failed');
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
        setError('Scan failed on the server');
        return false;
      }
    }
    setError('Scan timed out');
    return false;
  };

  return (
    <main className="dash-shell">
    <div className="container scan-page">
      <h1>Scan Document</h1>

      <div className="scan-tabs">
        <button className={`tab ${tab === 'paste' ? 'active' : ''}`} onClick={() => setTab('paste')}>
          Paste Text
        </button>
        <button className={`tab ${tab === 'upload' ? 'active' : ''}`} onClick={() => setTab('upload')}>
          Upload File
        </button>
      </div>

      <form onSubmit={handleSubmit} className="scan-form">
        {tab === 'paste' ? (
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
            {wordCount > 1000 && <span className="word-limit"> — exceeds 1,000 word limit per scan</span>}
          </div>
          </>
        ) : (
          <div className="upload-zone">
            <input
              type="file"
              accept=".pdf,.docx,.txt"
              onChange={(e) => setFile(e.target.files[0])}
            />
            {file && <span className="file-name">{file.name}</span>}
          </div>
        )}

        <button type="submit" className="btn btn-primary" disabled={busy}>
          {busy ? (status || 'Scanning...') : 'Start Scan'}
        </button>
      </form>

      {error && <p className="error">{error}</p>}
    </div>
    </main>
  );
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}
