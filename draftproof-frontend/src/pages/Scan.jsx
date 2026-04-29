import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { uploadDocument, uploadText, startScan } from '../api/draftproofApi';

export default function Scan() {
  const [tab, setTab] = useState('paste');
  const [text, setText] = useState('');
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setBusy(true);
    setError(null);

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
      const { data: scan } = await startScan(doc.id);
      navigate(`/report/${scan.report_id}`);
    } catch (err) {
      setError(err.response?.data?.detail || 'Scan failed');
    } finally {
      setBusy(false);
    }
  };

  return (
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
          <textarea
            className="scan-textarea"
            placeholder="Paste your document text here..."
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={14}
          />
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
          {busy ? 'Scanning...' : 'Start Scan'}
        </button>
      </form>

      {error && <p className="error">{error}</p>}
    </div>
  );
}
