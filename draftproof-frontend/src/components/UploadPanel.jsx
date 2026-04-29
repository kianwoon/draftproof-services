import { useState, useRef } from 'react';
import { uploadDocument, startScan } from '../api/draftproofApi';
import { useNavigate } from 'react-router-dom';

export default function UploadPanel() {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    setUploading(true);
    setError(null);

    try {
      const { data: doc } = await uploadDocument(new FormData(e.target));
      const { data: scan } = await startScan(doc.id);
      navigate(`/report/${scan.report_id}`);
    } catch (err) {
      setError(err.response?.data?.detail || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="upload-panel">
      <form onSubmit={handleSubmit}>
        <input
          type="file"
          name="file"
          ref={inputRef}
          accept=".pdf,.docx,.txt"
          onChange={(e) => setFile(e.target.files[0])}
        />
        <button type="submit" className="btn btn-primary" disabled={!file || uploading}>
          {uploading ? 'Scanning...' : 'Upload & Scan'}
        </button>
      </form>
      {error && <p className="error">{error}</p>}
    </div>
  );
}
