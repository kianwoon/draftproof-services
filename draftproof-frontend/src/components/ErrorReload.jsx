import { useEffect, useState } from 'react';

const MAX_RETRIES = 3;
const RETRY_KEY = 'error_reload_retries';

export default function ErrorReload({ message }) {
  const [count, setCount] = useState(10);
  const retries = parseInt(sessionStorage.getItem(RETRY_KEY) || '0', 10);

  useEffect(() => {
    if (retries >= MAX_RETRIES) return;
    if (count <= 0) {
      sessionStorage.setItem(RETRY_KEY, String(retries + 1));
      window.location.reload();
      return;
    }
    const id = setTimeout(() => setCount(c => c - 1), 1000);
    return () => clearTimeout(id);
  }, [count, retries]);

  if (retries >= MAX_RETRIES) {
    return (
      <div className="error-reload">
        <p className="error">{message}</p>
        <button className="btn btn-secondary" onClick={() => {
          sessionStorage.removeItem(RETRY_KEY);
          window.location.reload();
        }}>
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="error-reload">
      <p className="error">{message}</p>
      <p className="error-reload-hint">Refreshing in {count}s…</p>
    </div>
  );
}
