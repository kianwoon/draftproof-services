import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';

const MAX_RETRIES = 3;
const RETRY_KEY = 'error_reload_retries';

export default function ErrorReload({ message }) {
  const [count, setCount] = useState(10);
  const { t } = useTranslation();
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
          {t('errorReload.retry')}
        </button>
      </div>
    );
  }

  return (
    <div className="error-reload">
      <p className="error">{message}</p>
      <p className="error-reload-hint">{t('errorReload.refreshing', { count, defaultValue: `Refreshing in ${count}s...` })}</p>
    </div>
  );
}
