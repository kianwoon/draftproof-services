import { useEffect, useState } from 'react';

export default function ErrorReload({ message }) {
  const [count, setCount] = useState(10);

  useEffect(() => {
    if (count <= 0) { window.location.reload(); return; }
    const id = setTimeout(() => setCount(c => c - 1), 1000);
    return () => clearTimeout(id);
  }, [count]);

  return (
    <div className="error-reload">
      <p className="error">{message}</p>
      <p className="error-reload-hint">Refreshing in {count}s…</p>
    </div>
  );
}
