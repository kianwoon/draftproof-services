import { useState, useEffect } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import api from '../api/draftproofApi';
import ErrorReload from '../components/ErrorReload';

export default function BuyTokens() {
  const [packs, setPacks] = useState([]);
  const [balance, setBalance] = useState(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState(null);
  const [serverError, setServerError] = useState(null);
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();

  useEffect(() => {
    api.get('/payments/packs').then(r => setPacks(r.data)).catch(() => {});
    api.get('/payments/balance').then(r => setBalance(r.data)).catch(() => {});

    if (searchParams.get('success')) {
      setMessage({ type: 'success', text: 'Payment successful! Your tokens have been added.' });
    } else if (searchParams.get('canceled')) {
      setMessage({ type: 'info', text: 'Payment was canceled.' });
    }
  }, [searchParams]);

  const handleBuy = async (packId) => {
    setLoading(true);
    try {
      const { data } = await api.post('/payments/checkout', { pack_id: packId });
      window.location.href = data.url;
    } catch (err) {
      const msg = err.response?.data?.detail || 'Failed to start checkout';
      const status = err.response?.status;
      if (status >= 400) { setServerError(msg); } else { setMessage({ type: 'error', text: msg }); }
      setLoading(false);
    }
  };

  return (
    <div className="container" style={{ paddingTop: 'calc(var(--header-h) + 4rem)', paddingBottom: '4rem' }}>
      <div style={{ textAlign: 'center', marginBottom: '2.5rem' }}>
        <h2 style={{ marginBottom: '0.5rem' }}>Buy Tokens</h2>
        <p style={{ color: 'var(--text-2)' }}>SGD $2.90 per token — each scan costs 1 token per document.</p>
        {balance !== null && (
          <p className="balance-display">Current balance: <strong>{balance.balance} tokens</strong></p>
        )}
      </div>

      {message && (
        <div className={`alert alert-${message.type}`}>
          {message.text}
        </div>
      )}

      {serverError && <ErrorReload message={serverError} />}

      <div className="pack-grid">
        {packs.map(pack => (
          <div key={pack.id} className="pack-card">
            <h3>{pack.name}</h3>
            <div className="pack-tokens">{pack.tokens} tokens</div>
            <div className="pack-price">SGD ${pack.price_sgd.toFixed(2)}</div>
            <div className="pack-unit">${(pack.price_sgd / pack.tokens).toFixed(2)} / token</div>
            <button
              className="btn btn-primary"
              onClick={() => handleBuy(pack.id)}
              disabled={loading}
              style={{ width: '100%', marginTop: '1rem' }}
            >
              {loading ? 'Redirecting...' : 'Buy Now'}
            </button>
          </div>
        ))}
      </div>

      <div style={{ textAlign: 'center', marginTop: '2rem' }}>
        <button className="btn btn-secondary" onClick={() => navigate('/dashboard')}>
          Back to Dashboard
        </button>
      </div>
    </div>
  );
}
