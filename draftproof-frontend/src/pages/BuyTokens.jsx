import { useState, useEffect } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import api from '../api/draftproofApi';
import { useAuth } from '../context/AuthContext';
import ErrorReload from '../components/ErrorReload';
import CodeTexture from '../components/CodeTexture';

export default function BuyTokens() {
  const [packs, setPacks] = useState([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState(null);
  const [serverError, setServerError] = useState(null);
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const { balance, refreshBalance } = useAuth();

  useEffect(() => {
    const ac = new AbortController();
    api.get('/payments/packs', { signal: ac.signal }).then(r => setPacks(r.data)).catch(() => {});

    if (searchParams.get('success')) {
      setMessage({ type: 'success', text: 'Payment successful! Your tokens have been added.' });
      refreshBalance();
    } else if (searchParams.get('canceled')) {
      setMessage({ type: 'info', text: 'Payment was canceled.' });
    }

    return () => ac.abort();
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
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="app-page">
      <div className="container">
        <section className="app-hero app-hero-dark buy-hero">
          <CodeTexture id="buyHero" />
          <div>
            <p className="eyebrow">Tokens</p>
            <h1>Buy review credits when you need them.</h1>
            <p>No subscription. Tokens stay in your account until you use them.</p>
          </div>
          {balance !== null && (
            <div className="app-hero-stat">
              <span>Current balance</span>
              <strong>{balance} token{balance === 1 ? '' : 's'}</strong>
              <small>1 token per 1,000 words</small>
            </div>
          )}
        </section>

        {message && (
          <div className={`alert alert-${message.type}`}>
            {message.text}
          </div>
        )}

        {serverError && <ErrorReload message={serverError} />}

        <div className="pack-grid">
          {packs.map(pack => (
            <div key={pack.id} className="pack-card">
              <p className="eyebrow">{pack.name}</p>
              <div className="pack-tokens">{pack.tokens} tokens</div>
              <div className="pack-price">USD ${pack.price_usd.toFixed(2)}</div>
              <div className="pack-unit">${(pack.price_usd / pack.tokens).toFixed(2)} / token</div>
              <button
                className="btn btn-primary"
                onClick={() => handleBuy(pack.id)}
                disabled={loading}
              >
                {loading ? 'Redirecting...' : 'Buy now'}
              </button>
            </div>
          ))}
        </div>

        <div className="page-actions-center">
          <button className="btn btn-secondary" onClick={() => navigate('/dashboard')}>
            Back to dashboard
          </button>
        </div>
      </div>
    </main>
  );
}
