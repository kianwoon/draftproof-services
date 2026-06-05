import { useState, useEffect } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import api from '../api/draftproofApi';
import { useAuth } from '../context/AuthContext';
import ErrorReload from '../components/ErrorReload';
import CodeTexture from '../components/CodeTexture';
import { TOKEN_CURRENCY_CODE } from '../pricingConfig';

export default function BuyTokens() {
  const { t } = useTranslation();
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
      setMessage({ type: 'success', text: t('buy.paymentSuccess') });
      refreshBalance();
    } else if (searchParams.get('canceled')) {
      setMessage({ type: 'info', text: t('buy.paymentCanceled') });
    }

    return () => ac.abort();
  }, [searchParams, refreshBalance, t]);

  const handleBuy = async (packId) => {
    setLoading(true);
    try {
      const { data } = await api.post('/payments/checkout', { pack_id: packId });
      window.location.href = data.url;
    } catch (err) {
      const msg = err.response?.data?.detail || t('buy.checkoutFailed');
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
            <p className="eyebrow">{t('buy.eyebrow')}</p>
            <h1>{t('buy.title')}</h1>
            <p>{t('buy.body')}</p>
          </div>
          {balance !== null && (
            <div className="app-hero-stat">
              <span>{t('buy.currentBalance')}</span>
              <strong>{t('common.token', { count: balance })}</strong>
              <small>{t('buy.tokenRate')}</small>
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
              <div className="pack-tokens">{t('buy.tokenCount', { count: pack.tokens })}</div>
              <div className="pack-price">{TOKEN_CURRENCY_CODE} ${pack.price_sgd.toFixed(2)}</div>
              <button
                className="btn btn-primary"
                onClick={() => handleBuy(pack.id)}
                disabled={loading}
              >
                {loading ? t('buy.redirecting') : t('buy.buyNow')}
              </button>
            </div>
          ))}
        </div>

        <div className="page-actions-center">
          <button className="btn btn-secondary" onClick={() => navigate('/dashboard')}>
            {t('buy.backDashboard')}
          </button>
        </div>
      </div>
    </main>
  );
}
