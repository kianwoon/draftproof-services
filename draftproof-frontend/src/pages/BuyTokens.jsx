import { useState, useEffect } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import api, { isAuthExpiryError } from '../api/draftproofApi';
import { useAuth } from '../context/AuthContext';
import ErrorReload from '../components/ErrorReload';
import CodeTexture from '../components/CodeTexture';
import { TOKEN_CURRENCY_CODE } from '../pricingConfig';

const RECOMMENDED_PACK_ID = 'standard';
const PROMO_PACK_ID = 'pro';

function formatNumber(value, locale) {
  return new Intl.NumberFormat(locale).format(value);
}

export default function BuyTokens() {
  const { t, i18n } = useTranslation();
  const [packs, setPacks] = useState([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState(null);
  const [serverError, setServerError] = useState(null);
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const { balance, refreshBalance } = useAuth();
  const locale = i18n.resolvedLanguage?.startsWith('zh') ? 'zh-CN' : 'en-SG';
  const balanceTokens = Number(balance || 0);
  const scanWordEstimate = balanceTokens * 1000;
  const revisionBlockEstimate = Math.floor(balanceTokens / 5);
  const usageItems = t('buy.usageItems', { returnObjects: true });
  const packNotes = t('buy.packNotes', { returnObjects: true });

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
      // Session expired → the global 401 interceptor is already redirecting to
      // /signin; skip the ErrorReload countdown so the redirect is immediate.
      if (isAuthExpiryError(err)) return;
      const msg = err.response?.data?.detail || t('buy.checkoutFailed');
      const status = err.response?.status;
      if (status >= 400) { setServerError(msg); } else { setMessage({ type: 'error', text: msg }); }
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="app-page buy-page">
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
              <small>{t('buy.balanceScanEstimate', { count: formatNumber(scanWordEstimate, locale) })}</small>
              <small>{t('buy.balanceRewriteEstimate', { count: formatNumber(revisionBlockEstimate, locale) })}</small>
            </div>
          )}
        </section>

        {message && (
          <div className={`alert alert-${message.type}`}>
            {message.text}
          </div>
        )}

        {serverError && <ErrorReload message={serverError} />}

        <section className="buy-usage-strip" aria-label={t('buy.usageAria')}>
          {usageItems.map((item) => (
            <div className="buy-usage-item" key={item.label}>
              <span>{item.label}</span>
              <strong>{item.value}</strong>
              <small>{item.detail}</small>
            </div>
          ))}
        </section>

        <div className="pack-grid">
          {packs.map(pack => {
            const isRecommended = pack.id === RECOMMENDED_PACK_ID;
            const isPromo = pack.id === PROMO_PACK_ID;
            const price = Number(pack.price_sgd || 0);
            return (
              <div
                key={pack.id}
                className={`pack-card${isRecommended ? ' pack-card-recommended' : ''}`}
              >
                <div className="pack-card-topline">
                  <p className="eyebrow">{pack.name}</p>
                  <div className="pack-card-badges">
                    {isRecommended && <span>{t('buy.recommended')}</span>}
                    {isPromo && <span>{t('buy.promo')}</span>}
                  </div>
                </div>
                <div className="pack-tokens">{t('buy.tokenCount', { count: pack.tokens })}</div>
                <div className="pack-price">{TOKEN_CURRENCY_CODE} ${price.toFixed(2)}</div>
                <p className="pack-note">{packNotes[pack.id] || t('buy.defaultPackNote')}</p>
                <button
                  className={`btn ${isRecommended ? 'btn-primary' : 'btn-secondary'}`}
                  onClick={() => handleBuy(pack.id)}
                  disabled={loading}
                >
                  {loading ? t('buy.redirecting') : t('buy.buyNow')}
                </button>
              </div>
            );
          })}
        </div>

        <div className="page-actions-center">
          <button className="buy-back-link" onClick={() => navigate('/scan')}>
            {t('buy.backDashboard')}
          </button>
        </div>
      </div>
    </main>
  );
}
