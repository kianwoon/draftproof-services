import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import CodeTexture from './CodeTexture';

export default function Footer() {
  const { t } = useTranslation();

  return (
    <div className="global-footer-wrap">
      <CodeTexture id="globalFooter" className="footer-code-texture" />
      <footer className="site-footer">
        <Link to="/" className="brand footer-brand">
          <span className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 32 32" role="img">
              <path d="M16 3 27 7v8c0 7.2-4.6 11.8-11 14C9.6 26.8 5 22.2 5 15V7l11-4Z" />
              <path d="m10.8 15.9 3.4 3.3 7.4-8" />
            </svg>
          </span>
          <span>DraftProof</span>
        </Link>

        <div className="footer-links">
          <Link to="/#product">{t('footer.product')}</Link>
          <Link to="/#engine">{t('footer.howItWorks')}</Link>
          <Link to="/#report">{t('footer.sampleReport')}</Link>
          <Link to="/pricing">{t('footer.pricing')}</Link>
          <Link to="/privacy">{t('footer.privacy')}</Link>
          <Link to="/security">{t('footer.security')}</Link>
        </div>

        <p>{t('footer.disclaimer')}</p>
      </footer>
    </div>
  );
}
