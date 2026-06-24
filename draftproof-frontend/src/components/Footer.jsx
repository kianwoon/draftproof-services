import { Link, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import CodeTexture from './CodeTexture';
import { getLocaleFromPathname, localizePath } from '../localeRouting';

export default function Footer() {
  const { t } = useTranslation();
  const location = useLocation();
  const locale = getLocaleFromPathname(location.pathname);
  const publicPath = (path) => localizePath(path, locale);

  return (
    <div className="global-footer-wrap">
      <CodeTexture id="globalFooter" className="footer-code-texture" />
      <footer className="site-footer">
        <Link to={publicPath('/')} className="brand footer-brand">
          <span className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 32 32" role="img">
              <path d="M16 3 27 7v8c0 7.2-4.6 11.8-11 14C9.6 26.8 5 22.2 5 15V7l11-4Z" />
              <path d="m10.8 15.9 3.4 3.3 7.4-8" />
            </svg>
          </span>
          <span>DraftProof</span>
        </Link>

        <div className="footer-links">
          <Link to={publicPath('/#product')}>{t('footer.product')}</Link>
          <Link to={publicPath('/#engine')}>{t('footer.howItWorks')}</Link>
          <Link to={publicPath('/#report')}>{t('footer.sampleReport')}</Link>
          <Link to={publicPath('/content-checker')}>{t('footer.essayChecker')}</Link>
          <Link to={publicPath('/academic-integrity-ai')}>{t('footer.academicIntegrity')}</Link>
          <Link to={publicPath('/pricing')}>{t('footer.pricing')}</Link>
          <Link to={publicPath('/faq')}>{t('footer.faq')}</Link>
          <Link to={publicPath('/privacy')}>{t('footer.privacy')}</Link>
          <Link to={publicPath('/terms')}>{t('footer.terms')}</Link>
          <Link to={publicPath('/support')}>{t('footer.support')}</Link>
          <Link to={publicPath('/eula')}>{t('footer.eula')}</Link>
          <Link to={publicPath('/security')}>{t('footer.security')}</Link>
          <a href="https://www.reddit.com/r/DraftProofApp/" target="_blank" rel="noopener noreferrer">{t('footer.community')}</a>
          <a href={`mailto:${t('footer.supportEmail')}`}>{t('footer.supportEmail')}</a>
        </div>

        <p>{t('footer.disclaimer')}</p>
      </footer>
    </div>
  );
}
