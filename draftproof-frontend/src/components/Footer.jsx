import { Link, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import PageFreshness from './PageFreshness';
import { getLocaleFromPathname, localizePath } from '../localeRouting';

export default function Footer() {
  const { t } = useTranslation();
  const location = useLocation();
  const locale = getLocaleFromPathname(location.pathname);
  const publicPath = (path) => localizePath(path, locale);

  return (
    <footer className="landing-footer">
      <div className="section-inner landing-footer-inner">
        <div>
          <Link to={publicPath('/')} className="footer-wordmark">DraftProof</Link>
          <p>{t('footer.disclaimer')}</p>
          <PageFreshness path="/" className="footer-freshness" />
        </div>
        <nav aria-label={t('footer.product')}>
          <a href={publicPath('/#product')}>{t('footer.product')}</a>
          <a href={publicPath('/#engine')}>{t('footer.howItWorks')}</a>
          <a href={publicPath('/#report')}>{t('footer.sampleReport')}</a>
          <Link to={publicPath('/content-checker')}>{t('footer.essayChecker')}</Link>
          <Link to={publicPath('/rewrite')}>{t('footer.rewrite')}</Link>
          <Link to={publicPath('/academic-integrity-ai')}>{t('footer.academicIntegrity')}</Link>
          <Link to={publicPath('/turnitin-ai-score')}>{t('footer.turnitinScore')}</Link>
          <Link to={publicPath('/ai-declaration')}>{t('footer.aiDeclaration')}</Link>
          <Link to={publicPath('/reduce-ai-detection')}>{t('footer.reduceDetection')}</Link>
          <Link to={publicPath('/pricing')}>{t('footer.pricing')}</Link>
          <Link to={publicPath('/faq')}>{t('footer.faq')}</Link>
          <Link to={publicPath('/privacy')}>{t('footer.privacy')}</Link>
          <Link to={publicPath('/terms')}>{t('footer.terms')}</Link>
          <Link to={publicPath('/support')}>{t('footer.support')}</Link>
          <Link to={publicPath('/eula')}>{t('footer.eula')}</Link>
          <Link to={publicPath('/security')}>{t('footer.security')}</Link>
          <a href="https://www.reddit.com/r/DraftProofApp/" target="_blank" rel="noopener noreferrer">{t('footer.community')}</a>
          <a href={`mailto:${t('footer.supportEmail')}`}>{t('footer.supportEmail')}</a>
        </nav>
      </div>
    </footer>
  );
}
