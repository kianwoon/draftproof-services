import { useEffect, useState } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useAuth } from '../context/AuthContext';
import LanguageSwitcher from './LanguageSwitcher';
import { getLocaleFromPathname, localizePath } from '../localeRouting';

export default function Header() {
  const { user, logout, balance } = useAuth();
  const { t } = useTranslation();
  const navigate = useNavigate();
  const location = useLocation();
  const [menuOpen, setMenuOpen] = useState(false);
  const locale = getLocaleFromPathname(location.pathname);
  const publicPath = (path) => localizePath(path, locale);
  const marketingLinks = [
    { to: publicPath('/#product'), label: t('nav.why') },
    { to: publicPath('/features'), label: t('nav.features') },
    { to: publicPath('/technology'), label: t('nav.technology') },
    { to: publicPath('/rewrite'), label: t('nav.rewrite') },
    { to: publicPath('/content-checker'), label: t('nav.essayChecker') },
    { to: publicPath('/pricing'), label: t('nav.pricing') },
    { to: publicPath('/faq'), label: t('nav.faq') },
    { to: publicPath('/#report'), label: t('nav.sampleReport') },
  ];
  // Signed-in users buy via "Buy Tokens"; pricing/marketing links are hidden.
  const signedInPublicLinks = [];
  const visiblePublicLinks = user ? signedInPublicLinks : marketingLinks;

  useEffect(() => { setMenuOpen(false); }, [location.hash, location.pathname]);

  const handleLogout = async () => {
    await logout();
    navigate(publicPath('/'), { replace: true });
  };

  const isLanding = location.pathname === '/' || location.pathname === '/zh';

  return (
    <header className={`site-header${isLanding ? ' site-header-dark' : ''}`} aria-label={t('nav.main')}>
      <Link to={publicPath('/')} className="brand" aria-label={t('nav.home')}>
        <span className="brand-mark" aria-hidden="true">
          <svg viewBox="0 0 32 32" role="img">
            <path d="M16 3 27 7v8c0 7.2-4.6 11.8-11 14C9.6 26.8 5 22.2 5 15V7l11-4Z" />
            <path d="m10.8 15.9 3.4 3.3 7.4-8" />
          </svg>
        </span>
        <span>DraftProof</span>
      </Link>

      <nav className="nav-links" aria-label={t('nav.primary')}>
        {user && <Link to="/scan">{t('nav.dashboard')}</Link>}
        {user && <Link to="/reports">{t('nav.viewReports')}</Link>}
        {user && <Link to="/buy">{t('nav.buyTokens')}</Link>}
        {user && <Link to="/history">{t('nav.history')}</Link>}
        {visiblePublicLinks.map((link) => (
          <Link key={link.to} to={link.to}>{link.label}</Link>
        ))}
      </nav>

      {user ? (
        <div className="header-user">
          <LanguageSwitcher compact />
          <Link to="/history" className="token-badge">
            {balance !== null ? t('common.token', { count: balance }) : '—'}
          </Link>
          <div className="header-account">
            <button
              type="button"
              className="header-account-trigger"
              aria-haspopup="menu"
              aria-label={user.email}
            >
              {user.avatar_url ? (
                <img src={user.avatar_url} alt="" className="user-avatar" />
              ) : (
                <div className="user-avatar-placeholder">
                  {user.email.charAt(0).toUpperCase()}
                </div>
              )}
            </button>
            <div className="header-account-menu" role="menu">
              <span className="header-account-email">{user.email}</span>
              <Link to="/api-keys" className="header-account-item" role="menuitem">
                <svg className="header-account-item-icon" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                  <circle cx="8" cy="15" r="4" />
                  <path d="M11.5 11.5 20 3M16 3h4v4M14 9l2.5 2.5" />
                </svg>
                {t('nav.apiKeys')}
              </Link>
              <button
                type="button"
                onClick={handleLogout}
                className="header-account-item header-account-signout"
                role="menuitem"
              >
                <svg className="header-account-item-icon" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                  <path d="M15 4H6a1 1 0 0 0-1 1v14a1 1 0 0 0 1 1h9" />
                  <path d="M10 12h11m0 0-3.5-3.5M21 12l-3.5 3.5" />
                </svg>
                {t('nav.signOut')}
              </button>
            </div>
          </div>
        </div>
      ) : (
        <div className="header-actions">
          <LanguageSwitcher compact />
          <Link to={publicPath('/signin')} className="btn btn-primary btn-small">
            {t('nav.startReview')}
          </Link>
        </div>
      )}

      <button
        className={`hamburger${menuOpen ? ' is-open' : ''}`}
        onClick={() => setMenuOpen(!menuOpen)}
        aria-label={t('nav.toggleMenu')}
        aria-expanded={menuOpen}
      >
        <span /><span /><span />
      </button>

      {menuOpen && (
        <div className="mobile-menu" onClick={() => setMenuOpen(false)}>
          <div className="mobile-menu-inner" onClick={(e) => e.stopPropagation()}>
            <LanguageSwitcher />
            <Link to={publicPath('/')} className="mobile-link" onClick={() => setMenuOpen(false)}>{t('nav.homeLink')}</Link>
            {user && <Link to="/scan" className="mobile-link">{t('nav.dashboard')}</Link>}
            {user && <Link to="/reports" className="mobile-link">{t('nav.viewReports')}</Link>}
            {user && <Link to="/buy" className="mobile-link">{t('nav.buyTokens')}</Link>}
            {user && <Link to="/history" className="mobile-link">{t('nav.history')}</Link>}
            {user && <Link to="/api-keys" className="mobile-link">{t('nav.apiKeys')}</Link>}
            {visiblePublicLinks.map((link) => (
              <Link key={link.to} to={link.to} className="mobile-link" onClick={() => setMenuOpen(false)}>{link.label}</Link>
            ))}
            <div className="mobile-menu-actions">
              {user ? (
                <button onClick={() => { handleLogout(); setMenuOpen(false); }} className="btn btn-secondary">{t('nav.signOut')}</button>
              ) : (
                <Link to={publicPath('/signin')} className="btn btn-primary">{t('nav.startReview')}</Link>
              )}
            </div>
          </div>
        </div>
      )}
    </header>
  );
}
