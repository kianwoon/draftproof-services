import { useEffect, useState, useRef } from 'react';
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
  const [scanOpen, setScanOpen] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const dropdownRef = useRef(null);
  const locale = getLocaleFromPathname(location.pathname);
  const publicPath = (path) => localizePath(path, locale);
  const marketingLinks = [
    { to: publicPath('/why'), label: t('nav.why') },
    { to: publicPath('/essay-checker'), label: t('nav.essayChecker') },
    { to: publicPath('/pricing'), label: t('nav.pricing') },
    { to: publicPath('/faq'), label: t('nav.faq') },
    { to: publicPath('/#report'), label: t('nav.sampleReport') },
  ];
  const signedInPublicLinks = [
    { to: publicPath('/pricing'), label: t('nav.pricing') },
  ];
  const visiblePublicLinks = user ? signedInPublicLinks : marketingLinks;

  useEffect(() => { setScanOpen(false); setMenuOpen(false); }, [location.pathname]);

  useEffect(() => {
    if (!scanOpen) return;
    const close = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) setScanOpen(false);
    };
    document.addEventListener('click', close);
    return () => document.removeEventListener('click', close);
  }, [scanOpen]);

  const handleLogout = async () => {
    await logout();
    navigate(publicPath('/'), { replace: true });
  };

  const isScanActive = ['/scan', '/reports'].includes(location.pathname);
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
        {user && <Link to="/dashboard">{t('nav.dashboard')}</Link>}
        {user && (
          <div className="nav-dropdown" ref={dropdownRef}
            onMouseEnter={() => setScanOpen(true)}
            onMouseLeave={() => setScanOpen(false)}
          >
            <button
              className={`nav-dropdown-trigger${isScanActive ? ' active' : ''}`}
              onClick={() => setScanOpen(!scanOpen)}
            >
              {t('nav.scan')}
              <svg width="10" height="6" viewBox="0 0 10 6" style={{ marginLeft: 5, transition: 'transform .2s', transform: scanOpen ? 'rotate(180deg)' : 'rotate(0)' }}>
                <path d="M1 1l4 4 4-4" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" />
              </svg>
            </button>
            {scanOpen && (
              <div className="nav-dropdown-menu">
                <Link to="/scan" className="nav-dropdown-item" onClick={() => setScanOpen(false)}>
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                    <path d="M2 4.5A2.5 2.5 0 014.5 2h7A2.5 2.5 0 0114 4.5v7a2.5 2.5 0 01-2.5 2.5h-7A2.5 2.5 0 012 11.5v-7z" stroke="currentColor" strokeWidth="1.4"/>
                    <path d="M5 8h6M8 5v6" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"/>
                  </svg>
                  {t('nav.scanning')}
                </Link>
                <Link to="/reports" className="nav-dropdown-item" onClick={() => setScanOpen(false)}>
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                    <path d="M3 2h7l3 3v8a1 1 0 01-1 1H4a1 1 0 01-1-1V3a1 1 0 011-1z" stroke="currentColor" strokeWidth="1.4"/>
                    <path d="M6 7h4M6 9.5h3" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/>
                  </svg>
                  {t('nav.viewReports')}
                </Link>
              </div>
            )}
          </div>
        )}
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
          {user.avatar_url ? (
            <img src={user.avatar_url} alt={user.email} className="user-avatar" />
          ) : (
            <div className="user-avatar-placeholder">
              {user.email.charAt(0).toUpperCase()}
            </div>
          )}
          <button onClick={handleLogout} className="btn btn-secondary btn-small">{t('nav.signOut')}</button>
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
            {user && <Link to="/dashboard" className="mobile-link">{t('nav.dashboard')}</Link>}
            {user && <Link to="/scan" className="mobile-link">{t('nav.scan')}</Link>}
            {user && <Link to="/reports" className="mobile-link">{t('nav.reports')}</Link>}
            {user && <Link to="/buy" className="mobile-link">{t('nav.buyTokens')}</Link>}
            {user && <Link to="/history" className="mobile-link">{t('nav.history')}</Link>}
            {visiblePublicLinks.map((link) => (
              <Link key={link.to} to={link.to} className="mobile-link">{link.label}</Link>
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
