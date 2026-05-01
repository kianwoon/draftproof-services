import { useEffect, useState, useRef } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function Header() {
  const { user, logout, balance } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [scanOpen, setScanOpen] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const dropdownRef = useRef(null);

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
    navigate('/', { replace: true });
  };

  const isScanActive = ['/scan', '/reports'].includes(location.pathname);

  return (
    <header className="site-header" aria-label="Main navigation">
      <Link to="/" className="brand" aria-label="DraftProof home">
        <span className="brand-mark" aria-hidden="true">
          <svg viewBox="0 0 32 32" role="img">
            <path d="M16 3 27 7v8c0 7.2-4.6 11.8-11 14C9.6 26.8 5 22.2 5 15V7l11-4Z" />
            <path d="m10.8 15.9 3.4 3.3 7.4-8" />
          </svg>
        </span>
        <span>DraftProof</span>
      </Link>

      <nav className="nav-links" aria-label="Primary">
        {user ? <Link to="/dashboard">Dashboard</Link> : <Link to="/">Home</Link>}
        {user && (
          <div className="nav-dropdown" ref={dropdownRef}
            onMouseEnter={() => setScanOpen(true)}
            onMouseLeave={() => setScanOpen(false)}
          >
            <button
              className={`nav-dropdown-trigger${isScanActive ? ' active' : ''}`}
              onClick={() => setScanOpen(!scanOpen)}
            >
              Scan
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
                  Scanning
                </Link>
                <Link to="/reports" className="nav-dropdown-item" onClick={() => setScanOpen(false)}>
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                    <path d="M3 2h7l3 3v8a1 1 0 01-1 1H4a1 1 0 01-1-1V3a1 1 0 011-1z" stroke="currentColor" strokeWidth="1.4"/>
                    <path d="M6 7h4M6 9.5h3" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/>
                  </svg>
                  View Reports
                </Link>
              </div>
            )}
          </div>
        )}
        {user && <Link to="/buy">Buy Tokens</Link>}
        {user && <Link to="/history">History</Link>}
        <Link to="/why">Why</Link>
        <Link to="/pricing">Pricing</Link>
        <Link to="/#engine">How it works</Link>
      </nav>

      {user ? (
        <div className="header-user">
          <Link to="/history" className="token-badge">
            {balance !== null ? `${balance} tokens` : '—'}
          </Link>
          <span className="user-email">{user.email}</span>
          <button onClick={handleLogout} className="btn btn-secondary btn-small">Sign out</button>
        </div>
      ) : (
        <Link to="/signin" className="btn btn-primary btn-small">
          Sign in
        </Link>
      )}

      <button
        className={`hamburger${menuOpen ? ' is-open' : ''}`}
        onClick={() => setMenuOpen(!menuOpen)}
        aria-label="Toggle menu"
        aria-expanded={menuOpen}
      >
        <span /><span /><span />
      </button>

      {menuOpen && (
        <div className="mobile-menu" onClick={() => setMenuOpen(false)}>
          <div className="mobile-menu-inner" onClick={(e) => e.stopPropagation()}>
            {user ? <Link to="/dashboard" className="mobile-link">Dashboard</Link> : <Link to="/" className="mobile-link">Home</Link>}
            {user && <Link to="/scan" className="mobile-link">Scan</Link>}
            {user && <Link to="/reports" className="mobile-link">Reports</Link>}
            {user && <Link to="/buy" className="mobile-link">Buy Tokens</Link>}
            {user && <Link to="/history" className="mobile-link">History</Link>}
            <Link to="/why" className="mobile-link">Why</Link>
            <Link to="/pricing" className="mobile-link">Pricing</Link>
            <Link to="/#engine" className="mobile-link">How it works</Link>
            <div className="mobile-menu-actions">
              {user ? (
                <button onClick={() => { handleLogout(); setMenuOpen(false); }} className="btn btn-secondary">Sign out</button>
              ) : (
                <Link to="/signin" className="btn btn-primary">Sign in</Link>
              )}
            </div>
          </div>
        </div>
      )}
    </header>
  );
}
