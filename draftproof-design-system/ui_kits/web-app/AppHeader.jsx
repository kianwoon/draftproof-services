// AppHeader — the signed-in product header (light variant). Brand + primary nav
// with a hover "Scan" dropdown, plus the user cluster: language switcher, token
// balance badge, avatar, sign out. `go(screen)` drives the kit's screen router;
// `active` highlights the current section. Markup mirrors production Header.jsx.
const { useState } = React;

function BrandMark() {
  return (
    <span className="brand-mark" aria-hidden="true">
      <svg viewBox="0 0 32 32" role="img">
        <path d="M16 3 27 7v8c0 7.2-4.6 11.8-11 14C9.6 26.8 5 22.2 5 15V7l11-4Z" />
        <path d="m10.8 15.9 3.4 3.3 7.4-8" />
      </svg>
    </span>
  );
}

function AppHeader({ go, active, balance, email }) {
  const [scanOpen, setScanOpen] = useState(false);
  const isScanActive = active === 'scan' || active === 'report';
  const initial = (email || '?').charAt(0).toUpperCase();

  return (
    <header className="site-header" aria-label="Main navigation">
      <a href="#" className="brand" aria-label="DraftProof home"
        onClick={(e) => { e.preventDefault(); go('dashboard'); }}>
        <BrandMark />
        <span>DraftProof</span>
      </a>

      <nav className="nav-links" aria-label="Primary">
        <a href="#" className={active === 'dashboard' ? 'is-current' : ''}
          onClick={(e) => { e.preventDefault(); go('dashboard'); }}>Dashboard</a>

        <div className="nav-dropdown"
          onMouseEnter={() => setScanOpen(true)}
          onMouseLeave={() => setScanOpen(false)}>
          <button className={`nav-dropdown-trigger${isScanActive ? ' active' : ''}`}
            onClick={() => setScanOpen((v) => !v)}>
            Scan
            <svg width="10" height="6" viewBox="0 0 10 6" style={{ marginLeft: 5, transition: 'transform .2s', transform: scanOpen ? 'rotate(180deg)' : 'rotate(0)' }}>
              <path d="M1 1l4 4 4-4" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" />
            </svg>
          </button>
          {scanOpen && (
            <div className="nav-dropdown-menu">
              <a href="#" className="nav-dropdown-item"
                onClick={(e) => { e.preventDefault(); setScanOpen(false); go('scan'); }}>
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                  <path d="M2 4.5A2.5 2.5 0 014.5 2h7A2.5 2.5 0 0114 4.5v7a2.5 2.5 0 01-2.5 2.5h-7A2.5 2.5 0 012 11.5v-7z" stroke="currentColor" strokeWidth="1.4"/>
                  <path d="M5 8h6M8 5v6" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"/>
                </svg>
                Scanning
              </a>
              <a href="#" className="nav-dropdown-item"
                onClick={(e) => { e.preventDefault(); setScanOpen(false); go('report'); }}>
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                  <path d="M3 2h7l3 3v8a1 1 0 01-1 1H4a1 1 0 01-1-1V3a1 1 0 011-1z" stroke="currentColor" strokeWidth="1.4"/>
                  <path d="M6 7h4M6 9.5h3" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/>
                </svg>
                View Reports
              </a>
            </div>
          )}
        </div>

        <a href="#" onClick={(e) => e.preventDefault()}>Buy Tokens</a>
        <a href="#" onClick={(e) => e.preventDefault()}>History</a>
        <a href="#" onClick={(e) => e.preventDefault()}>Pricing</a>
        <a href="#" onClick={(e) => e.preventDefault()}>FAQ</a>
      </nav>

      <div className="header-user">
        <span className="language-switcher language-switcher-compact" aria-hidden="true">
          <span>Language</span>
          <select defaultValue="en" aria-label="Language">
            <option value="en">EN</option>
            <option value="zh">中文</option>
          </select>
        </span>
        <a href="#" className="token-badge" onClick={(e) => e.preventDefault()}>
          {balance} tokens
        </a>
        <div className="user-avatar-placeholder">{initial}</div>
        <button onClick={() => go('signin')} className="btn btn-secondary btn-small">Sign out</button>
      </div>
    </header>
  );
}
