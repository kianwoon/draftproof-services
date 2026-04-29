import { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import api from '../api/draftproofApi';

export default function Header() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [balance, setBalance] = useState(null);

  useEffect(() => {
    if (!user) { setBalance(null); return; }
    api.get('/payments/balance')
      .then(r => setBalance(r.data.balance))
      .catch(() => setBalance(null));
  }, [user]);

  const handleLogout = async () => {
    await logout();
    navigate('/', { replace: true });
  };

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
        {user && <Link to="/scan">Scan</Link>}
        {user && <Link to="/buy">Buy Tokens</Link>}
        <Link to="/pricing">Pricing</Link>
        <a href="#engine">How it works</a>
      </nav>

      {user ? (
        <div className="header-user">
          <Link to="/buy" className="token-badge">
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
    </header>
  );
}
