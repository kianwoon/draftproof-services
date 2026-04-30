import { Link, Navigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function Dashboard() {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="container" style={{ paddingTop: 'calc(var(--header-h) + 4rem)', textAlign: 'center' }}>
        <p>Loading...</p>
      </div>
    );
  }

  if (!user) {
    return <Navigate to="/" replace />;
  }

  const initials = user.email?.charAt(0).toUpperCase() || '?';

  return (
    <main className="dash-shell">
      <div className="container">
        {/* Welcome */}
        <section className="dash-welcome">
          <div className="dash-avatar">{initials}</div>
          <div>
            <h1>Welcome back</h1>
            <p className="dash-email">{user.email}</p>
          </div>
        </section>

        {/* Quick actions */}
        <section className="dash-actions">
          <Link to="/scan" className="dash-action-card dash-action-primary">
            <div className="dash-action-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                <polyline points="14 2 14 8 20 8"/>
                <line x1="12" y1="18" x2="12" y2="12"/>
                <line x1="9" y1="15" x2="15" y2="15"/>
              </svg>
            </div>
            <div>
              <h3>New Scan</h3>
              <p>Paste your text for a pre-submission integrity review</p>
            </div>
          </Link>
        </section>

        {/* Getting started */}
        <section className="dash-section">
          <h2>Getting started</h2>
          <div className="dash-steps">
            <div className="dash-step">
              <span className="step-num">1</span>
              <div>
                <strong>Paste your text</strong>
                <p>Copy and paste your document text into the scan page</p>
              </div>
            </div>
            <div className="dash-step">
              <span className="step-num">2</span>
              <div>
                <strong>Review the scan results</strong>
                <p>See citation gaps, source integrity, and phrasing signals</p>
              </div>
            </div>
            <div className="dash-step">
              <span className="step-num">3</span>
              <div>
                <strong>Fix and resubmit</strong>
                <p>Use suggested rewrites to improve before submission</p>
              </div>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
