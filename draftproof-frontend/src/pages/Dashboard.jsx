import { Link, Navigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import CodeTexture from '../components/CodeTexture';

export default function Dashboard() {
  const { user, loading, balance } = useAuth();

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
  const tokenLabel = balance === null ? 'Checking' : `${balance} token${balance === 1 ? '' : 's'}`;

  return (
    <main className="dash-shell">
      <div className="container">
        <section className="dash-hero">
          <CodeTexture id="dashboardHero" />
          <div className="dash-welcome">
            {user.avatar_url ? (
              <img src={user.avatar_url} alt={user.email} className="dash-avatar dash-avatar-img" />
            ) : (
              <div className="dash-avatar">{initials}</div>
            )}
            <div>
              <p className="eyebrow">DraftProof Workspace</p>
              <h1>Welcome back</h1>
              <p className="dash-email">{user.email}</p>
            </div>
          </div>

          <div className="dash-hero-panel" aria-label="Account summary">
            <div>
              <span>Available balance</span>
              <strong>{tokenLabel}</strong>
            </div>
            <Link to="/buy" className="btn btn-ghost btn-small">Buy tokens</Link>
          </div>
        </section>

        <section className="dash-grid" aria-label="Dashboard actions">
          <Link to="/scan" className="dash-primary-card">
            <div>
              <span className="brand-pill">Pre-submission review</span>
              <h2>Start a new integrity scan</h2>
              <p>
                Paste your draft and check citation gaps, source grounding, generic phrasing,
                and review-only authorship signals before submission.
              </p>
            </div>
            <div className="dash-card-footer">
              <span>1 token per 1,000 words</span>
              <strong>Start scan</strong>
            </div>
          </Link>

          <div className="dash-side-stack">
            <Link to="/reports" className="dash-small-card">
              <span className="dash-action-icon" aria-hidden="true">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                  <path d="M7 3.8h7.2L18 7.6v12.6H7V3.8Z" />
                  <path d="M14 3.8v4h4M9.5 11h5M9.5 14h5M9.5 17h3" />
                </svg>
              </span>
              <div>
                <h3>View reports</h3>
                <p>Return to prior scans and download report PDFs.</p>
              </div>
            </Link>

            <Link to="/history" className="dash-small-card">
              <span className="dash-action-icon" aria-hidden="true">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                  <path d="M12 6v6l3.5 2" />
                  <path d="M20 12a8 8 0 1 1-2.35-5.65" />
                  <path d="M20 4v5h-5" />
                </svg>
              </span>
              <div>
                <h3>Purchase history</h3>
                <p>Review token purchases and account activity.</p>
              </div>
            </Link>
          </div>
        </section>

        <section className="dash-section">
          <div className="dash-section-heading">
            <p className="eyebrow">Workflow</p>
            <h2>From draft to defensible report</h2>
          </div>

          <ol className="dash-steps">
            <li className="dash-step">
              <span className="step-num">1</span>
              <strong>Paste your text</strong>
              <p>Use the scan page for draft text you want checked before submission.</p>
            </li>
            <li className="dash-step">
              <span className="step-num">2</span>
              <strong>Review the signals</strong>
              <p>Check citation gaps, source integrity, grounding, and phrasing signals.</p>
            </li>
            <li className="dash-step">
              <span className="step-num">3</span>
              <strong>Fix what matters</strong>
              <p>Use report guidance to strengthen evidence, clarity, and academic tone.</p>
            </li>
            <li className="dash-step">
              <span className="step-num">4</span>
              <strong>Keep the report</strong>
              <p>Save the PDF as a review trail for yourself, your class, or your team.</p>
            </li>
          </ol>
        </section>
      </div>
    </main>
  );
}
