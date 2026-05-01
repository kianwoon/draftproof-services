import { useAuth } from '../context/AuthContext';
import { googleAuthUrl, microsoftAuthUrl } from '../api/authApi';
import { Navigate, useSearchParams, useNavigate } from 'react-router-dom';

export default function SignIn() {
  const { user, loading } = useAuth();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();

  if (loading) return (
    <div className="page-loading">
      <div className="reports-spinner" />
      <p>Loading...</p>
    </div>
  );
  if (user) return <Navigate to="/" replace />;

  const errorMsg = searchParams.get('error');
  const knownErrors = {
    'Session expired. Please sign in again.': true,
    'Access denied. Your email domain is not supported.': true,
    'Something went wrong during sign-in': true,
    'Something went wrong. Please try again.': true,
    'Invalid request. Please try again.': true,
    'Too many requests. Please wait a moment.': true,
  };
  const safeErrorMsg = errorMsg && knownErrors[errorMsg] ? errorMsg : null;
  const next = searchParams.get('next');
  if (next && !sessionStorage.getItem('auth_next')) {
    sessionStorage.setItem('auth_next', next);
  }

  return (
    <div className="container" style={{ paddingTop: 'calc(var(--header-h) + 4rem)' }}>
      <div className="signin-card">
        <h2 style={{ marginBottom: '0.5rem' }}>Sign in to DraftProof</h2>
        <p style={{ marginBottom: '2rem' }}>
          Use your Google or Microsoft account to continue.
        </p>

        {safeErrorMsg && (
          <div className="alert alert-error" style={{ marginBottom: '1.5rem' }}>
            {safeErrorMsg}
            <button
              onClick={() => navigate('/signin', { replace: true })}
              style={{ float: 'right', background: 'none', border: 'none', cursor: 'pointer', fontSize: '1.1rem', lineHeight: 1 }}
            >
              &times;
            </button>
          </div>
        )}

        <div className="signin-buttons">
          <a href={googleAuthUrl} className="btn btn-signin btn-google">
            <svg viewBox="0 0 24 24" width="20" height="20">
              <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z"/>
              <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
              <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
              <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
            </svg>
            Sign in with Google
          </a>

          <a href={microsoftAuthUrl} className="btn btn-signin btn-microsoft">
            <svg viewBox="0 0 24 24" width="20" height="20">
              <rect x="1" y="1" width="10" height="10" fill="#F25022"/>
              <rect x="13" y="1" width="10" height="10" fill="#7FBA00"/>
              <rect x="1" y="13" width="10" height="10" fill="#00A4EF"/>
              <rect x="13" y="13" width="10" height="10" fill="#FFB900"/>
            </svg>
            Sign in with Microsoft
          </a>
        </div>

        <p className="signin-note">
          Only Gmail and Hotmail/Outlook email addresses are accepted.
        </p>
      </div>
    </div>
  );
}
