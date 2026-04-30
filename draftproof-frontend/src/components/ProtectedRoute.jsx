import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function ProtectedRoute({ children }) {
  const { user, loading } = useAuth();
  const location = useLocation();

  if (loading) return <div className="container" style={{paddingTop: 'calc(var(--header-h) + 3rem)'}}><p>Loading...</p></div>;
  if (!user) {
    sessionStorage.setItem('auth_next', location.pathname);
    const msg = 'Session expired. Please sign in again.';
    return <Navigate to={`/signin?error=${encodeURIComponent(msg)}`} replace />;
  }

  return children;
}
