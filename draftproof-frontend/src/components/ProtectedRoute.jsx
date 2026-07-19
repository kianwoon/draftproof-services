import { Navigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

// Renders children immediately instead of blocking on getMe() resolving first.
// The JWT cookie authenticates each API call independently server-side, so
// waiting for /api/auth/me before the page's own data fetch just adds a
// second sequential network round trip on every protected page (measured:
// ~450ms per round trip from a geographically distant client, so this was
// roughly doubling load time for /scan and /reports). If the session turns
// out to be invalid, the page's first API call gets a 401 and the existing
// global interceptor (draftproofApi.js) redirects to /signin — the same
// safety net already relied on for mid-session cookie expiry. This can
// briefly render a protected page's shell for a signed-out user hitting the
// URL directly, until getMe() resolves or the first API call 401s.
export default function ProtectedRoute({ children }) {
  const { user, loading } = useAuth();

  if (!loading && !user) return <Navigate to="/signin" replace />;

  return children;
}
