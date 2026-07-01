import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { getMe, logout as authLogout } from '../api/authApi';
import api, { registerSessionExpiredHandler } from '../api/draftproofApi';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const navigate = useNavigate();
  const [user, setUser] = useState(null);
  // SSR-aware: on the server there is no session to resolve, so we render as
  // "loaded" so the Landing page prerenders with content (HomeRedirect returns
  // null while loading). On the client we still start in the loading state and
  // resolve via the effect below — identical to prior behavior. Safe because the
  // client mounts with createRoot (no hydration), so there is nothing to mismatch.
  const [loading, setLoading] = useState(() => typeof window !== 'undefined');
  const [balance, setBalance] = useState(null);
  const [reservedBalance, setReservedBalance] = useState(0);

  useEffect(() => {
    getMe()
      .then(({ data }) => setUser(data))
      .catch(() => setUser(null))
      .finally(() => setLoading(false));
  }, []);

  // Fired by the draftproofApi 401 interceptor when the session has expired
  // mid-use. Clears auth state (so ProtectedRoute also redirects declaratively)
  // and sends the user straight to sign-in — no ErrorReload reload countdown.
  // We stash the current location in auth_next so AuthCallback returns them here
  // after re-signing in, matching the existing Scan/Report expiry flow.
  const handleSessionExpired = useCallback(() => {
    setUser(null);
    setBalance(null);
    setReservedBalance(0);
    const { pathname, search } = window.location;
    const onSignIn = pathname === '/signin' || pathname === '/zh/signin';
    if (!onSignIn) {
      sessionStorage.setItem('auth_next', `${pathname}${search}`);
      navigate('/signin?error=Session expired. Please sign in again.', { replace: true });
    }
    // Best-effort server-side cookie clear; the session is already invalid, so
    // failure is fine. Runs after navigate so the redirect is never delayed.
    authLogout().catch(() => {});
  }, [navigate]);

  useEffect(() => {
    registerSessionExpiredHandler(handleSessionExpired);
    return () => registerSessionExpiredHandler(null);
  }, [handleSessionExpired]);

  const refreshBalance = useCallback(() => {
    if (!user) {
      setBalance(null);
      setReservedBalance(0);
      return;
    }
    api.get('/payments/balance')
      .then((r) => {
        const grossBalance = Number(r.data.balance) || 0;
        const reserved = Number(r.data.reserved) || 0;
        setBalance(Math.max(0, grossBalance - reserved));
        setReservedBalance(Math.max(0, reserved));
      })
      .catch((err) => {
        if (err.response?.status === 404) {
          setBalance(0);
          setReservedBalance(0);
          return;
        }
        setBalance(null);
        setReservedBalance(0);
      });
  }, [user]);

  useEffect(() => { refreshBalance(); }, [refreshBalance]);

  const logout = useCallback(async () => {
    try {
      await authLogout();
    } catch {
      // cookie may already be gone
    }
    setUser(null);
    setBalance(null);
    setReservedBalance(0);
  }, []);

  return (
    <AuthContext.Provider value={{ user, loading, logout, balance, reservedBalance, refreshBalance }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);
