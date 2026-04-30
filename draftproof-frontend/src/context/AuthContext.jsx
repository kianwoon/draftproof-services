import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { getMe, logout as authLogout } from '../api/authApi';
import api from '../api/draftproofApi';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [balance, setBalance] = useState(null);

  useEffect(() => {
    getMe()
      .then(({ data }) => setUser(data))
      .catch(() => setUser(null))
      .finally(() => setLoading(false));
  }, []);

  const refreshBalance = useCallback(() => {
    if (!user) { setBalance(null); return; }
    api.get('/payments/balance')
      .then(r => setBalance(r.data.balance))
      .catch(() => setBalance(null));
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
  }, []);

  return (
    <AuthContext.Provider value={{ user, loading, logout, balance, refreshBalance }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);
