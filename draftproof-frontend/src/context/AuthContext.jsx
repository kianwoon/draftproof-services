import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { getMe, logout as authLogout } from '../api/authApi';
import api from '../api/draftproofApi';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [balance, setBalance] = useState(null);
  const [reservedBalance, setReservedBalance] = useState(0);

  useEffect(() => {
    getMe()
      .then(({ data }) => setUser(data))
      .catch(() => setUser(null))
      .finally(() => setLoading(false));
  }, []);

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
