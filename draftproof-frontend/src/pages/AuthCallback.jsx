import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function AuthCallback() {
  const navigate = useNavigate();
  const { setUser } = useAuth();

  useEffect(() => {
    import('../api/authApi').then(({ getMe }) => {
      getMe()
        .then(({ data }) => {
          setUser(data);
          navigate('/scan', { replace: true });
        })
        .catch(() => navigate('/signin', { replace: true }));
    });
  }, [navigate, setUser]);

  return (
    <div className="container" style={{ paddingTop: 'calc(var(--header-h) + 4rem)' }}>
      <p>Signing you in...</p>
    </div>
  );
}
