import { useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function AuthCallback() {
  const navigate = useNavigate();
  const { setUser } = useAuth();
  const [searchParams] = useSearchParams();

  useEffect(() => {
    // If backend redirected here with an error, send user to sign-in with message
    const error = searchParams.get('error');
    if (error) {
      navigate(`/signin?error=${encodeURIComponent(error)}`, { replace: true });
      return;
    }

    import('../api/authApi').then(({ getMe }) => {
      getMe()
        .then(({ data }) => {
          setUser(data);
          const next = sessionStorage.getItem('auth_next') || '/scan';
          sessionStorage.removeItem('auth_next');
          navigate(next, { replace: true });
        })
        .catch(() => {
          navigate('/signin?error=Session expired. Please sign in again.', { replace: true });
        });
    });
  }, [navigate, setUser, searchParams]);

  return (
    <div className="page-loading">
      <div className="reports-spinner" />
      <p>Signing you in...</p>
    </div>
  );
}
