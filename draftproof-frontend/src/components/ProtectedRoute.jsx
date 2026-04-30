import { Navigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function ProtectedRoute({ children }) {
  const { user, loading } = useAuth();

  if (loading) return <div className="container" style={{paddingTop: 'calc(var(--header-h) + 3rem)'}}><p>Loading...</p></div>;
  if (!user) return <Navigate to="/signin" replace />;

  return children;
}
