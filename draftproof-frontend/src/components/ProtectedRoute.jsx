import { Navigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useAuth } from '../context/AuthContext';

export default function ProtectedRoute({ children }) {
  const { user, loading } = useAuth();
  const { t } = useTranslation();

  if (loading) return (
    <div className="page-loading">
      <div className="reports-spinner" />
      <p>{t('common.loading')}</p>
    </div>
  );
  if (!user) return <Navigate to="/signin" replace />;

  return children;
}
