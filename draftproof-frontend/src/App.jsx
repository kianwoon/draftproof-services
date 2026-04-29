import { Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import Header from './components/Header';
import Footer from './components/Footer';
import ProtectedRoute from './components/ProtectedRoute';
import Landing from './pages/Landing';
import Dashboard from './pages/Dashboard';
import SignIn from './pages/SignIn';
import AuthCallback from './pages/AuthCallback';
import Scan from './pages/Scan';
import Report from './pages/Report';
import Reports from './pages/Reports';
import Pricing from './pages/Pricing';
import BuyTokens from './pages/BuyTokens';

function HomeRedirect() {
  const { user, loading } = useAuth();
  if (loading) return null;
  return user ? <Navigate to="/dashboard" replace /> : <Landing />;
}

export default function App() {
  const { pathname } = useLocation();
  const hideFooter = pathname === '/';

  return (
    <AuthProvider>
      <Header />
      <Routes>
        <Route path="/" element={<HomeRedirect />} />
        <Route path="/dashboard" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
        <Route path="/pricing" element={<Pricing />} />
        <Route path="/buy" element={<ProtectedRoute><BuyTokens /></ProtectedRoute>} />
        <Route path="/signin" element={<SignIn />} />
        <Route path="/auth/callback" element={<AuthCallback />} />
        <Route path="/scan" element={<ProtectedRoute><Scan /></ProtectedRoute>} />
        <Route path="/reports" element={<ProtectedRoute><Reports /></ProtectedRoute>} />
        <Route path="/report/:id" element={<ProtectedRoute><Report /></ProtectedRoute>} />
      </Routes>
      {!hideFooter && <Footer />}
    </AuthProvider>
  );
}
